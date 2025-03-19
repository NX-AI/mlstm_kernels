#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from collections.abc import Callable

import torch

LOGGER = logging.getLogger(__name__)


def wrap_chunkwise__arbitrary_sequence_length(
    mlstm_chunkwise_kernel: Callable,
    mlstm_sequence_kernel: Callable,
    mlstm_step_kernel: Callable,
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    f: torch.Tensor,  # (B, NH, S)
    i: torch.Tensor,  # (B, NH, S)
    c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None,  # (B, NH, DHQK)
    m_initial: torch.Tensor = None,  # (B, NH, 1)
    return_last_states: bool = True,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
    enable_logging: bool = False,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):  # matH (B, NH, S, DHHV), tuple[matC_state_last (B, NH, DHQK, DHHV), vecN_states_last (B, NH, DHQK), scaMinter_states_last (B, NH, 1)]
    """This function computes the last hidden state and matH outputs of the mLSTM, independently of the sequence length.

    For this it uses three kernels:
    - mlstm_chunkwise_kernel: mlstm chunkwise kernels that processes chunks of a given chunk size in parallel.
    - mlstm_sequence_kernel: mlstm kernel that processes the remaining sequence length in a single step recurrence.
    - mlstm_step_kernel: mlstm kernel that processes a sequence length of 1 in a single step.

    It tries to maximize the chunksizes to improve performance.
    It will start with the given chunk size and then divides the chunksize by 2 until the chunk size is smaller than 16.
    At every chunksize it will process the maximal number of chunks that fit into the remaining sequence length.

    E.g. for chunk_size = 64, this function will try the chunksizes [64, 32, 16] if necessary.

    For the remaining sequence length, which is smaller than 16, we use a different kernel that computes the mLSTM
    in a single step and loop over this in pytorch.

    Args:
        mlstm_chunkwise_kernel: The mLSTM chunkwise kernel that processes chunks of a given chunk size in parallel
        mlstm_sequence_kernel: The mLSTM kernel that processes the remaining sequence length in a single step recurrence
        q: The query tensor (B, NH, S, DHQK)
        k: The key tensor (B, NH, S, DHQK)
        v: The value tensor (B, NH, S, DHHV)
        f: The forget gate tensor (B, NH, S)
        i: The input gate tensor (B, NH, S)
        c_initial: The initial cell state tensor (B, NH, DHQK, DHHV)
        n_initial: The initial hidden state tensor (B, NH, DHQK)
        m_initial: The initial memory state tensor (B, NH, 1)
        return_last_states: If True, the function will return the last states of the mLSTM
        eps: The epsilon value used for numerical stability
        autocast_kernel_dtype: The dtype used for the kernel computation
        chunk_size: The chunk size used for the chunkwise kernel
        enable_logging: If True, the function will log debug information. Default is False.

    Returns:
        The last hidden state tensor (B, NH, S, DHHV) or a tuple containing the last hidden state tensor and the last states of the mLSTM
        Last states are (c (B, NH, DHQK, DHHV), n (B, NH, DHQK), m (B, NH, 1)).
    """

    B, NH, S, DHQK = k.shape
    DHHV = v.shape[-1]

    chunk_sizes = []
    kcs = chunk_size
    while kcs >= 16:
        chunk_sizes.append(kcs)
        kcs = kcs // 2

    # Note: we are in a compiled region, so we cannot log without a graph break
    # therefore we make this optional
    LOGGER.debug(f"Trying chunk_sizes={chunk_sizes}") if enable_logging else None

    c_state = (
        c_initial
        if c_initial is not None
        else torch.zeros(B, NH, DHQK, DHHV, device=k.device, dtype=torch.float32)
    )
    n_state = (
        n_initial
        if n_initial is not None
        else torch.zeros(B, NH, DHQK, device=k.device, dtype=torch.float32)
    )
    m_state = (
        m_initial
        if m_initial is not None
        else torch.zeros(B, NH, 1, device=k.device, dtype=torch.float32)
    )

    if S > 1:
        # process the sequence length in chunks
        LOGGER.debug(
            "Regular Mode: Processing sequence length in chunks"
        ) if enable_logging else None
        h_outs = []
        seq_len_start_idx = 0
        for chunk_size_iter in chunk_sizes:
            LOGGER.debug(
                f"c_state.shape={c_state.shape}, n_state.shape={n_state.shape}, m_state.shape={m_state.shape}"
            ) if enable_logging else None
            LOGGER.debug(
                f"c_state.stride(0)= {c_state.stride(0)}, c_state.stride(1)= {c_state.stride(1)}, c_state.stride(2)= {c_state.stride(2)}, matC_cur.stride(3)= {c_state.stride(3)}"
            ) if enable_logging else None
            remaining_seq_len = S - seq_len_start_idx
            num_chunks = remaining_seq_len // chunk_size_iter
            if num_chunks == 0:
                continue
            LOGGER.debug(
                f"chunk_size={chunk_size_iter}, remaining_seq_len={remaining_seq_len}"
            ) if enable_logging else None
            if remaining_seq_len < chunk_size_iter:
                LOGGER.debug(
                    f"Skipping chunk_size={chunk_size_iter} as remaining_seq_len={remaining_seq_len} < {chunk_size_iter}"
                ) if enable_logging else None
                continue
            iter_seq_len = chunk_size_iter * num_chunks
            seq_len_idx = seq_len_start_idx + iter_seq_len
            LOGGER.debug(
                f"Mid OR Final: compute last state for seq[{seq_len_start_idx}:{seq_len_idx}], NC={num_chunks}, chunk_size={chunk_size_iter}"
            ) if enable_logging else None
            h_out, (c_state, n_state, m_state) = mlstm_chunkwise_kernel(
                q=q[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                k=k[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                v=v[..., seq_len_start_idx:seq_len_idx, :].contiguous(),
                f=f[..., seq_len_start_idx:seq_len_idx].contiguous(),
                i=i[..., seq_len_start_idx:seq_len_idx].contiguous(),
                c_initial=c_state,
                n_initial=n_state,
                m_initial=m_state,
                chunk_size=chunk_size_iter,
                return_last_states=True,
                autocast_kernel_dtype=autocast_kernel_dtype,
                eps=eps,
            )
            seq_len_start_idx += iter_seq_len
            h_outs.append(h_out)
            if remaining_seq_len % chunk_size_iter == 0:
                LOGGER.debug(
                    f"Finished processing sequence length in chunks, seq_len_start_idx={seq_len_start_idx}, S={S}"
                ) if enable_logging else None
                break

        remaining_seq_len = S - seq_len_start_idx

        if remaining_seq_len > 0:
            LOGGER.debug(
                f"Final: Recurrent step mode: compute last state for seq[{seq_len_start_idx}:{S}], remaining_seq_len={remaining_seq_len}"
            ) if enable_logging else None
            # we use here matK as q as this kernel does not need a query, since we do not care about the outputs only about the last state
            h_out, (c_state, n_state, m_state) = mlstm_sequence_kernel(
                q=q[..., seq_len_start_idx:S, :].contiguous(),
                k=k[..., seq_len_start_idx:S, :].contiguous(),
                v=v[..., seq_len_start_idx:S, :].contiguous(),
                i=i[..., seq_len_start_idx:S].contiguous(),
                f=f[..., seq_len_start_idx:S].contiguous(),
                c_initial=c_state,
                n_initial=n_state,
                m_initial=m_state,
                return_last_states=True,
                eps=eps,
            )
            h_outs.append(h_out)
        h_out = torch.concatenate(h_outs, dim=2)

    else:
        assert (
            S == 1
        ), f"Received empty sequence (S={S}), require at least single element in the sequence."
        # process the sequence length in a single step
        # while this case is also captured by the regular mode above,
        # it avoids the overhead of the loop and calls the step kernel directly
        LOGGER.debug(
            "Single step mode: Processing sequence length in a single step"
        ) if enable_logging else None
        # The step function does not want a sequence dimension
        # qkv shape is (B, NH, DHQK/DHV)
        # i, f shape is (B, NH, 1)
        h_out, (c_state, n_state, m_state) = mlstm_step_kernel(
            q=q.squeeze(2),
            k=k.squeeze(2),
            v=v.squeeze(2),
            i=i,
            f=f,
            c=c_state,
            n=n_state,
            m=m_state,
            eps=eps,
        )
        h_out = h_out[:, :, None, :]

    if return_last_states:
        return h_out, (c_state, n_state, m_state)
    else:
        return h_out


def wrap_chunkwise__pad_zeros(
    mlstm_chunkwise_kernel: Callable,
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    f: torch.Tensor,  # (B, NH, S)
    i: torch.Tensor,  # (B, NH, S)
    c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None,  # (B, NH, DHQK)
    m_initial: torch.Tensor = None,  # (B, NH, 1)
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.bfloat16,
    chunk_size: int = 64,
    **kwargs,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    assert not return_last_states, (
        "We are padding zeros, so we cannot return last states,",
        "as they would be not the true last states.",
    )

    B, NH, S, DHQK = q.shape  # (B, NH, S, DHQK)
    S_unpadded = S
    # padding to chunk size for kernels
    if S % chunk_size != 0:
        S_padded = ((S + chunk_size - 1) // chunk_size) * chunk_size
        q_pad = q.new_zeros(B, NH, S_padded, q.shape[3])
        k_pad = k.new_zeros(B, NH, S_padded, k.shape[3])
        v_pad = v.new_zeros(B, NH, S_padded, v.shape[3])
        i_pad = i.new_zeros(B, NH, S_padded)
        f_pad = f.new_zeros(B, NH, S_padded)
        q_pad[:, :, :S_unpadded, :] = q
        k_pad[:, :, :S_unpadded, :] = k
        v_pad[:, :, :S_unpadded, :] = v
        i_pad[:, :, :S_unpadded] = i
        f_pad[:, :, :S_unpadded] = f
    else:
        q_pad = q
        k_pad = k
        v_pad = v
        i_pad = i
        f_pad = f

    matH = mlstm_chunkwise_kernel(
        q=q_pad,
        k=k_pad,
        v=v_pad,
        i=i_pad,
        f=f_pad,
        c_initial=c_initial,
        n_initial=n_initial,
        m_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        autocast_kernel_dtype=autocast_kernel_dtype,
        chunk_size=chunk_size,
        **kwargs,
    )
    matH = matH[:, :, :S_unpadded, :]
    return matH
