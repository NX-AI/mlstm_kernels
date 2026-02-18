#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import numpy as np
from collections.abc import Callable
from mlstm_kernels.torch.chunkwise import (
    mlstm_chunkwise__limit_chunk,
)

from mlstm_kernels.torch.recurrent import (
    mlstm_recurrent_sequence__triton_step_fused_fw,
)


def template_test_keep_state(
    B: int,
    NH: int,
    S: int,
    seq_pad: int,
    DHQK: int,
    DHHV: int,
    chunkwise_kernel: Callable,
    sequence_kernel: Callable,
    chunk_size: int = 64,
    dtype_inputs: str = "float32",
    device: str = "cuda",
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> None:
    torch.manual_seed(42)

    # baseline inputs
    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    i = torch.randn(B, NH, S)
    f = 3.0 + torch.randn(B, NH, S)

    # start with no (zero) initial state
    c_initial = torch.zeros(B, NH, DHQK, DHHV)
    n_initial = torch.zeros(B, NH, DHQK)
    m_initial = torch.zeros(B, NH, 1)

    # padded inputs
    S_padded = S + seq_pad
    qkv_pad_value = 0.0
    igate_pad_value = -25.0
    fgate_pad_value = 25.0

    q_padded = torch.full((B, NH, S_padded, DHQK), fill_value=qkv_pad_value)
    k_padded = torch.full((B, NH, S_padded, DHQK), fill_value=qkv_pad_value)
    v_padded = torch.full((B, NH, S_padded, DHHV), fill_value=qkv_pad_value)
    i_padded = torch.full((B, NH, S_padded), fill_value=igate_pad_value)
    f_padded = torch.full((B, NH, S_padded), fill_value=fgate_pad_value)
    q_padded[:, :, :S, :] = q
    k_padded[:, :, :S, :] = k
    v_padded[:, :, :S, :] = v
    i_padded[:, :, :S] = i
    f_padded[:, :, :S] = f

    dtype_inputs = getattr(torch, dtype_inputs)
    device = torch.device(device)
    (q, k, v, i, f) = tuple(
        map(lambda x: x.to(dtype=dtype_inputs, device=device), (q, k, v, i, f))
    )
    (c_initial, n_initial, m_initial) = tuple(
        map(
            lambda x: x.to(dtype=torch.float32, device=device),
            (c_initial, n_initial, m_initial),
        )
    )

    (q_padded, k_padded, v_padded, i_padded, f_padded) = tuple(
        map(
            lambda x: x.to(dtype=dtype_inputs, device=device),
            (q_padded, k_padded, v_padded, i_padded, f_padded),
        )
    )

    # run the chunkwise kernel with the padded inputs and initial states
    h_padded, (c_last_padded, n_last_padded, m_last_padded) = chunkwise_kernel(
        q=q_padded,
        k=k_padded,
        v=v_padded,
        i=i_padded,
        f=f_padded,
        c_initial=c_initial,
        n_initial=n_initial,
        m_initial=m_initial,
        return_last_states=True,
        chunk_size=chunk_size,
        autocast_kernel_dtype=torch.float32,
        eps=1e-6,
    )

    # run the sequence kernel with the unpadded inputs and the last states from the padded chunkwise kernel
    h_sequence, (c_last_sequence, n_last_sequence, m_last_sequence) = sequence_kernel(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        c_initial=c_initial,
        n_initial=n_initial,
        m_initial=m_initial,
        return_last_states=True,
        eps=1e-6,
        dtype_state=torch.float32,
    )

    c_last_padded = c_last_padded.cpu().detach().numpy()
    n_last_padded = n_last_padded.cpu().detach().numpy()
    m_last_padded = m_last_padded.cpu().detach().numpy()

    c_last_sequence = c_last_sequence.cpu().detach().numpy()
    n_last_sequence = n_last_sequence.cpu().detach().numpy()
    m_last_sequence = m_last_sequence.cpu().detach().numpy()

    # compare the last states from the padded chunkwise kernel to the sequence kernel
    np.testing.assert_allclose(c_last_padded, c_last_sequence, rtol=rtol, atol=atol)
    np.testing.assert_allclose(n_last_padded, n_last_sequence, rtol=rtol, atol=atol)
    np.testing.assert_allclose(m_last_padded, m_last_sequence, rtol=rtol, atol=atol)


def test_keep_state():
    """This tests verifies that we can pad variable sequences to a fixed length and still get the correct final states."""
    template_test_keep_state(
        B=1,
        NH=1,
        S=50,
        seq_pad=1024 - 50,
        DHQK=64,
        DHHV=128,
        chunkwise_kernel=mlstm_chunkwise__limit_chunk,
        sequence_kernel=mlstm_recurrent_sequence__triton_step_fused_fw,
        dtype_inputs="float32",
        device="cuda",
        atol=1e-3,
        rtol=5e-3,
    )
