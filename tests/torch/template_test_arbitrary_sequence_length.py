#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from mlstm_kernels.torch.kernel_wrappers import (
    wrap_chunkwise__arbitrary_sequence_length,
)


def template_test_wrap_chunkwise__arbitrary_sequence_length(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
    dtype_inputs: str,
    dtype_state: str = "float32",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    eps: float = 1e-6,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    torch.manual_seed(42)
    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    i = torch.randn(B, NH, S)
    f = 3.0 + torch.randn(B, NH, S)

    dtype_inputs = getattr(torch, dtype_inputs)
    device = torch.device(device)
    (q, k, v, i, f) = tuple(
        map(lambda x: x.to(dtype=dtype_inputs, device=device), (q, k, v, i, f))
    )

    # run the parallel baseline
    h_parallel_bl = parallel_baseline(q, k, v, i, f, eps=eps)
    # run the sequence baseline
    h_seq_bl, (c_last_seq_bl, n_last_seq_bl, m_last_seq_bl) = sequence_baseline(
        q, k, v, i, f, return_last_states=True, eps=eps
    )

    # create the arbitrary sequence length chunkwise function
    chunkwise_arbitrary_seq_len_fn = partial(
        wrap_chunkwise__arbitrary_sequence_length,
        mlstm_chunkwise_kernel=chunkwise_target,
        mlstm_sequence_kernel=partial(sequence_target, dtype_state=getattr(torch, dtype_state)),
        mlstm_step_kernel=partial(step_target, dtype_state=getattr(torch, dtype_state)),
        dtype_state=getattr(torch, dtype_state)
    )

    # run the chunkwise arbitrary seq_len fn
    h_cw_absl, (c_last_cw_absl, n_last_cw_absl, m_last_cw_absl) = (
        chunkwise_arbitrary_seq_len_fn(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            return_last_states=True,
            chunk_size=chunk_size,
            eps=eps,
        )
    )

    assert c_last_cw_absl.dtype == getattr(torch, dtype_state), f"Got: {c_last_cw_absl.dtype}, Expected: {getattr(torch, dtype_state)}"
    assert n_last_cw_absl.dtype == getattr(torch, dtype_state)
    assert m_last_cw_absl.dtype == getattr(torch, dtype_state)

    h_parallel_bl = h_parallel_bl.cpu().detach().numpy()
    h_seq_bl = h_seq_bl.cpu().detach().numpy()
    h_cw_absl = h_cw_absl.cpu().detach().numpy()

    c_last_seq_bl = c_last_seq_bl.cpu().detach().numpy()
    n_last_seq_bl = n_last_seq_bl.cpu().detach().numpy()
    m_last_seq_bl = m_last_seq_bl.cpu().detach().numpy()

    c_last_cw_absl = c_last_cw_absl.float().cpu().detach().numpy()
    n_last_cw_absl = n_last_cw_absl.float().cpu().detach().numpy()
    m_last_cw_absl = m_last_cw_absl.float().cpu().detach().numpy()


    # match baselines
    np.testing.assert_allclose(
        h_parallel_bl,
        h_seq_bl,
        rtol=rtol,
        atol=atol,
        err_msg="h_out of the two baselines parallel and sequence do not match.",
    )

    # match chunkwise to parallel
    np.testing.assert_allclose(
        h_cw_absl,
        h_parallel_bl,
        rtol=rtol,
        atol=atol,
        err_msg="h_out chunkwise and parallel do not match.",
    )

    # match chunkwise to sequence
    np.testing.assert_allclose(
        h_cw_absl,
        h_seq_bl,
        rtol=rtol,
        atol=atol,
        err_msg="h_out chunkwise and sequence do not match.",
    )
    np.testing.assert_allclose(
        c_last_cw_absl,
        c_last_seq_bl,
        rtol=rtol,
        atol=atol,
        err_msg="c_last chunkwise and sequence do not match.",
    )
    np.testing.assert_allclose(
        n_last_cw_absl,
        n_last_seq_bl,
        rtol=rtol,
        atol=atol,
        err_msg="n_last chunkwise and sequence do not match.",
    )
    np.testing.assert_allclose(
        m_last_cw_absl,
        m_last_seq_bl,
        rtol=rtol,
        atol=atol,
        err_msg="m_last chunkwise and sequence do not match.",
    )


def template_test_wrap_chunkwise__arbitrary_sequence_length_single_step(
    B: int,
    NH: int,
    DHQK: int,
    DHHV: int,
    step_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
    dtype: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    eps: float = 1e-6,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function, for a single step."""
    torch.manual_seed(42)
    S = 1

    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    i = torch.randn(B, NH, S)
    f = 3.0 + torch.randn(B, NH, S)
    c = torch.zeros(B, NH, DHQK, DHHV)
    n = torch.zeros(B, NH, DHQK)
    m = torch.zeros(B, NH, 1)

    dtype = torch.float32
    device = torch.device(device)
    (q, k, v, i, f, c, n, m) = tuple(
        map(lambda x: x.to(dtype=dtype, device=device), (q, k, v, i, f, c, n, m))
    )

    # run the step baseline
    h_step_bl, (c_step_bl, n_step_bl, m_step_bl) = step_baseline(
        q=q.squeeze(2), k=k.squeeze(2), v=v.squeeze(2), i=i, f=f, c=c, n=n, m=m, eps=eps
    )
    h_step_bl = h_step_bl[:, :, None, :]

    # create the arbitrary sequence length chunkwise function
    chunkwise_arbitrary_seq_len_fn = partial(
        wrap_chunkwise__arbitrary_sequence_length,
        mlstm_chunkwise_kernel=chunkwise_target,
        mlstm_sequence_kernel=sequence_target,
        mlstm_step_kernel=step_target,
    )

    # run the chunkwise arbitrary seq_len fn
    h_cw_absl, (c_last_cw_absl, n_last_cw_absl, m_last_cw_absl) = (
        chunkwise_arbitrary_seq_len_fn(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
            return_last_states=True,
            chunk_size=64,
        )
    )

    h_cw_absl = h_cw_absl.cpu().detach().numpy()

    c_last_cw_absl = c_last_cw_absl.cpu().detach().numpy()
    n_last_cw_absl = n_last_cw_absl.cpu().detach().numpy()
    m_last_cw_absl = m_last_cw_absl.cpu().detach().numpy()

    h_step_bl = h_step_bl.cpu().detach().numpy()
    c_step_bl = c_step_bl.cpu().detach().numpy()
    n_step_bl = n_step_bl.cpu().detach().numpy()
    m_step_bl = m_step_bl.cpu().detach().numpy()

    # match to step baseline
    np.testing.assert_allclose(
        h_cw_absl,
        h_step_bl,
        rtol=rtol,
        atol=atol,
        err_msg="h_out chunkwise and step do not match.",
    )
    np.testing.assert_allclose(
        c_last_cw_absl,
        c_step_bl,
        rtol=rtol,
        atol=atol,
        err_msg="c_last chunkwise and step do not match.",
    )
    np.testing.assert_allclose(
        n_last_cw_absl,
        n_step_bl,
        rtol=rtol,
        atol=atol,
        err_msg="n_last chunkwise and step do not match.",
    )
    np.testing.assert_allclose(
        m_last_cw_absl,
        m_step_bl,
        rtol=rtol,
        atol=atol,
        err_msg="m_last chunkwise and step do not match.",
    )
