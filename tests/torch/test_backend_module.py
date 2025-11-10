#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import pytest
import torch

from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig


def template_test_backend_module(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    mode: str,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
    dtype: str,
    device: str,
    return_last_states: bool = True,
    inference_state_dtype: str = "float32",
):
    torch.manual_seed(42)
    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    i = torch.randn(B, NH, S)
    f = 3.0 + torch.randn(B, NH, S)

    dtype = getattr(torch, dtype)
    device = torch.device(device)
    (q, k, v, i, f) = tuple(
        map(lambda x: x.to(dtype=dtype, device=device), (q, k, v, i, f))
    )

    config = mLSTMBackendConfig(
        chunkwise_kernel=chunkwise_kernel,
        sequence_kernel=sequence_kernel,
        step_kernel=step_kernel,
        chunk_size=chunk_size,
        eps=1e-6,
        mode=mode,
        return_last_states=return_last_states,
        inference_state_dtype=inference_state_dtype,
    )
    print("config", config)

    backend = mLSTMBackend(config)
    expected_h_shape = (B, NH, S, DHHV)
    expected_c_state_shape = (B, NH, DHQK, DHHV)
    expected_n_state_shape = (B, NH, DHQK)
    expected_m_state_shape = (B, NH, 1)

    if return_last_states:
        h, (c_state, n_state, m_state) = backend(q, k, v, i, f)
        assert (
            c_state.shape == expected_c_state_shape
        ), f"C state shape mismatch. Expected: {expected_c_state_shape}, got: {c_state.shape}"
        assert (
            n_state.shape == expected_n_state_shape
        ), f"N state shape mismatch. Expected: {expected_n_state_shape}, got: {n_state.shape}"
        assert (
            m_state.shape == expected_m_state_shape
        ), f"M state shape mismatch. Expected: {expected_m_state_shape}, got: {m_state.shape}"

        if mode == "inference":
            assert (
                c_state.dtype == getattr(torch, inference_state_dtype)
            ), f"C state dtype mismatch. Expected: {inference_state_dtype}, got: {c_state.dtype}"
            assert (
                n_state.dtype == getattr(torch, inference_state_dtype)
            ), f"N state dtype mismatch. Expected: {inference_state_dtype}, got: {n_state.dtype}"
            assert (
                m_state.dtype == getattr(torch, inference_state_dtype)
            ), f"M state dtype mismatch. Expected: {inference_state_dtype}, got: {m_state.dtype}"

    else:
        h = backend(q, k, v, i, f)

    assert (
        h.shape == expected_h_shape
    ), f"H shape mismatch. Expected: {expected_h_shape}, got: {h.shape}"

    assert h.dtype == dtype, f"H dtype mismatch. Expected: {dtype}, got: {h.dtype}"


@pytest.mark.parametrize(
    "B, NH, S, DHQK, DHHV, chunk_size",
    [[1, 2, 128, 64, 128, 64], [1, 2, 256, 128, 128, 64]],
)
@pytest.mark.parametrize("mode", ["train"])
@pytest.mark.parametrize(
    "chunkwise_kernel",
    [
        "chunkwise--native_autograd",
        "chunkwise--native_custbw",
        "chunkwise--triton_limit_chunk",
        "chunkwise--triton_xl_chunk",
    ],
)
@pytest.mark.parametrize(
    "sequence_kernel",
    ["native_sequence__native"],
)
@pytest.mark.parametrize(
    "step_kernel",
    ["native"],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("device", ["cuda"])
def test_backend_module_train(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    mode: str,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
    dtype: str,
    device: str,
):
    template_test_backend_module(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        mode=mode,
        chunkwise_kernel=chunkwise_kernel,
        sequence_kernel=sequence_kernel,
        step_kernel=step_kernel,
        dtype=dtype,
        device=device,
    )


@pytest.mark.parametrize(
    "B, NH, S, DHQK, DHHV, chunk_size",
    [[1, 2, 40, 64, 128, 64], [1, 2, 1, 128, 128, 64]],
)
@pytest.mark.parametrize("mode", ["train_with_padding"])
@pytest.mark.parametrize(
    "chunkwise_kernel",
    [
        "chunkwise--triton_limit_chunk",
        "chunkwise--triton_xl_chunk",
    ],
)
@pytest.mark.parametrize(
    "sequence_kernel",
    ["native_sequence__native"],
)
@pytest.mark.parametrize(
    "step_kernel",
    ["native"],
)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("device", ["cuda"])
def test_backend_module_train_with_padding(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    mode: str,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
    dtype: str,
    device: str,
):
    template_test_backend_module(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        mode=mode,
        chunkwise_kernel=chunkwise_kernel,
        sequence_kernel=sequence_kernel,
        step_kernel=step_kernel,
        dtype=dtype,
        device=device,
        return_last_states=False,
    )


@pytest.mark.parametrize(
    "B, NH, S, DHQK, DHHV, chunk_size",
    [
        [1, 2, 43, 64, 128, 64],
        [1, 2, 1, 128, 128, 64],
    ],
)
@pytest.mark.parametrize("mode", ["inference"])
@pytest.mark.parametrize(
    "chunkwise_kernel",
    [
        "chunkwise--triton_limit_chunk",
        "chunkwise--triton_xl_chunk",
    ],
)
@pytest.mark.parametrize(
    "sequence_kernel",
    [
        "native_sequence__native",
        "native_sequence__triton",
    ],
)
@pytest.mark.parametrize(
    "step_kernel",
    [
        "native",
        "triton",
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "float16"])
@pytest.mark.parametrize("inference_state_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_backend_module_inference(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    mode: str,
    chunkwise_kernel: str,
    sequence_kernel: str,
    step_kernel: str,
    dtype: str,
    inference_state_dtype: str,
    device: str,
):
    template_test_backend_module(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        mode=mode,
        chunkwise_kernel=chunkwise_kernel,
        sequence_kernel=sequence_kernel,
        step_kernel=step_kernel,
        dtype=dtype,
        device=device,
        return_last_states=True,
        inference_state_dtype=inference_state_dtype,
    )
