#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging

import pytest
import torch

from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton_xl_chunk"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_triton_chunkwise_xl_chunk_vs_native_parallel_stablef_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=mlstm_chunkwise__xl_chunk,
        baseline_name="native_parallel_stablef_custbw",
        target_name="triton_chunkwise_xl_chunk",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=2e-2,
        rtol_fw=5e-2,
        atol_fwbw=42e-1,  # we need to increase this tolerance for vecF.grad (max diff val 0.267...)
        rtol_fwbw=0.5,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_inter_vs_intra_chunks(S, B, NH, DHQK, DHHV):
    torch.manual_seed(2025)
    q = torch.randn(B, NH, S, DHQK, device="cuda", requires_grad=True)
    k = torch.randn(B, NH, S, DHQK, device="cuda", requires_grad=True)
    v = torch.randn(B, NH, S, DHHV, device="cuda", requires_grad=True)
    i = torch.randn(B, NH, S, device="cuda", requires_grad=True)
    f = torch.randn(B, NH, S, device="cuda", requires_grad=True)

    h_ref = mlstm_chunkwise__xl_chunk(
        q, k, v, i, f,
        chunk_size=128, chunk_size_inter=128, chunk_size_intra=128,
        siz_b_L_parallel=64, siz_b_L_loop=64,
        siz_b_DH_parallel=DHHV, siz_b_DH_loop=DHHV,
    )

    dh = torch.randn_like(h_ref)
    dq_ref, dk_ref, dv_ref, di_ref, df_ref = torch.autograd.grad(
        [h_ref], [q, k, v, i, f], [dh]
    )

    h = mlstm_chunkwise__xl_chunk(
        q, k, v, i, f,
        chunk_size=128, chunk_size_inter=64, chunk_size_intra=128,
        siz_b_L_parallel=64, siz_b_L_loop=64,
        siz_b_DH_parallel=DHHV, siz_b_DH_loop=DHHV
    )
    dq, dk, dv, di, df = torch.autograd.grad(
        [h], [q, k, v, i, f], [dh]
    )

    torch.testing.assert_close(h, h_ref)
    torch.testing.assert_close(dq, dq_ref)
    torch.testing.assert_close(dk, dk_ref)
    torch.testing.assert_close(dv, dv_ref)
    torch.testing.assert_close(di, di_ref)
    torch.testing.assert_close(df, df_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
def test_state_passing(mlstm_state_passing_test, state_passing_qkvif):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_state_passing_test(
        kernel_fn=mlstm_chunkwise__xl_chunk,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        rtol=2e-3,
        atol=2e-3,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
@pytest.mark.parametrize("chunk_size", [64, 128, 256])
def test_triton_chunkwise_xl_chunk_backend_module_vs_native_parallel_stablef_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
    chunk_size,
):
    """This is a integration test for the mlstm siging chunkwise kernel in the
    backend module. It compares the output of the mlstm siging chunkwise kernel with the native pytroch impl."""
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")

    # create backend module:
    from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig

    backend_cfg = mLSTMBackendConfig(
        chunkwise_kernel="chunkwise--triton_xl_chunk",
        mode="train",
        chunk_size=chunk_size,
        return_last_states=False,
    )
    chunkwise_siging_backend_module = mLSTMBackend(config=backend_cfg)

    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=chunkwise_siging_backend_module,
        baseline_name="native_parallel_stablef_custbw_norm",
        target_name=f"triton_chunkwise_xl_chunk-cs{chunk_size}",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=2e-2,
        rtol_fw=5e-2,
        atol_fwbw=4e-1,  # we need to increase this tolerance for vecF.grad (max diff val 0.575306)
        rtol_fwbw=0.5,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
        pass_EPS_to_kernels=False,
    )
