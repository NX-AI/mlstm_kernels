#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging

import pytest
import torch

from mlstm_kernels.triton.chunkwise.limit_chunk import (
    mlstm_chunkwise__parallel_bw_dQKV_kernel,
    mlstm_chunkwise__parallel_fw_H_kernel,
    mlstm_chunkwise__recurrent_bw_dC_kernel,
    mlstm_chunkwise__recurrent_fw_C_kernel,
)
from mlstm_kernels.torch.chunkwise.triton_limit_chunk import (
    mlstm_chunkwise__limit_chunk,
)
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton_limit_chunk"


@pytest.mark.parametrize(
    "kernel",
    [
        mlstm_chunkwise__recurrent_fw_C_kernel,
        mlstm_chunkwise__parallel_fw_H_kernel,
        mlstm_chunkwise__recurrent_bw_dC_kernel,
        mlstm_chunkwise__parallel_bw_dQKV_kernel,
    ],
)
def test_batch_size_does_not_specialize(kernel):
    batch_param = next(param for param in kernel.params if param.name == "B")
    assert not batch_param.is_constexpr
    assert batch_param.do_not_specialize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_triton_chunkwise_limit_chunk_vs_native_parallel_stablef_fp32(
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
        target_fn=mlstm_chunkwise__limit_chunk,
        baseline_name="native_parallel_stablef_custbw",
        target_name="triton_chunkwise_limit_chunk",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=2e-2,
        rtol_fw=5e-2,
        atol_fwbw=3e-1,
        rtol_fwbw=8e-1,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
def test_state_passing(mlstm_state_passing_test, state_passing_qkvif):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_state_passing_test(
        kernel_fn=mlstm_chunkwise__limit_chunk,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        rtol=1e-5,
        atol=1e-5,
        device="cuda",
    )
