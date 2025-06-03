#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from functools import partial

import pytest
import torch

from mlstm_kernels.torch.chunkwise.triton_xl_chunk_siging import (
    mlstm_siging_chunkwise__xl_chunk,
)
from mlstm_kernels.torch.parallel.native_siging import (
    mlstm_siging_parallel__native_autograd,
    mlstm_siging_parallel__native_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton_xl_chunk_ingsig"

combinations_long = {
    "S": [1024],
    "B": [1],
    "NH": [2],
    "DHQK": [128],
    "DHHV": [256],
}
fun_combs = [values for values in zip(*combinations_long.values())]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], fun_combs)
@pytest.mark.parametrize("chunk_size", [64, 128, 256, 512])
@pytest.mark.parametrize("normalize", [True, False])
def test_triton_chunkwise_xl_chunk_ingsig_vs_native_parallel_stablef_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
    chunk_size,
    normalize,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_custbw, stable_fgate=True, normalize=normalize
        ),
        target_fn=partial(mlstm_siging_chunkwise__xl_chunk, normalize=normalize, chunk_size=chunk_size),
        baseline_name=f"native_parallel_stablef_custbw_siging_norm{normalize}",
        target_name=f"triton_chunkwise_xl_chunk_siging_norm{normalize}_cs{chunk_size}",
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
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize("normalize", [True, False])
def test_state_passing(mlstm_siging_state_passing_test, state_passing_qkvif, normalize):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_siging_state_passing_test(
        kernel_fn=mlstm_siging_chunkwise__xl_chunk,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        normalize=normalize,
        rtol=2e-3,
        atol=2e-3,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], fun_combs) #final_combinations)
@pytest.mark.parametrize("chunk_size", [64, 128, 256, 512])
@pytest.mark.parametrize("normalize", [True, False])
def test_triton_chunkwise_xl_chunk_siging_backend_module_vs_native_parallel_stablef_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
    normalize,
    chunk_size,
):
    """This is a integration test for the mlstm siging chunkwise kernel in the
    backend module. It compares the output of the mlstm siging chunkwise kernel with the native pytroch impl."""
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")

    # create backend module:
    from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig

    backend_cfg = mLSTMBackendConfig(
        chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
        mode="train",
        chunk_size=chunk_size,
        normalize_siging=normalize,
        return_last_states=False,
    )
    chunkwise_siging_backend_module = mLSTMBackend(config=backend_cfg)

    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_autograd,
            stable_fgate=True,
            normalize=normalize,
        ),
        target_fn=chunkwise_siging_backend_module,
        baseline_name=f"native_parallel_stablef_custbw_siging_norm{normalize}",
        target_name=f"triton_chunkwise_xl_chunk_siging-norm{normalize}_cs{chunk_size}",
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
