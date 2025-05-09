#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from mlstm_kernels.jax.chunkwise.triton_xl_chunk_siging import (
    mlstm_siging_chunkwise__xl_chunk,
)
from mlstm_kernels.jax.parallel.native_siging import (
    mlstm_siging_parallel__native_autograd,
)

from ...conftest import combinations_other_list, final_combinations
from ..template_test_against_pytorch import check_jax_against_pytorch_reference

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-jax_xl_chunk_siging"


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
@pytest.mark.parametrize("normalize", [True, False])
def test_jax_native_chunkwise_vs_triton_xl_chunk_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV, normalize
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_autograd,
            stable_fgate=True,
            normalize=normalize,
        ),
        target_fn=partial(mlstm_siging_chunkwise__xl_chunk, normalize=normalize),
        baseline_name=f"native_parallel_siging_stablef_norm{normalize}_autograd",
        target_name=f"triton_xl_chunk_siging_norm{normalize}",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=3e-3,
        rtol_fw=5e-2,
        atol_fwbw=0.2,  # we need those high tolerances for the forget gate gradient Max absolute difference: 0.2168259
        rtol_fwbw=0.1,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        use_jit=True,
    )


@pytest.mark.skipif(pytest.short_test, reason="Short test.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_other_list)
@pytest.mark.parametrize("normalize", [False])
@pytest.mark.parametrize("chunk_size", [256])
def test_jax_native_chunkwise_vs_triton_xl_chunk_other(
    test_session_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
    normalize,
    chunk_size,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_autograd,
            stable_fgate=True,
            normalize=normalize,
        ),
        target_fn=partial(
            mlstm_siging_chunkwise__xl_chunk, normalize=normalize, chunk_size=chunk_size
        ),
        baseline_name=f"native_parallel_siging_stablef_norm{normalize}_autograd",
        target_name=f"triton_xl_chunk_siging_norm{normalize}_cs{chunk_size}",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=3e-3,
        rtol_fw=5e-2,
        atol_fwbw=0.23,  # we need those high tolerances for the forget gate gradient Max absolute difference: 0.2168259
        rtol_fwbw=0.1,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        use_jit=True,
    )


# @pytest.mark.parametrize("normalize", [True, False])
# def test_vs_torch_xl_chunk_siging(test_output_folder, normalize):
#     test_data_file = (
#         test_output_folder
#         / f"chunkwise-triton_xl_chunk_ingsig_triton_chunkwise_xl_chunk_siging_norm{normalize}-vs-native_parallel_stablef_custbw_siging_norm{normalize}_S256B1NH2DHQK64DHHV128.npz"
#     )

#     check_jax_against_pytorch_reference(
#         torch_test_data_file=test_data_file,
#         jax_mlstm_parallel_fn=partial(mlstm_siging_chunkwise__xl_chunk, normalize=normalize),
#         atol_fw=4e-4,
#         rtol_fw=1e-2,
#         atol_fwbw=2e-2,
#         rtol_fwbw=5e-2,
#     )


# TODO adapt the state passing test for siging mLSTM
# @pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
# @pytest.mark.parametrize("mlstm_kernel", [mlstm_siging_chunkwise__xl_chunk])
# def test_mlstm_chunkwise_state_passing(
#     default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
#     mlstm_state_passing_test: callable,
#     mlstm_kernel: callable,
# ):
#     """Compare single forward vs chunked one with states passed between steps."""
#     # Repeat the inputs to have longer sequence length.
#     default_qkvif = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=2), default_qkvif)
#     mlstm_state_passing_test(mlstm_kernel, *default_qkvif, num_chunks=4, rtol=5e-2, atol=5e-3)
