#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from collections.abc import Callable

import pytest
import torch

from mlstm_kernels.torch.chunkwise import (
    mlstm_chunkwise__limit_chunk,
    mlstm_chunkwise__native_autograd,
    mlstm_chunkwise__xl_chunk,
)
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
)
from mlstm_kernels.torch.recurrent import (
    mlstm_recurrent_step__native,
    mlstm_recurrent_step__triton,
)
from mlstm_kernels.torch.recurrent.native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)

from .template_test_arbitrary_sequence_length import (
    template_test_wrap_chunkwise__arbitrary_sequence_length,
    template_test_wrap_chunkwise__arbitrary_sequence_length_single_step,
)


# CPU test
@pytest.mark.parametrize(
    "sequence_length, chunk_size",
    [[65, 32], [93, 64], [7, 16], [500, 64]],
)  # [93, 7, 3, 461])
@pytest.mark.parametrize(
    "parallel_baseline, sequence_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_parallel__native_stablef_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_chunkwise__native_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_recurrent_step__native,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cpu"])
def test_wrap_chunkwise__arbitrary_sequence_length_cpu(
    caplog,
    sequence_length: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, S, DHQK, DHHV = 1, 1, sequence_length, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        parallel_baseline=parallel_baseline,
        sequence_baseline=sequence_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype_inputs="float32",
        atol=1e-5,
        rtol=1e-5,
        eps=1e-6,
    )


# CPU test
@pytest.mark.parametrize(
    "step_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_recurrent_step__native,
            mlstm_chunkwise__native_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_recurrent_step__native,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cpu"])
def test_wrap_chunkwise__arbitrary_sequence_length_single_step_cpu(
    caplog,
    step_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, DHQK, DHHV = 1, 1, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length_single_step(
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        step_baseline=step_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype="float32",
        atol=1e-5,
        rtol=1e-5,
        eps=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "sequence_length, chunk_size",
    [[64, 32], [75, 32], [93, 64], [7, 16], [500, 64]],
)
@pytest.mark.parametrize(
    "parallel_baseline, sequence_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_parallel__native_stablef_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_chunkwise__limit_chunk,
            mlstm_recurrent_sequence__triton_step_fused_fw,
            mlstm_recurrent_step__triton,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_wrap_chunkwise__arbitrary_sequence_length_limit_chunk(
    caplog,
    sequence_length: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, S, DHQK, DHHV = 1, 1, sequence_length, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        parallel_baseline=parallel_baseline,
        sequence_baseline=sequence_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype_inputs="float32",
        atol=5e-2,
        rtol=1e-2,
        eps=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "B, NH, DHQK, DHHV", [[2, 4, 16, 32], [1, 3, 16, 64], [1, 1, 32, 64]]
)
@pytest.mark.parametrize(
    "step_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_recurrent_step__native,
            mlstm_chunkwise__native_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_recurrent_step__triton,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_wrap_chunkwise__arbitrary_sequence_length_single_step_fused_step(
    caplog,
    B: int,
    NH: int,
    DHQK: int,
    DHHV: int,
    step_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    caplog.set_level(logging.DEBUG)
    template_test_wrap_chunkwise__arbitrary_sequence_length_single_step(
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        step_baseline=step_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype="float32",
        atol=1e-5,
        rtol=1e-5,
        eps=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "sequence_length, chunk_size",
    [
        [64, 32],
        [75, 32],
        [93, 64],
        [7, 16],
        [500, 64],
        [500, 256],
        [512, 512],
    ],
)
@pytest.mark.parametrize(
    "parallel_baseline, sequence_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_parallel__native_stablef_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_chunkwise__xl_chunk,
            mlstm_recurrent_sequence__triton_step_fused_fw,
            mlstm_recurrent_step__triton,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_wrap_chunkwise__arbitrary_sequence_length_xl_chunk(
    caplog,
    sequence_length: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, S, DHQK, DHHV = 1, 1, sequence_length, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        parallel_baseline=parallel_baseline,
        sequence_baseline=sequence_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype_inputs="float32",
        atol=5e-2,
        rtol=1e-2,
        eps=1e-6,
    )

# TODO(max) support configurable state dtype for chunkwise kernels.
# Note: We expect this to fail since currently the triton chunkwise kernels.
# always use float32 states for numerical stability during training
# However it could be possible to use bfloat16 states in forward and only in backward
# we use float32 states.
# Since for these cases the sequence length is divisible by the chunk_size we only
# call chunkwise kernels.
@pytest.mark.xfail
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "sequence_length, chunk_size",
    [
        [64, 32],
        [512, 512],
    ],
)
@pytest.mark.parametrize(
    "parallel_baseline, sequence_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_parallel__native_stablef_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_chunkwise__xl_chunk,
            mlstm_recurrent_sequence__triton_step_fused_fw,
            mlstm_recurrent_step__triton,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_wrap_chunkwise__absl_state_dtype_bf16_xfail(
    caplog,
    sequence_length: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, S, DHQK, DHHV = 1, 1, sequence_length, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        parallel_baseline=parallel_baseline,
        sequence_baseline=sequence_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype_inputs="float32",
        dtype_state="bfloat16",
        atol=5.7e-1, # only the case [75-32] needs this large tolerances
        rtol=1e-2,
        eps=1e-6,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "sequence_length, chunk_size",
    [
        # [64, 32],
        [75, 32],
        [93, 64],
        [7, 16],
        [500, 64],
        [500, 256],
        # [512, 512],
    ],
)
@pytest.mark.parametrize(
    "parallel_baseline, sequence_baseline, chunkwise_target, sequence_target, step_target",
    [
        [
            mlstm_parallel__native_stablef_autograd,
            mlstm_recurrent_sequence__native_fw,
            mlstm_chunkwise__xl_chunk,
            mlstm_recurrent_sequence__triton_step_fused_fw,
            mlstm_recurrent_step__triton,
        ],
    ],
)
@pytest.mark.parametrize("device", ["cuda"])
def test_wrap_chunkwise__absl_state_dtype_bf16(
    caplog,
    sequence_length: int,
    chunk_size: int,
    parallel_baseline: Callable,
    sequence_baseline: Callable,
    chunkwise_target: Callable,
    sequence_target: Callable,
    step_target: Callable,
    device: str,
):
    """Tests the wrap_chunkwise__arbitrary_sequence_length function.

    As baselines it uses the native parallel implementation and the sequence implementation,
    which support arbitrary sequence lengths by default.
    """
    caplog.set_level(logging.DEBUG)
    B, NH, S, DHQK, DHHV = 1, 1, sequence_length, 16, 32
    template_test_wrap_chunkwise__arbitrary_sequence_length(
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        chunk_size=chunk_size,
        parallel_baseline=parallel_baseline,
        sequence_baseline=sequence_baseline,
        chunkwise_target=chunkwise_target,
        sequence_target=sequence_target,
        step_target=step_target,
        device=device,
        dtype_inputs="float32",
        dtype_state="bfloat16",
        atol=5.7e-1, # only the case [75-32] needs this large tolerances
        rtol=1e-2,
        eps=1e-6,
    )