# Copyright JKU Linz 2024
# Author: Maximilian Beck
from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ....kernel_utils import contiguous
from ._triton_bw import _mlstm_chunkwise_bw
from ._triton_fw import _mlstm_chunkwise_fw

# Triton.

# Forward and backward pass of the mLSTM chunkwise formulation.

# Notation:
# Dimensions:
#     B: batch size
#     NH: number of heads
#     S: sequence length
#     DH: hidden dimension
#     NC: number of chunks
#     L: chunk size

# Variables:
#     vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
#     vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
#     scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.


## PyTorch Autograd Function - Boilerplate
def _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=torch.float16) -> Callable:
    class _mlstm_chunkwise_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,  # (B, NH, S, DHQK)
            matK: torch.Tensor,  # (B, NH, S, DHQK)
            matV: torch.Tensor,  # (B, NH, S, DHV)
            vecI: torch.Tensor,  # (B, NH, S)
            vecF: torch.Tensor,  # (B, NH, S)
            matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
            vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
            scaM_initial: torch.Tensor = None,  # (B, NH)
            qk_scale: float = None,
            return_last_states: bool = False,
            RECOMPUTE_STATES_IN_BW: bool = True,
            CHUNK_SIZE: int = 64,
            EPS: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            B, NH, S, DHQK = matQ.shape
            if qk_scale is None:
                qk_scale = DHQK**-0.5

            matH_out, vecN_out, vecM_out, last_states, all_states = _mlstm_chunkwise_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                scaM_initial=scaM_initial,
                qk_scale=qk_scale,
                return_last_states=return_last_states,
                return_all_states=(not RECOMPUTE_STATES_IN_BW),
                EPS=EPS,
                CHUNK_SIZE=CHUNK_SIZE,
            )

            if return_last_states:
                (matC_last, vecN_last, scaM_last) = last_states
            else:
                (matC_last, vecN_last, scaM_last) = (None, None, None)

            if all_states is not None:
                matC_all, vecN_all, scaM_all = all_states
            else:
                matC_all, vecN_all, scaM_all = (None, None, None)

            ctx.save_for_backward(
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                scaM_initial,
                matC_all,
                vecN_all,
                scaM_all,
                vecN_out,
                vecM_out,
                torch.tensor(CHUNK_SIZE),
                torch.tensor(EPS),
            )
            return matH_out, matC_last, vecN_last, scaM_last

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, matDeltaH, matDeltaC_last, vecDeltaN_last, scaDeltaM_last):
            (
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                scaM_initial,
                matC_all,
                vecN_all,
                scaM_all,
                vecN_out,
                vecM_out,
                CHUNK_SIZE,
                EPS,
            ) = ctx.saved_tensors
            B, NH, S, DHQK = matQ.shape
            DHV = matV.shape[-1]

            (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                scaDeltaM_initial,
            ) = _mlstm_chunkwise_bw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                scaM_initial=scaM_initial,
                matC_all=matC_all,
                vecN_all=vecN_all,
                scaM_all=scaM_all,
                vecN_out=vecN_out,
                vecM_out=vecM_out,
                matDeltaH=matDeltaH,
                matDeltaC_last=matDeltaC_last,
                vecDeltaN_last=vecDeltaN_last,
                scaDeltaM_last=scaDeltaM_last,
                CHUNK_SIZE=int(CHUNK_SIZE),
                EPS=float(EPS),
            )

            return (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                scaDeltaM_initial,
                None,
                None,
                None,
                None,
                None,
            )

    return _mlstm_chunkwise_fwbw


_mlstm_chunkwise_fwbw_float32 = _mlstm_chunkwise_fwbw_generator(
    autocast_kernel_dtype=torch.float32
)
_mlstm_chunkwise_fwbw_float16 = _mlstm_chunkwise_fwbw_generator(
    autocast_kernel_dtype=torch.float16
)
_mlstm_chunkwise_fwbw_bfloat16 = _mlstm_chunkwise_fwbw_generator(
    autocast_kernel_dtype=torch.bfloat16
)


def _get_chunkwise_fwbw_kernel(autocast_kernel_dtype: torch.dtype) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_chunkwise_fwbw_float32
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_chunkwise_fwbw_float16
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_chunkwise_fwbw_bfloat16
    else:
        raise ValueError(f"Unsupported kernel dtype {autocast_kernel_dtype}.")


# TODO mbeck: remove this function (use the other one below)
def mlstm_chunkwise_fwbw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    RECOMPUTE_STATES_IN_BW: bool = True,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.float16,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    _mlstm_chunkwise_fwbw = _get_chunkwise_fwbw_kernel(autocast_kernel_dtype)
    matH_out, matC_last, vecN_last, scaM_last = _mlstm_chunkwise_fwbw.apply(
        matQ,
        matK,
        matV,
        vecI,
        vecF,
        matC_initial,
        vecN_initial,
        scaM_initial,
        qk_scale,
        return_last_states,
        RECOMPUTE_STATES_IN_BW,
        CHUNK_SIZE,
        EPS,
    )
    if return_last_states:
        return matH_out, (matC_last, vecN_last, scaM_last)
    else:
        return matH_out


def mlstm_chunkwise_max_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 64,
    autocast_kernel_dtype: torch.dtype = torch.float32,
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    _mlstm_chunkwise_fwbw = _get_chunkwise_fwbw_kernel(autocast_kernel_dtype)
    matH_out, matC_last, vecN_last, scaM_last = _mlstm_chunkwise_fwbw.apply(
        q,
        k,
        v,
        i,
        f,
        c_initial,
        n_initial,
        m_initial,
        None,
        return_last_states,
        True,
        chunk_size,
        eps,
    )
    if return_last_states:
        return matH_out, (matC_last, vecN_last, scaM_last)
    else:
        return matH_out