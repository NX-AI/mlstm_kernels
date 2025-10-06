#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ....torch.utils import contiguous
from .bw import mlstm_chunkwise_bw
from .fw import mlstm_chunkwise_fw


## PyTorch Autograd Function - Boilerplate
def _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype: torch.dtype, chunk_size: int, eps: float) -> Callable:
    class _mlstm_chunkwise_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,  # (B, NH, S, DHQK)
            matK: torch.Tensor,  # (B, NH, S, DHQK)
            matV: torch.Tensor,  # (B, NH, S, DHHV)
            vecI: torch.Tensor,  # (B, NH, S)
            vecF: torch.Tensor,  # (B, NH, S)
            matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
            vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
            scaM_initial: torch.Tensor = None,  # (B, NH, 1)
            qk_scale: float = None,
            return_last_states: bool = False,
            RECOMPUTE_STATES_IN_BW: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # matH_out (B, NH, S, DHHV), matC_last (B, NH, DHQK, DHHV), vecN_last (B, NH, DHQK), scaM_last (B, NH, 1)
            B, NH, S, DHQK = matQ.shape
            if qk_scale is None:
                qk_scale = DHQK**-0.5

            matH_out, vecN_out, vecM_out, last_states, all_states = mlstm_chunkwise_fw(
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
                EPS=eps,
                CHUNK_SIZE=chunk_size,
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
            ) = ctx.saved_tensors
            B, NH, S, DHQK = matQ.shape
            DHHV = matV.shape[-1]

            (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                scaDeltaM_initial,
            ) = mlstm_chunkwise_bw(
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
                CHUNK_SIZE=chunk_size,
                EPS=eps,
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
            )

    return _mlstm_chunkwise_fwbw

def mlstm_chunkwise__limit_chunk(
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
    _mlstm_chunkwise_fwbw = _mlstm_chunkwise_fwbw_generator(autocast_kernel_dtype=autocast_kernel_dtype, chunk_size=chunk_size, eps=eps)
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
    )
    if return_last_states:
        return matH_out, (matC_last, vecN_last, scaM_last)
    else:
        return matH_out
