#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the kernel that combines the recurrent and parallel part of the forward pass of the mLSTM chunkwise formulation.
It should allow arbitrary large chunk sizes and head dimensions.
"""

import torch

from ....triton.chunkwise.kernel_param_heuristics import (
    get_xl_chunk_kernel_params,
)
from ...utils import contiguous_noctx
from .bw_parallel_dK import mlstm_chunkwise__parallel_bw_dK
from .bw_parallel_dQ import mlstm_chunkwise__parallel_bw_dQ
from .bw_parallel_dV import mlstm_chunkwise__parallel_bw_dV
from .bw_recurrent import mlstm_chunkwise__recurrent_bw_dC
from .chunkwise_gates import (
    compute_chunkwise_log_gates_vecB_vecA,
    compute_gate_grads_vecDeltaI_vecDeltaF,
)
from .fw_recurrent import mlstm_chunkwise__recurrent_fw_C


@contiguous_noctx
def mlstm_chunkwise_bw(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    ## Backward arguments
    matCstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1) * DHQK, DHHV)
    vecNstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1) * DHQK)
    scaMstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1))
    vecN_out: torch.Tensor = None,  # (B, NH, S)
    vecM_out: torch.Tensor = None,  # (B, NH, S)
    matDeltaH_out: torch.Tensor = None,  # (B, NH, S, DHHV)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    ## Other arguments
    qk_scale: float = None,
    chunk_size: int = 128,
    chunk_size_inter: int | None = None,
    chunk_size_intra: int | None = None,
    siz_b_L_parallel: int | None = None,
    siz_b_L_loop: int | None = None,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    eps: float = 0.0,
):
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    kernel_chunk_params = get_xl_chunk_kernel_params(
        sequence_length=S,
        target_chunk_size=chunk_size,
        siz_b_L_loop=siz_b_L_loop,
        siz_b_L_parallel=siz_b_L_parallel,
        chunk_size_inter=chunk_size_inter,
        chunk_size_intra=chunk_size_intra,
    )

    #! recompute the "all" states if needed
    if matCstate_all is None:
        assert (
            (matCstate_all is None)
            and (vecNstate_all is None)
            and (scaMstate_all is None)
        ), "Either all or none of the states must be provided."

        matCstate_all, vecNstate_all, scaMstate_all = mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecF=vecF,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            chunk_size=kernel_chunk_params.chunk_size_inter,
            save_states_every_nth_chunk=kernel_chunk_params.save_states_every_nth_chunk,
            num_stages=num_stages_inter,
            num_warps=num_warps_inter,
        )

    #! recurrent backward: compute the deltaC (& deltaN) gradients
    # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    matDeltaC_states = mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecF=vecF,  # (B, NH, S)
        scaM_inter=scaMstate_all,  # (B, NH, NCintra+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH_out,  # (B, NH, S, DHHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHHV)
        qk_scale=qk_scale,
        eps=eps,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    #! parallel backward: compute the deltaQ, deltaK, deltaV gradients
    vecB, vecA = compute_chunkwise_log_gates_vecB_vecA(
        chunk_size=kernel_chunk_params.chunk_size_intra, vecI=vecI, vecF=vecF
    )
    grad_output_dtype = matQ.dtype
    #! compute deltaV
    matDeltaV = mlstm_chunkwise__parallel_bw_dV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        scaMstate_all=scaMstate_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=kernel_chunk_params.chunk_size_intra,
        siz_b_LQ=kernel_chunk_params.siz_b_L_loop,
        siz_b_LKV=kernel_chunk_params.siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    #! compute deltaK
    matDeltaK = mlstm_chunkwise__parallel_bw_dK(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        scaMstate_all=scaMstate_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=kernel_chunk_params.chunk_size_intra,
        siz_b_LQ=kernel_chunk_params.siz_b_L_loop,
        siz_b_LKV=kernel_chunk_params.siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    #! compute deltaQ
    matDeltaQ = mlstm_chunkwise__parallel_bw_dQ(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        scaMstate_all=scaMstate_all,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        chunk_size=kernel_chunk_params.chunk_size_intra,
        siz_b_LQ=kernel_chunk_params.siz_b_L_parallel,
        siz_b_LKV=kernel_chunk_params.siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=grad_output_dtype,
    )

    vecDeltaI, vecDeltaF = compute_gate_grads_vecDeltaI_vecDeltaF(
        matQ=matQ, matK=matK, matDeltaQ=matDeltaQ, matDeltaK=matDeltaK, vecF=vecF
    )

    # vecDeltaI = torch.zeros((B, NH, S), dtype=vecI.dtype, device=vecI.device)

    matDeltaC_initial = (
        matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    )

    vecDeltaN_initial = (
        torch.zeros_like(vecN_initial) if vecN_initial is not None else None
    )
    scaDeltaM_initial = (
        torch.zeros_like(scaM_initial) if scaM_initial is not None else None
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
    )
