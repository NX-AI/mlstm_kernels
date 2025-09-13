#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton

from ....triton.chunkwise.xl_chunk import mlstm_chunkwise__recurrent_bw_dC_kernel
from ....triton.kernel_param_heuristics import get_head_dim_block_size
from ....utils.kernels import is_power_of_2
from ...utils import torch2triton_dtype


def mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    vecM_combine: torch.Tensor,  # (B, NH, S)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    qk_scale: float = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
) -> torch.Tensor:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    """Computes only the deltaC gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matDeltaH.shape[-1]
    _dtype, _device = matQ.dtype, matQ.device
    NC = scaM_inter.shape[-1] - 1
    L = S // NC
    assert is_power_of_2(L), "Chunk size must be a power of 2."
    assert S % L == 0, "S must be divisible by chunk_size."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    matDeltaC_states = torch.empty(
        (B, NH, (NC + 1) * DHQK, DHHV),
        dtype=torch.float32,
        device=_device,
    )

    siz_b_DHQK = get_head_dim_block_size(head_dim=DHQK, min_block_size=64)
    siz_b_DHHV = get_head_dim_block_size(head_dim=DHHV, min_block_size=64)

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK == 64 else 2

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    mlstm_chunkwise__recurrent_bw_dC_kernel[grid](
        matQ=matQ,
        vecF=vecF,
        scaM_inter=scaM_inter,
        vecM_combine=vecM_combine,
        matDeltaH=matDeltaH,
        vecN_out=vecN_out,
        matDeltaC_last=matDeltaC_last,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        str_matQ_B_NH=matQ.stride(1),
        str_matQ_S=matQ.stride(2),
        str_matQ_DHQK=matQ.stride(3),
        str_vecF_B_NH=vecF.stride(1),
        str_scaM_inter_B_NH=scaM_inter.stride(1),
        str_scaM_inter_NC=scaM_inter.stride(2),
        str_vecM_combine_B_NH=vecM_combine.stride(1),
        str_vecM_combine_S=vecM_combine.stride(2),
        str_matDeltaH_B_NH=matDeltaH.stride(1),
        str_matDeltaH_S=matDeltaH.stride(2),
        str_matDeltaH_DHHV=matDeltaH.stride(3),
        str_vecN_out_B_NH=vecN_out.stride(1),
        str_vecN_out_S=vecN_out.stride(2),
        str_matDeltaC_last_B_NH=matDeltaC_last.stride(1) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHQK=matDeltaC_last.stride(2) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHHV=matDeltaC_last.stride(3) if USE_LAST_STATE else 0,
        str_matDeltaC_states_B_NH=matDeltaC_states.stride(1),
        str_matDeltaC_states_NCDHQK=matDeltaC_states.stride(2),
        str_matDeltaC_states_DHHV=matDeltaC_states.stride(3),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        USE_LAST_STATE=USE_LAST_STATE,
        DTYPE=torch2triton_dtype(_dtype),
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matDeltaC_states
