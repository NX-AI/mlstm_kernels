#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""PyTorch.

# Backward pass of the mLSTM chunkwise formulation.

# Notation:
# Dimensions:
#     B: batch size
#     NH: number of heads
#     S: sequence length
#     DH: hidden dimension
#     NC: number of chunks
#     L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from .fw import mlstm_chunkwise__recurrent_fw_C


def mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    vecM_combine: torch.Tensor,  # (B, NH, S)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    qk_scale: float = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHV)
    """Computes only the deltaC gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.
    """
    B, NH, S, DHQK, DHV = *matQ.shape, matDeltaH.shape[-1]
    NC = num_chunks
    L = chunk_size
    _dtype, _device = matQ.dtype, matQ.device

    matDeltaC_states = torch.zeros(
        (B, NH, (NC + 1) * DHQK, DHV), dtype=_dtype, device=_device
    )

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    if matDeltaC_last is not None:
        matDeltaC_k = matDeltaC_last
    else:
        matDeltaC_k = torch.zeros((B, NH, DHQK, DHV), dtype=_dtype, device=_device)

    scaG = vecB[..., -1]  # (B, NH, NC)

    for k in range(NC, 0, -1):  # goes until 1
        # store the matDeltaC_k from the previous iteration
        # in the first iteration, this is the delta error from the last chunk
        matDeltaC_states[:, :, k * DHQK : (k + 1) * DHQK, :] = matDeltaC_k.clone()

        # load
        scaG_k = scaG[:, :, (k - 1), None]
        scaM_inter_kminus1 = scaM_inter[:, :, (k - 1), None]
        scaM_inter_k = scaM_inter[:, :, k, None]
        scaGbar_k = torch.exp(scaG_k + scaM_inter_kminus1 - scaM_inter_k)[:, :, None]

        vecB_k = vecB[:, :, (k - 1), :]  # (B, NH, L)
        vecM_combine_k = vecM_combine[:, :, (k - 1) * L : k * L]  # (B, NH, L)
        # print(f"vecB_k shape: {vecB_k.shape}, vecM_combine_k shape: {vecM_combine_k.shape}, scaM_inter_kminus1 shape: {scaM_inter_kminus1.shape}, scaM_inter_k shape: {scaM_inter_k.shape}")
        vecBbar_k = torch.exp(vecB_k + scaM_inter_kminus1 - vecM_combine_k)[
            :, :, :, None
        ]  # (B, NH, L, 1)

        matQ_k = matQ[:, :, (k - 1) * L : k * L, :]  # (B, NH, L, DHQK)
        matQbar_k = matQ_k * vecBbar_k * qk_scale

        vecN_k = vecN_out[:, :, (k - 1) * L : k * L, None]  # (B, NH, L, 1)
        matDeltaH_k = matDeltaH[:, :, (k - 1) * L : k * L, :] / (
            vecN_k + eps
        )  # (B, NH, L, DHV)

        # matDeltaC_k-1 update
        matDeltaC_kminus1 = (
            scaGbar_k * matDeltaC_k + matQbar_k.transpose(-2, -1) @ matDeltaH_k
        )  # (B, NH, DHQK, DHV)

        # move to the next iteration
        matDeltaC_k = matDeltaC_kminus1

    # store the matDeltaC_k from the last iteration
    matDeltaC_states[:, :, :DHQK, :] = matDeltaC_k

    return matDeltaC_states


def _mlstm_chunkwise__parallel_bw_dQKV(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecM_combine: torch.Tensor,  # (B, NH, S) = (B, NH, NC * L)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHV)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHV)
    matDeltaC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHV)
    qk_scale: float = None,
    chunk_size: int = 64,
    num_chunks: int = 1,
    eps: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHV)
    B, NH, S, DHQK, DHV = *matQ.shape, matV.shape[-1]
    NC = num_chunks
    L = chunk_size
    _dtype, _device = matQ.dtype, matQ.device

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    #! intra chunk gradients
    # load / prepare the inputs
    matDeltaH = matDeltaH / (vecN_out[:, :, :, None] + eps)

    matDeltaH = rearrange(matDeltaH, "b nh (nc l) dh -> b nh nc l dh", l=L)

    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=L)

    vecM_combine = rearrange(vecM_combine, "b nh (nc l) -> b nh nc l", l=L)

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    # recompute the gate matrix D
    matF = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]
    matF_mask = torch.where(ltr, matF, -float("inf"))
    matDtilde = matF_mask + vecI[:, :, :, None, :]
    matDbar = torch.exp(matDtilde - vecM_combine[:, :, :, :, None])

    # recompute the S matrix
    matS = (matQ @ matK.transpose(-2, -1)) * qk_scale
    matSbar = matS * matDbar

    # compute the intra delta gradients
    matDeltaV_intra = matSbar.transpose(-2, -1) @ matDeltaH

    matDeltaSbar = matDeltaH @ matV.transpose(-2, -1)
    matDeltaS = matDeltaSbar * matDbar

    matDeltaQ_intra = (matDeltaS @ matK) * qk_scale
    matDeltaK_intra = ((matDeltaS).transpose(-2, -1) @ matQ) * qk_scale

    #! inter chunk gradients
    # load / prepare the inputs
    matDeltaC_states = rearrange(
        matDeltaC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )
    matC_states = rearrange(
        matC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )

    vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)

    # compute the gates vecA, vecB
    scaM_inter_kminus1 = scaM_inter[:, :, :-1, None]
    scaM_inter_k = scaM_inter[:, :, 1:, None]
    vecBbar = torch.exp(vecB + scaM_inter_kminus1 - vecM_combine)[:, :, :, :, None]
    vecAbar = torch.exp(vecA - scaM_inter_k)[:, :, :, :, None]

    # compute the inter delta gradients
    matDeltaV_inter = (matK * vecAbar) @ matDeltaC_states

    matDeltaK_inter = (matV * vecAbar) @ (matDeltaC_states.transpose(-2, -1))
    matDeltaQ_inter = (matDeltaH * vecBbar) @ (matC_states * qk_scale).transpose(-2, -1)

    # combine the delta gradients
    matDeltaQ = matDeltaQ_intra + matDeltaQ_inter
    matDeltaK = matDeltaK_intra + matDeltaK_inter
    matDeltaV = matDeltaV_intra + matDeltaV_inter

    matDeltaQ = rearrange(matDeltaQ, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaK = rearrange(matDeltaK, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaV = rearrange(matDeltaV, "b nh nc l dh -> b nh (nc l) dh")
    return matDeltaQ, matDeltaK, matDeltaV


def mlstm_chunkwise_bw(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    ## Backward arguments
    matC_all: torch.Tensor = None,  # (B, NH, NC * DHQK, DHV)
    vecN_all: torch.Tensor = None,  # (B, NH, NC * DHQK)
    scaM_all: torch.Tensor = None,  # (B, NH, NC+1)
    vecN_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    vecM_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    matDeltaH: torch.Tensor = None,  # (B, NH, S, DHV)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecDeltaN_last: torch.Tensor = None,  # (B, NH, DHQK)
    scaDeltaM_last: torch.Tensor = None,  # (B, NH)
    ## Common arguments
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
):
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]

    assert (
        S % CHUNK_SIZE == 0
    ), f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."

    NC = S // CHUNK_SIZE

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    #! recompute the "all" states if needed
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."
        matC_all, vecN_all, scaM_all = mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecB=vecB,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            qk_scale=qk_scale,
            chunk_size=CHUNK_SIZE,
            num_chunks=NC,
        )

    # recurrent backward: compute the deltaC gradients
    matDeltaC_states = mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecB=vecB,  # (B, NH, NC, L)
        scaM_inter=scaM_all,  # (B, NH, NC+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        chunk_size=CHUNK_SIZE,
        num_chunks=NC,
        eps=EPS,
    )  # (B, NH, NC * DHQK, DHV)

    # save bw dC inputs and outputs for debugging
    inp_dict_bw_dC = dict(
        matDeltaC_states=matDeltaC_states,
        matQ=matQ,
        vecB=vecB,
        vecF=vecF,
        scaM_inter=scaM_all,
        vecM_combine=vecM_out,
        matDeltaH=matDeltaH,
        vecN_out=vecN_out,
        matDeltaC_last=matDeltaC_last,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )
    save_file = f"/home/max/myrepos/nxai_public/mlstm_kernels_internal_cleaned/notebooks_kernel_dev/xl_chunk/debug_xl_chunk_bw_dC_inputs_cs{CHUNK_SIZE}.pt"
    torch.save(inp_dict_bw_dC, save_file)
    print(f"Saved inputs to {save_file}")

    # parallel backward: compute the deltaQ, deltaK, deltaV, deltaI gradients
    matC_k_states = matC_all[:, :, :-DHQK, :]  # take the first NC states
    matDeltaC_k_states = matDeltaC_states[:, :, DHQK:, :]  # take the last NC states

    matDeltaQ, matDeltaK, matDeltaV = _mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC+1)
        matC_states=matC_k_states,  # (B, NH, NC * DHQK, DHV)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_k_states,  # (B, NH, NC * DHQK, DHV)
        qk_scale=qk_scale,
        chunk_size=CHUNK_SIZE,
        num_chunks=NC,
        eps=EPS,
    )

    # postprocessing: compute deltaF and deltaI gradients
    vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    # compute deltaI
    # both are equivalent:
    vecDeltaI = (matV * matDeltaV).sum(-1)
    # vecDeltaI = (matK * matDeltaK).sum(-1)

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
