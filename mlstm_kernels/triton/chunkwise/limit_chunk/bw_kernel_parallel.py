#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton.

Backward parallel kernel of the mLSTM chunkwise formulation.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length (K, V)
    T: sequence length (Q)
    DHQK: hidden dimension (Q, K)
    DHHV: hidden dimension (H, V)
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
    matD, D: gating matrix for the parallel form.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B"])
def mlstm_chunkwise__parallel_bw_dQKV_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    vecM_combine,  # (B, NH, S)
    scaM_inter,  # (B, NH, NC+1)
    matC_states,  # (B, NH, (NC+1) * DHQK, DHHV) # take only the first NC states
    matDeltaH,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV) # take only the last NC states
    matDeltaQ,  # (B, NH, S, DHQK)
    matDeltaK,  # (B, NH, S, DHQK)
    matDeltaV,  # (num_b_DHQK, B, NH, S, DHHV)
    qk_scale,
    str_matQK_B_NH,  # shared with matQ, matDeltaQ, matK, matDeltaK
    str_matQK_S,
    str_matQK_DHQK,
    str_matDV_num_b_DHQK,
    str_matHV_B_NH,  # shared with matDeltaV, matDeltaH
    str_matHV_S,
    str_matHV_DHHV,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_vecM_combine_B_NH,
    str_vecM_combine_S,
    str_scaM_inter_B_NH,
    str_scaM_inter_NC,
    str_matC_states_B_NH,
    str_matC_states_NCDHQK,
    str_matC_states_DHHV,
    str_vecN_out_B_NH,
    str_vecN_out_S,
    str_matDeltaC_states_B_NH,
    str_matDeltaC_states_NCDHQK,
    str_matDeltaC_states_DHHV,
    B,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 1e-6,
):
    idx_b_DHQK, idx_b_NC, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # [intra] recompute matDbar
    # load gates (L,)
    vecB_val = tl.load(vecB + idx_b_BNH * S + idx_b_NC * L + tl.arange(0, L)).to(
        tl.float32
    )  # (L,)
    vecI_val = tl.load(vecI + idx_b_BNH * S + idx_b_NC * L + tl.arange(0, L)).to(
        tl.float32
    )  # (L,)
    # load vecM_combine (L,)
    vecM_combine_val = tl.load(
        vecM_combine + idx_b_BNH * S + idx_b_NC * L + tl.arange(0, L)
    )

    # compute gate matrix matDbar (L, L)
    idx_mask = tl.arange(0, L)
    mask = idx_mask[:, None] >= idx_mask[None, :]
    matF_full_val = vecB_val[:, None] - vecB_val[None, :]
    matF_mask_val = tl.where(mask, matF_full_val, -float("inf"))
    matDtilde_val = matF_mask_val + vecI_val[None, :]
    matDbar_val = tl.exp(matDtilde_val - vecM_combine_val[:, None])

    # [inter,intra] compute vecAbar, vecBbar
    # load scaM_inter_k, scaM_inter_km1
    scaM_inter_km1_val = tl.load(scaM_inter + idx_b_BNH * (NC + 1) + idx_b_NC)  # (1,)
    scaM_inter_k_val = tl.load(
        scaM_inter + idx_b_BNH * (NC + 1) + (idx_b_NC + 1)
    )  # (1,)

    # scaG is the last val of vecB
    scaG_val = tl.load(vecB + idx_b_BNH * S + idx_b_NC * L + L - 1)  # (1,)
    vecA_val = scaG_val - vecB_val + vecI_val  # (L,)

    vecAbar_val = tl.exp(vecA_val - scaM_inter_k_val)  # (L,)
    vecBbar_val = tl.exp(vecB_val + scaM_inter_km1_val - vecM_combine_val)  # (L, )

    # [intra] recompute matS, matSbar
    # NOTE: we compute only a part of matS as we do not sum over the full DHQK dim
    # we have to sum the matV after the kernel.
    matQ_ptr = tl.make_block_ptr(
        base=matQ + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    matK_ptr = tl.make_block_ptr(
        base=matK + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)  # (L, siz_b_DHQK)
    matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)  # (siz_b_DHQK, L)
    matS_val = tl.dot(matQ_val, tl.trans(matK_val)) * qk_scale
    matSbar_val = (matS_val * matDbar_val).to(DTYPE)  # (L, L)

    matKbar_val = (matK_val.to(tl.float32) * vecAbar_val[:, None]).to(
        DTYPE
    )  # (L, siz_b_DHQK)

    vecN_out_ptr = vecN_out + idx_b_BNH * S + idx_b_NC * L + tl.arange(0, L)
    vecN_out_val = tl.load(vecN_out_ptr).to(tl.float32)  # (L,)
    matDeltaQ_inter_val = tl.zeros((L, siz_b_DHQK), dtype=tl.float32)
    matDeltaK_inter_val = tl.zeros((L, siz_b_DHQK), dtype=tl.float32)
    matDeltaSbar_val = tl.zeros((L, L), dtype=tl.float32)

    for idx_b_DHHV in range(tl.cdiv(DHHV, siz_b_DHHV)):
        # ? pointers for iteration
        # load pointers:
        matV_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        matDeltaH_ptr = tl.make_block_ptr(
            base=matDeltaH + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        matC_km1_ptr = tl.make_block_ptr(
            base=matC_states
            + idx_b_BNH * str_matC_states_B_NH
            + idx_b_NC * DHQK * DHHV,
            shape=(DHHV, DHQK),  # transposed
            strides=(str_matC_states_DHHV, str_matC_states_NCDHQK),
            offsets=(idx_b_DHHV * siz_b_DHHV, idx_b_DHQK * siz_b_DHQK),
            block_shape=(siz_b_DHHV, siz_b_DHQK),
            order=(0, 1),
        )
        # (idx_b_NC + 1) since we take only the last NC states
        matDeltaC_k_ptr = tl.make_block_ptr(
            base=matDeltaC_states
            + idx_b_BNH * str_matDeltaC_states_B_NH
            + (idx_b_NC + 1) * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # store pointers:
        matDeltaV_ptr = tl.make_block_ptr(
            base=matDeltaV
            + idx_b_DHQK * str_matDV_num_b_DHQK
            + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # ? end pointers

        # [inter,intra] matDeltaV
        # matDeltaH = matDeltaH / vecN_out
        matDeltaH_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(
            tl.float32
        )  # (L, siz_b_DHHV)
        # matDeltaH_val = matDeltaH_val / (vecN_out_val[:, None] + EPS)  # (L, siz_b_DHHV)

        # [inter] matDeltaK += (matV @ matDeltaC_k.transpose()) * vecAbar
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)  # (L, siz_b_DHHV)
        matDeltaC_k_val = tl.load(matDeltaC_k_ptr, boundary_check=(0, 1)).to(
            DTYPE
        )  # (siz_b_DHQK, siz_b_DHHV)

        matDeltaK_inter_val += tl.dot(
            matV_val, tl.trans(matDeltaC_k_val)
        )  # (L, siz_b_DHQK)

        # [inter] matDeltaQ += matDeltaH @ matC_km1.transpose() * vecBbar
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(
            DTYPE
        )  # (siz_b_DHHV, siz_b_DHQK)
        matDeltaQ_inter_val += (
            tl.dot((matDeltaH_val).to(DTYPE), matC_km1_val) * qk_scale
        )  # (L, siz_b_DHQK)

        # [intra] matDeltaS += (matDeltaH @ matV.transpose()) * matDbar
        matDeltaSbar_val += tl.dot(
            (matDeltaH_val / (vecN_out_val[:, None] + EPS)).to(DTYPE),
            tl.trans(matV_val),
        )  # (L, L)

        # [inter, intra] matDeltaV = matSbar.transpose() @ matDeltaH + matKbar @ matDeltaC_k
        matDeltaV_val = tl.dot(
            tl.trans(matSbar_val),
            (matDeltaH_val / (vecN_out_val[:, None] + EPS)).to(DTYPE),
        ) + tl.dot(matKbar_val, matDeltaC_k_val)
        # store matDeltaV (NOTE: each idx_b_DHQK stores its own contribution in HBM, it will be summed outside the kernel)
        tl.store(
            matDeltaV_ptr,
            matDeltaV_val.to(matDeltaV_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
    matDeltaQ_inter_val = (
        matDeltaQ_inter_val * vecBbar_val[:, None] / (vecN_out_val[:, None] + EPS)
    )
    matDeltaK_inter_val = matDeltaK_inter_val * vecAbar_val[:, None]

    matDeltaS_val = (matDeltaSbar_val * matDbar_val).to(DTYPE)

    # [intra] matDeltaK = matDeltaS.transpose() @ matQ * qk_scale
    matDeltaK_val = matDeltaK_inter_val + (
        tl.dot(tl.trans(matDeltaS_val), (matQ_val).to(DTYPE)) * qk_scale
    )
    # [intra] matDeltaQ = matDeltaS @ matK * qk_scale
    matDeltaQ_val = matDeltaQ_inter_val + ((tl.dot(matDeltaS_val, matK_val)) * qk_scale)

    # store matDeltaQ, matDeltaK
    matDeltaK_ptr = tl.make_block_ptr(
        base=matDeltaK + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    matDeltaQ_ptr = tl.make_block_ptr(
        base=matDeltaQ + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    tl.store(
        matDeltaK_ptr,
        matDeltaK_val.to(matDeltaK.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        matDeltaQ_ptr,
        matDeltaQ_val.to(matDeltaQ.dtype.element_ty),
        boundary_check=(0, 1),
    )
