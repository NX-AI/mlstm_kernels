#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton.

Forward parallel kernel of the mLSTM chunkwise formulation.

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
def mlstm_chunkwise__parallel_fw_H_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    matC_states,  # (B, NH, NC * DHQK, DHHV)
    vecN_states,  # (B, NH, NC * DHQK)
    scaMinter_states,  # (B, NH, NC)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matHout,  # (B, NH, S, DHHV)
    vecNout,  # (B, NH, S)
    vecMout,  # (B, NH, S)
    qk_scale,
    str_matQK_B_NH,
    str_matQK_S,
    str_matQK_DHQK,
    str_matHV_B_NH,
    str_matHV_S,
    str_matHV_DHHV,
    str_matCstates_B_NH,
    str_matCstates_NCDHQK,
    str_matCstates_DHHV,
    str_vecNstates_B_NH,
    str_vecNstates_NCDHQK,
    str_scaMinterstates_B_NH,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_vecMN_B_NH,
    str_vecMN_S,
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
    idx_b_DHHV, idx_b_NC, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # load vecB (L,)
    vecB_val = tl.load(
        vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)
    ).to(tl.float32)

    # load vecI (L,)
    vecI_val = tl.load(
        vecI + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)
    ).to(tl.float32)

    # load scaMinter_km1 (1,)
    scaMinter_km1_val = tl.load(
        scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + idx_b_NC
    ).to(tl.float32)

    # compute gate matrix matDbar (L, L)
    idx_mask = tl.arange(0, L)
    mask = idx_mask[:, None] >= idx_mask[None, :]
    matD_full_val = vecB_val[:, None] - vecB_val[None, :] + vecI_val[None, :]
    matD_val = tl.where(mask, matD_full_val, -float("inf"))

    # compute vecM_k_intra (L,) & vecM_k_combine (L,)
    vecM_intra_val = tl.max(matD_val, axis=1)
    vecM_combine_val = tl.maximum(vecB_val + scaMinter_km1_val, vecM_intra_val)
    # tl.static_print("vecM_combine_val", vecM_combine_val[:, None])
    # tl.static_print("matD_val", matD_val)
    matDbar_val = tl.exp(matD_val - vecM_combine_val[:, None])

    # compute vecBbar (L,)
    vecBbar_val = tl.exp(vecB_val + scaMinter_km1_val - vecM_combine_val)
    # tl.static_print("vecBbar_val", vecBbar_val)
    # tl.device_print("vecBbar_val",vecBbar_val)

    ## loop over DHQK blocks
    matS_val = tl.zeros((L, L), dtype=tl.float32)
    matH_inter_val = tl.zeros((L, siz_b_DHHV), dtype=tl.float32)
    vecH_inter_denom_val = tl.zeros((L,), dtype=tl.float32)
    for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
        # tl.device_print("idx_b_DHQK", idx_b_DHQK)
        # define pointers for iteration
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
            shape=(DHQK, S),
            strides=(str_matQK_DHQK, str_matQK_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matC_km1_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + idx_b_NC * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecN_km1_ptr = (
            vecN_states
            + idx_b_BNH * str_vecNstates_B_NH
            + idx_b_NC * DHQK
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )

        # load matQ block (L, siz_b_DHQK)
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)
        # load matK transposed block (L, siz_b_DHQK)
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)

        # accumulate matS (L, L)
        matS_val += (tl.dot(matQ_val, matK_val) * qk_scale).to(DTYPE)

        # compute matQbar (L, siz_b_DHQK)
        # tl.static_print("matQ_val", matQ_val)
        # tl.static_print("vecBbar_val", vecBbar_val[:, None])
        # tl.static_print("qk_scale", qk_scale)
        matQbar_val = (matQ_val * vecBbar_val[:, None] * qk_scale).to(DTYPE)

        # load matC_kminus1_tile (siz_b_DHQK, siz_b_DHHV)
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(DTYPE)
        # accumulate matH_k_inter (L, siz_b_DHHV)
        matH_inter_val += tl.dot(matQbar_val, matC_km1_val)

        # load vecN_km1 (siz_b_DHQK,)
        vecN_km1_val = tl.load(vecN_km1_ptr).to(DTYPE)
        # accumulate vecH_k_inter_denom (L,)
        vecH_inter_denom_val += tl.sum(
            (matQbar_val * vecN_km1_val[None, :]).to(tl.float32), axis=1
        )

    ## loop end

    # compute matSbar (L, L)
    matSbar_val = (matS_val * matDbar_val).to(DTYPE)

    # load matV (L, siz_b_DHHV)
    matV_ptr = tl.make_block_ptr(
        base=matV + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)

    # compute matH_k_intra (L, siz_b_DHHV)
    matH_intra_val = tl.dot(matSbar_val, matV_val)
    # compute vecH_k_intra_denom (L,)
    vecH_intra_denom_val = tl.sum(matSbar_val.to(tl.float32), axis=1)

    # compute matH_k_num (L, siz_b_DHHV)
    matH_num_val = matH_inter_val + matH_intra_val

    # compute H_k_denom (L,)
    vecH_denom_val = tl.maximum(
        tl.abs(vecH_inter_denom_val + vecH_intra_denom_val), tl.exp(-vecM_combine_val)
    )

    # compute matH_k_out (L, siz_b_DHHV)
    matHout_val = matH_num_val / (vecH_denom_val[:, None] + EPS)

    # store matH_k_out, vecN_k_denom, vecM_k_combine
    matHout_ptr = tl.make_block_ptr(
        base=matHout + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    vecNout_ptr = (
        vecNout
        + idx_b_BNH * str_vecMN_B_NH
        + (idx_b_NC * L + tl.arange(0, L)) * str_vecMN_S
    )
    vecMout_ptr = (
        vecMout
        + idx_b_BNH * str_vecMN_B_NH
        + (idx_b_NC * L + tl.arange(0, L)) * str_vecMN_S
    )
    tl.store(matHout_ptr, matHout_val.to(DTYPE), boundary_check=(0, 1))
    tl.store(vecNout_ptr, vecH_denom_val.to(tl.float32))
    tl.store(vecMout_ptr, vecM_combine_val.to(tl.float32))
