#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Triton.

Forward recurrent kernel of the mLSTM chunkwise formulation.

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


@triton.jit
def mlstm_chunkwise__recurrent_fw_C_kernel(
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    matC_initial,  # (B, NH, DHQK, DHHV)
    vecN_initial,  # (B, NH, DHQK)
    scaMinter_initial,  # (B, NH)
    matC_states,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states,  # (B, NH, (NC + 1))
    str_matK_B_NH,
    str_matK_S,
    str_matK_DHQK,
    str_matV_B_NH,
    str_matV_S,
    str_matV_DHHV,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_matCstates_B_NH,
    str_matCstates_NCDHQK,
    str_matCstates_DHHV,
    str_vecNstates_B_NH,
    str_vecNstates_NCDHQK,
    str_scaMinterstates_B_NH,
    str_scaMinterstates_NC,
    str_matCinitial_B_NH,
    str_matCinitial_DHQK,
    str_matCinitial_DHHV,
    str_vecNinitial_B_NH,
    str_vecNinitial_DHQK,
    str_scaMinterinitial_B_NH,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    idx_b_DHQK, idx_b_DHHV, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # create running states in shared memory
    matC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)
    vecN_k_val = tl.zeros((siz_b_DHQK,), dtype=tl.float32)
    scaMinter_k_val = 0.0  # tl.zeros((1,), dtype=tl.float32)

    if USE_INITIAL_STATE:
        # each thread block loads a (siz_b_DHQK, siz_b_DHHV) block from matC_initial
        matCinitial_ptr = tl.make_block_ptr(
            base=matC_initial + idx_b_BNH * str_matCinitial_B_NH,
            shape=(DHQK, DHHV),
            strides=(str_matCinitial_DHQK, str_matCinitial_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # each thread block loads a (siz_b_DHQK,) chunk from vecN_initial
        vecNinitial_ptr = (
            vecN_initial
            + idx_b_BNH * str_vecNinitial_B_NH
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH

        # load initial states
        matC_k_val = tl.load(matCinitial_ptr, boundary_check=(0, 1)).to(tl.float32)
        vecN_k_val = tl.load(vecNinitial_ptr).to(tl.float32)
        scaMinter_k_val = tl.load(scaMinterinitial_ptr).to(tl.float32)

    # iterate over chunks
    for k in range(NC):
        # load matK in transposed form
        matK_k_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * str_matK_B_NH,
            shape=(DHQK, S),
            strides=(str_matK_DHQK, str_matK_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, k * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matV_k_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matV_B_NH,
            shape=(S, DHHV),
            strides=(str_matV_S, str_matV_DHHV),
            offsets=(k * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # create pointer for matCstates_k, vecNstates_k, scaMinterstates_k
        # each thread block stores a (siz_b_DHQK, siz_b_DHHV) block to matC_states_k
        matCstates_k_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + k * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecNstates_k_ptr = (
            vecN_states
            + idx_b_BNH * str_vecNstates_B_NH
            + k * DHQK
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        scaMinterstates_k_ptr = (
            scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + k
        )

        # store the states from the previous iteration
        tl.store(
            matCstates_k_ptr, matC_k_val.to(dtype=tl.float32), boundary_check=(0, 1)
        )
        if idx_b_DHHV == 0:
            tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=tl.float32))
        if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
            tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=tl.float32))

        # load / compute vecA_k, scaG_k
        # last element of vecB in k-th chunk
        vecB_last_k_val = tl.load(
            vecB + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + (L - 1)
        ).to(tl.float32)
        vecB_k_val = tl.load(
            vecB + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + tl.arange(0, L)
        ).to(tl.float32)

        vecI_k_val = tl.load(
            vecI + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + tl.arange(0, L)
        ).to(tl.float32)

        vecA_k_val = (vecB_last_k_val - vecB_k_val) + vecI_k_val
        scaG_k_val = vecB_last_k_val

        # scaM_inter_k update
        scaAmax_k_val = tl.max(vecA_k_val)
        scaMinter_next_val = tl.maximum(scaG_k_val + scaMinter_k_val, scaAmax_k_val)

        # load matK_k, matV_k
        matK_k_val = tl.load(matK_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matV_k_val = tl.load(matV_k_ptr, boundary_check=(0, 1)).to(DTYPE)

        # matC_k update
        vecAbar_k_val = tl.exp(vecA_k_val - scaMinter_next_val)
        scaGbar_k_val = tl.exp(scaG_k_val + scaMinter_k_val - scaMinter_next_val)

        matKbar_k_val = (matK_k_val * vecAbar_k_val[None, :]).to(DTYPE)

        matC_k_val = scaGbar_k_val * matC_k_val + tl.dot(matKbar_k_val, matV_k_val)

        # vecN_k update
        vecN_k_val = scaGbar_k_val * vecN_k_val + tl.sum(
            matKbar_k_val.to(tl.float32), axis=1
        )

        # move to next iteration
        scaMinter_k_val = scaMinter_next_val

    # store the states from the last iteration
    matCstates_k_ptr = tl.make_block_ptr(
        base=matC_states + idx_b_BNH * str_matCstates_B_NH + NC * DHQK * DHHV,
        shape=(DHQK, DHHV),
        strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
        offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(1, 0),
    )
    vecNstates_k_ptr = (
        vecN_states
        + idx_b_BNH * str_vecNstates_B_NH
        + NC * DHQK
        + idx_b_DHQK * siz_b_DHQK
        + tl.arange(0, siz_b_DHQK)
    )
    scaMinterstates_k_ptr = scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + NC
    tl.store(matCstates_k_ptr, matC_k_val.to(dtype=tl.float32), boundary_check=(0, 1))
    if idx_b_DHHV == 0:
        tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=tl.float32))
    if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
        tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=tl.float32))
