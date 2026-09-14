#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the recurrent part of the backward pass of the mLSTM chunkwise formulation.
It is adapted such that it allows to write out the deltaC states and optionally the deltaN states
at every NC-th chunk.
"""

import triton
import triton.language as tl


@triton.jit
def mlstm_chunkwise__recurrent_bw_dC_kernel(
    matQ,  # (B, NH, S, DHQK)
    vecF,  # (B, NH, NC * L) = (B, NH, S)
    scaM_inter,  # (B, NH, NC+1)
    vecM_combine,  # (B, NH, S)
    matDeltaH,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S)
    matDeltaC_last,  # (B, NH, DHQK, DHHV)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    qk_scale: tl.constexpr,
    str_matQ_B_NH: tl.constexpr,
    str_matQ_S: tl.constexpr,
    str_matQ_DHQK: tl.constexpr,
    str_vecF_B_NH: tl.constexpr,
    str_scaM_inter_B_NH: tl.constexpr,
    str_scaM_inter_NC: tl.constexpr,
    str_vecM_combine_B_NH: tl.constexpr,
    str_vecM_combine_S: tl.constexpr,
    str_matDeltaH_B_NH: tl.constexpr,
    str_matDeltaH_S: tl.constexpr,
    str_matDeltaH_DHHV: tl.constexpr,
    str_vecN_out_B_NH: tl.constexpr,
    str_vecN_out_S: tl.constexpr,
    str_matDeltaC_last_B_NH: tl.constexpr,
    str_matDeltaC_last_DHQK: tl.constexpr,
    str_matDeltaC_last_DHHV: tl.constexpr,
    str_matDeltaC_states_B_NH: tl.constexpr,
    str_matDeltaC_states_NCDHQK: tl.constexpr,
    str_matDeltaC_states_DHHV: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    USE_LAST_STATE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 1e-6,
):
    idx_b_DHQK, idx_b_DHHV, idx_b_NH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # create running deltaC error state in shared memory
    matDeltaC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)

    if USE_LAST_STATE:
        # each thread block loads a (siz_b_DHQK, siz_b_DHHV) tile of matDeltaC_last
        matDeltaC_last_ptr = tl.make_block_ptr(
            base=matDeltaC_last + idx_b_NH * str_matDeltaC_last_B_NH,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_last_DHQK, str_matDeltaC_last_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # load last state
        matDeltaC_k_val = tl.load(matDeltaC_last_ptr, boundary_check=(1, 0)).to(
            tl.float32
        )

    # iterate over chunks from last to first
    for k in range(NC, 0, -1):
        # ? define pointers
        # load matQ in transposed form
        matQ_k_ptr = tl.make_block_ptr(
            base=matQ + idx_b_NH * str_matQ_B_NH,
            shape=(DHQK, S),
            strides=(str_matQ_DHQK, str_matQ_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, (k - 1) * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matDeltaH_ptr = tl.make_block_ptr(
            base=matDeltaH + idx_b_NH * str_matDeltaH_B_NH,
            shape=(S, DHHV),
            strides=(str_matDeltaH_S, str_matDeltaH_DHHV),
            offsets=((k - 1) * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # ? end pointers

        # * store matDeltaC_k_val from previous iteration in HBM
        matDeltaCstates_k_ptr = tl.make_block_ptr(
            base=matDeltaC_states
            + idx_b_NH * str_matDeltaC_states_B_NH
            + k * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        tl.store(
            matDeltaCstates_k_ptr,
            matDeltaC_k_val.to(tl.float32),
            boundary_check=(0, 1),
        )

        # * compute matDeltaC_km1_val
        # load scaG_k, vecB_k, scaM_inter_km1, scaM_inter_k, vecM_combine_k
        # load vecF
        vecF_val = tl.load(
            vecF + idx_b_NH * str_vecF_B_NH + (k - 1) * L + tl.arange(0, L),
        ).to(tl.float32)
        vecFlogsig_val = tl.log(tl.sigmoid(vecF_val))

        vecB_val = tl.cumsum(vecFlogsig_val, axis=0)  # (L,)
        # scaG_k_val is the sum of all forget gates in the current chunk
        scaG_k_val = tl.sum(vecFlogsig_val, axis=0)  # (1,)

        scaM_inter_km1_val = tl.load(
            scaM_inter + idx_b_NH * str_scaM_inter_B_NH + (k - 1)
        ).to(tl.float32)
        scaM_inter_k_val = tl.load(scaM_inter + idx_b_NH * str_scaM_inter_B_NH + k).to(
            tl.float32
        )
        vecM_combine_k_val = tl.load(
            vecM_combine
            + idx_b_NH * str_vecM_combine_B_NH
            + (k - 1) * L
            + tl.arange(0, L)
        ).to(tl.float32)

        # compute scaGbar_k, vecBbar_k
        scaGbar_k_val = tl.exp(scaG_k_val + scaM_inter_km1_val - scaM_inter_k_val)
        vecBbar_k_val = tl.exp(vecB_val + scaM_inter_km1_val - vecM_combine_k_val)

        # compute matQbar_k (DHQK, L) (Note: matQ_k is transposed)
        matQ_k_val = tl.load(matQ_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matQbar_k_val = (matQ_k_val * vecBbar_k_val[None, :] * qk_scale).to(DTYPE)

        # load vecN_out_k, matDeltaH_k
        vecN_out_k_val = tl.load(
            vecN_out + idx_b_NH * str_vecN_out_B_NH + (k - 1) * L + tl.arange(0, L)
        ).to(tl.float32)  # (L,)
        matDeltaH_k_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(
            tl.float32
        )  # (L, DHHV)
        # compute matDeltaHinter_k
        matDeltaH_k_val = (matDeltaH_k_val / (vecN_out_k_val[:, None] + EPS)).to(DTYPE)

        # compute matDeltaC_km1
        matDeltaC_k_val = scaGbar_k_val * matDeltaC_k_val + tl.dot(
            matQbar_k_val, matDeltaH_k_val
        )

    # * store the first state from the last iteration
    matDeltaCstates_0_ptr = tl.make_block_ptr(
        base=matDeltaC_states + idx_b_NH * str_matDeltaC_states_B_NH + 0,
        shape=(DHQK, DHHV),
        strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
        offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(
        matDeltaCstates_0_ptr, matDeltaC_k_val.to(tl.float32), boundary_check=(0, 1)
    )
