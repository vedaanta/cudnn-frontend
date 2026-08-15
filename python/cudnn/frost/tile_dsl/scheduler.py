# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from .constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL  # noqa: F401

from typing import NamedTuple

from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .barrier import PipelineState, advance, wait, arrive_expect_tx


@cute.jit
def read_tile_id_arrive(mb, cga_size: int):
    if cga_size == 1:
        if nvvm.elect_sync():
            nvvm.mbarrier_arrive(mb)
    else:
        lane_stride = 32 // cga_size
        lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
        for i in cutlass.range_constexpr(cga_size):
            target_lane = i * lane_stride
            if lane == cutlass.Int32(target_lane):
                peer_mb = nvvm.mapa(mb, cutlass.Int32(i))
                # RELEASE (not relaxed) arrive: this signal gates the
                # scheduler's next try_cancel REFILL of the payload slot this
                # warp reads, so the warp's prior payload loads must be
                # ordered-before the arrive is visible (canonical CLC
                # consumer_release is likewise a release arrive).
                nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER)


class Sched(NamedTuple):
    mb_scheduler: object
    mb_read_tile_id: object
    tile_id_smem: object
    bidx_init: object
    bidy_init: object
    bidz_init: object


@cute.jit
def read_clc_payload(sched, idx):
    """Decode one try_cancel response slot from a SINGLE atomic 128-bit load
    (LDS.128) of the opaque response, mirroring the canonical
    ``cute.arch.clc_response`` decode (one vector load + register extracts).

    The single vectorized load gives every consumer warp a CONSISTENT
    (ctaid, valid) snapshot — three separate 32-bit raw loads could tear
    against the asynchronously landing response bytes.

    Returns ``(first_ctaid_x, first_ctaid_y, is_valid)`` — is_valid is 0/1.
    The trailing cross-proxy fence orders this generic-proxy read before the
    scheduler's NEXT async-proxy try_cancel write into the slot (required by
    the canonical CLC pattern, see the DSL's dynamic_persistent_tile_scheduler
    "insert_fence" note).
    """
    vec = sched.tile_id_smem.load(idx * cutlass.Int32(8), vector_size=4, alignment=16)
    nvvm.fence_proxy("async.shared", space="cta")
    return vec[0], vec[1], vec[2] & cutlass.Int32(1)


@cute.jit
def scheduler_warp_loop(sched, sched_stages: int, is_cga_first_cta, cga_size: int = 1):
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)

        # Canonical CLC ordering (CUTLASS PipelineClcFetchAsync): the FIRST
        # CTA's scheduler warp arms arrive+expect_tx(16) on EVERY CTA's full
        # barrier, program-ordered BEFORE its try_cancel issue, so the
        # multicast response's complete-tx can never land on a peer barrier
        # whose expect_tx hasn't been armed yet.  (Per-CTA local arming left
        # the non-leader's expect_tx unordered against the leader's issue.)
        if nvvm.elect_sync() and is_cga_first_cta:
            for i in cutlass.range_constexpr(cga_size):
                if cutlass.const_expr(i == 0):
                    arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
                else:
                    peer_mb = nvvm.mapa(sched.mb_scheduler.subview(state.idx), cutlass.Int32(i))
                    nvvm.mbarrier_arrive_expect_tx(peer_mb, 16, scope=nvvm.MemScope.CLUSTER)
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        _m, _n, is_valid = read_clc_payload(sched, state.idx)

        state = advance(state, sched_stages)
