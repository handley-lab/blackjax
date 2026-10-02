# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FSM scheduler for vectorized slice sampling.

Implements the finite-state-machine execution of Dance et al.,
"Efficiently Vectorized MCMC on Modern Accelerators" (arXiv:2503.17405),
specialized to a nested-sampling slice transition (initial commit of the
implementation used in Jim; not yet wired into :mod:`blackjax.mcmc.slice`).
A static *segment* (a
contiguous run of slices sharing one waveform-rebuild class) runs as one
``lax.while_loop`` whose body evaluates the likelihood exactly once per
lane per tick. Under ``vmap`` the while-loop batching rule masks finished
lanes, so each lane advances through its own (slice, phase) sequence
instead of waiting at per-ordinal loop barriers.

Bitwise contract: for the same per-slice key sequence this reproduces the
BlackJAX slice kernel (with Jim's cached stepping-out) exactly: the same
endpoint states, expansion and shrink counts, accept flags, and recorded
brackets. All uniforms are pre-drawn from BlackJAX's key-split tree; JAX's
counter-based Threefry PRNG makes draw *order* irrelevant. RBG-family keys
are intentionally unsupported because their non-standard ``vmap`` semantics
are incompatible with asynchronous lane advancement.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from blackjax.types import Array

__all__ = [
    "SegmentSchedule",
    "SegmentInfo",
    "HybridSegmentSchedule",
    "HybridSegmentInfo",
    "slice_randoms_from_keys",
    "make_screened_evaluator",
    "run_segment",
    "run_hybrid_segment",
]

_LEFT, _RIGHT, _SHRINK, _DONE, _INDEPENDENCE = 0, 1, 2, 3, 4


class SegmentSchedule(NamedTuple):
    """Pre-drawn per-slice randomness for one segment, in execution order."""

    directions: Array
    level_u: Array
    bracket_u: Array
    bracket_v: Array
    shrink_key_data: Array


class SegmentInfo(NamedTuple):
    """Per-slice diagnostics matching BlackJAX ``SliceInfo`` semantics."""

    is_accepted: Array
    num_expansions: Array
    num_shrink: Array
    bracket_left: Array
    bracket_right: Array


class HybridSegmentSchedule(NamedTuple):
    """Static slice/MH operations sharing one cache-rebuild class."""

    is_independence: Array
    is_complementary_de: Array
    parameter_index: Array
    proposal_lower: Array
    proposal_upper: Array
    operation_key_data: Array
    slice_info_index: Array
    independence_info_index: Array
    complementary_de_info_index: Array
    complementary_de_displacement: Array
    complementary_de_donor_indices: Array
    complementary_de_donor_policy_violation: Array
    directions: Array
    level_u: Array
    bracket_u: Array
    bracket_v: Array
    shrink_key_data: Array
    n_slices: int
    n_independence: int
    n_complementary_de: int
    complementary_de_complement_size: Array


class HybridSegmentInfo(NamedTuple):
    """Filtered diagnostics for one mixed slice/independence segment."""

    slice_info: SegmentInfo
    independence_acceptances: Array
    complementary_de_acceptances: Array
    complementary_de_donor_indices: Array
    complementary_de_donor_policy_violations: Array
    complementary_de_position_before: Array
    complementary_de_proposal_position: Array
    complementary_de_complement_size: Array


def slice_randoms_from_keys(slice_keys):
    """Split per-slice keys exactly as the BlackJAX slice kernel does.

    Returns ``(prop_keys, level_u, bracket_u, bracket_v, shrink_key_data)``.
    Callers draw directions from ``prop_keys`` and hand the remaining draws to
    :class:`SegmentSchedule`.
    """

    key_impl = str(jax.random.key_impl(slice_keys))
    if key_impl != "threefry2x32":
        raise ValueError(
            "The FSM scheduler requires Threefry PRNG keys; "
            f"{key_impl!r} has batching semantics that cannot preserve "
            "BlackJAX slice paths under asynchronous lane advancement."
        )

    def one(key):
        prop_key, slice_key = jax.random.split(key)
        level_key, interval_key, shrink_key = jax.random.split(slice_key, 3)
        u_key, jk_key = jax.random.split(interval_key)
        return (
            prop_key,
            jax.random.uniform(level_key),
            jax.random.uniform(u_key),
            jax.random.uniform(jk_key),
            jax.random.key_data(shrink_key),
        )

    return jax.vmap(one)(slice_keys)


def make_screened_evaluator(evaluate, upper_bound, log_prior, *, minimum_batch=32):
    """Make a scalar evaluator with an explicitly compacting vmap rule.

    Packs active, unscreened proposals before a batched expensive evaluation.
    The screen is a one-sided likelihood upper bound. A nonfinite bound always
    falls back to the ordinary evaluator. Surviving proposals use the original
    likelihood and cache; no approximation is returned in an accepted state.

    The resulting call accepts position, cache, contour, prior slice level,
    and active flag. The likelihood bound must apply to the same target used
    by ``evaluate``. Strict rejection at bound <= contour matches strict NS
    contours; this interface deliberately does not support auxiliary marks.
    """
    if minimum_batch < 1:
        raise ValueError("minimum_batch must be positive")

    def survivor(position, contour, level, active):
        prior_ok = active & (log_prior(position) >= level)

        def screen(_):
            bound = upper_bound(position)
            return ~jnp.isfinite(bound) | (bound > contour)

        return jax.lax.cond(prior_ok, screen, lambda _: jnp.asarray(False), None)

    @custom_vmap
    def screened(position, cache, contour, level, active):
        keep = survivor(position, contour, level, active)
        rejected = (jnp.asarray(-jnp.inf), jnp.asarray(-jnp.inf), cache)
        return jax.lax.cond(
            keep, lambda _: evaluate(position, cache), lambda _: rejected, None
        )

    @screened.def_vmap
    def batched(axis_size, in_batched, position, cache, contour, level, active):
        values = jax.tree.map(
            lambda value, batch: (
                value
                if batch
                else jnp.broadcast_to(value, (axis_size,) + jnp.shape(value))
            ),
            (position, cache, contour, level, active),
            tuple(in_batched),
        )
        position, cache, contour, level, active = values
        keep = jax.vmap(survivor)(position, contour, level, active)
        count = jnp.sum(keep, dtype=jnp.int32)
        indices = jnp.nonzero(keep, size=axis_size, fill_value=axis_size)[0]
        rejected = (
            jnp.full((axis_size,), -jnp.inf),
            jnp.full((axis_size,), -jnp.inf),
            cache,
        )
        widths = []
        width = min(minimum_batch, axis_size)
        while width < axis_size:
            widths.append(width)
            width *= 2
        widths.append(axis_size)

        def branch(width):
            def run(_):
                selected = indices[:width]
                # Padding reads a valid row and drops its write. It can never
                # overwrite lane zero or a previously screened lane.
                safe = jnp.minimum(selected, axis_size - 1)
                result = jax.vmap(evaluate)(
                    position[safe], jax.tree.map(lambda v: v[safe], cache)
                )
                return jax.tree.map(
                    lambda empty, value: empty.at[selected].set(value, mode="drop"),
                    rejected,
                    result,
                )

            return run

        branch_index = jnp.sum(count > jnp.asarray(widths[:-1]), dtype=jnp.int32)
        result = jax.lax.cond(
            count > 0,
            lambda _: jax.lax.switch(branch_index, [branch(w) for w in widths], None),
            lambda _: rejected,
            None,
        )
        return result, jax.tree.map(lambda _: True, result)

    return screened


class _Carry(NamedTuple):
    slice_idx: Array
    phase: Array
    state: Any
    level: Array
    left: Array
    right: Array
    j_remaining: Array
    k_remaining: Array
    shrink_key_data: Array
    n_expand: Array
    n_shrink: Array
    info: SegmentInfo


class _HybridCarry(NamedTuple):
    op_idx: Array
    phase: Array
    state: Any
    level: Array
    left: Array
    right: Array
    j_remaining: Array
    k_remaining: Array
    shrink_key_data: Array
    n_expand: Array
    n_shrink: Array
    slice_info: SegmentInfo
    independence_acceptances: Array
    complementary_de_acceptances: Array
    complementary_de_position_before: Array
    complementary_de_proposal_position: Array


def _slice_entry(schedule, idx, logdensity, max_expansions, width, shrink_only=False):
    """Bundle initialization of the slice at ``idx``."""
    level = logdensity + jnp.log(schedule.level_u[idx])
    left = -width * schedule.bracket_u[idx]
    right = left + width
    if shrink_only:
        j = jnp.asarray(0)
        k = jnp.asarray(0)
    else:
        j = jnp.floor(max_expansions * schedule.bracket_v[idx]).astype(int)
        k = (max_expansions - 1) - j
    return level, left, right, j, k, schedule.shrink_key_data[idx]


def _normalize_exhausted_endpoints(c: _Carry) -> _Carry:
    """Advance past endpoint predicates whose expansion budget is exhausted."""
    # Normalize left before right: both zero budgets can enter SHRINK in one
    # tick without changing the bracket, accepted state, or random key.
    phase = jnp.where((c.phase == _LEFT) & (c.j_remaining <= 0), _RIGHT, c.phase)
    skip_right = (phase == _RIGHT) & (c.k_remaining <= 0)
    phase = jnp.where(skip_right, _SHRINK, phase)
    # Capture the original bracket before this tick's first shrink proposal
    # can reject and shorten it, just as the ordinary RIGHT transition does.
    info = c.info._replace(
        bracket_left=jnp.where(
            skip_right,
            c.info.bracket_left.at[c.slice_idx].set(c.left),
            c.info.bracket_left,
        ),
        bracket_right=jnp.where(
            skip_right,
            c.info.bracket_right.at[c.slice_idx].set(c.right),
            c.info.bracket_right,
        ),
    )
    return c._replace(phase=phase, info=info)


def run_segment(
    schedule: SegmentSchedule,
    state,
    loglikelihood_0,
    *,
    eval_candidate: Callable,
    wrap_position: Callable,
    max_expansions: int,
    max_shrinkage: int,
    width: float = 1.0,
    shrink_only: bool = False,
    contour_predicate: Callable | None = None,
    likelihood_upper_bound: Callable | None = None,
    screen_log_prior: Callable | None = None,
    screen_minimum_batch: int = 32,
    skip_exhausted_endpoints: bool = False,
):
    """Run every slice of one segment for a single chain.

    ``eval_candidate(position, cache) -> (logdensity, loglikelihood, cache)``
    is the segment's only likelihood site. ``state`` carries ``position``,
    ``logdensity``, ``loglikelihood``, ``loglikelihood_birth``, and ``cache``.

    ``shrink_only`` is a static Python bool: when set, every slice enters the
    shrink phase directly with a fixed unit bracket and never performs
    stepping-out expansions (Neal 2003 sec 4.1).

    ``contour_predicate(loglikelihood)`` optionally replaces the scalar strict
    comparison. The caller must close it over one fixed candidate mark for the
    entire segment.

    ``skip_exhausted_endpoints`` omits pure candidate calls whose endpoint
    predicate cannot affect stepping out because its side's budget is zero.
    Accepted states, keys, brackets, and expansion/shrink counters are unchanged;
    physical callback counts change. It has no effect on shrink-only segments.
    """
    assert max_shrinkage >= 1
    if likelihood_upper_bound is not None:
        if contour_predicate is not None or screen_log_prior is None:
            raise ValueError(
                "likelihood screening needs an ordinary contour and a prior callback"
            )
        screened_evaluate = make_screened_evaluator(
            eval_candidate,
            likelihood_upper_bound,
            screen_log_prior,
            minimum_batch=screen_minimum_batch,
        )

    n_slices = schedule.level_u.shape[0]
    zero_i = jnp.zeros((n_slices,), dtype=jnp.result_type(int))
    info0 = SegmentInfo(
        is_accepted=jnp.zeros((n_slices,), dtype=bool),
        num_expansions=zero_i,
        num_shrink=zero_i,
        bracket_left=jnp.zeros((n_slices,)),
        bracket_right=jnp.zeros((n_slices,)),
    )
    entry_phase = _SHRINK if shrink_only else _LEFT
    level, left, right, j, k, shrink_key_data = _slice_entry(
        schedule, 0, state.logdensity, max_expansions, width, shrink_only=shrink_only
    )
    carry = _Carry(
        slice_idx=jnp.asarray(0),
        phase=jnp.asarray(entry_phase),
        state=state,
        level=level,
        left=left,
        right=right,
        j_remaining=j,
        k_remaining=k,
        shrink_key_data=shrink_key_data,
        n_expand=jnp.asarray(0),
        n_shrink=jnp.asarray(0),
        info=info0,
    )

    def cond(c):
        return c.phase != _DONE

    def body(c):
        if skip_exhausted_endpoints:
            c = _normalize_exhausted_endpoints(c)
        key = jax.random.wrap_key_data(
            c.shrink_key_data,
            impl="threefry2x32",
        )
        next_key, subkey = jax.random.split(key)
        t_shrink = c.left + jax.random.uniform(subkey) * (c.right - c.left)
        in_left = c.phase == _LEFT
        in_right = c.phase == _RIGHT
        in_shrink = c.phase == _SHRINK
        shrink_key_data = jnp.where(
            in_shrink,
            jax.random.key_data(next_key),
            c.shrink_key_data,
        )
        t = jnp.where(
            in_left,
            c.left,
            jnp.where(
                in_right,
                c.right,
                jnp.where(in_shrink, t_shrink, 0.0),
            ),
        )

        direction = schedule.directions[c.slice_idx]
        position = wrap_position(c.state.position + t * direction)
        if likelihood_upper_bound is None:
            logdensity, loglikelihood, cache = eval_candidate(position, c.state.cache)
        else:
            logdensity, loglikelihood, cache = screened_evaluate(
                position, c.state.cache, loglikelihood_0, c.level, c.phase != _DONE
            )
        if contour_predicate is None:
            above_contour = loglikelihood > loglikelihood_0
        else:
            above_contour = contour_predicate(loglikelihood)
        inside = (logdensity >= c.level) & above_contour

        expand_left = in_left & inside & (c.j_remaining > 0)
        to_right = in_left & ~expand_left
        expand_right = in_right & inside & (c.k_remaining > 0)
        to_shrink = in_right & ~expand_right

        found = in_shrink & inside
        n_shrink = c.n_shrink + in_shrink.astype(c.n_shrink.dtype)
        exhausted = in_shrink & ~inside & (n_shrink >= max_shrinkage)
        slice_done = found | exhausted
        keep_shrinking = in_shrink & ~slice_done
        n_expand = c.n_expand + (expand_left | expand_right).astype(c.n_expand.dtype)

        left = jnp.where(
            expand_left,
            c.left - width,
            jnp.where(keep_shrinking & (t < 0.0), t, c.left),
        )
        right = jnp.where(
            expand_right,
            c.right + width,
            jnp.where(keep_shrinking & (t >= 0.0), t, c.right),
        )

        candidate = c.state._replace(
            position=position,
            logdensity=logdensity,
            loglikelihood=loglikelihood,
            cache=cache,
        )
        new_state = jax.tree.map(
            lambda new, old: jnp.where(found, new, old), candidate, c.state
        )

        info = c.info._replace(
            bracket_left=jnp.where(
                to_shrink,
                c.info.bracket_left.at[c.slice_idx].set(left),
                c.info.bracket_left,
            ),
            bracket_right=jnp.where(
                to_shrink,
                c.info.bracket_right.at[c.slice_idx].set(right),
                c.info.bracket_right,
            ),
        )
        info = jax.tree.map(
            lambda arr, val: jnp.where(slice_done, arr.at[c.slice_idx].set(val), arr),
            info,
            SegmentInfo(
                is_accepted=found,
                num_expansions=n_expand,
                num_shrink=n_shrink,
                bracket_left=info.bracket_left[c.slice_idx],
                bracket_right=info.bracket_right[c.slice_idx],
            ),
        )

        next_idx = c.slice_idx + slice_done.astype(c.slice_idx.dtype)
        segment_done = slice_done & (next_idx >= n_slices)
        entry_idx = jnp.minimum(next_idx, n_slices - 1)
        e_level, e_left, e_right, e_j, e_k, e_key_data = _slice_entry(
            schedule,
            entry_idx,
            new_state.logdensity,
            max_expansions,
            width,
            shrink_only=shrink_only,
        )
        advance = slice_done & ~segment_done

        phase = jnp.where(
            segment_done,
            _DONE,
            jnp.where(
                advance,
                entry_phase,
                jnp.where(
                    to_right,
                    _RIGHT,
                    jnp.where(
                        to_shrink | keep_shrinking,
                        _SHRINK,
                        c.phase,
                    ),
                ),
            ),
        )
        return _Carry(
            slice_idx=jnp.where(advance, next_idx, c.slice_idx),
            phase=phase,
            state=new_state,
            level=jnp.where(advance, e_level, c.level),
            left=jnp.where(advance, e_left, left),
            right=jnp.where(advance, e_right, right),
            j_remaining=jnp.where(
                advance,
                e_j,
                c.j_remaining - expand_left.astype(c.j_remaining.dtype),
            ),
            k_remaining=jnp.where(
                advance,
                e_k,
                c.k_remaining - expand_right.astype(c.k_remaining.dtype),
            ),
            shrink_key_data=jnp.where(advance, e_key_data, shrink_key_data),
            n_expand=jnp.where(advance, 0, n_expand),
            n_shrink=jnp.where(advance, 0, n_shrink),
            info=info,
        )

    final = jax.lax.while_loop(cond, body, carry)
    return final.state, final.info


def run_hybrid_segment(
    schedule: HybridSegmentSchedule,
    state,
    loglikelihood_0,
    *,
    eval_candidate: Callable,
    wrap_position: Callable,
    max_expansions: int,
    max_shrinkage: int,
    width: float = 1.0,
    contour_predicate: Callable | None = None,
):
    """Run ordered slice and one-tick independence-MH operations in one FSM.

    An optional ``contour_predicate`` is shared by slice membership and every
    MH hard-constraint check in the segment.
    """
    assert max_shrinkage >= 1

    n_ops = schedule.is_independence.shape[0]
    if n_ops < 1:
        raise ValueError("A hybrid FSM segment must contain at least one operation")
    slice_storage = max(schedule.n_slices, 1)
    independence_storage = max(schedule.n_independence, 1)
    complementary_de_storage = max(schedule.n_complementary_de, 1)
    zero_i = jnp.zeros((slice_storage,), dtype=jnp.result_type(int))
    slice_info0 = SegmentInfo(
        is_accepted=jnp.zeros((slice_storage,), dtype=bool),
        num_expansions=zero_i,
        num_shrink=zero_i,
        bracket_left=jnp.zeros((slice_storage,)),
        bracket_right=jnp.zeros((slice_storage,)),
    )
    level, left, right, j, k, shrink_key_data = _slice_entry(
        schedule, 0, state.logdensity, max_expansions, width
    )
    carry = _HybridCarry(
        op_idx=jnp.asarray(0),
        phase=jnp.where(
            schedule.is_independence[0] | schedule.is_complementary_de[0],
            _INDEPENDENCE,
            _LEFT,
        ),
        state=state,
        level=level,
        left=left,
        right=right,
        j_remaining=j,
        k_remaining=k,
        shrink_key_data=shrink_key_data,
        n_expand=jnp.asarray(0),
        n_shrink=jnp.asarray(0),
        slice_info=slice_info0,
        independence_acceptances=jnp.zeros((independence_storage,), dtype=bool),
        complementary_de_acceptances=jnp.zeros((complementary_de_storage,), dtype=bool),
        complementary_de_position_before=jnp.zeros(
            (complementary_de_storage,) + state.position.shape,
            dtype=state.position.dtype,
        ),
        complementary_de_proposal_position=jnp.zeros(
            (complementary_de_storage,) + state.position.shape,
            dtype=state.position.dtype,
        ),
    )

    def cond(c):
        return c.phase != _DONE

    def body(c):
        is_independence = schedule.is_independence[c.op_idx]
        is_complementary_de = schedule.is_complementary_de[c.op_idx]
        is_metropolis = is_independence | is_complementary_de

        shrink_key = jax.random.wrap_key_data(
            c.shrink_key_data,
            impl="threefry2x32",
        )
        next_shrink_key, shrink_subkey = jax.random.split(shrink_key)
        t_shrink = c.left + jax.random.uniform(shrink_subkey) * (c.right - c.left)
        in_left = (c.phase == _LEFT) & ~is_metropolis
        in_right = (c.phase == _RIGHT) & ~is_metropolis
        in_shrink = (c.phase == _SHRINK) & ~is_metropolis
        shrink_key_data = jnp.where(
            in_shrink,
            jax.random.key_data(next_shrink_key),
            c.shrink_key_data,
        )
        t = jnp.where(
            in_left,
            c.left,
            jnp.where(in_right, c.right, jnp.where(in_shrink, t_shrink, 0.0)),
        )
        slice_position = wrap_position(
            c.state.position + t * schedule.directions[c.op_idx]
        )

        operation_key = jax.random.wrap_key_data(
            schedule.operation_key_data[c.op_idx],
            impl="threefry2x32",
        )
        proposal_key, accept_key = jax.random.split(operation_key)
        proposed_value = jax.random.uniform(
            proposal_key,
            minval=schedule.proposal_lower[c.op_idx],
            maxval=schedule.proposal_upper[c.op_idx],
        ).astype(c.state.position.dtype)
        independence_position = c.state.position.at[
            schedule.parameter_index[c.op_idx]
        ].set(proposed_value)
        complementary_de_position = wrap_position(
            c.state.position + schedule.complementary_de_displacement[c.op_idx]
        )
        position = jnp.where(
            is_independence,
            independence_position,
            jnp.where(
                is_complementary_de,
                complementary_de_position,
                slice_position,
            ),
        )

        logdensity, loglikelihood, cache = eval_candidate(position, c.state.cache)
        if contour_predicate is None:
            above_contour = loglikelihood > loglikelihood_0
        else:
            above_contour = contour_predicate(loglikelihood)
        inside = (logdensity >= c.level) & above_contour
        independence_accepted = (
            is_independence
            & above_contour
            & (
                jnp.log(jax.random.uniform(accept_key))
                < logdensity - c.state.logdensity
            )
        )
        complementary_de_accepted = (
            is_complementary_de
            & ~schedule.complementary_de_donor_policy_violation[
                schedule.complementary_de_info_index[c.op_idx]
            ]
            & above_contour
            & (
                jnp.log(jax.random.uniform(accept_key))
                < logdensity - c.state.logdensity
            )
        )

        expand_left = in_left & inside & (c.j_remaining > 0)
        to_right = in_left & ~expand_left
        expand_right = in_right & inside & (c.k_remaining > 0)
        to_shrink = in_right & ~expand_right
        found = in_shrink & inside
        n_shrink = c.n_shrink + in_shrink.astype(c.n_shrink.dtype)
        exhausted = in_shrink & ~inside & (n_shrink >= max_shrinkage)
        slice_done = found | exhausted
        keep_shrinking = in_shrink & ~slice_done
        n_expand = c.n_expand + (expand_left | expand_right).astype(c.n_expand.dtype)

        left = jnp.where(
            expand_left,
            c.left - width,
            jnp.where(keep_shrinking & (t < 0.0), t, c.left),
        )
        right = jnp.where(
            expand_right,
            c.right + width,
            jnp.where(keep_shrinking & (t >= 0.0), t, c.right),
        )
        candidate = c.state._replace(
            position=position,
            logdensity=logdensity,
            loglikelihood=loglikelihood,
            cache=cache,
        )
        take_candidate = found | independence_accepted | complementary_de_accepted
        new_state = jax.tree.map(
            lambda new, old: jnp.where(take_candidate, new, old),
            candidate,
            c.state,
        )

        slice_idx = schedule.slice_info_index[c.op_idx]
        write_bracket = ~is_independence & to_shrink
        slice_info = c.slice_info._replace(
            bracket_left=jnp.where(
                write_bracket,
                c.slice_info.bracket_left.at[slice_idx].set(left),
                c.slice_info.bracket_left,
            ),
            bracket_right=jnp.where(
                write_bracket,
                c.slice_info.bracket_right.at[slice_idx].set(right),
                c.slice_info.bracket_right,
            ),
        )
        write_slice = ~is_independence & slice_done
        slice_info = jax.tree.map(
            lambda arr, val: jnp.where(write_slice, arr.at[slice_idx].set(val), arr),
            slice_info,
            SegmentInfo(
                is_accepted=found,
                num_expansions=n_expand,
                num_shrink=n_shrink,
                bracket_left=slice_info.bracket_left[slice_idx],
                bracket_right=slice_info.bracket_right[slice_idx],
            ),
        )
        independence_idx = schedule.independence_info_index[c.op_idx]
        independence_acceptances = jnp.where(
            is_independence,
            c.independence_acceptances.at[independence_idx].set(independence_accepted),
            c.independence_acceptances,
        )
        complementary_de_idx = schedule.complementary_de_info_index[c.op_idx]
        complementary_de_acceptances = jnp.where(
            is_complementary_de,
            c.complementary_de_acceptances.at[complementary_de_idx].set(
                complementary_de_accepted
            ),
            c.complementary_de_acceptances,
        )
        complementary_de_position_before = jnp.where(
            is_complementary_de,
            c.complementary_de_position_before.at[complementary_de_idx].set(
                c.state.position
            ),
            c.complementary_de_position_before,
        )
        complementary_de_proposal_position = jnp.where(
            is_complementary_de,
            c.complementary_de_proposal_position.at[complementary_de_idx].set(position),
            c.complementary_de_proposal_position,
        )

        op_done = is_metropolis | slice_done
        next_idx = c.op_idx + op_done.astype(c.op_idx.dtype)
        segment_done = op_done & (next_idx >= n_ops)
        entry_idx = jnp.minimum(next_idx, n_ops - 1)
        e_level, e_left, e_right, e_j, e_k, e_key_data = _slice_entry(
            schedule,
            entry_idx,
            new_state.logdensity,
            max_expansions,
            width,
        )
        advance = op_done & ~segment_done
        entry_phase = jnp.where(
            schedule.is_independence[entry_idx]
            | schedule.is_complementary_de[entry_idx],
            _INDEPENDENCE,
            _LEFT,
        )
        phase = jnp.where(
            segment_done,
            _DONE,
            jnp.where(
                advance,
                entry_phase,
                jnp.where(
                    to_right,
                    _RIGHT,
                    jnp.where(to_shrink | keep_shrinking, _SHRINK, c.phase),
                ),
            ),
        )
        return _HybridCarry(
            op_idx=jnp.where(advance, next_idx, c.op_idx),
            phase=phase,
            state=new_state,
            level=jnp.where(advance, e_level, c.level),
            left=jnp.where(advance, e_left, left),
            right=jnp.where(advance, e_right, right),
            j_remaining=jnp.where(
                advance,
                e_j,
                c.j_remaining - expand_left.astype(c.j_remaining.dtype),
            ),
            k_remaining=jnp.where(
                advance,
                e_k,
                c.k_remaining - expand_right.astype(c.k_remaining.dtype),
            ),
            shrink_key_data=jnp.where(advance, e_key_data, shrink_key_data),
            n_expand=jnp.where(advance, 0, n_expand),
            n_shrink=jnp.where(advance, 0, n_shrink),
            slice_info=slice_info,
            independence_acceptances=independence_acceptances,
            complementary_de_acceptances=complementary_de_acceptances,
            complementary_de_position_before=complementary_de_position_before,
            complementary_de_proposal_position=complementary_de_proposal_position,
        )

    final = jax.lax.while_loop(cond, body, carry)
    return final.state, HybridSegmentInfo(
        slice_info=jax.tree.map(lambda x: x[: schedule.n_slices], final.slice_info),
        independence_acceptances=final.independence_acceptances[
            : schedule.n_independence
        ],
        complementary_de_acceptances=final.complementary_de_acceptances[
            : schedule.n_complementary_de
        ],
        complementary_de_donor_indices=schedule.complementary_de_donor_indices[
            : schedule.n_complementary_de
        ],
        complementary_de_donor_policy_violations=(
            schedule.complementary_de_donor_policy_violation[
                : schedule.n_complementary_de
            ]
        ),
        complementary_de_position_before=final.complementary_de_position_before[
            : schedule.n_complementary_de
        ],
        complementary_de_proposal_position=(
            final.complementary_de_proposal_position[: schedule.n_complementary_de]
        ),
        complementary_de_complement_size=schedule.complementary_de_complement_size,
    )
