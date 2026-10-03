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
"""Evaluation-driven slice sampling, following Iman Faisal's FSM scheduler.

A scalar tick advances interval construction, shrinkage and acceptance checks.
The caller schedules ticks and successive moves. The slice function has one
shared call site.

Interval strategies implement Neal (2003), Figures 3--6. The proposal defines
the slice independently of the interval strategy.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax, random

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.slice import direction_proposal, init
from blackjax.types import Array

_LEFT, _RIGHT, _SHRINK, _EXPAND_LEFT, _EXPAND_RIGHT, _CHECK_LEFT, _CHECK_RIGHT = range(
    7
)
_DONE = -1


class SliceInfo(NamedTuple):
    is_accepted: Array
    num_evaluations: Array
    num_expansions: Array
    num_shrink: Array


class SteppingOutState(NamedTuple):
    position: Any
    logdensity: Array
    phase: Array = _LEFT
    left: Array = 0.0
    right: Array = 0.0
    level: Array = 0.0
    left_steps: Array = 0
    right_steps: Array = 0


class DoublingState(NamedTuple):
    position: Any
    logdensity: Array
    phase: Array = _LEFT
    left: Array = 0.0
    right: Array = 0.0
    level: Array = 0.0
    remaining: Array = 0
    other_inside: Array = False
    expanded_left: Array = 0.0
    expanded_right: Array = 0.0
    check_left: Array = 0.0
    check_right: Array = 0.0
    trial_t: Array = 0.0


def _shrink_bracket(state, t):
    left = jnp.where(t < 0, t, state.left)
    right = jnp.where(t >= 0, t, state.right)
    return state._replace(left=left, right=right, phase=_SHRINK)


def _next_phase(state):
    j, k = state.left_steps, state.right_steps
    phase = jnp.where((state.phase == _LEFT) & (j <= 0), _RIGHT, state.phase)
    phase = jnp.where((phase == _RIGHT) & (k <= 0), _SHRINK, phase)
    return state._replace(phase=phase)


def init_stepping_out(rng_key, state, width, max_expansions):
    level_key, bracket_key, budget_key = random.split(rng_key, 3)
    dtype = state.logdensity.dtype
    left = -width * random.uniform(bracket_key, dtype=dtype)
    level = state.logdensity + jnp.log(random.uniform(level_key, dtype=dtype))
    state = state._replace(
        phase=_LEFT,
        left=left,
        right=left + width,
        level=level,
    )
    j = jnp.floor(max_expansions * random.uniform(budget_key)).astype(int)
    k = max_expansions - 1 - j
    return _next_phase(state._replace(left_steps=j, right_steps=k))


def build_stepping_out_kernel(slice_fn, width):
    """Fixed-width stepping-out with Neal's randomly divided expansion budget.

    The initial bracket counts towards max_expansions; zero and one both perform
    no expansion. Shrinkage is not truncated.
    """

    def kernel(rng_key, state):
        span = state.right - state.left
        shrinking = state.phase == _SHRINK
        t = lax.switch(
            state.phase,
            (
                lambda state: state.left,
                lambda state: state.right,
                lambda state: random.uniform(
                    rng_key,
                    dtype=state.left.dtype,
                    minval=state.left,
                    maxval=state.right,
                ),
            ),
            state,
        )
        candidate, is_valid = slice_fn(t)
        inside = is_valid & (candidate.logdensity >= state.level)
        j, k = state.left_steps, state.right_steps

        def expand_left(_):
            new_state = state._replace(
                left=state.left - width * inside,
                phase=jnp.where(inside, _LEFT, _RIGHT),
                left_steps=jnp.where(inside, j - 1, 0),
            )
            return _next_phase(new_state)

        def expand_right(_):
            new_state = state._replace(
                right=state.right + width * inside,
                phase=jnp.where(inside, _RIGHT, _SHRINK),
                right_steps=jnp.where(inside, k - 1, 0),
            )
            return _next_phase(new_state)

        def accept_or_shrink(_):
            return lax.cond(
                inside,
                lambda state: state._replace(phase=_DONE, **candidate._asdict()),
                lambda state: _shrink_bracket(state, t),
                state,
            )

        state = lax.switch(
            state.phase, (expand_left, expand_right, accept_or_shrink), None
        )
        info = SliceInfo(
            state.phase == _DONE,
            1,
            (state.right - state.left > span).astype(int),
            jnp.asarray(shrinking, dtype=int),
        )
        return state, info

    return kernel


def init_doubling(rng_key, state, width, max_expansions):
    level_key, bracket_key = random.split(rng_key)
    dtype = state.logdensity.dtype
    left = -width * random.uniform(bracket_key, dtype=dtype)
    level = state.logdensity + jnp.log(random.uniform(level_key, dtype=dtype))
    state = state._replace(
        phase=jnp.where(max_expansions > 0, _LEFT, _SHRINK),
        left=left,
        right=left + width,
        level=level,
        remaining=max_expansions,
        other_inside=False,
        expanded_left=left,
        expanded_right=left + width,
        check_left=left,
        check_right=left + width,
        trial_t=left,
    )
    return state


def build_doubling_kernel(slice_fn, width):
    """Random-side doubling and Neal's reverse-construction acceptance test.

    Each tick evaluates an initial endpoint, a newly doubled endpoint, a
    shrinkage candidate or a reverse-construction endpoint.
    """

    def expand(rng_key, state, inside, left_endpoint):
        def extend(state):
            side = random.bernoulli(rng_key)
            span = state.right - state.left
            left = state.left - jnp.where(side, span, 0)
            right = state.right + jnp.where(side, 0, span)
            state = state._replace(
                phase=jnp.where(side, _EXPAND_LEFT, _EXPAND_RIGHT),
                left=left,
                right=right,
                remaining=state.remaining - 1,
                other_inside=jnp.where(
                    side == left_endpoint, state.other_inside, inside
                ),
                expanded_left=left,
                expanded_right=right,
            )
            return state

        expanding = (state.remaining > 0) & (state.other_inside | inside)
        return lax.cond(
            expanding, extend, lambda state: state._replace(phase=_SHRINK), state
        )

    def bisect(state):
        def body(state):
            mid = (state.check_left + state.check_right) / 2
            return state._replace(
                check_left=jnp.where(state.trial_t >= mid, mid, state.check_left),
                check_right=jnp.where(state.trial_t < mid, mid, state.check_right),
            )

        wide = lambda state: state.check_right - state.check_left > 1.1 * width
        separated = lambda state: (state.check_left > 0) | (state.check_right <= 0)
        needs_check = wide(state)
        state = lax.cond(needs_check, body, lambda state: state, state)
        state = lax.while_loop(
            lambda state: wide(state) & ~separated(state), body, state
        )
        check = needs_check & separated(state)
        return state._replace(phase=jnp.where(check, _CHECK_LEFT, _DONE))

    def kernel(rng_key, state):
        span = state.right - state.left
        shrinking = state.phase == _SHRINK
        t = lax.switch(
            state.phase,
            (
                lambda state: state.left,
                lambda state: state.right,
                lambda state: random.uniform(
                    rng_key,
                    dtype=state.left.dtype,
                    minval=state.left,
                    maxval=state.right,
                ),
                lambda state: state.left,
                lambda state: state.right,
                lambda state: state.check_left,
                lambda state: state.check_right,
            ),
            state,
        )
        candidate, is_valid = slice_fn(t)
        inside = is_valid & (candidate.logdensity >= state.level)

        def first_left(_):
            return state._replace(phase=_RIGHT, other_inside=inside)

        def first_right(_):
            return expand(rng_key, state, inside, False)

        def check_or_shrink(_):

            def check(state):
                state = state._replace(
                    check_left=state.expanded_left,
                    check_right=state.expanded_right,
                    trial_t=t,
                )
                return bisect(state._replace(**candidate._asdict()))

            return lax.cond(
                inside,
                check,
                lambda state: _shrink_bracket(state, t),
                state,
            )

        def endpoint(_):
            return expand(rng_key, state, inside, state.phase == _EXPAND_LEFT)

        def check_left(_):
            return state._replace(phase=_CHECK_RIGHT, other_inside=inside)

        def check_right(_):
            return lax.cond(
                jnp.logical_not(state.other_inside | inside),
                lambda state: _shrink_bracket(state, state.trial_t),
                bisect,
                state,
            )

        state = lax.switch(
            state.phase,
            (
                first_left,
                first_right,
                check_or_shrink,
                endpoint,
                endpoint,
                check_left,
                check_right,
            ),
            None,
        )

        info = SliceInfo(
            state.phase == _DONE,
            1,
            (state.right - state.left > span).astype(int),
            jnp.asarray(shrinking, dtype=int),
        )
        return state, info

    return kernel


def as_top_level_api(
    logdensity_fn,
    *,
    proposal_generator=direction_proposal(),
    width=1.0,
    interval=build_doubling_kernel,
    max_expansions=10,
) -> SamplingAlgorithm:
    """Complete one slice move by advancing evaluation ticks to acceptance.

    ``interval`` selects ``build_stepping_out_kernel`` or
    ``build_doubling_kernel``. Shrinkage is not truncated.

    .. code:: python

        sampler = slice_fsm.as_top_level_api(logdensity_fn)
        state = sampler.init(position)
        state, info = jax.jit(sampler.step)(rng_key, state)
    """
    state_type, init_move = {
        build_stepping_out_kernel: (SteppingOutState, init_stepping_out),
        build_doubling_kernel: (DoublingState, init_doubling),
    }[interval]

    def kernel(rng_key, state, logdensity_fn):
        rng_key, move_key = random.split(rng_key)
        slice_key, proposal_key = random.split(move_key)
        slice_fn = proposal_generator(proposal_key, state.position, logdensity_fn)
        tick = interval(slice_fn, width)
        particle = state_type(state.position, state.logdensity)
        particle = init_move(slice_key, particle, width, max_expansions)

        def body(carry):
            rng_key, particle, info = carry
            rng_key, step_key = random.split(rng_key)
            particle, tick_info = tick(step_key, particle)
            info = SliceInfo(
                tick_info.is_accepted,
                info.num_evaluations + tick_info.num_evaluations,
                info.num_expansions + tick_info.num_expansions,
                info.num_shrink + tick_info.num_shrink,
            )
            return rng_key, particle, info

        _, particle, info = lax.while_loop(
            lambda carry: ~carry[2].is_accepted,
            body,
            (rng_key, particle, SliceInfo(jnp.asarray(False), 0, 0, 0)),
        )
        state = state._replace(
            position=particle.position, logdensity=particle.logdensity
        )
        return state, info

    return build_sampling_algorithm(kernel, init, logdensity_fn)
