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

A scalar kernel advances through interval construction, shrinkage, acceptance
checks and successive moves in one loop. Under vmap, chains need not wait for
one another between these stages. The slice function has one shared call site.

Interval strategies implement Neal (2003), Figures 3--6. The proposal defines
the slice independently of the interval strategy.
"""

from collections import namedtuple
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from blackjax.types import Array, PRNGKey

_LEFT, _RIGHT, _SHRINK, _EXPAND, _CHECK_LEFT, _CHECK_RIGHT = range(6)
_DONE = -1


class SliceInfo(NamedTuple):
    num_evaluations: Array
    num_expansions: Array
    num_shrink: Array


class SliceState(NamedTuple):
    position: Any
    logdensity: Array
    rng_key: PRNGKey = None
    proposal: Any = ()
    index: Array = 0
    phase: Array = _LEFT
    left: Array = 0.0
    right: Array = 0.0
    t: Array = 0.0
    level: Array = 0.0
    num_evaluations: Array = 0
    num_expansions: Array = 0
    num_shrink: Array = 0


SteppingOutState = namedtuple(
    "SteppingOutState",
    (*SliceState._fields, "left_steps", "right_steps"),
    defaults=(*SliceState._field_defaults.values(), 0, 0),
)

DoublingState = namedtuple(
    "DoublingState",
    (
        *SliceState._fields,
        "remaining",
        "side",
        "left_inside",
        "right_inside",
        "expanded_left",
        "expanded_right",
        "check_left",
        "check_right",
        "separated",
        "trial_t",
    ),
    defaults=(
        *SliceState._field_defaults.values(),
        0,
        False,
        False,
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        False,
        0.0,
    ),
)


def init(rng_key, state, width):
    rng_key, level_key, bracket_key = random.split(rng_key, 3)
    dtype = state.logdensity.dtype
    left = -width * random.uniform(bracket_key, dtype=dtype)
    level = state.logdensity + jnp.log(random.uniform(level_key, dtype=dtype))
    return state._replace(
        rng_key=rng_key,
        phase=_LEFT,
        left=left,
        right=left + width,
        t=left,
        level=level,
    )


def _draw(s):
    rng_key, subkey = random.split(s.rng_key)
    t = s.left + random.uniform(subkey, dtype=s.left.dtype) * (s.right - s.left)
    return s._replace(rng_key=rng_key, phase=_SHRINK, t=t)


def _reject(s, t):
    left = jnp.where(t < 0, t, s.left)
    right = jnp.where(t >= 0, t, s.right)
    return _draw(s._replace(left=left, right=right))


def stepping_out(width, max_expansions):
    """Fixed-width stepping-out with Neal's randomly divided expansion budget.

    The initial bracket counts towards max_expansions; zero and one both perform
    no expansion. Shrinkage is not truncated.
    """

    def init_fn(rng_key, state):
        rng_key, budget_key = random.split(rng_key)
        s = init(rng_key, state, width)
        j = jnp.floor(max_expansions * random.uniform(budget_key)).astype(int)
        k = max_expansions - 1 - j
        return request(s._replace(left_steps=j, right_steps=k))

    def request(s):
        j, k = s.left_steps, s.right_steps
        phase = jnp.where((s.phase == _LEFT) & (j <= 0), _RIGHT, s.phase)
        phase = jnp.where((phase == _RIGHT) & (k <= 0), _SHRINK, phase)
        s = s._replace(phase=phase, t=jnp.where(phase == _LEFT, s.left, s.right))
        return lax.cond(phase == _SHRINK, _draw, lambda s: s, s)

    def update(s, candidate, inside):
        j, k = s.left_steps, s.right_steps

        def left(_):
            expand = inside & (j > 0)
            next_s = s._replace(
                left=s.left - width * expand,
                phase=jnp.where(expand, _LEFT, _RIGHT),
                num_expansions=s.num_expansions + expand,
            )
            return request(next_s._replace(left_steps=jnp.where(inside, j - 1, 0)))

        def right(_):
            expand = inside & (k > 0)
            next_s = s._replace(
                right=s.right + width * expand,
                phase=jnp.where(expand, _RIGHT, _SHRINK),
                num_expansions=s.num_expansions + expand,
            )
            return request(next_s._replace(right_steps=jnp.where(inside, k - 1, 0)))

        def shrink(_):
            next_s = s._replace(num_shrink=s.num_shrink + 1)
            next_s = lax.cond(
                inside,
                lambda s: s._replace(phase=_DONE, **candidate._asdict()),
                lambda s: _reject(s, s.t),
                next_s,
            )
            return next_s

        return lax.switch(s.phase, (left, right, shrink), None)

    return init_fn, update


def doubling(width, max_expansions):
    """Random-side doubling and Neal's reverse-construction acceptance test.

    Phases 0/1 evaluate initial endpoints, 2 draws shrinkage candidates, 3 evaluates
    a newly doubled endpoint, and 4/5 evaluate reverse-construction endpoints.
    """

    def init_fn(rng_key, state):
        s = init(rng_key, state, width)
        s = s._replace(
            remaining=max_expansions,
            side=False,
            left_inside=False,
            right_inside=False,
            expanded_left=s.left,
            expanded_right=s.right,
            check_left=s.left,
            check_right=s.right,
            separated=False,
            trial_t=s.t,
        )
        return lax.cond(max_expansions > 0, lambda s: s, _draw, s)

    def expand(s):
        def extend(s):
            rng_key, subkey = random.split(s.rng_key)
            side = random.bernoulli(subkey)
            span = s.right - s.left
            left = s.left - jnp.where(side, span, 0)
            right = s.right + jnp.where(side, 0, span)
            s = s._replace(
                rng_key=rng_key,
                phase=_EXPAND,
                left=left,
                right=right,
                t=jnp.where(side, left, right),
                num_expansions=s.num_expansions + 1,
                remaining=s.remaining - 1,
                side=side,
                expanded_left=left,
                expanded_right=right,
            )
            return s

        expanding = (s.remaining > 0) & (s.left_inside | s.right_inside)
        return lax.cond(expanding, extend, _draw, s)

    def bisect(s):
        def body(s):
            mid = (s.check_left + s.check_right) / 2
            separated = s.separated | ((0 < mid) != (s.trial_t < mid))
            return s._replace(
                check_left=jnp.where(s.trial_t >= mid, mid, s.check_left),
                check_right=jnp.where(s.trial_t < mid, mid, s.check_right),
                separated=separated,
            )

        wide = lambda s: s.check_right - s.check_left > 1.1 * width
        needs_check = wide(s)
        s = lax.cond(needs_check, body, lambda s: s, s)
        s = lax.while_loop(lambda s: wide(s) & ~s.separated, body, s)
        check = needs_check & s.separated
        return s._replace(phase=jnp.where(check, _CHECK_LEFT, _DONE), t=s.check_left)

    def update(s, candidate, inside):
        def first_left(_):
            return s._replace(phase=_RIGHT, t=s.right, left_inside=inside)

        def first_right(_):
            return expand(s._replace(right_inside=inside))

        def shrink(_):
            next_s = s._replace(num_shrink=s.num_shrink + 1)

            def check(s):
                s = s._replace(
                    check_left=s.expanded_left,
                    check_right=s.expanded_right,
                    separated=False,
                    trial_t=s.t,
                )
                return bisect(s._replace(**candidate._asdict()))

            return lax.cond(
                inside,
                check,
                lambda s: _reject(s, s.t),
                next_s,
            )

        def endpoint(_):
            s_next = s._replace(
                left_inside=jnp.where(s.side, inside, s.left_inside),
                right_inside=jnp.where(s.side, s.right_inside, inside),
            )
            return expand(s_next)

        def check_left(_):
            return s._replace(phase=_CHECK_RIGHT, t=s.check_right, left_inside=inside)

        def check_right(_):
            return lax.cond(
                jnp.logical_not(s.left_inside | inside),
                lambda s: _reject(s, s.trial_t),
                bisect,
                s,
            )

        return lax.switch(
            s.phase,
            (first_left, first_right, shrink, endpoint, check_left, check_right),
            None,
        )

    return init_fn, update


def build_kernel(
    proposal_generator,
    interval=stepping_out,
    *,
    width=1.0,
    max_expansions=10,
):
    """Build a scalar, vmappable kernel returning ``(state, SliceInfo)``.

    ``proposal_generator(rng_key, position, move_index) -> slice_fn`` constructs
    ``slice_fn(t) -> (state, is_valid)``. The state contains position and logdensity.

    ``kernel(rng_key, state, num_steps)`` completes every prescribed move. Per-chain
    keys are independent of other chains' progress. Shrinkage continues until an
    acceptable candidate is found; max_expansions limits only bracket construction.
    """
    init, update = interval(width, max_expansions)

    def kernel(rng_key, state, num_steps=1):
        def init_move(rng_key, state):
            rng_key, proposal_key = random.split(rng_key)
            state = init(rng_key, state)
            slice_fn = proposal_generator(proposal_key, state.position, state.index)
            graph, shape = jax.make_jaxpr(slice_fn, return_shape=True)(state.t)
            return (
                state._replace(proposal=graph.consts),
                graph.jaxpr,
                jax.tree.structure(shape),
            )

        def body(state):
            result = jax.core.eval_jaxpr(graph, state.proposal, state.t)
            proposed_state, is_valid = jax.tree.unflatten(out_tree, result)
            inside = is_valid & (proposed_state.logdensity >= state.level)
            state = update(state, proposed_state, inside)
            state = state._replace(num_evaluations=state.num_evaluations + 1)

            def finish_move(state):
                state = state._replace(index=state.index + 1)
                return lax.cond(
                    state.index < num_steps,
                    lambda s: init_move(s.rng_key, s)[0],
                    lambda s: s,
                    state,
                )

            return lax.cond(state.phase == _DONE, finish_move, lambda s: s, state)

        state = state._replace(
            index=0, num_evaluations=0, num_expansions=0, num_shrink=0
        )
        state, graph, out_tree = init_move(rng_key, state)
        state = lax.while_loop(lambda s: s.index < num_steps, body, state)
        info = SliceInfo(state.num_evaluations, state.num_expansions, state.num_shrink)
        return state, info

    return kernel
