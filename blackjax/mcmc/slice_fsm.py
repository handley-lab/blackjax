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
one another between these stages. The evaluator has one shared call site.

Interval strategies implement Neal (2003), Figures 3--6. The proposal defines
the path independently of the interval strategy and target evaluation.
"""

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


class _Slice(NamedTuple):
    key: PRNGKey
    phase: Array
    left: Array
    right: Array
    t: Array
    level: Array
    trial: Any
    num_expansions: Array
    num_shrink: Array


def _initialise(key, particle, width):
    key, level_key, bracket_key = random.split(key, 3)
    dtype = particle.logdensity.dtype
    left = -width * random.uniform(bracket_key, dtype=dtype)
    level = particle.logdensity + jnp.log(random.uniform(level_key, dtype=dtype))
    return _Slice(key, _LEFT, left, left + width, left, level, particle, 0, 0)


def _draw(s):
    key, subkey = random.split(s.key)
    t = s.left + random.uniform(subkey, dtype=s.left.dtype) * (s.right - s.left)
    return s._replace(key=key, phase=_SHRINK, t=t)


def _reject(s, t):
    left = jnp.where(t < 0, t, s.left)
    right = jnp.where(t >= 0, t, s.right)
    return _draw(s._replace(left=left, right=right))


def stepping_out(width, max_expansions):
    """Fixed-width stepping-out with Neal's randomly divided expansion budget.

    The initial bracket counts towards max_expansions; zero and one both perform
    no expansion. Shrinkage is not truncated.
    """

    def start(key, particle):
        key, budget_key = random.split(key)
        s = _initialise(key, particle, width)
        j = jnp.floor(max_expansions * random.uniform(budget_key)).astype(int)
        k = max_expansions - 1 - j
        return request(s, (j, k))

    def request(s, budget):
        j, k = budget
        phase = jnp.where((s.phase == _LEFT) & (j <= 0), _RIGHT, s.phase)
        phase = jnp.where((phase == _RIGHT) & (k <= 0), _SHRINK, phase)
        s = s._replace(phase=phase, t=jnp.where(phase == _LEFT, s.left, s.right))
        return lax.cond(phase == _SHRINK, _draw, lambda s: s, s), budget

    def advance(s, budget, candidate, inside):
        j, k = budget

        def left(_):
            expand = inside & (j > 0)
            next_s = s._replace(
                left=s.left - width * expand,
                phase=jnp.where(expand, _LEFT, _RIGHT),
                num_expansions=s.num_expansions + expand,
            )
            return request(next_s, (jnp.where(inside, j - 1, 0), k))

        def right(_):
            expand = inside & (k > 0)
            next_s = s._replace(
                right=s.right + width * expand,
                phase=jnp.where(expand, _RIGHT, _SHRINK),
                num_expansions=s.num_expansions + expand,
            )
            return request(next_s, (j, jnp.where(inside, k - 1, 0)))

        def shrink(_):
            next_s = s._replace(num_shrink=s.num_shrink + 1)
            next_s = lax.cond(
                inside,
                lambda s: s._replace(phase=_DONE, trial=candidate),
                lambda s: _reject(s, s.t),
                next_s,
            )
            return next_s, budget

        return lax.switch(s.phase, (left, right, shrink), None)

    return start, advance


class _Doubling(NamedTuple):
    remaining: Array
    side: Array
    left_inside: Array
    right_inside: Array
    left: Array
    right: Array
    check_left: Array
    check_right: Array
    separated: Array
    trial_t: Array


def doubling(width, max_expansions):
    """Random-side doubling and Neal's reverse-construction acceptance test.

    Phases 0/1 evaluate initial endpoints, 2 draws shrinkage candidates, 3 evaluates
    a newly doubled endpoint, and 4/5 evaluate reverse-construction endpoints.
    """

    def start(key, particle):
        s = _initialise(key, particle, width)
        d = _Doubling(
            max_expansions,
            False,
            False,
            False,
            s.left,
            s.right,
            s.left,
            s.right,
            False,
            s.t,
        )
        return lax.cond(max_expansions > 0, lambda s: s, _draw, s), d

    def expand(s, d):
        def extend(pair):
            s, d = pair
            key, subkey = random.split(s.key)
            side = random.bernoulli(subkey)
            span = s.right - s.left
            left = s.left - jnp.where(side, span, 0)
            right = s.right + jnp.where(side, 0, span)
            s = s._replace(
                key=key,
                phase=_EXPAND,
                left=left,
                right=right,
                t=jnp.where(side, left, right),
                num_expansions=s.num_expansions + 1,
            )
            d = d._replace(remaining=d.remaining - 1, side=side, left=left, right=right)
            return s, d

        expanding = (d.remaining > 0) & (d.left_inside | d.right_inside)
        return lax.cond(
            expanding, extend, lambda pair: (_draw(pair[0]), pair[1]), (s, d)
        )

    def bisect(s, d):
        def body(d):
            mid = (d.check_left + d.check_right) / 2
            separated = d.separated | ((0 < mid) != (d.trial_t < mid))
            return d._replace(
                check_left=jnp.where(d.trial_t >= mid, mid, d.check_left),
                check_right=jnp.where(d.trial_t < mid, mid, d.check_right),
                separated=separated,
            )

        wide = lambda d: d.check_right - d.check_left > 1.1 * width
        needs_check = wide(d)
        d = lax.cond(needs_check, body, lambda d: d, d)
        d = lax.while_loop(lambda d: wide(d) & ~d.separated, body, d)
        check = needs_check & d.separated
        return s._replace(phase=jnp.where(check, _CHECK_LEFT, _DONE), t=d.check_left), d

    def advance(s, d, candidate, inside):
        def first_left(_):
            return s._replace(phase=_RIGHT, t=s.right), d._replace(left_inside=inside)

        def first_right(_):
            return expand(s, d._replace(right_inside=inside))

        def shrink(_):
            next_s = s._replace(num_shrink=s.num_shrink + 1)

            def check(pair):
                s, d = pair
                d = d._replace(
                    check_left=d.left,
                    check_right=d.right,
                    separated=False,
                    trial_t=s.t,
                )
                return bisect(s._replace(trial=candidate), d)

            return lax.cond(
                inside,
                check,
                lambda pair: (_reject(pair[0], pair[0].t), pair[1]),
                (next_s, d),
            )

        def endpoint(_):
            next_d = d._replace(
                left_inside=jnp.where(d.side, inside, d.left_inside),
                right_inside=jnp.where(d.side, d.right_inside, inside),
            )
            return expand(s, next_d)

        def check_left(_):
            return s._replace(phase=_CHECK_RIGHT, t=d.check_right), d._replace(
                left_inside=inside
            )

        def check_right(_):
            return lax.cond(
                jnp.logical_not(d.left_inside | inside),
                lambda pair: (_reject(pair[0], pair[1].trial_t), pair[1]),
                lambda pair: bisect(*pair),
                (s, d),
            )

        return lax.switch(
            s.phase,
            (first_left, first_right, shrink, endpoint, check_left, check_right),
            None,
        )

    return start, advance


class _Chain(NamedTuple):
    key: PRNGKey
    index: Array
    particle: Any
    direction: Any
    slice: _Slice
    interval: Any
    info: SliceInfo


def build_kernel(
    evaluate, proposal, strategy=stepping_out, *, width=1.0, max_expansions=10
):
    """Build a scalar, vmappable kernel returning ``(particle, SliceInfo)``.

    ``evaluate(position) -> (particle, is_valid)`` supplies a particle with position
    and logdensity fields, optionally containing other evaluated quantities.
    ``proposal`` is a pair of pure functions:
    ``generate(key, position, move_index) -> direction`` and
    ``move(position, direction, t) -> position``. Direction may be any fixed-shape
    pytree. Each path must preserve the target's base measure and support the
    reverse move with the same probability.

    ``kernel(key, particle, num_steps)`` completes every prescribed move. Per-chain
    keys are independent of other chains' progress. Shrinkage continues until an
    acceptable candidate is found; max_expansions limits only bracket construction.
    """
    generate, move = proposal
    start, advance = strategy(width, max_expansions)

    def kernel(key, particle, num_steps=1):
        def enter(key, particle, index, info):
            key, direction_key, slice_key = random.split(key, 3)
            direction = generate(direction_key, particle.position, index)
            s, interval = start(slice_key, particle)
            return _Chain(key, index, particle, direction, s, interval, info)

        def body(c):
            position = move(c.particle.position, c.direction, c.slice.t)
            candidate, valid = evaluate(position)
            inside = valid & (candidate.logdensity >= c.slice.level)
            s, interval = advance(c.slice, c.interval, candidate, inside)
            done = s.phase == _DONE
            info = SliceInfo(
                c.info.num_evaluations + 1,
                c.info.num_expansions + jnp.where(done, s.num_expansions, 0),
                c.info.num_shrink + jnp.where(done, s.num_shrink, 0),
            )
            c = c._replace(slice=s, interval=interval, info=info)

            def finish(c):
                index = c.index + 1
                c = c._replace(index=index, particle=s.trial)
                return lax.cond(
                    index < num_steps,
                    lambda c: enter(c.key, c.particle, c.index, c.info),
                    lambda c: c,
                    c,
                )

            return lax.cond(done, finish, lambda c: c, c)

        c = enter(key, particle, 0, SliceInfo(0, 0, 0))
        c = lax.while_loop(lambda c: c.index < num_steps, body, c)
        return c.particle, c.info

    return kernel
