# Copyright 2020- The Blackjax Authors.
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
"""Asynchronous chains composed from scalar slice kernels."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random

from blackjax.mcmc import slice_fsm
from blackjax.mcmc.slice import SliceState
from blackjax.mcmc.slice_fsm import SliceInfo


def build_kernel(init_fn, build_kernel_fn):
    """Compose slice advances into a chain without barriers between moves.

    ``init_fn(key, particle)`` initializes an FSM carrying the full particle
    in its position. ``build_kernel_fn(slice_fn)`` builds one scalar advance.
    The returned kernel accepts the usual slice proposal generator and a
    number of completed moves, and can be vmapped over independent chains.
    """

    def kernel(rng_key, particle, logdensity_fn, proposal_generator, num_inner_steps):
        rng_key, proposal_key, init_key = random.split(rng_key, 3)
        state = init_fn(init_key, particle)
        info = SliceInfo(jnp.asarray(False), 0, 0, 0)

        def body(carry):
            rng_key, proposal_key, origin, state, count, info = carry
            rng_key, step_key, next_key, init_key = random.split(rng_key, 4)
            proposal = proposal_generator(proposal_key, origin.position, logdensity_fn)

            def slice_fn(t):
                candidate, valid = proposal(t)
                return SliceState(candidate, candidate.logdensity), valid

            state, step_info = build_kernel_fn(slice_fn)(step_key, state)
            count += step_info.is_accepted.astype(int)
            info = jax.tree.map(jnp.add, info, step_info)
            info = info._replace(is_accepted=count == num_inner_steps)

            def next_move(_):
                particle = state.position
                return next_key, particle, init_fn(init_key, particle)

            proposal_key, origin, state = lax.cond(
                step_info.is_accepted & (count < num_inner_steps),
                next_move,
                lambda _: (proposal_key, origin, state),
                None,
            )
            return rng_key, proposal_key, origin, state, count, info

        carry = (rng_key, proposal_key, particle, state, 0, info)
        _, _, _, state, _, info = lax.while_loop(
            lambda carry: carry[4] < num_inner_steps, body, carry
        )
        return state.position, info

    return kernel


def build_doubling_kernel(max_expansions=10, width=1.0):
    """Build asynchronous chains using doubling slice moves."""

    def init_fn(rng_key, particle):
        state = slice_fsm.DoublingState(particle, particle.logdensity)
        return slice_fsm.init_doubling(rng_key, state, width, max_expansions)

    return build_kernel(init_fn, partial(slice_fsm.build_doubling_kernel, width=width))


def build_stepping_out_kernel(max_expansions=10, width=1.0):
    """Build asynchronous chains using fixed-width stepping-out slice moves."""

    def init_fn(rng_key, particle):
        state = slice_fsm.SteppingOutState(particle, particle.logdensity)
        return slice_fsm.init_stepping_out(rng_key, state, width, max_expansions)

    return build_kernel(
        init_fn, partial(slice_fsm.build_stepping_out_kernel, width=width)
    )
