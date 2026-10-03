"""Nested sampling with independently advancing slice chains."""

import jax.numpy as jnp
from jax import lax, random

from blackjax.mcmc.slice import SliceState
from blackjax.mcmc.slice_fsm import SliceInfo
from blackjax.ns import from_mcmc
from blackjax.ns.nss import covariance_proposal, live_covariance_factor


def build_kernel(
    init_state_fn,
    num_inner_steps,
    init_slice_fn,
    build_tick_fn,
    num_delete=1,
    proposal=covariance_proposal,
    inner_kernel_params=live_covariance_factor,
):
    """Advance each replacement chain through its moves without move barriers.

    ``init_slice_fn(key, particle)`` initializes an FSM whose position is the
    complete particle. ``build_tick_fn(slice_fn)`` builds one evaluation tick.
    """

    def chain(rng_key, particle, loglikelihood_0, **parameters):
        generate = proposal(init_state_fn, loglikelihood_0, **parameters)
        rng_key, proposal_key, init_key = random.split(rng_key, 3)
        state = init_slice_fn(init_key, particle)
        info = SliceInfo(jnp.asarray(False), 0, 0, 0)

        def body(carry):
            rng_key, proposal_key, origin, state, count, info = carry
            rng_key, tick_key, next_key, init_key = random.split(rng_key, 4)
            proposed = generate(proposal_key, origin.position, None)

            def slice_fn(t):
                candidate, valid = proposed(t)
                return SliceState(candidate, candidate.logdensity), valid

            state, tick_info = build_tick_fn(slice_fn)(tick_key, state)
            count += tick_info.is_accepted.astype(int)
            info = SliceInfo(
                count == num_inner_steps,
                info.num_evaluations + tick_info.num_evaluations,
                info.num_expansions + tick_info.num_expansions,
                info.num_shrink + tick_info.num_shrink,
            )

            def next_move(_):
                particle = state.position
                return next_key, particle, init_slice_fn(init_key, particle)

            proposal_key, origin, state = lax.cond(
                tick_info.is_accepted & (count < num_inner_steps),
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

    return from_mcmc.build_kernel(chain, 1, inner_kernel_params, num_delete)
