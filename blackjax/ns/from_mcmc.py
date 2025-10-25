from functools import partial
from typing import Callable, NamedTuple

import jax

from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.base import delete_fn as default_delete_fn


class MCMCUpdateInfo(NamedTuple):
    """Thin layer to hold all the info pertaining to the update step."""

    mcmc_states: NamedTuple
    mcmc_infos: NamedTuple


def update_with_mcmc_take_last(
    constrained_mcmc_step_fn,
    num_mcmc_steps,
):
    """An update strategy for NS that uses MCMC to update the particles.
    For now we will not keep the states as they will be too large to store.
    """

    def update_function(rng_key, state, loglikelihood_0, step_parameters):
        # shared_mcmc_step_fn = partial(
        #     constrained_mcmc_step_fn,
        #     loglikelihood_0=loglikelihood_0,
        #     **step_parameters,
        # )
        shared_mcmc_step_fn = constrained_mcmc_step_fn

        def mcmc_kernel(rng_key, state):
            def body_fn(state, rng_key):
                new_state, info = shared_mcmc_step_fn(rng_key, state)
                return new_state, info  # (new_state, info)

            keys = jax.random.split(rng_key, num_mcmc_steps)
            final_state, infos = jax.lax.scan(body_fn, state, keys)
            return final_state, infos  # MCMCUpdateInfo(infos[0], infos[1])

        return jax.vmap(mcmc_kernel, in_axes=(0, 0, None, None))(
            rng_key, state, loglikelihood_0, step_parameters
        )

    return update_function


def build_kernel(
    init_state_fn: Callable,
    mcmc_init_fn: Callable,
    mcmc_step_fn: Callable,
    num_inner_steps: int,
    update_inner_kernel_params_fn: Callable,
    num_delete: int = 1,
) -> Callable:
    """Builds the Nested Slice Sampling kernel. wrapping any mcmc algorithm"""

    def constrained_mcmc_step_fn(rng_key, state, loglikelihood_0, **params):
        rng_key, prop_key = jax.random.split(rng_key, 2)
        mcmc_state = mcmc_init_fn(rng_key, state.position, state.logprior)
        new_mcmc_state, mcmc_info = mcmc_step_fn(prop_key, mcmc_state, **params)
        new_state = init_state_fn(new_mcmc_state.position)
        new_state = jax.lax.cond(
            new_state.loglikelihood > loglikelihood_0,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )

        return new_state, mcmc_info

    inner_kernel = update_with_mcmc_take_last(constrained_mcmc_step_fn, num_inner_steps)

    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn,
    )
    return kernel
