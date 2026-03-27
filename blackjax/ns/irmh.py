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
"""Nested Sampling with Independent Random Metropolis-Hastings (NS-IRMH).

Uses a Gaussian proposal fitted to live points as an independent MH kernel,
run through the standard from_mcmc NS infrastructure. The proposal parameters
(mean and covariance) are updated adaptively at each NS step.
"""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import build_kernel as from_mcmc_build_kernel
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_means,
)
from blackjax.types import ArrayTree

__all__ = ["as_top_level_api", "build_kernel", "init", "update_irmh_params"]


def update_irmh_params(
    rng_key,
    state: NSState,
    info: NSInfo,
    inner_kernel_params=None,
):
    """Update IRMH proposal parameters from current live points.

    Parameters
    ----------
    rng_key
        PRNG key (unused).
    state
        Current NSState.
    info
        Info from last step (unused).
    inner_kernel_params
        Previous parameters (unused).
    """
    positions = state.particles.position
    mean = particles_means(positions)
    cov = jnp.atleast_2d(particles_covariance_matrix(positions))
    return {"mean": mean, "cov": cov}


class IRMHState(NamedTuple):
    """State of the IRMH chain."""

    position: ArrayTree
    logdensity: float


class IRMHInfo(NamedTuple):
    """Info from an IRMH step."""

    acceptance_rate: float
    is_accepted: bool
    proposal: IRMHState


def irmh_init(position, logdensity_fn):
    """Initialize an IRMH state."""
    return IRMHState(position=position, logdensity=logdensity_fn(position))


def irmh_step(rng_key, state, logdensity_fn, *, mean, cov):
    """Single IRMH step: propose from Gaussian, accept with MH ratio.

    The MH acceptance ratio for an independent proposal q is:
        α = min(1, [π(x')/q(x')] / [π(x)/q(x)])
    where π is the target (logdensity_fn, i.e. the prior).
    """
    proposal_pos = jax.random.multivariate_normal(rng_key, mean, cov)
    proposal_logdensity = logdensity_fn(proposal_pos)
    proposal_log_q = jax.scipy.stats.multivariate_normal.logpdf(proposal_pos, mean, cov)
    current_log_q = jax.scipy.stats.multivariate_normal.logpdf(
        state.position, mean, cov
    )

    # MH ratio: [π(x')/q(x')] / [π(x)/q(x)]
    log_ratio = (proposal_logdensity - proposal_log_q) - (
        state.logdensity - current_log_q
    )
    acceptance_rate = jnp.minimum(1.0, jnp.exp(log_ratio))

    log_uniform = jnp.log(jax.random.uniform(jax.random.fold_in(rng_key, 1)))
    is_accepted = log_uniform < log_ratio

    proposal = IRMHState(position=proposal_pos, logdensity=proposal_logdensity)
    new_state = jax.lax.cond(
        is_accepted,
        lambda _: proposal,
        lambda _: state,
        operand=None,
    )
    info = IRMHInfo(
        acceptance_rate=acceptance_rate,
        is_accepted=is_accepted,
        proposal=proposal,
    )
    return new_state, info


def build_kernel(
    init_state_fn: Callable,
    logdensity_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    update_inner_kernel_params_fn: Callable = update_irmh_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
) -> Callable:
    """Builds a Nested Sampling kernel using IRMH.

    Parameters
    ----------
    init_state_fn
        Function to initialize a NS particle state from a position.
    logdensity_fn
        Log-density function (typically the prior log-probability).
    num_inner_steps
        Number of IRMH steps per particle replacement.
    num_delete
        Number of particles to replace per NS iteration.
    update_inner_kernel_params_fn
        Function to update proposal parameters adaptively.
    delete_fn
        Function to select which particles to delete.
    update_strategy
        MCMC update strategy (default: take last state after scan).
    """
    return from_mcmc_build_kernel(
        init_state_fn=init_state_fn,
        logdensity_fn=logdensity_fn,
        mcmc_init_fn=irmh_init,
        mcmc_step_fn=irmh_step,
        num_inner_steps=num_inner_steps,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        num_delete=num_delete,
        delete_fn=delete_fn,
    )


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_irmh_params,
    delete_fn: Callable = default_delete_fn,
) -> SamplingAlgorithm:
    """Creates a Nested Sampling algorithm using IRMH.

    Uses an independent Metropolis-Hastings kernel with a Gaussian proposal
    fitted to the live points. The proposal mean and covariance are updated
    adaptively at each NS step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability.
    loglikelihood_fn
        A function that computes the log-likelihood.
    num_inner_steps
        Number of IRMH steps per particle replacement.
    num_delete
        Number of particles to delete and replace at each NS step.
    init_state_strategy_fn
        A function to initialize particle state from positions.
    update_inner_kernel_params_fn
        A function to update proposal parameters from particles.
    delete_fn
        Particle deletion function.

    Returns
    -------
    SamplingAlgorithm
        A ``SamplingAlgorithm`` tuple containing ``init`` and ``step`` functions.
    """
    init_state_fn = partial(
        init_state_strategy_fn,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    kernel = build_kernel(
        init_state_fn,
        logdensity_fn=logprior_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
