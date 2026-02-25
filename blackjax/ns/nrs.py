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
"""Nested Rejection Sampling (NRS) algorithm.

A specific implementation of Nested Sampling that uses importance-weighted
rejection sampling from a Gaussian proposal as the inner kernel. The Gaussian
is fitted to the live points' empirical mean and covariance.
"""

from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_gaussian import update_with_gaussian_proposal
from blackjax.ns.utils import get_first_row
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_means,
)
from blackjax.types import ArrayTree

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
]


def update_inner_kernel_params(
    rng_key: jax.random.PRNGKey,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Update inner kernel parameters from current particles.

    Computes the empirical mean and covariance matrix from the live particles
    in the flattened coordinate system used by ``particles_as_rows``.

    Parameters
    ----------
    rng_key
        PRNG key (unused but required by interface).
    state
        The current NSState containing live particles.
    info
        Information from the last NS step (unused but kept for interface).
    inner_kernel_params
        Previous inner kernel parameters (unused but kept for interface).

    Returns
    -------
    Dict[str, ArrayTree]
        Dictionary containing 'mean' and 'cov' in the flattened coordinate system.
    """
    positions = state.particles.position
    mean = particles_means(positions)
    cov = jnp.atleast_2d(particles_covariance_matrix(positions))
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])
    return {"mean": mean, "cov": cov}


def build_kernel(
    init_state_fn: Callable,
    unravel_fn: Callable,
    num_delete: int = 1,
    num_proposals: int = 1000,
    max_rounds: int = 100,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
) -> Callable:
    """Builds the Nested Rejection Sampling kernel.

    Parameters
    ----------
    init_state_fn
        Scalar function: position -> StateWithLogLikelihood.
    unravel_fn
        Function mapping a flat array to a PyTree position.
    num_delete
        Number of particles to replace per step.
    num_proposals
        Batch size per round of proposal generation.
    max_rounds
        Maximum rounds before declaring failure.
    update_inner_kernel_params_fn
        Function to update inner kernel parameters (mean, cov).
    delete_fn
        Particle deletion function.

    Returns
    -------
    Callable
        A kernel function for Nested Rejection Sampling.
    """
    inner_kernel = update_with_gaussian_proposal(
        init_state_fn, unravel_fn, num_delete, num_proposals, max_rounds
    )

    delete_fn = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_delete: int = 1,
    num_proposals: int = 1000,
    max_rounds: int = 100,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
) -> SamplingAlgorithm:
    """Creates a Nested Rejection Sampling (NRS) algorithm.

    Uses importance-weighted rejection sampling from a Gaussian proposal
    fitted to the live points. The Gaussian parameters (mean and covariance)
    are updated adaptively at each NS step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_delete
        The number of particles to delete and replace at each NS step.
    num_proposals
        The number of proposals drawn per batch. Larger values improve
        acceptance rate but use more memory.
    max_rounds
        Maximum number of proposal batches before declaring failure.
    init_state_strategy_fn
        A function to initialize particle state from positions.
    update_inner_kernel_params_fn
        A function to update inner kernel parameters from particles.
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

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        )

    # Deferred kernel construction: unravel_fn depends on position structure.
    # Built once on first step call, cached for subsequent calls.
    _kernel_cache = {}

    def step_fn(rng_key, state):
        if "kernel" not in _kernel_cache:
            prototype_pos = get_first_row(state.particles.position)
            _, unravel_fn = ravel_pytree(prototype_pos)
            _kernel_cache["kernel"] = build_kernel(
                init_state_fn,
                unravel_fn,
                num_delete,
                num_proposals,
                max_rounds,
                update_inner_kernel_params_fn=update_inner_kernel_params_fn,
                delete_fn=delete_fn,
            )
        return _kernel_cache["kernel"](rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
