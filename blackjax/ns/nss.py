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
"""Public API for Nested Slice Sampling."""

from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import SliceState
from blackjax.mcmc.ss import build_kernel as build_slice_kernel
from blackjax.mcmc.ss import default_stepper_fn
from blackjax.mcmc.ss import (
    sample_direction_from_covariance as ss_sample_direction_from_covariance,
)
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import new_state_and_info
from blackjax.ns.utils import get_first_row, repeat_kernel
from blackjax.smc.tuning.from_particles import (
    particles_as_rows,
    particles_covariance_matrix,
)
from blackjax.types import ArrayTree, PRNGKey

__all__ = [
    "init",
    "as_top_level_api",
    "build_kernel",
]


def sample_direction_from_covariance(
    rng_key: PRNGKey, params: Dict[str, ArrayTree]
) -> ArrayTree:
    """Generate a normalized slice direction for NSS.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    params
        Dictionary containing 'cov' PyTree with covariance matrix rows.

    Returns
    -------
    A Mahalanobis-normalized direction vector.
    """
    cov = params["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov)
    d = ss_sample_direction_from_covariance(rng_key, cov)
    return unravel_fn(d)


def compute_covariance_from_particles(
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Adapt the slice direction proposal parameters.

    Parameters
    ----------
    state
        Current NSState containing live particles.
    info
        NSInfo from the last Nested Sampling step.
    inner_kernel_params
        Dictionary of parameters for the inner kernel.

    Returns
    -------
    Dictionary containing covariance PyTree.
    """
    cov_matrix = jnp.atleast_2d(particles_covariance_matrix(state.particles))
    single_particle = get_first_row(state.particles)
    _, unravel_fn = ravel_pytree(single_particle)
    cov_pytree = jax.vmap(unravel_fn)(cov_matrix)
    return {"cov": cov_pytree}


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
) -> Callable:
    """Build a Nested Slice Sampling kernel.

    Parameters
    ----------
    logprior_fn
        Function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        Function that computes the log-likelihood of a single particle.
    num_inner_steps
        Number of HRSS steps to run for each new particle generation.
    num_delete
        Number of particles to delete and replace at each NS step.
    stepper_fn
        Stepper function for the HRSS kernel.
    adapt_direction_params_fn
        Function that adapts the slice direction proposal parameters.
    generate_slice_direction_fn
        Function that generates a normalized direction for HRSS.

    Returns
    -------
    A kernel function for Nested Slice Sampling.
    """

    slice_kernel = build_slice_kernel(stepper_fn)

    @repeat_kernel(num_inner_steps)
    def inner_kernel(
        rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params
    ):
        """Inner kernel for NSS using constrained slice sampling."""
        # Do constrained slice sampling
        slice_state = SliceState(position=state.position, logdensity=state.logprior)
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key, params)
        logdensity_fn = logprior_fn
        constraint_fn = lambda x: jnp.array([loglikelihood_fn(x)])
        constraint = jnp.array([loglikelihood_0])
        strict = jnp.array([True])
        new_slice_state, slice_info = slice_kernel(
            rng_key, slice_state, logdensity_fn, d, constraint_fn, constraint, strict
        )

        # Pass the relevant information back to PartitionedState and PartitionedInfo
        return new_state_and_info(
            position=new_slice_state.position,
            logprior=new_slice_state.logdensity,
            loglikelihood=slice_info.constraint[0],
            info=slice_info,
        )

    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    update_inner_kernel_params_fn = adapt_direction_params_fn
    kernel = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn,
    )
    return kernel


def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for Nested Slice Sampling.

    Parameters
    ----------
    logprior_fn
        Function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        Function that computes the log-likelihood of a single particle.
    num_inner_steps
        Number of HRSS steps to run for each new particle generation.
    num_delete
        Number of particles to delete and replace at each NS step.
    stepper_fn
        Stepper function for the HRSS kernel.
    adapt_direction_params_fn
        Function that adapts the slice direction proposal parameters.
    generate_slice_direction_fn
        Function that generates a normalized direction for HRSS.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        num_inner_steps,
        num_delete,
        stepper_fn=stepper_fn,
        adapt_direction_params_fn=adapt_direction_params_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
    )
    init_fn = partial(
        init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=adapt_direction_params_fn,
    )
    step_fn = kernel

    return SamplingAlgorithm(init_fn, step_fn)
