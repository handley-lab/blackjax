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
"""Nested Slice Sampling (NSS) algorithm.

This module implements the Nested Slice Sampling algorithm, which combines the
Nested Sampling framework with an inner Hit-and-Run Slice Sampling (HRSS) kernel
for exploring the constrained prior distribution at each likelihood level.

The key idea is to leverage the efficiency of slice sampling for constrained
sampling tasks. The parameters of the HRSS kernel, specifically the covariance
matrix for proposing slice directions, are adaptively tuned based on the current
set of live particles.
"""

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
    """Default function to generate a normalized slice direction for NSS.

    This function uses a mathematically simplified approach to generate direction
    vectors uniformly distributed on a hypersphere:
    1. Sample from standard multivariate normal N(0, I)
    2. Normalize to unit vector (uniform on hypersphere)
    3. Transform by S^(1/2) where S is the covariance matrix

    This is equivalent to the traditional approach of sampling from N(0, S) and
    normalizing by Mahalanobis norm, but is more numerically stable and efficient.

    Parameters
    ----------
    rng_key
        A JAX PRNG key.
    params
        Keyword arguments, must contain:
        - `cov`: A PyTree (structured like a particle) whose leaves are rows
                 of the covariance matrix.
        - `chol`: A PyTree with the square root (Cholesky factor) of the
                  covariance matrix.

    Returns
    -------
    ArrayTree
        A direction vector uniformly distributed on a hypersphere (PyTree,
        matching the structure of a single particle), to be used by the slice sampler.
    """
    cov = params["cov"]
    chol = params["chol"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    cov = particles_as_rows(cov)
    chol = particles_as_rows(chol)
    d = ss_sample_direction_from_covariance(rng_key, cov, chol)
    return unravel_fn(d)


def compute_covariance_from_particles(
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Default function to adapt/tune the slice direction proposal parameters.

    This function computes the empirical covariance matrix from the current set of
    live particles in `state.particles`. This covariance matrix is then returned
    and can be used by the slice direction generation function (e.g.,
    `default_generate_slice_direction_fn`) in the next Nested Sampling iteration.

    Parameters
    ----------
    state
        The current `NSState` of the Nested Sampler, containing the live particles.
    info
        The `NSInfo` from the last Nested Sampling step (currently unused by this function).
    inner_kernel_params
        A dictionary of parameters for the inner kernel (currently unused by this function).

    Returns
    -------
    Dict[str, ArrayTree]
        A dictionary `{'cov': cov_pytree, 'chol': chol_pytree}`.
        `cov_pytree` is a PyTree with the same structure as a single particle
        containing the covariance matrix. `chol_pytree` contains the Cholesky
        decomposition (square root) of the covariance matrix. If the full DxD
        covariance matrix of the flattened particles is `M_flat`, and `unravel_fn`
        is the function to un-flatten a D-vector to the particle's PyTree structure,
        then `cov_pytree` is equivalent to `jax.vmap(unravel_fn)(M_flat)`.
        This means each leaf of `cov_pytree` will have a shape `(D, *leaf_original_dims)`.
    """
    cov_matrix = jnp.atleast_2d(particles_covariance_matrix(state.particles))
    cov_matrix *= cov_matrix.shape[0] + 2
    chol_matrix = jnp.linalg.cholesky(cov_matrix)
    single_particle = get_first_row(state.particles)
    _, unravel_fn = ravel_pytree(single_particle)
    cov_pytree = jax.vmap(unravel_fn)(cov_matrix)
    chol_pytree = jax.vmap(unravel_fn)(chol_matrix)
    return {"cov": cov_pytree, "chol": chol_pytree}


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> Callable:
    """Builds the Nested Slice Sampling kernel.

    This function creates a Nested Slice Sampling kernel that uses
    Hit-and-Run Slice Sampling (HRSS) as its inner kernel. The parameters
    for the HRSS direction proposal (specifically, the covariance matrix)
    are adaptively tuned at each step using `adapt_direction_params_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_inner_steps
        The number of HRSS steps to run for each new particle generation.
        This should be a multiple of the dimension of the parameter space.
    num_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> x_new` for the HRSS kernel.
        Defaults to `default_stepper_fn`.
    adapt_direction_params_fn
        A function `(ns_state, ns_info) -> dict_of_params` that computes/adapts
        the parameters (e.g., covariance matrix) for the slice direction proposal,
        based on the current NS state. Defaults to `compute_covariance_from_particles`.
    generate_slice_direction_fn
        A function `(rng_key, **params) -> direction_pytree` that generates a
        normalized direction for HRSS, using parameters from `adapt_direction_params_fn`.
        Defaults to `sample_direction_from_covariance`.
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase. Defaults to 10.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.
        Defaults to 100.

    Returns
    -------
    Callable
        A kernel function for Nested Slice Sampling that takes an `rng_key` and
        the current `NSState` and returns a tuple containing the new `NSState` and
        the `NSInfo` for the step.
    """

    slice_kernel = build_slice_kernel(stepper_fn, max_steps, max_shrinkage)

    @repeat_kernel(num_inner_steps)
    def inner_kernel(
        rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params
    ):
        # Do constrained slice sampling
        slice_state = SliceState(
            position=state.position,
            logdensity=state.logprior,
            constraint=jnp.array([state.loglikelihood]),
        )
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
            loglikelihood=new_slice_state.constraint[0],
            info=slice_info,
        )

    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    # Vectorize the inner kernel for parallel execution
    in_axes = (0, 0, None, None, None, None)

    update_inner_kernel_params_fn = adapt_direction_params_fn
    kernel = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        jax.vmap(inner_kernel, in_axes=in_axes),
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
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates an adaptive Nested Slice Sampling (NSS) algorithm.

    This function configures a Nested Sampling algorithm that uses Hit-and-Run
    Slice Sampling (HRSS) as its inner kernel. The parameters for the HRSS
    direction proposal (specifically, the covariance matrix) are adaptively tuned
    at each step using `adapt_direction_params_fn`.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_inner_steps
        The number of HRSS steps to run for each new particle generation.
        This should be a multiple of the dimension of the parameter space.
    num_delete
        The number of particles to delete and replace at each NS step.
        Defaults to 1.
    stepper_fn
        The stepper function `(x, direction, t) -> x_new` for the HRSS kernel.
        Defaults to `default_stepper`.
    adapt_direction_params_fn
        A function `(ns_state, ns_info) -> dict_of_params` that computes/adapts
        the parameters (e.g., covariance matrix) for the slice direction proposal,
        based on the current NS state. Defaults to `compute_covariance_from_particles`.
    generate_slice_direction_fn
        A function `(rng_key, **params) -> direction_pytree` that generates a
        normalized direction for HRSS, using parameters from `adapt_direction_params_fn`.
        Defaults to `sample_direction_from_covariance`.
    max_steps
        The maximum number of steps to take when expanding the interval in
        each direction during the stepping-out phase. Defaults to 10.
    max_shrinkage
        The maximum number of shrinking steps to perform to avoid infinite loops.
        Defaults to 100.

    Returns
    -------
    SamplingAlgorithm
        A `SamplingAlgorithm` tuple containing `init` and `step` functions for
        the configured Nested Slice Sampler. The state managed by this
        algorithm is `NSState`.
    """

    kernel = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        num_inner_steps,
        num_delete,
        stepper_fn=stepper_fn,
        adapt_direction_params_fn=adapt_direction_params_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )

    def init_fn(position, rng_key=None):
        # Vectorize the functions for parallel evaluation over particles
        return init(
            position,
            logprior_fn=jax.vmap(logprior_fn),
            loglikelihood_fn=jax.vmap(loglikelihood_fn),
            update_inner_kernel_params_fn=adapt_direction_params_fn,
        )

    step_fn = kernel

    return SamplingAlgorithm(init_fn, step_fn)
