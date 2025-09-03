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
"""Public API for base Nested Sampling."""

from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel", "NSState", "NSInfo", "delete_fn"]


class NSState(NamedTuple):
    """State of the Nested Sampler.

    Attributes
    ----------
    particles
        Current positions of the live particles.
    loglikelihood
        Log-likelihood values of the particles.
    loglikelihood_birth
        Log-likelihood threshold at particle birth.
    logprior
        Log-prior values of the particles.
    pid
        Particle IDs for tracking lineage.
    logX
        Current log prior volume estimate.
    logZ
        Accumulated log evidence estimate.
    logZ_live
        Log evidence contribution from live particles.
    inner_kernel_params
        Parameters for the inner kernel.
    """

    particles: ArrayLikeTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the particles
    pid: Array  # particle IDs
    logX: Array  # The current log-volume estimate
    logZ: Array  # The accumulated evidence estimate
    logZ_live: Array  # The current evidence estimate
    inner_kernel_params: Dict  # Parameters for the inner kernel


class NSInfo(NamedTuple):
    """Additional information on the Nested Sampling transition.

    Attributes
    ----------
    particles
        Dead particles from the current step.
    loglikelihood
        Log-likelihood values of dead particles.
    loglikelihood_birth
        Birth log-likelihood thresholds of dead particles.
    logprior
        Log-prior values of dead particles.
    inner_kernel_info
        Information from the inner kernel update step.
    """

    particles: ArrayTree
    loglikelihood: Array  # The log-likelihood of the particles
    loglikelihood_birth: Array  # The log-likelihood threshold at particle birth
    logprior: Array  # The log-prior density of the particles
    inner_kernel_info: NamedTuple  # Information from the inner kernel update step


class PartitionedState(NamedTuple):
    """State container for partitioned loglikelihood and logprior.

    Attributes
    ----------
    position
        Current positions of particles in the inner kernel.
    logprior
        Log-prior values for particles.
    loglikelihood
        Log-likelihood values for particles.
    """

    position: ArrayLikeTree  # Current positions of particles in the inner kernel
    logprior: Array  # Log-prior values for particles in the inner kernel
    loglikelihood: Array  # Log-likelihood values for particles in the inner kernel


class PartitionedInfo(NamedTuple):
    """Transition information with partitioned loglikelihood and logprior.

    Attributes
    ----------
    position
        Final positions after the transition.
    logprior
        Log-prior values at final positions.
    loglikelihood
        Log-likelihood values at final positions.
    info
        Additional transition diagnostic information.
    """

    position: ArrayTree
    logprior: ArrayTree
    loglikelihood: ArrayTree
    info: NamedTuple


def new_state_and_info(position, logprior, loglikelihood, info):
    """Create new PartitionedState and PartitionedInfo from transition results.

    Parameters
    ----------
    position
        The particle positions after the transition step.
    logprior
        The log-prior densities at the new positions.
    loglikelihood
        The log-likelihood values at the new positions.
    info
        Additional transition-specific information from the step.

    Returns
    -------
    A tuple containing the new partitioned state and associated information.
    """
    new_state = PartitionedState(
        position=position,
        logprior=logprior,
        loglikelihood=loglikelihood,
    )
    info = PartitionedInfo(
        position=position,
        logprior=logprior,
        loglikelihood=loglikelihood,
        info=info,
    )
    return new_state, info


def init(
    particles: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = -jnp.nan,
    logX: Optional[Array] = 0.0,
    logZ: Optional[Array] = -jnp.inf,
) -> NSState:
    """Initialize the Nested Sampler state.

    Parameters
    ----------
    particles
        Initial set of particles drawn from the prior.
    logprior_fn
        Function that computes the log-prior of a single particle.
    loglikelihood_fn
        Function that computes the log-likelihood of a single particle.
    loglikelihood_birth
        Initial log-likelihood birth threshold.
    logX
        Initial log prior volume estimate.
    logZ
        Initial log evidence estimate.

    Returns
    -------
    The initial state of the Nested Sampler.
    """
    loglikelihood = jax.vmap(loglikelihood_fn)(particles)
    loglikelihood_birth = loglikelihood_birth * jnp.ones_like(loglikelihood)
    logprior = jax.vmap(logprior_fn)(particles)
    pid = jnp.arange(len(loglikelihood))
    dtype = loglikelihood.dtype
    logX = jnp.array(logX, dtype=dtype)
    logZ = jnp.array(logZ, dtype=dtype)
    logZ_live = logmeanexp(loglikelihood) + logX
    inner_kernel_params: Dict = {}
    return NSState(
        particles,
        loglikelihood,
        loglikelihood_birth,
        logprior,
        pid,
        logX,
        logZ,
        logZ_live,
        inner_kernel_params,
    )


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    inner_kernel: Callable,
) -> Callable:
    """Build a Nested Sampling kernel.

    Parameters
    ----------
    logprior_fn
        Function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        Function that computes the log-likelihood of a single particle.
    delete_fn
        Function that identifies particles to delete and selects starting points.
    inner_kernel
        Kernel function used to generate new particles.

    Returns
    -------
    A kernel function for Nested Sampling.
    """

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        # Delete, and grab all the dead information
        rng_key, delete_fn_key = jax.random.split(rng_key)
        dead_idx, target_update_idx, start_idx = delete_fn(delete_fn_key, state)
        dead_particles = jax.tree.map(lambda x: x[dead_idx], state.particles)
        dead_loglikelihood = state.loglikelihood[dead_idx]
        dead_loglikelihood_birth = state.loglikelihood_birth[dead_idx]
        dead_logprior = state.logprior[dead_idx]

        # Resample the live particles
        loglikelihood_0 = dead_loglikelihood.max()
        rng_key, sample_key = jax.random.split(rng_key)
        sample_keys = jax.random.split(sample_key, len(start_idx))
        particles = jax.tree.map(lambda x: x[start_idx], state.particles)
        logprior = state.logprior[start_idx]
        loglikelihood = state.loglikelihood[start_idx]
        inner_state = PartitionedState(particles, logprior, loglikelihood)
        in_axes = (0, 0, None, None, None, None)
        new_inner_state, inner_info = jax.vmap(inner_kernel, in_axes=in_axes)(
            sample_keys,
            inner_state,
            logprior_fn,
            loglikelihood_fn,
            loglikelihood_0,
            state.inner_kernel_params,
        )

        # Update the particles
        particles = jax.tree_util.tree_map(
            lambda p, n: p.at[target_update_idx].set(n),
            state.particles,
            new_inner_state.position,
        )
        loglikelihood = state.loglikelihood.at[target_update_idx].set(
            new_inner_state.loglikelihood
        )
        loglikelihood_birth = state.loglikelihood_birth.at[target_update_idx].set(
            loglikelihood_0 * jnp.ones(len(target_update_idx))
        )
        logprior = state.logprior.at[target_update_idx].set(new_inner_state.logprior)
        pid = state.pid.at[target_update_idx].set(state.pid[start_idx])

        # Update the run-time information
        logX, logZ, logZ_live = update_ns_runtime_info(
            state.logX, state.logZ, loglikelihood, dead_loglikelihood
        )

        # Return updated state and info
        state = NSState(
            particles,
            loglikelihood,
            loglikelihood_birth,
            logprior,
            pid,
            logX,
            logZ,
            logZ_live,
            state.inner_kernel_params,
        )
        info = NSInfo(
            dead_particles,
            dead_loglikelihood,
            dead_loglikelihood_birth,
            dead_logprior,
            inner_info,
        )
        return state, info

    return kernel


def delete_fn(
    rng_key: PRNGKey, state: NSState, num_delete: int
) -> tuple[Array, Array, Array]:
    """Identify particles to delete and select live particles for resampling.

    Parameters
    ----------
    rng_key
        JAX PRNG key for choosing live particles.
    state
        Current state of the Nested Sampler.
    num_delete
        Number of particles to delete and replace.

    Returns
    -------
    A tuple containing dead indices, target update indices, and start indices.
    """
    loglikelihood = state.loglikelihood
    neg_dead_loglikelihood, dead_idx = jax.lax.top_k(-loglikelihood, num_delete)
    constraint = loglikelihood > -neg_dead_loglikelihood.min()
    weights = jnp.array(constraint, dtype=jnp.float32)
    start_idx = jax.random.choice(
        rng_key,
        len(weights),
        shape=(num_delete,),
        p=weights / weights.sum(),
        replace=True,
    )
    target_update_idx = dead_idx
    return dead_idx, target_update_idx, start_idx


def update_ns_runtime_info(
    logX: Array, logZ: Array, loglikelihood: Array, dead_loglikelihood: Array
) -> tuple[Array, Array, Array]:
    """Update the Nested Sampling runtime information."""
    num_particles = len(loglikelihood)
    num_deleted = len(dead_loglikelihood)
    num_live = jnp.arange(num_particles, num_particles - num_deleted, -1)
    delta_logX = -1 / num_live
    logX = logX + jnp.cumsum(delta_logX)
    log_delta_X = logX + jnp.log(1 - jnp.exp(delta_logX))
    log_delta_Z = dead_loglikelihood + log_delta_X

    delta_logZ = logsumexp(log_delta_Z)
    logZ = jnp.logaddexp(logZ, delta_logZ)
    logZ_live = logmeanexp(loglikelihood) + logX[-1]
    return logX[-1], logZ, logZ_live


def logmeanexp(x: Array) -> Array:
    """Compute log of mean of exp(x)."""
    return logsumexp(x) - jnp.log(len(x))
