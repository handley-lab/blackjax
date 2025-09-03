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
"""Utility functions for Nested Sampling."""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.types import Array, ArrayTree, PRNGKey


def log1mexp(x: Array) -> Array:
    """Compute log(1 - exp(x)) in a numerically stable way.

    Parameters
    ----------
    x
        Input array or scalar.

    Returns
    -------
    The value of log(1 - exp(x)).
    """
    return jnp.where(
        x > -0.6931472,  # approx log(2)
        jnp.log(-jnp.expm1(x)),
        jnp.log1p(-jnp.exp(x)),
    )


def compute_num_live(info: NSInfo) -> Array:
    """Compute the effective number of live points at each death contour.

    Parameters
    ----------
    info
        NSInfo object containing birth and death log-likelihoods.

    Returns
    -------
    Array of effective number of live points.
    """
    birth_logL = info.loglikelihood_birth
    death_logL = info.loglikelihood

    birth_events = jnp.column_stack(
        (birth_logL, jnp.ones_like(birth_logL, dtype=jnp.int32))
    )
    death_events = jnp.column_stack(
        (death_logL, -jnp.ones_like(death_logL, dtype=jnp.int32))
    )
    combined = jnp.concatenate([birth_events, death_events], axis=0)
    logL_col = combined[:, 0]
    n_col = combined[:, 1]
    not_nan_sort_key = ~jnp.isnan(logL_col)
    logL_sort_key = logL_col
    n_sort_key = n_col
    sorted_indices = jnp.lexsort((n_sort_key, logL_sort_key, not_nan_sort_key))
    sorted_n_col = n_col[sorted_indices]
    cumsum = jnp.cumsum(sorted_n_col)
    cumsum = jnp.maximum(cumsum, 0)
    death_mask_sorted = sorted_n_col == -1
    num_live = cumsum[death_mask_sorted] + 1

    return num_live


def logX(rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100) -> tuple[Array, Array]:
    """Simulate the stochastic evolution of log prior volumes.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    dead_info
        NSInfo object containing dead particles.
    shape
        Number of Monte Carlo samples to generate.

    Returns
    -------
    Tuple of log prior volumes and log volume elements.
    """
    rng_key, subkey = jax.random.split(rng_key)
    min_val = jnp.finfo(dead_info.loglikelihood.dtype).tiny
    r = jnp.log(
        jax.random.uniform(
            subkey, shape=(dead_info.loglikelihood.shape[0], shape)
        ).clip(min_val, 1 - min_val)
    )

    num_live = compute_num_live(dead_info)
    t = r / num_live[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate([logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0)
    log_diff = logXm - logXp
    logdX = log1mexp(log_diff) + logXp - jnp.log(2)
    return logX, logdX


def log_weights(
    rng_key: PRNGKey, dead_info: NSInfo, shape: int = 100, beta: float = 1.0
) -> Array:
    """Calculate the log importance weights for Nested Sampling results.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    dead_info
        NSInfo object containing dead particles.
    shape
        Number of Monte Carlo samples.
    beta
        Inverse temperature.

    Returns
    -------
    Array of log importance weights.
    """
    sort_indices = jnp.argsort(dead_info.loglikelihood)
    unsort_indices = jnp.empty_like(sort_indices)
    unsort_indices = unsort_indices.at[sort_indices].set(jnp.arange(len(sort_indices)))
    dead_info_sorted = jax.tree.map(lambda x: x[sort_indices], dead_info)
    _, log_dX = logX(rng_key, dead_info_sorted, shape)
    log_w = log_dX + beta * dead_info_sorted.loglikelihood[..., jnp.newaxis]
    return log_w[unsort_indices]


def finalise(live: NSState, dead: list[NSInfo]) -> NSInfo:
    """Combine dead particle history with final live points.

    Parameters
    ----------
    live
        Final NSState containing live particles.
    dead
        List of NSInfo objects from NS steps.

    Returns
    -------
    Combined NSInfo object.
    """

    all_pytrees_to_combine = dead + [
        NSInfo(
            live.particles,
            live.loglikelihood,
            live.loglikelihood_birth,
            live.logprior,
            dead[-1].inner_kernel_info,
        )
    ]
    combined_dead_info = jax.tree.map(
        lambda *args: jnp.concatenate(args),
        all_pytrees_to_combine[0],
        *all_pytrees_to_combine[1:],
    )
    return combined_dead_info


def ess(rng_key: PRNGKey, dead_info_map: NSInfo) -> Array:
    """Compute the Effective Sample Size (ESS) from log-weights.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    dead_info_map
        NSInfo object containing all particles.

    Returns
    -------
    The mean Effective Sample Size.
    """
    logw = log_weights(rng_key, dead_info_map).mean(axis=-1)
    logw -= logw.max()
    l_sum_w = jax.scipy.special.logsumexp(logw)
    l_sum_w_sq = jax.scipy.special.logsumexp(2 * logw)
    ess = jnp.exp(2 * l_sum_w - l_sum_w_sq)
    return ess


def sample(rng_key: PRNGKey, dead_info_map: NSInfo, shape: int = 1000) -> ArrayTree:
    """Resample particles according to their importance weights.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    dead_info_map
        NSInfo object containing all particles.
    shape
        Number of posterior samples to draw.

    Returns
    -------
    PyTree of resampled particles.
    """
    logw = log_weights(rng_key, dead_info_map).mean(axis=-1)
    indices = jax.random.choice(
        rng_key,
        dead_info_map.loglikelihood.shape[0],
        p=jnp.exp(logw.squeeze() - jnp.max(logw)),
        shape=(shape,),
        replace=True,
    )
    return jax.tree.map(lambda leaf: leaf[indices], dead_info_map.particles)


def get_first_row(x: ArrayTree) -> ArrayTree:
    """Extract the first row of each leaf in a PyTree.

    Parameters
    ----------
    x
        PyTree of arrays.

    Returns
    -------
    PyTree with first slice of each leaf.
    """
    return jax.tree.map(lambda x: x[0], x)


def repeat_kernel(num_repeats: int):
    """Decorator to repeat a kernel function multiple times."""

    def decorator(kernel):
        @functools.wraps(kernel)
        def repeated_kernel(rng_key: PRNGKey, state, *args, **kwargs):
            def body_fn(state, rng_key):
                return kernel(rng_key, state, *args, **kwargs)

            keys = jax.random.split(rng_key, num_repeats)
            return jax.lax.scan(body_fn, state, keys)

        return repeated_kernel

    return decorator


def uniform_prior(
    rng_key: PRNGKey, num_live: int, bounds: Dict[str, Tuple[float, float]]
) -> Tuple[ArrayTree, Callable]:
    """Create a uniform prior for parameters.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    num_live
        Number of live particles to sample.
    bounds
        Dictionary mapping parameter names to bounds.

    Returns
    -------
    Tuple of sampled particles and log-prior function.
    """

    def logprior_fn(params):
        logprior = 0.0
        for p, (a, b) in bounds.items():
            x = params[p]
            logprior += jax.scipy.stats.uniform.logpdf(x, a, b - a)
        return logprior

    def prior_sample(rng_key):
        init_keys = jax.random.split(rng_key, len(bounds))
        params = {}
        for rng_key, (p, (a, b)) in zip(init_keys, bounds.items()):
            params[p] = jax.random.uniform(rng_key, minval=a, maxval=b)
        return params

    init_keys = jax.random.split(rng_key, num_live)
    particles = jax.vmap(prior_sample)(init_keys)

    return particles, logprior_fn
