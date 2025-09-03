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
"""Public API for Hit-and-Run Slice Sampling."""

from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = [
    "SliceState",
    "SliceInfo",
    "init",
    "build_kernel",
    "build_hrss_kernel",
    "hrss_as_top_level_api",
]


class SliceState(NamedTuple):
    """State of the Slice Sampling algorithm.

    Attributes
    ----------
    position
        Current position of the chain.
    logdensity
        Log-density at the current position.
    logslice
        Log-height defining the slice.
    """

    position: ArrayLikeTree
    logdensity: float
    logslice: float = jnp.inf


class SliceInfo(NamedTuple):
    """Additional information on the Slice Sampling transition.

    Attributes
    ----------
    constraint
        Constraint values at the final position.
    l_steps
        Number of left expansion steps.
    r_steps
        Number of right expansion steps.
    s_steps
        Number of shrinking steps.
    evals
        Total number of log-density evaluations.
    """

    constraint: Array = jnp.array([])
    l_steps: int = 0
    r_steps: int = 0
    s_steps: int = 0
    evals: int = 0


def init(position: ArrayTree, logdensity_fn: Callable) -> SliceState:
    """Initialize the Slice Sampler state.

    Parameters
    ----------
    position
        The initial position of the chain.
    logdensity_fn
        A function that computes the log-density of the target distribution.

    Returns
    -------
    SliceState
        The initial state of the Slice Sampler.
    """
    return SliceState(position, logdensity_fn(position))


def build_kernel(
    stepper_fn: Callable,
) -> Callable:
    """Build a Slice Sampling kernel.

    Parameters
    ----------
    stepper_fn
        Function that computes a new position given position, direction and step size.

    Returns
    -------
    A kernel function for Slice Sampling.
    """

    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        logdensity_fn: Callable,
        d: ArrayTree,
        constraint_fn: Callable,
        constraint: Array,
        strict: Array,
    ) -> tuple[SliceState, SliceInfo]:
        """Generate a new sample with the Slice Sampling kernel."""
        rng_key, vs_key, hs_key = jax.random.split(rng_key, 3)
        intermediate_state, vs_info = vertical_slice(vs_key, state)
        new_state, hs_info = horizontal_slice(
            hs_key,
            intermediate_state,
            d,
            stepper_fn,
            logdensity_fn,
            constraint_fn,
            constraint,
            strict,
        )

        info = SliceInfo(
            constraint=hs_info.constraint,
            l_steps=hs_info.l_steps,
            r_steps=hs_info.r_steps,
            s_steps=hs_info.s_steps,
            evals=vs_info.evals + hs_info.evals,
        )

        return new_state, info

    return kernel


def vertical_slice(rng_key: PRNGKey, state: SliceState) -> tuple[SliceState, SliceInfo]:
    """Define the vertical slice.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    state
        Current slice sampling state.

    Returns
    -------
    Updated state with slice height and info.
    """
    logslice = state.logdensity + jnp.log(jax.random.uniform(rng_key))
    new_state = state._replace(logslice=logslice)
    info = SliceInfo()
    return new_state, info


def horizontal_slice(
    rng_key: PRNGKey,
    state: SliceState,
    d: ArrayTree,
    stepper_fn: Callable,
    logdensity_fn: Callable,
    constraint_fn: Callable,
    constraint: Array,
    strict: Array,
) -> tuple[SliceState, SliceInfo]:
    """Propose a new sample using stepping-out and shrinking.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    state
        Current slice sampling state.
    d
        Direction for proposing moves.
    stepper_fn
        Function that computes new position.
    logdensity_fn
        Log-density function of target distribution.
    constraint_fn
        Function that evaluates additional constraints.
    constraint
        Constraint threshold values.
    strict
        Boolean flags for strict vs non-strict constraints.

    Returns
    -------
    New state and sampling process information.
    """
    # Initial bounds
    rng_key, subkey = jax.random.split(rng_key)
    u = jax.random.uniform(subkey)
    x0 = state.position

    def body_fun(carry):
        _, s, t, n = carry
        t += s
        x = stepper_fn(x0, d, t)
        logdensity_x = logdensity_fn(x)
        constraint_x = constraint_fn(x)
        constraints = jnp.where(
            strict, constraint_x > constraint, constraint_x >= constraint
        )
        constraints = jnp.append(constraints, logdensity_x >= state.logslice)
        within = jnp.all(constraints)
        n += 1
        return within, s, t, n

    def cond_fun(carry):
        within = carry[0]
        return within

    # Expand
    _, _, l, l_steps = jax.lax.while_loop(cond_fun, body_fun, (True, -1, -u, 0))
    _, _, r, r_steps = jax.lax.while_loop(cond_fun, body_fun, (True, +1, 1 - u, 0))

    # Shrink
    def shrink_body_fun(carry):
        _, l, r, _, _, _, rng_key, s_steps = carry
        s_steps += 1

        rng_key, subkey = jax.random.split(rng_key)
        u = jax.random.uniform(subkey, minval=l, maxval=r)
        x = stepper_fn(x0, d, u)

        logdensity_x = logdensity_fn(x)
        constraint_x = constraint_fn(x)
        constraints = jnp.where(
            strict, constraint_x > constraint, constraint_x >= constraint
        )
        constraints = jnp.append(constraints, logdensity_x >= state.logslice)
        within = jnp.all(constraints)

        l = jnp.where(u < 0, u, l)
        r = jnp.where(u > 0, u, r)

        return within, l, r, x, logdensity_x, constraint_x, rng_key, s_steps

    def shrink_cond_fun(carry):
        within = carry[0]
        return ~within

    carry = (False, l, r, x0, -jnp.inf, constraint, rng_key, 0)
    carry = jax.lax.while_loop(shrink_cond_fun, shrink_body_fun, carry)
    _, l, r, x, logdensity_x, constraint_x, rng_key, s_steps = carry
    slice_state = SliceState(x, logdensity_x)
    evals = l_steps + r_steps + s_steps
    slice_info = SliceInfo(constraint_x, l_steps, r_steps, s_steps, evals)
    return slice_state, slice_info


def build_hrss_kernel(
    generate_slice_direction_fn: Callable,
    stepper_fn: Callable,
) -> Callable:
    """Build a Hit-and-Run Slice Sampling kernel.

    Parameters
    ----------
    generate_slice_direction_fn
        Function that generates a direction vector for hit-and-run.
    stepper_fn
        Function that computes new position.

    Returns
    -------
    A kernel function for Hit-and-Run Slice Sampling.
    """
    slice_kernel = build_kernel(stepper_fn)

    def kernel(
        rng_key: PRNGKey, state: SliceState, logdensity_fn: Callable
    ) -> tuple[SliceState, SliceInfo]:
        """Generate a new sample with the HRSS kernel."""
        rng_key, prop_key = jax.random.split(rng_key, 2)
        d = generate_slice_direction_fn(prop_key)
        constraint_fn = lambda x: jnp.array([])
        constraint = jnp.array([])
        strict = jnp.array([])
        return slice_kernel(
            rng_key, state, logdensity_fn, d, constraint_fn, constraint, strict
        )

    return kernel


def default_stepper_fn(x: ArrayTree, d: ArrayTree, t: float) -> ArrayTree:
    """Default stepper function for slice sampling.

    Parameters
    ----------
    x
        Starting position.
    d
        Direction of movement.
    t
        Step size along the direction.

    Returns
    -------
    The new position.
    """
    return jax.tree.map(lambda x, d: x + t * d, x, d)


def sample_direction_from_covariance(rng_key: PRNGKey, cov: Array) -> Array:
    """Generate a normalized direction from a multivariate Gaussian.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    cov
        Covariance matrix for the multivariate Gaussian.

    Returns
    -------
    A normalized direction vector.
    """
    d = jax.random.multivariate_normal(rng_key, mean=jnp.zeros(cov.shape[0]), cov=cov)
    invcov = jnp.linalg.inv(cov)
    norm = jnp.sqrt(jnp.einsum("...i,...ij,...j", d, invcov, d))
    d = d / norm[..., None]
    return d


def hrss_as_top_level_api(
    logdensity_fn: Callable,
    cov: Array,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for Hit-and-Run Slice Sampling.

    Parameters
    ----------
    logdensity_fn
        Log-density function of the target distribution.
    cov
        Covariance matrix for the direction proposal.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """
    generate_slice_direction_fn = partial(sample_direction_from_covariance, cov=cov)
    kernel = build_hrss_kernel(generate_slice_direction_fn, default_stepper_fn)
    init_fn = partial(init, logdensity_fn=logdensity_fn)
    step_fn = partial(kernel, logdensity_fn=logdensity_fn)
    return SamplingAlgorithm(init_fn, step_fn)
