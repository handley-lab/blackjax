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
"""Public API for adaptive Nested Sampling."""

from typing import Callable, Dict, Optional

import jax.numpy as jnp

from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import build_kernel as base_build_kernel
from blackjax.ns.base import init as base_init
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey

__all__ = ["init", "build_kernel"]


def init(
    particles: ArrayLikeTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_birth: Array = -jnp.nan,
    update_inner_kernel_params_fn: Optional[Callable] = None,
) -> NSState:
    """Initialize the adaptive Nested Sampler state.

    Parameters
    ----------
    particles
        Initial set of particles drawn from the prior.
    loglikelihood_fn
        Function that computes the log-likelihood of a single particle.
    logprior_fn
        Function that computes the log-prior of a single particle.
    loglikelihood_birth
        Initial log-likelihood birth threshold.
    update_inner_kernel_params_fn
        Function that updates inner kernel parameters.

    Returns
    -------
    The initial state of the Nested Sampler.
    """
    state = base_init(particles, logprior_fn, loglikelihood_fn, loglikelihood_birth)
    if update_inner_kernel_params_fn is not None:
        inner_kernel_params = update_inner_kernel_params_fn(state, None, {})
        state = state._replace(inner_kernel_params=inner_kernel_params)
    return state


def build_kernel(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    delete_fn: Callable,
    inner_kernel: Callable,
    update_inner_kernel_params_fn: Callable[
        [NSState, NSInfo, Dict[str, ArrayTree]], Dict[str, ArrayTree]
    ],
) -> Callable:
    """Build an adaptive Nested Sampling kernel.

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
    update_inner_kernel_params_fn
        Function that updates inner kernel parameters after each step.

    Returns
    -------
    A kernel function for adaptive Nested Sampling.
    """

    base_kernel = base_build_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_kernel,
    )

    def kernel(rng_key: PRNGKey, state: NSState) -> tuple[NSState, NSInfo]:
        """Generate a new sample with the adaptive NS kernel."""
        new_state, info = base_kernel(rng_key, state)

        inner_kernel_params = update_inner_kernel_params_fn(
            new_state, info, new_state.inner_kernel_params
        )
        new_state = new_state._replace(inner_kernel_params=inner_kernel_params)
        return new_state, info

    return kernel
