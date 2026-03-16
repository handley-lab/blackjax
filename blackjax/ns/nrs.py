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

Importance-weighted rejection sampling from a proposal distribution fitted to
live points. By default uses a Gaussian proposal. A fixed-size buffer of
proposals is maintained; non-surviving slots are overwritten each round.
When a heavier weight is observed, all stored proposals are re-tested against
the new maximum, preserving independence of survivors.
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_means,
)
from blackjax.types import Array, ArrayTree

__all__ = [
    "RejectionInfo",
    "_build_rejection_inner_kernel",
    "as_top_level_api",
    "build_kernel",
    "ellipsoid_proposal",
    "gaussian_proposal",
    "init",
    "update_ellipsoid_params",
    "update_gaussian_params",
]


class RejectionInfo(NamedTuple):
    """Information returned by the proposal inner kernel.

    Attributes
    ----------
    num_proposals_total
        Total proposals drawn across all batches.
    num_accepted_total
        Proposals passing the likelihood constraint (before rejection test).
    num_survivors
        Proposals surviving the final rejection test (after w_max rescaling).
    num_rounds
        Number of batches processed.
    log_w_max
        Final maximum log-importance-weight observed across all batches.
    success
        Whether num_delete survivors were obtained.
    """

    num_proposals_total: Array
    num_accepted_total: Array
    num_survivors: Array
    num_rounds: Array
    log_w_max: Array
    success: Array


def gaussian_proposal(key, n, *, mean, cov):
    """Draw samples from a multivariate Gaussian and evaluate their log-densities.

    Parameters
    ----------
    key
        PRNG key.
    n
        Number of samples to draw.
    mean
        Mean of the Gaussian proposal.
    cov
        Covariance matrix of the Gaussian proposal.

    Returns
    -------
    tuple
        (samples, log_proposal) where samples has shape (n, d) and
        log_proposal has shape (n,).
    """
    samples = jax.random.multivariate_normal(key, mean, cov, shape=(n,))
    log_proposal = jax.scipy.stats.multivariate_normal.logpdf(samples, mean, cov)
    return samples, log_proposal


def update_gaussian_params(
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
    return {"mean": mean, "cov": cov}


def ellipsoid_proposal(key, n, *, mean, L, log_volume):
    """Draw samples uniformly from an ellipsoid and evaluate their log-densities.

    The ellipsoid is defined by ``{x : (x - mean)^T Σ^{-1} (x - mean) <= 1}``
    where ``Σ = L L^T`` (L is lower-triangular Cholesky factor, already scaled
    to enclose all live points).

    Parameters
    ----------
    key
        PRNG key.
    n
        Number of samples to draw.
    mean
        Centre of the ellipsoid.
    L
        Cholesky factor of the (scaled) covariance matrix defining the ellipsoid.
    log_volume
        Log-volume of the ellipsoid (precomputed for efficiency).

    Returns
    -------
    tuple
        (samples, log_proposal) where samples has shape (n, d) and
        log_proposal has shape (n,).
    """
    d = mean.shape[0]
    key1, key2 = jax.random.split(key)
    # Sample uniformly from the unit ball: direction * radius^(1/d)
    directions = jax.random.normal(key1, (n, d))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    radii = jax.random.uniform(key2, (n, 1)) ** (1.0 / d)
    unit_ball = directions * radii
    # Transform to ellipsoid
    samples = mean[None, :] + unit_ball @ L.T
    log_proposal = jnp.full(n, -log_volume)
    return samples, log_proposal


def _log_unit_ball_volume(d):
    """Log-volume of the d-dimensional unit ball."""
    return (d / 2.0) * jnp.log(jnp.pi) - jax.lax.lgamma(d / 2.0 + 1.0)


def update_ellipsoid_params(
    rng_key,
    state: NSState,
    info: NSInfo,
    inner_kernel_params: Optional[Dict[str, ArrayTree]] = None,
) -> Dict[str, ArrayTree]:
    """Update ellipsoid parameters from current live points.

    Fits a bounding ellipsoid to the live points following the MultiNest
    strategy (Feroz et al. 2009):
    1. Compute the empirical mean and covariance.
    2. Find the minimum enlargement factor so the ellipsoid contains all
       live points (maximum Mahalanobis distance).
    3. Further enlarge if necessary so the ellipsoid volume is at least
       the estimated remaining prior volume ``exp(-i/N)`` where ``i`` is
       the iteration count and ``N`` the number of live points.

    Parameters
    ----------
    rng_key
        PRNG key (unused but required by interface).
    state
        The current NSState containing live particles.
    info
        Information from the last NS step (unused but kept for interface).
    inner_kernel_params
        Previous inner kernel parameters. Used to retrieve and increment
        the iteration counter ``i``.

    Returns
    -------
    Dict[str, ArrayTree]
        Dictionary containing 'mean', 'L' (scaled Cholesky factor),
        'log_volume' of the bounding ellipsoid, and 'i' iteration counter.
    """
    positions = state.particles.position
    n_live = positions.shape[0]
    mean = particles_means(positions)
    cov = jnp.atleast_2d(particles_covariance_matrix(positions))
    L_cov = jnp.linalg.cholesky(cov)

    # Mahalanobis distances: solve L_cov @ z = (x - mean) for z, then ||z||
    centered = positions - mean[None, :]
    z = jax.scipy.linalg.solve_triangular(L_cov, centered.T, lower=True).T
    mahal_sq = jnp.sum(z**2, axis=-1)
    max_mahal = jnp.sqrt(mahal_sq.max())

    # Minimum bounding ellipsoid
    d = jnp.array(mean.shape[0], dtype=float)
    log_ball = _log_unit_ball_volume(d)
    log_vol_bounding = (
        log_ball + jnp.sum(jnp.log(jnp.diag(L_cov))) + d * jnp.log(max_mahal)
    )

    # Estimated remaining prior volume: exp(-i/N)
    i = inner_kernel_params.get("i", jnp.array(0.0)) + 1.0
    log_vol_prior = -i / n_live

    # Enlarge to whichever is larger
    log_vol_target = jnp.maximum(log_vol_bounding, log_vol_prior)

    # Scale factor: V_target = V_ball * det(L_cov) * scale^d
    # => log(scale) = (log_vol_target - log_ball - log|det(L_cov)|) / d
    log_det_L = jnp.sum(jnp.log(jnp.diag(L_cov)))
    log_scale = (log_vol_target - log_ball - log_det_L) / d
    scale = jnp.exp(log_scale)

    L_scaled = L_cov * scale
    log_volume = log_vol_target

    return {"mean": mean, "L": L_scaled, "log_volume": log_volume, "i": i}


def _build_rejection_inner_kernel(
    init_state_fn: Callable,
    unravel_fn: Callable,
    proposal_fn: Callable,
    num_delete: int,
    num_proposals: int,
    max_rounds: int = 100,
):
    """Factory for a simple rejection-based update strategy.

    Draws samples from the proposal, accepts those with likelihood above the
    threshold, and collects until ``num_delete`` are found. No importance
    weighting — every accepted sample is kept with equal probability.

    Parameters
    ----------
    init_state_fn
        Scalar function: position -> StateWithLogLikelihood.
    unravel_fn
        Function mapping a flat array to a PyTree position.
    proposal_fn
        Proposal function with signature ``(key, n, **params) -> (samples, log_proposal)``.
        The ``log_proposal`` return value is ignored.
    num_delete
        Number of replacement particles to produce.
    num_proposals
        Batch size: number of proposals drawn per round.
    max_rounds
        Maximum number of batches before declaring failure.

    Returns
    -------
    Callable
        Update function with signature
        ``(rng_key, state, loglikelihood_0, **params) -> (new_particles, info)``.
    """

    def update_function(rng_key, state, loglikelihood_0, **params):
        NEG_INF = jnp.array(-jnp.inf)
        proposal_params = {k: v for k, v in params.items() if k != "i"}

        @jax.vmap
        def eval_states(x):
            return init_state_fn(unravel_fn(x), loglikelihood_birth=loglikelihood_0)

        prototype = jax.tree.map(lambda x: x[0], state.particles)

        def cond_fn(carry):
            _, _, _, _, surviving, _, _, round_count = carry
            return (surviving.sum() < num_delete) & (round_count < max_rounds)

        def body_fn(carry):
            states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd = carry
            key, batch_key, u_key = jax.random.split(key, 3)

            samples, log_proposal = proposal_fn(
                batch_key, num_proposals, **proposal_params
            )
            batch_states = eval_states(samples)

            # Prior weight: π(x)/q(x); set to -inf if likelihood too low
            batch_log_w = batch_states.logdensity - log_proposal
            in_contour = batch_states.loglikelihood > loglikelihood_0
            batch_log_w = jnp.where(in_contour, batch_log_w, NEG_INF)
            batch_log_u = jnp.log(jax.random.uniform(u_key, shape=(num_proposals,)))

            # Pool with existing survivors and re-test all against new log_w_max
            all_states = jax.tree.map(
                lambda a, b: jnp.concatenate([a, b]), states, batch_states
            )
            all_log_w = jnp.concatenate([log_w, batch_log_w])
            all_log_u = jnp.concatenate([log_u, batch_log_u])
            log_w_max = jnp.maximum(log_w_max, all_log_w.max())
            all_surviving = all_log_u < all_log_w - log_w_max

            num_surv = all_surviving.sum()
            (surv_idx,) = jnp.nonzero(all_surviving, size=num_proposals, fill_value=0)
            surviving = jnp.arange(num_proposals) < num_surv
            states = jax.tree.map(lambda a: a[surv_idx], all_states)
            log_w = jnp.where(surviving, all_log_w[surv_idx], NEG_INF)
            log_u = all_log_u[surv_idx]
            n_acc = n_acc + in_contour.sum()
            rnd = rnd + 1

            return states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd

        states = jax.tree.map(
            lambda x: jnp.zeros((num_proposals,) + x.shape), prototype
        )
        log_w = jnp.full(num_proposals, NEG_INF)
        log_u = jnp.zeros(num_proposals)
        log_w_max = NEG_INF
        surviving = jnp.zeros(num_proposals, dtype=bool)
        n_acc = jnp.array(0)
        key = rng_key
        rnd = jnp.array(0)

        carry = states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd
        final = jax.lax.while_loop(cond_fn, body_fn, carry)
        states, _, _, log_w_max, surviving, n_acc, _, num_rounds = final

        num_survivors = surviving.sum()
        (survivor_idx,) = jnp.nonzero(surviving, size=num_delete, fill_value=0)
        new_particles = jax.tree.map(lambda x: x[survivor_idx], states)

        info = RejectionInfo(
            num_proposals_total=num_rounds * num_proposals,
            num_accepted_total=n_acc,
            num_survivors=num_survivors,
            num_rounds=num_rounds,
            log_w_max=log_w_max,
            success=num_survivors >= num_delete,
        )
        return new_particles, info

    return update_function


def _build_proposal_inner_kernel(
    init_state_fn: Callable,
    unravel_fn: Callable,
    proposal_fn: Callable,
    num_delete: int,
    num_proposals: int,
    max_rounds: int = 100,
):
    """Factory for a proposal-based update strategy.

    Parameters
    ----------
    init_state_fn
        Scalar function: position -> StateWithLogLikelihood. Must accept
        ``loglikelihood_birth`` keyword argument.
    unravel_fn
        Function mapping a flat array to a PyTree position, obtained from
        ``jax.flatten_util.ravel_pytree``.
    proposal_fn
        Proposal function with signature ``(key, n, **params) -> (samples, log_proposal)``
        where samples has shape ``(n, d)`` and log_proposal has shape ``(n,)``.
    num_delete
        Number of replacement particles to produce.
    num_proposals
        Batch size: number of proposals drawn per round.
    max_rounds
        Maximum number of batches before declaring failure.

    Returns
    -------
    Callable
        Update function with signature
        ``(rng_key, state, loglikelihood_0, **params) -> (new_particles, info)``.
        When ``info.success`` is False, the algorithm failed to find enough
        survivors. The returned ``new_particles`` may contain duplicates
        (from ``jnp.nonzero`` fill_value=0) and should not be used without
        checking ``info.success``.
    """

    def update_function(rng_key, state, loglikelihood_0, **params):
        NEG_INF = jnp.array(-jnp.inf)
        proposal_params = {k: v for k, v in params.items() if k != "i"}

        @jax.vmap
        def eval_states(x):
            return init_state_fn(unravel_fn(x), loglikelihood_birth=loglikelihood_0)

        prototype = jax.tree.map(lambda x: x[0], state.particles)

        def cond_fn(carry):
            _, _, _, _, surviving, _, _, round_count = carry
            return (surviving.sum() < num_delete) & (round_count < max_rounds)

        def body_fn(carry):
            states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd = carry
            key, batch_key, u_key = jax.random.split(key, 3)

            samples, log_proposal = proposal_fn(
                batch_key, num_proposals, **proposal_params
            )
            batch_states = eval_states(samples)
            batch_log_w = batch_states.logdensity - log_proposal
            valid = batch_states.loglikelihood > loglikelihood_0
            batch_log_w = jnp.where(valid, batch_log_w, NEG_INF)
            batch_log_u = jnp.log(jax.random.uniform(u_key, shape=(num_proposals,)))

            all_states = jax.tree.map(
                lambda a, b: jnp.concatenate([a, b]), states, batch_states
            )
            all_log_w = jnp.concatenate([log_w, batch_log_w])
            all_log_u = jnp.concatenate([log_u, batch_log_u])
            log_w_max = jnp.maximum(log_w_max, all_log_w.max())
            all_surviving = all_log_u < all_log_w - log_w_max

            num_surv = all_surviving.sum()
            (surv_idx,) = jnp.nonzero(all_surviving, size=num_proposals, fill_value=0)
            surviving = jnp.arange(num_proposals) < num_surv
            states = jax.tree.map(lambda a: a[surv_idx], all_states)
            log_w = jnp.where(surviving, all_log_w[surv_idx], NEG_INF)
            log_u = all_log_u[surv_idx]
            n_acc = n_acc + valid.sum()
            rnd = rnd + 1

            return states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd

        states = jax.tree.map(
            lambda x: jnp.zeros((num_proposals,) + x.shape), prototype
        )
        log_w = jnp.full(num_proposals, NEG_INF)
        log_u = jnp.zeros(num_proposals)
        log_w_max = NEG_INF
        surviving = jnp.zeros(num_proposals, dtype=bool)
        n_acc = jnp.array(0)
        key = rng_key
        rnd = jnp.array(0)

        carry = states, log_w, log_u, log_w_max, surviving, n_acc, key, rnd
        final = jax.lax.while_loop(cond_fn, body_fn, carry)
        states, _, _, log_w_max, surviving, n_acc, _, num_rounds = final

        num_survivors = surviving.sum()
        (survivor_idx,) = jnp.nonzero(surviving, size=num_delete, fill_value=0)
        new_particles = jax.tree.map(lambda x: x[survivor_idx], states)

        info = RejectionInfo(
            num_proposals_total=num_rounds * num_proposals,
            num_accepted_total=n_acc,
            num_survivors=num_survivors,
            num_rounds=num_rounds,
            log_w_max=log_w_max,
            success=num_survivors >= num_delete,
        )
        return new_particles, info

    return update_function


def build_kernel(
    init_state_fn: Callable,
    unravel_fn: Callable,
    num_proposals: int,
    num_delete: int = 1,
    max_rounds: int = 100,
    proposal_fn: Callable = gaussian_proposal,
    update_inner_kernel_params_fn: Callable = update_gaussian_params,
    delete_fn: Callable = default_delete_fn,
    inner_kernel_builder: Callable = _build_proposal_inner_kernel,
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
    proposal_fn
        Proposal function with signature ``(key, n, **params) -> (samples, log_proposal)``.
    update_inner_kernel_params_fn
        Function to update inner kernel parameters.
    delete_fn
        Particle deletion function.
    inner_kernel_builder
        Factory function for the inner kernel. Defaults to
        ``_build_proposal_inner_kernel`` (importance-weighted rejection).
        Use ``_build_rejection_inner_kernel`` for simple rejection.

    Returns
    -------
    Callable
        A kernel function for Nested Rejection Sampling.
    """
    inner_kernel = inner_kernel_builder(
        init_state_fn, unravel_fn, proposal_fn, num_delete, num_proposals, max_rounds
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
    prototype_position: ArrayTree,
    num_proposals: int,
    num_delete: int = 1,
    max_rounds: int = 100,
    proposal_fn: Callable = gaussian_proposal,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_gaussian_params,
    delete_fn: Callable = default_delete_fn,
    inner_kernel_builder: Callable = _build_proposal_inner_kernel,
) -> SamplingAlgorithm:
    """Creates a Nested Rejection Sampling (NRS) algorithm.

    Uses importance-weighted rejection sampling from a proposal distribution
    fitted to the live points. By default uses a Gaussian proposal whose
    mean and covariance are updated adaptively at each NS step.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    prototype_position
        A single example position (PyTree) used to determine the flattening
        structure for proposals. Typically ``positions[0]``.
    num_delete
        The number of particles to delete and replace at each NS step.
    num_proposals
        The number of proposals drawn per batch.
    max_rounds
        Maximum number of proposal batches before declaring failure.
    proposal_fn
        Proposal function with signature ``(key, n, **params) -> (samples, log_proposal)``.
    init_state_strategy_fn
        A function to initialize particle state from positions.
    update_inner_kernel_params_fn
        A function to update inner kernel parameters from particles.
    delete_fn
        Particle deletion function.
    inner_kernel_builder
        Factory function for the inner kernel. Defaults to
        ``_build_proposal_inner_kernel`` (importance-weighted rejection).
        Use ``_build_rejection_inner_kernel`` for simple rejection.

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

    _, unravel_fn = ravel_pytree(prototype_position)

    kernel = build_kernel(
        init_state_fn,
        unravel_fn,
        num_proposals,
        num_delete,
        max_rounds,
        proposal_fn=proposal_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        inner_kernel_builder=inner_kernel_builder,
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
