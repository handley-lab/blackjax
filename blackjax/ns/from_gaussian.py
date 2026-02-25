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
"""Gaussian proposal inner kernel for Nested Sampling.

Importance-weighted rejection sampling from a Gaussian proposal fitted to
live points. Proposals passing the likelihood constraint are accepted into
a buffer with stored log-uniforms. When a heavier weight is observed, all
stored proposals are re-tested against the new maximum, preserving
independence of survivors.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.types import Array

__all__ = ["update_with_gaussian_proposal", "GaussianProposalInfo", "count_survivors"]


class GaussianProposalInfo(NamedTuple):
    """Information returned by the Gaussian proposal inner kernel.

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


def count_survivors(log_weights, log_uniforms, log_w_max):
    """Compute survival mask for importance-weighted rejection.

    A proposal survives iff it has a finite weight and its stored log-uniform
    satisfies ``log_u < log_w - log_w_max``.

    Parameters
    ----------
    log_weights
        Log-importance-weights for each proposal.
    log_uniforms
        Stored log-uniforms for each proposal.
    log_w_max
        Current maximum log-weight.

    Returns
    -------
    Array
        Boolean mask of surviving proposals.
    """
    valid = jnp.isfinite(log_weights)
    delta = jnp.where(
        jnp.isfinite(log_w_max), log_weights - log_w_max, jnp.array(-jnp.inf)
    )
    surviving = valid & (log_uniforms < delta)
    return surviving


def update_with_gaussian_proposal(
    init_state_fn: Callable,
    unravel_fn: Callable,
    num_delete: int,
    num_proposals: int,
    max_rounds: int = 100,
):
    """Factory for a Gaussian proposal update strategy.

    Parameters
    ----------
    init_state_fn
        Scalar function: position -> StateWithLogLikelihood. Must accept
        ``loglikelihood_birth`` keyword argument.
    unravel_fn
        Function mapping a flat array to a PyTree position, obtained from
        ``jax.flatten_util.ravel_pytree``.
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
        ``(rng_key, state, loglikelihood_0, *, mean, cov) -> (new_particles, info)``.
        When ``info.success`` is False, the algorithm failed to find enough
        survivors. The returned ``new_particles`` may contain duplicates
        (from ``jnp.nonzero`` fill_value=0) and should not be used without
        checking ``info.success``.
    """

    buffer_size = num_proposals * max_rounds

    def update_function(rng_key, state, loglikelihood_0, *, mean, cov):
        NEG_INF = jnp.array(-jnp.inf)

        # Pre-allocate buffers (fixed shape for JIT)
        prototype = jax.tree.map(lambda x: x[0], state.particles)
        buffer_states = jax.tree.map(
            lambda x: jnp.zeros((buffer_size,) + x.shape), prototype
        )
        buffer_log_weights = jnp.full(buffer_size, NEG_INF)
        buffer_log_uniforms = jnp.zeros(buffer_size)

        init_carry = (
            buffer_states,
            buffer_log_weights,
            buffer_log_uniforms,
            jnp.array(0),  # num_buffered
            NEG_INF,  # log_w_max
            jnp.array(0),  # num_accepted_total
            rng_key,
            jnp.array(0),  # round_count
        )

        def cond_fn(carry):
            _, log_w, log_u, _, log_w_max, _, _, round_count = carry
            surviving = count_survivors(log_w, log_u, log_w_max)
            return (surviving.sum() < num_delete) & (round_count < max_rounds)

        def body_fn(carry):
            states, log_w, log_u, num_buf, log_w_max, n_acc, key, rnd = carry
            key, batch_key, u_key = jax.random.split(key, 3)

            # Draw proposals from Gaussian (parallel)
            flat_proposals = jax.random.multivariate_normal(
                batch_key, mean, cov, shape=(num_proposals,)
            )
            positions = jax.vmap(unravel_fn)(flat_proposals)
            batch_states = jax.vmap(
                lambda x: init_state_fn(x, loglikelihood_birth=loglikelihood_0)
            )(positions)

            # Constraint mask + log-importance-weights
            accepted = batch_states.loglikelihood > loglikelihood_0
            log_proposal = jax.scipy.stats.multivariate_normal.logpdf(
                flat_proposals, mean, cov
            )
            log_weights = batch_states.logdensity - log_proposal
            valid = accepted & jnp.isfinite(log_weights)
            log_weights = jnp.where(valid, log_weights, NEG_INF)

            # Log-uniforms for rejection test (clip away from 0 to avoid -inf)
            u = jax.random.uniform(u_key, shape=(num_proposals,))
            u = jnp.clip(u, jnp.finfo(u.dtype).tiny, 1.0)
            log_uniforms = jnp.log(u)

            # Append to buffer. Invariant: num_buf == rnd * num_proposals,
            # and cond_fn prevents rnd == max_rounds from entering body_fn,
            # so idx is always within [0, buffer_size).
            idx = jnp.arange(num_proposals) + num_buf
            states = jax.tree.map(
                lambda s, b: s.at[idx].set(b), states, batch_states
            )
            log_w = log_w.at[idx].set(log_weights)
            log_u = log_u.at[idx].set(log_uniforms)

            # Update tracking (all log-space, no overflow)
            batch_log_w_max = jnp.where(valid, log_weights, NEG_INF).max()
            new_log_w_max = jnp.maximum(log_w_max, batch_log_w_max)
            new_n_acc = n_acc + valid.sum()

            return (
                states,
                log_w,
                log_u,
                num_buf + num_proposals,
                new_log_w_max,
                new_n_acc,
                key,
                rnd + 1,
            )

        final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        states, log_w, log_u, num_buf, log_w_max, n_acc, _, num_rounds = final

        # Extract first num_delete survivors
        surviving = count_survivors(log_w, log_u, log_w_max)
        (survivor_idx,) = jnp.nonzero(surviving, size=num_delete, fill_value=0)
        new_particles = jax.tree.map(lambda x: x[survivor_idx], states)

        info = GaussianProposalInfo(
            num_proposals_total=num_buf,
            num_accepted_total=n_acc,
            num_survivors=surviving.sum(),
            num_rounds=num_rounds,
            log_w_max=log_w_max,
            success=surviving.sum() >= num_delete,
        )
        return new_particles, info

    return update_function
