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
"""Gradient-Guided Hamiltonian Nested Sampling.

This module implements a JAX port of the Hamiltonian Nested Sampling algorithm
from `GGNS <https://github.com/Pablo-Lemos/GGNS>`_ (Pablo Lemos et al.).  The
key idea is to use gradient information from the log-likelihood to guide
trajectory proposals inside the NS likelihood contour.

Algorithm outline
-----------------
For each replacement particle:

1. Start from an existing live point inside the current likelihood contour.
2. Initialise a random unit-velocity vector, optionally perturbed by a small
   Gaussian noise ``sigma_vel`` for ergodicity.
3. Repeatedly take leapfrog steps of size ``dt``.
4. When a step exits the likelihood contour (``logL < logL_min``), **reflect
   the velocity off the likelihood surface** using the gradient:

   .. math::
       v' = v - 2",(v \cdot \hat n)\hat n, \quad
       \hat n = \frac{\nabla \log L}{\|\nabla \log L\|}

5. When a step exits the prior box ``[lower, upper]``, reflect the velocity
   component-wise off the boundary wall.
6. Continue until ``max_reflections`` reflections have occurred *or* the
   ``max_steps`` budget is exhausted.
7. Collect all trajectory points that were (a) inside the contour *and* (b)
   occurred after ``min_reflections`` reflections have been accumulated.
8. Return one of those valid points selected uniformly at random.

Setting ``num_delete=1`` gives **static** Hamiltonian NS (equivalent to
``HamiltonianStaticNS`` in GGNS); setting ``num_delete > 1`` gives **dynamic**
Hamiltonian NS (equivalent to ``HamiltonianNS`` in GGNS).  A single function
covers both, matching the GGNS design.

Notes
-----
- All array shapes are **static** at JIT compile time: trajectory buffers have
a fixed size ``max_steps`` filled with a validity mask.
- The reflection loop uses :func:`jax.lax.while_loop`; branching uses
:func:`jax.lax.cond` / :func:`jnp.where`.
- Gradients are obtained via :func:`jax.value_and_grad`.
- Positions must be *flat* 1-D arrays of shape ``(nparams,)``.  For dict-style
  positions, flatten them first (e.g. with :func:`jax.flatten_util.ravel_pytree`).
"""

from functools import partial
from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp

from blackjax import SamplingAlgorithm
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init
from blackjax.ns.base import NSInfo, NSState, StateWithLogLikelihood
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.types import Array, ArrayLikeTree, PRNGKey

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
]

# ------------------------------------------------------------------------------
# Module-level constants
# ------------------------------------------------------------------------------

# Threshold below which ||grad|| is treated as zero (no reliable normal vector).
_GRAD_NORM_THRESHOLD: float = 1e-10

# Numerical floor added to ||grad|| denominator to avoid division by zero
# in the inactive branch of jnp.where (which traces both sides).
_GRAD_NORM_EPS: float = 1e-30

# Default initial leapfrog step size.
_DEFAULT_DT: float = 1e-2

# dt adaptation thresholds and scale factors (from GGNS).
# out_frac > _OUT_FRAC_HIGH  => step is too large, shrink dt.
# out_frac < _OUT_FRAC_LOW   => step is too small, grow dt.
_OUT_FRAC_HIGH: float = 0.15
_OUT_FRAC_LOW: float = 0.05
_DT_SHRINK: float = 0.9
_DT_GROW: float = 1.1
_DT_MIN: float = 1e-5
_DT_MAX: float = 10.0


class HamiltonianInfo(NamedTuple):
    """Per-step information returned by the Hamiltonian trajectory sampler.

    Attributes
    ----------
    out_frac
        Fraction of trajectory steps that were *outside* the likelihood contour
        (or outside the prior box).  Used by :func:`update_inner_kernel_params`
        to adapt the step size ``dt``.
    """

    out_frac: Array


def hamiltonian_reflection_step(
    rng_key: PRNGKey,
    state: StateWithLogLikelihood,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    loglikelihood_0: Array,
    dt: float,
    min_reflections: int,
    max_reflections: int,
    sigma_vel: float,
    lower: Array,
    upper: Array,
    max_steps: int,
) -> tuple[StateWithLogLikelihood, HamiltonianInfo]:
    """Run a single gradient-guided Hamiltonian trajectory with reflections.

    Implements one "MCMC step" for the NS inner kernel: starting from ``state``,
it runs a Hamiltonian trajectory that reflects off the likelihood contour
and prior boundaries, collects valid interior points, and returns one of
those selected at random.

    Parameters
    ----------
    rng_key
        JAX PRNG key.
    state
        Current particle state (a single particle, not a batch).
    loglikelihood_fn
        Log-likelihood function for a single position array.
    logprior_fn
        Log-prior function for a single position array.
    loglikelihood_0
        Current NS likelihood threshold.  Accepted positions must satisfy
        ``logL > loglikelihood_0``.
    dt
        Leapfrog step size.
    min_reflections
        Minimum number of reflections before a trajectory point is considered
        valid for selection.
    max_reflections
        Target number of reflections that triggers trajectory termination.
    sigma_vel
        Standard deviation of Gaussian noise added to the initial velocity for
        ergodicity.
    lower
        Lower prior-box bounds, shape ``(nparams,)``.
    upper
        Upper prior-box bounds, shape ``(nparams,)``.
    max_steps
        Hard upper limit on the number of leapfrog steps (static Python int).

    Returns
    -------
    tuple[StateWithLogLikelihood, HamiltonianInfo]
        The new particle state (or the original ``state`` if no valid point was
        found) and diagnostic information.
    """
    pos = state.position  # shape (nparams,)

    # ---------------------------------------------------------------------- #
    # 1. Initialise velocity                                                   #
    # ---------------------------------------------------------------------- #
    rng_key, vel_key, noise_key, sel_key = jax.random.split(rng_key, 4)
    vel = jax.random.normal(vel_key, shape=pos.shape)
    vel_norm = jnp.linalg.norm(vel)
    vel = jnp.where(vel_norm > 0, vel / vel_norm, vel)

    # Small perturbation for ergodicity
    noise = jax.random.normal(noise_key, shape=vel.shape) * sigma_vel
    vel = vel + noise
    vel_norm = jnp.linalg.norm(vel)
    vel = jnp.where(vel_norm > 0, vel / vel_norm, vel)

    # ---------------------------------------------------------------------- #
    # 2. Pre-allocate fixed-size trajectory buffers                            #
    # ---------------------------------------------------------------------- #
    buf_pos = jnp.zeros((max_steps, *pos.shape))
    buf_logL = jnp.full(max_steps, -jnp.inf)
    buf_valid = jnp.zeros(max_steps, dtype=jnp.bool_)

    init_carry = (
        pos,
        vel,
        jnp.int32(0),   # num_reflections
        jnp.int32(0),   # step_count
        jnp.int32(0),   # out_count (steps outside contour or prior)
        buf_pos,
        buf_logL,
        buf_valid,
    )

    # ---------------------------------------------------------------------- #
    # 3. Trajectory loop via jax.lax.while_loop                               #
    # ---------------------------------------------------------------------- #

    def cond_fn(carry):
        _, _, num_reflections, step_count, _, _, _, _ = carry
        return (num_reflections < max_reflections) & (step_count < max_steps)

    def body_fn(carry):
        pos, vel, num_reflections, step_count, out_count, buf_pos, buf_logL, buf_valid = carry

        # --- leapfrog step ---
        new_pos = pos + vel * dt

        # --- gradient of log-likelihood ---
        logL, grad = jax.value_and_grad(loglikelihood_fn)(new_pos)

        # --- constraint checks ---
        in_prior = jnp.all((new_pos >= lower) & (new_pos <= upper))
        in_contour = logL > loglikelihood_0

        # --- component-wise prior-wall reflection ---
        beyond_lower = new_pos < lower
        beyond_upper = new_pos > upper
        vel_prior = vel * jnp.where(beyond_lower | beyond_upper, -1.0, 1.0)

        # --- likelihood-surface reflection: v' = v - 2*(v·n)*n ---
        norm_grad = jnp.linalg.norm(grad)
        normal = jnp.where(
            norm_grad > _GRAD_NORM_THRESHOLD,
            grad / (norm_grad + _GRAD_NORM_EPS),  # +eps avoids division by zero
            jnp.zeros_like(grad),
        )
        vel_ll = vel - 2.0 * jnp.dot(vel, normal) * normal

        # --- choose which (if any) reflection to apply ---
        need_prior_reflect = ~in_prior
        need_ll_reflect = ~in_contour & in_prior
        reflected = need_prior_reflect | need_ll_reflect

        new_vel = jnp.where(
            need_prior_reflect,
            vel_prior,
            jnp.where(need_ll_reflect, vel_ll, vel),
        )

        new_num_reflections = num_reflections + reflected.astype(jnp.int32)
        new_out_count = out_count + (~(in_contour & in_prior)).astype(jnp.int32)

        # --- valid if inside contour AND past min_reflections ---
        is_valid = in_contour & in_prior & (new_num_reflections >= min_reflections)

        # --- update trajectory buffers (always write; use buf_valid to filter) ---
        new_buf_pos = buf_pos.at[step_count].set(new_pos)
        new_buf_logL = buf_logL.at[step_count].set(logL)
        new_buf_valid = buf_valid.at[step_count].set(is_valid)

        return (
            new_pos,
            new_vel,
            new_num_reflections,
            step_count + 1,
            new_out_count,
            new_buf_pos,
            new_buf_logL,
            new_buf_valid,
        )

    final_carry = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    (_, _, _, step_f, out_count_f, buf_pos_f, buf_logL_f, buf_valid_f) = final_carry

    # ---------------------------------------------------------------------- #
    # 4. Compute out_frac                                                      #
    # ---------------------------------------------------------------------- #
    out_frac = out_count_f.astype(jnp.float32) / jnp.maximum(
        step_f.astype(jnp.float32), 1.0
    )

    # ---------------------------------------------------------------------- #
    # 5. Select a random valid trajectory point                                #
    # ---------------------------------------------------------------------- #
    valid_weights = buf_valid_f.astype(jnp.float32)
    total_valid = valid_weights.sum()
    has_valid = total_valid > 0.0

    # Uniform fall-back weights when no valid point exists
    safe_weights = jnp.where(has_valid, valid_weights, jnp.ones(max_steps))
    selected_idx = jax.random.choice(
        sel_key, max_steps, p=safe_weights / safe_weights.sum()
    )

    selected_pos = buf_pos_f[selected_idx]
    selected_logL = buf_logL_f[selected_idx]

    # Re-compute log-prior for selected position (cheap)
    selected_logprior = logprior_fn(selected_pos)

    # ---------------------------------------------------------------------- #
    # 6. Build new state; fall back to original if no valid point found        #
    # ---------------------------------------------------------------------- #
    new_state = StateWithLogLikelihood(
        position=jnp.where(has_valid, selected_pos, pos),
        logdensity=jnp.where(has_valid, selected_logprior, state.logdensity),
        loglikelihood=jnp.where(has_valid, selected_logL, state.loglikelihood),
        loglikelihood_birth=jnp.where(
            has_valid,
            jnp.full_like(state.loglikelihood_birth, loglikelihood_0),
            state.loglikelihood_birth,
        ),
    )

    return new_state, HamiltonianInfo(out_frac=out_frac)

def update_inner_kernel_params(
    rng_key: PRNGKey,
    state: NSState,
    info: Optional[NSInfo],
    inner_kernel_params: Dict,
) -> Dict:
    """Adapt the Hamiltonian step size ``dt`` based on trajectory diagnostics.

    The adaptation rule mirrors that of GGNS:

    * ``out_frac > 0.15``  => ``dt *= 0.9``   (too many outside steps; shrink)
    * ``out_frac < 0.05``  => ``dt *= 1.1``   (too few outside steps; grow)
    * ``dt`` is clipped to ``[1e-5, 10.0]``.

    On the very first call (``info is None``), the initial ``dt`` stored in
    ``inner_kernel_params['dt']`` (or ``1e-2`` if absent) is returned unchanged.

    Parameters
    ----------
    rng_key
        Unused PRNG key (kept for interface consistency).
    state
        Current NS state (unused).
    info
        ``NSInfo`` from the last NS step.  Its ``update_info`` field must
        contain the ``(states, HamiltonianInfo)`` tuple produced by
        :func:`~blackjax.ns.from_mcmc.update_with_mcmc_take_last`.
    inner_kernel_params
        Current parameter dictionary; must contain ``'dt'`` after the first
        call.

    Returns
    -------
    dict
        Updated ``{'dt': new_dt}``.
    """
    dt = inner_kernel_params.get("dt", jnp.array(_DEFAULT_DT))

    if info is None:
        return {"dt": dt}

    # info.update_info is a HamiltonianInfo with out_frac of shape
    # (num_delete, num_inner_steps) — produced by vmap over mcmc_kernel
    # whose scan accumulates HamiltonianInfo per inner step.
    ham_infos = info.update_info
    out_frac = jnp.mean(ham_infos.out_frac)

    dt = jnp.where(out_frac > _OUT_FRAC_HIGH, dt * _DT_SHRINK, dt)
    dt = jnp.where(out_frac < _OUT_FRAC_LOW, dt * _DT_GROW, dt)
    dt = jnp.clip(dt, _DT_MIN, _DT_MAX)
    return {"dt": dt}

def build_kernel(
    init_state_fn: Callable,
    loglikelihood_fn: Callable,
    logprior_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    min_reflections: int = 2,
    max_reflections: int = 10,
    sigma_vel: float = 0.0,
    lower: Array = None,
    upper: Array = None,
    max_steps: int = 200,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
) -> Callable:
    """Build a Hamiltonian Nested Sampling kernel.

    See :func:`as_top_level_api` for full parameter documentation.
    """

    def constrained_ham_fn(rng_key, state, loglikelihood_0, dt):
        return hamiltonian_reflection_step(
            rng_key,
            state,
            loglikelihood_fn,
            logprior_fn,
            loglikelihood_0,
            dt=dt,
            min_reflections=min_reflections,
            max_reflections=max_reflections,
            sigma_vel=sigma_vel,
            lower=lower,
            upper=upper,
            max_steps=max_steps,
        )

    inner_kernel = update_strategy(constrained_ham_fn, num_inner_steps, num_delete)

    _delete_fn = partial(delete_fn, num_delete=num_delete)

    kernel = build_adaptive_kernel(
        _delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    return kernel

def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int = 1,
    dt_ini: float = _DEFAULT_DT,
    min_reflections: int = 2,
    max_reflections: int = 10,
    sigma_vel: float = 0.0,
    lower: ArrayLikeTree = None,
    upper: ArrayLikeTree = None,
    max_steps: int = 200,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
) -> SamplingAlgorithm:
    """Creates a Gradient-Guided Hamiltonian Nested Sampling algorithm.

    The algorithm uses gradient-guided Hamiltonian trajectories with likelihood
    surface reflections to propose new live points, as described in GGNS
    (Pablo Lemos et al., 2023).

    Setting ``num_delete=1`` gives **static** Hamiltonian NS (one particle
    replaced per step); setting ``num_delete > 1`` gives **dynamic** Hamiltonian
    NS (batch replacements per step, better GPU utilisation).

    Parameters
    ----------
    logprior_fn
        Log-prior function for a single flat position array of shape
        ``(nparams,)``.
    loglikelihood_fn
        Log-likelihood function for a single flat position array.  Gradients
        are computed via :func:`jax.value_and_grad`, so this must be
        JAX-differentiable.
    num_inner_steps
        Number of Hamiltonian trajectories run sequentially per replacement
        particle.  Each trajectory takes up to ``max_steps`` leapfrog steps.
    num_delete
        Number of live points removed and replaced per NS step.
        ``num_delete=1`` => static NS; ``num_delete > 1`` => dynamic NS.
        Defaults to 1.
    dt_ini
        Initial leapfrog step size.  Adapted automatically by
        :func:`update_inner_kernel_params`.  Defaults to ``1e-2``.
    min_reflections
        Minimum number of reflections before a trajectory point is eligible
        for selection as the new live point.  Defaults to 2.
    max_reflections
        Number of reflections after which the trajectory terminates.
        Defaults to 10.
    sigma_vel
        Standard deviation of the Gaussian noise added to the initial velocity
        vector for ergodicity.  Set to 0 (default) for pure deterministic
        trajectories.
    lower
        Array of lower prior-box bounds, shape ``(nparams,)``.  Required.
    upper
        Array of upper prior-box bounds, shape ``(nparams,)``.  Required.
    max_steps
        Hard upper limit on the number of leapfrog steps per trajectory
        (static Python int, determines the trajectory buffer size).
        Defaults to 200.
    update_inner_kernel_params_fn
        Function that adapts ``dt`` from trajectory diagnostics.  Defaults to
        :func:`update_inner_kernel_params`.
    delete_fn
        Function to select which live points to remove.  Defaults to
        :func:`~blackjax.ns.base.delete_fn`.
    update_strategy
        Strategy for updating live points via MCMC.  Defaults to
        :func:`~blackjax.ns.from_mcmc.update_with_mcmc_take_last`.

    Returns
    -------
    SamplingAlgorithm
        A :class:`~blackjax.base.SamplingAlgorithm` with ``init`` and ``step``
        functions.  The step function signature is
        ``step(rng_key, state) -> (new_state, info)``.

    Raises
    ------
    ValueError
        If ``lower`` or ``upper`` is ``None``.
    """
    if lower is None or upper is None:
        raise ValueError(
            "Hamiltonian NS requires explicit prior bounds.  "
            "Pass `lower` and `upper` as arrays of shape (nparams,)."
        )

    lower_arr = jnp.asarray(lower)
    upper_arr = jnp.asarray(upper)

    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )

    # Bind dt_ini into the params-update function so the first call works
    def _update_inner_kernel_params(rng_key, state, info, params):
        if not params:
            params = {"dt": jnp.array(dt_ini)}
        return update_inner_kernel_params_fn(rng_key, state, info, params)

    kernel = build_kernel(
        init_state_fn=init_state_fn,
        loglikelihood_fn=loglikelihood_fn,
        logprior_fn=logprior_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        min_reflections=min_reflections,
        max_reflections=max_reflections,
        sigma_vel=sigma_vel,
        lower=lower_arr,
        upper=upper_arr,
        max_steps=max_steps,
        update_inner_kernel_params_fn=_update_inner_kernel_params,
        delete_fn=delete_fn,
        update_strategy=update_strategy,
    )

    def init_fn(position, rng_key=None):
        return init(
            position,
            init_state_fn=jax.vmap(init_state_fn),
            update_inner_kernel_params_fn=_update_inner_kernel_params,
            rng_key=rng_key,
        )

    def step_fn(rng_key, state):
        return kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)