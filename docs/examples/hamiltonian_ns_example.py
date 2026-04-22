"""Hamiltonian Nested Sampling — Worked Example.

This script demonstrates the four Nested Sampling variants available in
BlackJAX on a simple 2-D Gaussian problem for which the log-evidence is
known analytically:

1. ``blackjax.nss``             — static Nested Slice Sampling (HRSS inner kernel)
2. ``blackjax.dynamic_nss``     — dynamic Nested Slice Sampling (``num_delete > 1``)
3. ``blackjax.ns_hamiltonian``  — gradient-guided Hamiltonian NS, static
4. ``blackjax.ns_hamiltonian``  — gradient-guided Hamiltonian NS, dynamic

Usage::

    python docs/examples/hamiltonian_ns_example.py

Requirements: blackjax, jax, scipy (optional, for comparison).
"""

import os
import sys
import time

# Ensure the project root is on the path when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    import numpy as np
    from scipy.special import erf

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import blackjax
import blackjax.ns.utils as ns_utils

# -----------------------------------------------------------------------
# Problem definition: 2-D Gaussian likelihood, uniform prior on [-5, 5]^2
# -----------------------------------------------------------------------
NDIM = 2
LOWER = jnp.full(NDIM, -5.0)
UPPER = jnp.full(NDIM, 5.0)
MU = jnp.zeros(NDIM)
SIGMA = 1.0

if _SCIPY_AVAILABLE:
    prior_prob = (1.0 / 10.0) ** NDIM
    factor = (2.0 * np.pi * SIGMA**2) ** (NDIM / 2.0)
    p_in = erf(5.0 / (SIGMA * np.sqrt(2))) ** NDIM
    ANALYTIC_LOG_Z: float = float(np.log(prior_prob * factor * p_in))
else:
    ANALYTIC_LOG_Z = None


def logprior_fn(x):
    """Uniform log-prior on [-5, 5]^2."""
    return jnp.where(
        jnp.all((x >= LOWER) & (x <= UPPER)),
        -jnp.log(jnp.prod(UPPER - LOWER)),
        -jnp.inf,
    )


def loglikelihood_fn(x):
    """Unnormalised 2-D Gaussian log-likelihood."""
    return -0.5 * jnp.sum(((x - MU) / SIGMA) ** 2)


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------
NUM_LIVE = 200
NUM_INNER = 5


def run_ns(sampler, label, max_steps=2000, tol=3.0):
    """Run a full NS loop and return (logZ_estimate, elapsed_seconds, dead)."""
    rng_key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(
        rng_key, shape=(NUM_LIVE, NDIM), minval=-5.0, maxval=5.0
    )

    state = jax.jit(sampler.init)(positions)
    step_fn = jax.jit(sampler.step)

    rng_key, subkey = jax.random.split(rng_key)
    state, _ = step_fn(subkey, state)

    dead_list = []
    t0 = time.perf_counter()

    for i in range(max_steps):
        rng_key, subkey = jax.random.split(rng_key)
        state, info = step_fn(subkey, state)
        dead_list.append(info)

        if state.integrator.logZ_live - state.integrator.logZ < -tol:
            break

    elapsed = time.perf_counter() - t0

    dead = ns_utils.finalise(state, dead_list, update_info=False)
    logZ_est = float(state.integrator.logZ)
    print(
        f"  {label:<40s}  logZ = {logZ_est:+.3f}  "
        f"(steps={i + 1}, t={elapsed:.1f}s)"
    )
    return logZ_est, elapsed, dead


# -----------------------------------------------------------------------
# Run all four variants
# -----------------------------------------------------------------------
print("=" * 72)
print("BlackJAX Nested Sampling — 2-D Gaussian benchmark")
if ANALYTIC_LOG_Z is not None:
    print(f"  Analytic log Z = {ANALYTIC_LOG_Z:+.4f}")
print("=" * 72)

results = {}

# 1. Static NSS
sampler_nss = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_inner_steps=NUM_INNER,
    num_delete=1,
)
results["nss (static)"] = run_ns(sampler_nss, "nss (static, num_delete=1)")

# 2. Dynamic NSS
NUM_DELETE_DYN = NUM_LIVE // 10
sampler_dnss = blackjax.dynamic_nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=NUM_DELETE_DYN,
    num_inner_steps=NUM_INNER,
)
results["dynamic_nss"] = run_ns(
    sampler_dnss, f"dynamic_nss (num_delete={NUM_DELETE_DYN})"
)

# 3. Hamiltonian NS — static
sampler_ham_static = blackjax.ns_hamiltonian(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_inner_steps=NUM_INNER,
    num_delete=1,
    dt_ini=0.3,
    min_reflections=2,
    max_reflections=10,
    sigma_vel=0.0,
    lower=LOWER,
    upper=UPPER,
    max_steps=150,
)
results["ns_hamiltonian (static)"] = run_ns(
    sampler_ham_static, "ns_hamiltonian (static, num_delete=1)"
)

# 4. Hamiltonian NS — dynamic
NUM_DELETE_HAM = NUM_LIVE // 10
sampler_ham_dyn = blackjax.ns_hamiltonian(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_inner_steps=NUM_INNER,
    num_delete=NUM_DELETE_HAM,
    dt_ini=0.3,
    min_reflections=2,
    max_reflections=10,
    sigma_vel=0.0,
    lower=LOWER,
    upper=UPPER,
    max_steps=150,
)
results["ns_hamiltonian (dynamic)"] = run_ns(
    sampler_ham_dyn, f"ns_hamiltonian (dynamic, num_delete={NUM_DELETE_HAM})"
)

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"{'Sampler':<40s}  {'logZ':>10s}  {'bias':>10s}  {'time':>8s}")
print("-" * 72)
for name, (logZ, elapsed, _) in results.items():
    bias = (f"{logZ - ANALYTIC_LOG_Z:+.3f}" if ANALYTIC_LOG_Z is not None else "N/A")
    print(f"  {name:<38s}  {logZ:+10.3f}  {bias:>10s}  {elapsed:>7.1f}s")
if ANALYTIC_LOG_Z is not None:
    print(f"  {'Analytic':<38s}  {ANALYTIC_LOG_Z:+10.4f}")
print("=" * 72)