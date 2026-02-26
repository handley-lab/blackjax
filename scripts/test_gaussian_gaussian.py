import time

import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.integrator import init_integrator, update_integrator
from blackjax.ns.utils import finalise
import tqdm

rng_key = jax.random.PRNGKey(0)

ndim = 5

# Prior: N(mu_prior, cov_prior) with non-trivial correlations
mu_prior = jnp.array([0.0, 0.5, -0.3, 0.2, -0.1])

L_prior = jnp.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.9, 0.0, 0.0, 0.0],
    [-0.2, 0.4, 0.8, 0.0, 0.0],
    [0.1, -0.3, 0.2, 0.7, 0.0],
    [0.5, 0.1, -0.1, 0.3, 0.6],
])
cov_prior = L_prior @ L_prior.T

# Likelihood: N(mu_like, cov_like) with different non-trivial correlations
mu_like = jnp.array([1.0, -0.5, 0.8, 0.0, 0.3])

L_like = jnp.array([
    [0.5, 0.0, 0.0, 0.0, 0.0],
    [-0.2, 0.4, 0.0, 0.0, 0.0],
    [0.1, -0.1, 0.3, 0.0, 0.0],
    [0.3, 0.2, -0.1, 0.4, 0.0],
    [-0.1, 0.1, 0.2, -0.2, 0.3],
])
cov_like = L_like @ L_like.T

# Analytic evidence: Z = N(mu_prior | mu_like, cov_prior + cov_like)
logZ_analytic = jax.scipy.stats.multivariate_normal.logpdf(
    mu_prior, mu_like, cov_prior + cov_like
)

# Analytic posterior: N(mu_post, cov_post)
prec_prior = jnp.linalg.inv(cov_prior)
prec_like = jnp.linalg.inv(cov_like)
cov_post = jnp.linalg.inv(prec_prior + prec_like)
mu_post = cov_post @ (prec_prior @ mu_prior + prec_like @ mu_like)

print(f"Prior mean: {mu_prior}")
print(f"Likelihood mean: {mu_like}")
print(f"Analytic logZ = {logZ_analytic:.6f}")
print(f"Analytic posterior mean: {mu_post}")
print()

logprior_fn = lambda x: jax.scipy.stats.multivariate_normal.logpdf(x, mu_prior, cov_prior)
loglikelihood_fn = lambda x: jax.scipy.stats.multivariate_normal.logpdf(x, mu_like, cov_like)

# Draw initial positions from the prior
rng_key, init_key = jax.random.split(rng_key)
n_live = 1000
num_delete = 50
positions = jax.random.multivariate_normal(init_key, mu_prior, cov_prior, shape=(n_live,))


def run_sampler(name, algo, positions, rng_key, update_info=True):
    live = algo.init(positions)
    step = jax.jit(algo.step)
    integrator = init_integrator(live.particles)
    dead_points = []
    infos = []

    # Warmup JIT
    rng_key, subkey = jax.random.split(rng_key)
    live, dead = step(subkey, live)
    integrator = update_integrator(integrator, live.particles, dead.particles)
    dead_points.append(dead)
    infos.append(dead.update_info)

    t0 = time.perf_counter()
    with tqdm.tqdm(desc=f"{name} dead points", unit=" pts") as pbar:
        pbar.update(len(dead.particles))
        while not integrator.logZ_live - integrator.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            live, dead = step(subkey, live)
            integrator = update_integrator(integrator, live.particles, dead.particles)
            dead_points.append(dead)
            infos.append(dead.update_info)
            pbar.update(len(dead.particles))
    wall_time = time.perf_counter() - t0

    ns_run = finalise(live, dead_points, update_info=update_info)
    return ns_run, infos, wall_time


# --- NSS ---
print("=" * 60)
print("NSS (Nested Slice Sampling)")
print("=" * 60)

num_inner_steps_nss = 20
rng_key, nss_key = jax.random.split(rng_key)
nss_algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps_nss,
)
nss_run, nss_infos, nss_time = run_sampler("NSS", nss_algo, positions, nss_key)

import anesthetic
nss_samples = anesthetic.NestedSamples(
    data=nss_run.particles.position,
    logL=nss_run.particles.loglikelihood,
    logL_birth=nss_run.particles.loglikelihood_birth,
)

nss_num_steps = len(nss_infos)
nss_num_dead = len(nss_run.particles.position)
nss_like_evals = nss_num_steps * num_delete * num_inner_steps_nss

print(f"NSS logZ = {nss_samples.logZ():.4f} +/- {nss_samples.logZ(12).std():.4f}")
print(f"NSS wall time = {nss_time:.2f}s")
print(f"NSS steps = {nss_num_steps}, dead points = {nss_num_dead}")
print(f"NSS likelihood evaluations = {nss_like_evals}")
print()

# --- NRS ---
print("=" * 60)
print("NRS (Nested Rejection Sampling)")
print("=" * 60)

num_proposals_nrs = 1000
rng_key, nrs_key = jax.random.split(rng_key)
nrs_algo = blackjax.nrs(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    prototype_position=positions[0],
    num_delete=num_delete,
)
nrs_run, nrs_infos, nrs_time = run_sampler("NRS", nrs_algo, positions, nrs_key, update_info=False)

nrs_samples = anesthetic.NestedSamples(
    data=nrs_run.particles.position,
    logL=nrs_run.particles.loglikelihood,
    logL_birth=nrs_run.particles.loglikelihood_birth,
)

nrs_num_steps = len(nrs_infos)
nrs_num_dead = len(nrs_run.particles.position)
nrs_like_evals = sum(int(info.num_proposals_total) for info in nrs_infos)
nrs_accepted = sum(int(info.num_accepted_total) for info in nrs_infos)
nrs_rounds = sum(int(info.num_rounds) for info in nrs_infos)

print(f"NRS logZ = {nrs_samples.logZ():.4f} +/- {nrs_samples.logZ(12).std():.4f}")
print(f"NRS wall time = {nrs_time:.2f}s")
print(f"NRS steps = {nrs_num_steps}, dead points = {nrs_num_dead}")
print(f"NRS likelihood evaluations = {nrs_like_evals}")
print(f"NRS accepted (L > L0) = {nrs_accepted}")
print(f"NRS total rounds = {nrs_rounds}")
print()

# --- Summary ---
print("=" * 60)
print("Comparison")
print("=" * 60)
print(f"{'':20s} {'NSS':>12s} {'NRS':>12s} {'Analytic':>12s}")
print(f"{'logZ':20s} {nss_samples.logZ():12.4f} {nrs_samples.logZ():12.4f} {logZ_analytic:12.4f}")
print(f"{'logZ error':20s} {nss_samples.logZ(12).std():12.4f} {nrs_samples.logZ(12).std():12.4f}")
print(f"{'Wall time (s)':20s} {nss_time:12.2f} {nrs_time:12.2f}")
print(f"{'Dead points':20s} {nss_num_dead:12d} {nrs_num_dead:12d}")
print(f"{'Like evals':20s} {nss_like_evals:12d} {nrs_like_evals:12d}")
print(f"{'Like evals/dead pt':20s} {nss_like_evals/nss_num_dead:12.1f} {nrs_like_evals/nrs_num_dead:12.1f}")
print()

# --- Save samples ---
nss_samples.to_csv("test_gaussian_gaussian_nss.csv")
nrs_samples.to_csv("test_gaussian_gaussian_nrs.csv")
print("Saved test_gaussian_gaussian_nss.csv")
print("Saved test_gaussian_gaussian_nrs.csv")

# --- Comparison plot ---
import matplotlib.pyplot as plt

axes = nss_samples.plot_2d(range(ndim), label="NSS")
nrs_samples.plot_2d(axes, label="NRS")
axes.iloc[-1, 0].legend(bbox_to_anchor=(ndim, ndim), loc='lower right')
plt.savefig("test_gaussian_gaussian.png", dpi=150, bbox_inches="tight")
print("Saved test_gaussian_gaussian.png")
