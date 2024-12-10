import distrax
import jax
import jax.numpy as jnp
import tqdm
from jax.scipy.linalg import inv, solve

from jax.scipy.stats import multivariate_normal
import blackjax
from blackjax.ns.utils import log_weights

jax.config.update("jax_enable_x64", True)


rng_key = jax.random.PRNGKey(1)
d = 5
#
cov1 = jnp.eye(d) * 0.25
cov2 = cov1
m1 = jnp.full(d, 2.0)
m2 = jnp.full(d, -2.0)
# make a dxd covariance matrix with 1.0 on the diagonal and 0.5 off-diagonal
# cov1 = jnp.eye(d) * 1.0 + 0.5 * (1.0 - jnp.eye(d))
# now an equal volume cov with anti correlation
# v = jnp.ones(d)
# v = v / jnp.linalg.norm(v)
# cov2 = cov1

prior = distrax.Uniform(low=-5 * jnp.ones(d), high=5 * jnp.ones(d))
prior = distrax.Independent(prior, reinterpreted_batch_ndims=1)


def loglikelihood(x):
    logpdf_one = multivariate_normal.logpdf(x, mean=m1, cov=cov1)
    logpdf_two = multivariate_normal.logpdf(x, mean=m2, cov=cov2)
    return jax.scipy.special.logsumexp(
        jnp.array([logpdf_one, logpdf_two]), axis=0
    )


# prior = distrax.Uniform(low=-5 * jnp.ones(t), high=5 * jnp.ones(t))
# prior = distrax.Independent(prior, reinterpreted_batch_ndims=1)

analytic = jax.scipy.special.logsumexp(
    jnp.array([prior.log_prob(jnp.ones(d)), prior.log_prob(jnp.ones(d))])
)

n_live = 100
n_delete = 1
num_mcmc_steps = d * 5
algo = blackjax.ns.adaptive.nss(
    logprior_fn=prior.log_prob,
    loglikelihood_fn=loglikelihood,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
)

# rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

# initial_particles = prior.sample(seed=init_key, sample_shape=(n_live,))
# state = algo.init(initial_particles, loglikelihood)
# sample_key, subkey = jax.random.split(sample_key)

# @jax.jit
# def one_step(carry, xs):
#     state, k = carry
#     k, subk = jax.random.split(k, 2)
#     state, dead_point = algo.step(subk, state)
#     return (state, k), dead_point


# # n_steps = 1000
# # (live, _), dead = jax.lax.scan((one_step), (state, rng_key), length=n_steps)

# dead = []

# for _ in tqdm.trange(30000):
#     if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:  # type: ignore[attr-defined]
#         break
#     (state, rng_key), dead_info = one_step((state, rng_key), None)
#     dead.append(dead_info)

length = 6000


@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = algo.step(subk, state)
    return (state, k), (state, dead_point)


def integrate(rng):
    rng, sample_key, run_key = jax.random.split(rng, 3)
    initial_state = prior._sample_n(sample_key, 100)
    state = algo.init(initial_state, loglikelihood)
    (live, _), (live_accum, dead) = jax.lax.scan(
        (one_step), (state, run_key), length=length
    )
    return live, dead, live_accum


# Now to scan over 500 different RNG seeds
n_replicas = 10
base_key = jax.random.PRNGKey(2)
keys = jax.random.split(base_key, n_replicas)
live, dead, live_accum = jax.vmap(integrate)(keys)


# dead = []

# for _ in tqdm.trange(5000):
#     if state.sampler_state.logZ_live - state.sampler_state.logZ < -3:  # type: ignore[attr-defined]
#         break
#     (state, rng_key), dead_info = one_step((state, rng_key), None)
#     dead.append(dead_info)

import anesthetic as ns
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

d1 = jnp.linalg.norm(live_accum.sampler_state.particles - m1, axis=-1)
d2 = jnp.linalg.norm(live_accum.sampler_state.particles - m2, axis=-1)
f, a = plt.subplots()
a.plot((d1 > d2).sum(axis=-1).T)
a.hlines(50, 0, length, color="k", ls="--", label="expected")
a.set_xlabel("Iteration")
a.legend()
a.set_ylabel("Number of particles in cluster 1")
plt.savefig("cluster_assignment.png")

model1 = multivariate_normal(mean=m1, cov=cov1)
model2 = multivariate_normal(mean=m2, cov=cov2)
true_samples = ns.MCMCSamples(
    jnp.concatenate([model1.rvs(size=1000), model2.rvs(size=1000)])
)

a = true_samples.plot_2d(np.arange(2))

means = []
stds = []
for i in range(n_replicas):
    samples = ns.NestedSamples(
        data=jnp.concatenate(dead.particles[i]),
        logL=jnp.concatenate(dead.logL[i]),
        logL_birth=jnp.concatenate(dead.logL_birth[i]),
    )
    zs = samples.logZ(100)
    means.append(zs.mean())
    stds.append(zs.std())
    a = samples.plot_2d(a, color="red", alpha=0.35)

samples.to_csv("samples.csv")

plt.savefig("ergodically_separate.png")
f, a = plt.subplots()
a.errorbar(np.arange(n_replicas), means, yerr=stds, fmt="o")
a.axhline(analytic, color="k", ls="--", label="Truth")
a.set_xlabel("Replica")
a.set_ylabel(r"$\log Z$")
a.legend()
plt.savefig("ergodically_separate_evidence.png")
