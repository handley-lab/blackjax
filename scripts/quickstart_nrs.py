import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.integrator import init_integrator, update_integrator
from blackjax.ns.utils import finalise
import tqdm

rng_key = jax.random.PRNGKey(0)

loglikelihood_fn = lambda x: jax.scipy.stats.multivariate_normal.logpdf(
    x, jnp.ones(5), jnp.eye(5) * 0.01
)
logprior_fn = lambda x: jax.scipy.stats.multivariate_normal.logpdf(
    x, jnp.zeros(5), jnp.eye(5)
)

rng_key, initialization_key = jax.random.split(rng_key)
positions = jax.random.normal(initialization_key, (1000, 5))

algo = blackjax.nrs(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    prototype_position=positions[0],
    num_delete=50,
)

live = algo.init(positions)
step = jax.jit(algo.step)
integrator = init_integrator(live.particles)

dead_points = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not integrator.logZ_live - integrator.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key)
        live, dead = step(subkey, live)
        integrator = update_integrator(integrator, live.particles, dead.particles)
        dead_points.append(dead)
        pbar.update(len(dead.particles))

ns_run = finalise(live, dead_points, update_info=False)

import anesthetic

nested_samples = anesthetic.NestedSamples(
    data=ns_run.particles.position,
    logL=ns_run.particles.loglikelihood,
    logL_birth=ns_run.particles.loglikelihood_birth,
)

print(f"logZ = {nested_samples.logZ():.3f}")
nested_samples.to_csv("quickstart_nrs.csv")
print("Saved quickstart_nrs.csv")

prior = nested_samples.set_beta(0.0).plot_2d(range(5), label="prior")
post = nested_samples.plot_2d(prior, label="posterior")
prior.iloc[-1, 0].legend(bbox_to_anchor=(len(prior), len(prior)), loc='lower right')

import matplotlib.pyplot as plt
plt.savefig("quickstart_nrs.png", dpi=150, bbox_inches="tight")
print("Saved quickstart_nrs.png")
