import blackjax
import jax
import jax.numpy as jnp
import tqdm


jax.config.update('jax_enable_x64', True) 

rng_key = jax.random.PRNGKey(0)
n_dims = 3
n_live = 1000
n_delete = 500
num_mcmc_steps = n_dims * 5

# | Define data and likelihood
x = jnp.linspace(-1, 1, 10)
m = 2.0
c = 1.0
sigma = 0.1
key, rng_key = jax.random.split(rng_key)
y =  m * x + c + sigma * jax.random.normal(key, (10,), dtype=jnp.float64)

@jax.jit
def loglikelihood_fn(p):
    return jax.scipy.stats.multivariate_normal.logpdf(y, p["m"] * x + p["c"], p["sigma"]**2)

# | Define the prior function

m_min, m_max = -10.0, 10.0
c_min, c_max = -10.0, 10.0
sigma_min, sigma_max = 0.0, 10.0

uniform_logprob = lambda x, a, b: jax.scipy.stats.uniform.logpdf(x, a, b-a)

def logprior_fn(p):
    logprior = 0.0
    logprior += uniform_logprob(p["m"], m_min, m_max)
    logprior += uniform_logprob(p["c"], c_min, c_max)
    logprior += uniform_logprob(p["sigma"], sigma_min, sigma_max)
    return logprior

# | Sample live points from the prior
rng_key, init_key = jax.random.split(rng_key, 2)
init_keys = jax.random.split(init_key, n_dims)
particles = {
        "m": jax.random.uniform(init_keys[0], (n_live,), minval=m_min, maxval=m_max),
        "c": jax.random.uniform(init_keys[1], (n_live,), minval=c_min, maxval=c_max),
        "sigma": jax.random.uniform(init_keys[2], (n_live,), minval=sigma_min, maxval=sigma_max),
        }

_, ravel_fn = jax.flatten_util.ravel_pytree({k: v[0] for k, v in particles.items()})

# | Initialize the Nested Sampling algorithm
nested_sampler = blackjax.ns.adaptive.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    n_delete=n_delete,
    num_mcmc_steps=num_mcmc_steps,
    ravel_fn=ravel_fn,
)

state = nested_sampler.init(particles, loglikelihood_fn)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.sampler_state.logZ_live - state.sampler_state.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

# | anesthetic post-processing
from anesthetic import NestedSamples
import numpy as np
dead = jax.tree.map(
        lambda *args: jnp.reshape(jnp.stack(args, axis=0), 
                                  (-1,) + args[0].shape[1:]),
        *dead)
live = state.sampler_state

logL = np.concatenate((dead.logL, live.logL), dtype=float)
logL_birth = np.concatenate((dead.logL_birth, live.logL_birth), dtype=float)
data = np.concatenate([
    np.column_stack([v for v in dead.particles.values()]),
    np.column_stack([v for v in live.particles.values()])
    ], axis=0)

columns = list(dead.particles.keys())
samples = NestedSamples(data, logL=logL, logL_birth=logL_birth, columns=columns)
samples.gui()
samples.to_csv('line.csv') 
