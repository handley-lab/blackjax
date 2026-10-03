from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.custom_batching import custom_vmap

from blackjax.mcmc import slice_fsm as fsm
from blackjax.ns import adaptive, nss
from blackjax.ns.base import init_state_strategy


@pytest.mark.parametrize("doubling", [False, True])
def test_async_replacement_chains(doubling):
    calls = []

    def logprior(x):
        return -jnp.sum(x**2) / 18

    def loglikelihood(x):
        return -jnp.sum((x - 0.5) ** 2)

    init_particle = partial(
        init_state_strategy, logprior_fn=logprior, loglikelihood_fn=loglikelihood
    )

    @custom_vmap
    def evaluate(origin, x, threshold):
        return init_particle(x, loglikelihood_birth=threshold)

    @evaluate.def_vmap
    def evaluate_batch(size, batched, origin, x, threshold):
        jax.debug.callback(lambda x: calls.append(np.asarray(x)), origin)
        result = jax.vmap(lambda x: init_particle(x, loglikelihood_birth=threshold))(x)
        return result, jax.tree.map(lambda _: True, result)

    def proposal(init_fn, threshold, covariance_factor):
        def generate(key, origin, logdensity):
            direction = nss.sample_direction_from_covariance_factor(
                key, origin, covariance_factor
            )

            def slice_fn(t):
                candidate = evaluate(origin, origin + t * direction, threshold)
                return candidate, candidate.loglikelihood > threshold

            return slice_fn

        return generate

    state_type = fsm.DoublingState if doubling else fsm.SteppingOutState
    init_move = fsm.init_doubling if doubling else fsm.init_stepping_out
    build_kernel = (
        fsm.build_doubling_kernel if doubling else fsm.build_stepping_out_kernel
    )

    def init_slice(key, particle):
        return init_move(key, state_type(particle, particle.logdensity), 1.0, 10)

    kernel = nss.build_async_kernel(
        init_particle,
        8,
        init_slice,
        partial(build_kernel, width=1.0),
        16,
        proposal=proposal,
    )
    state = adaptive.init(
        3 * jax.random.normal(jax.random.key(20), (128, 2)),
        jax.vmap(init_particle),
        update_inner_kernel_params_fn=nss.live_covariance_factor,
    )
    result, info = jax.jit(kernel)(jax.random.key(30), state)
    jax.block_until_ready(result)
    jax.effects_barrier()
    diagnostics = info.update_info
    assert diagnostics.num_evaluations.shape == (16,)
    assert np.all(diagnostics.is_accepted)
    assert len(calls) == int(jnp.max(diagnostics.num_evaluations))
    origins = np.asarray(calls)
    changes = np.any(origins[1:] != origins[:-1], axis=-1)
    completed = np.cumsum(changes, axis=0)
    assert np.any((completed.min(axis=1) == 0) & (completed.max(axis=1) >= 1))
    np.testing.assert_allclose(
        result.particles.loglikelihood,
        jax.vmap(loglikelihood)(result.particles.position),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.particles.logdensity,
        jax.vmap(logprior)(result.particles.position),
        rtol=1e-6,
    )
    born = ~jnp.isnan(result.particles.loglikelihood_birth)
    assert int(born.sum()) == 16
    threshold = jnp.sort(state.particles.loglikelihood)[15]
    np.testing.assert_array_equal(result.particles.loglikelihood_birth[born], threshold)
    assert jnp.all(result.particles.loglikelihood[born] > threshold)
