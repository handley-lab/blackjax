from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.custom_batching import custom_vmap

from blackjax.mcmc import slice_fsm
from blackjax.ns import adaptive, from_mcmc, nss
from blackjax.ns.base import delete_fn, init_state_strategy


def build_chain_ns(
    init_particle,
    num_steps,
    num_delete,
    slice_kernel,
    proposal=nss.covariance_proposal,
    inner_kernel_params=nss.live_covariance_factor,
):
    def chain(key, particle, num_inner_steps, loglikelihood_0, **parameters):
        generate = proposal(init_particle, loglikelihood_0, **parameters)
        return slice_kernel(key, particle, None, generate, num_inner_steps)

    return adaptive.build_kernel(
        partial(delete_fn, num_delete=num_delete),
        from_mcmc.update_with_mcmc_chains(chain, num_steps, num_delete),
        update_inner_kernel_params_fn=inner_kernel_params,
    )


def test_async_defaults_match_explicit_stepping_out():
    init_particle = partial(
        init_state_strategy,
        logprior_fn=lambda x: -jnp.sum(x**2) / 18,
        loglikelihood_fn=lambda x: -jnp.sum((x - 0.5) ** 2),
    )
    state = adaptive.init(
        jax.random.normal(jax.random.key(1), (32, 2)),
        jax.vmap(init_particle),
        update_inner_kernel_params_fn=nss.live_covariance_factor,
    )
    default = nss.build_async_kernel(init_particle, 3, 4, 5, 7)
    explicit = build_chain_ns(
        init_particle,
        3,
        4,
        slice_kernel=slice_fsm.build_chain(
            interval=slice_fsm.build_stepping_out_kernel,
            max_expansions=5,
            max_shrinkage=7,
        ),
    )
    key = jax.random.key(2)
    for actual, expected in zip(
        jax.tree.leaves(jax.jit(default)(key, state)),
        jax.tree.leaves(jax.jit(explicit)(key, state)),
    ):
        np.testing.assert_array_equal(actual, expected)


def test_async_shrinkage_exhaustion_retains_particle():
    from blackjax.mcmc.slice import SliceState

    particle = SliceState(jnp.array([0.0]), jnp.array(0.0))

    def proposal(key, position, logdensity_fn):
        return lambda t: (SliceState(position + t, jnp.array(0.0)), False)

    kernel = slice_fsm.build_chain(
        interval=slice_fsm.build_stepping_out_kernel, max_expansions=1, max_shrinkage=2
    )
    result, info = jax.jit(kernel, static_argnums=(2, 3, 4))(
        jax.random.key(3), particle, None, proposal, 3
    )
    np.testing.assert_array_equal(result.position, particle.position)
    np.testing.assert_array_equal(info.num_shrink, [2, 2, 2])
    np.testing.assert_array_equal(info.num_evaluations, [2, 2, 2])
    assert not jnp.any(info.is_accepted)


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

    build_kernel = (
        slice_fsm.build_doubling_kernel
        if doubling
        else slice_fsm.build_stepping_out_kernel
    )

    kernel = build_chain_ns(
        init_particle,
        8,
        16,
        slice_kernel=slice_fsm.build_chain(interval=build_kernel),
        proposal=proposal,
        inner_kernel_params=nss.live_covariance_factor,
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
    assert diagnostics.num_evaluations.shape == (16, 8)
    assert np.all(diagnostics.is_accepted)
    assert len(calls) == int(jnp.max(diagnostics.num_evaluations.sum(axis=1)))
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


@pytest.mark.parametrize("doubling", [False, True])
@pytest.mark.parametrize("factor", [False, True])
def test_covariance_proposal(doubling, factor):
    def logprior(x):
        return -jnp.sum(x**2) / 18

    def loglikelihood(x):
        return -jnp.sum((x - 0.5) ** 2)

    init_particle = partial(
        init_state_strategy, logprior_fn=logprior, loglikelihood_fn=loglikelihood
    )
    parameters = nss.live_covariance_factor if factor else nss.live_covariance
    builder = (
        slice_fsm.build_doubling_kernel
        if doubling
        else slice_fsm.build_stepping_out_kernel
    )
    kernel = jax.jit(
        build_chain_ns(
            init_particle,
            8,
            16,
            slice_kernel=slice_fsm.build_chain(interval=builder),
            inner_kernel_params=parameters,
        )
    )
    state = adaptive.init(
        3 * jax.random.normal(jax.random.key(20), (128, 2)),
        jax.vmap(init_particle),
        update_inner_kernel_params_fn=parameters,
    )
    for key in jax.random.split(jax.random.key(30), 3):
        state, info = kernel(key, state)
        assert jnp.all(info.update_info.is_accepted)
        np.testing.assert_allclose(
            state.particles.loglikelihood,
            jax.vmap(loglikelihood)(state.particles.position),
            rtol=1e-6,
        )
