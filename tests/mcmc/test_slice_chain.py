from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.mcmc import slice_fsm
from blackjax.mcmc.slice import direction_proposal, init


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("doubling", [False, True])
def test_wrapped_components(chain, doubling):
    initialize = slice_fsm.init_doubling if doubling else slice_fsm.init_stepping_out
    advance = (
        slice_fsm.build_doubling_kernel
        if doubling
        else slice_fsm.build_stepping_out_kernel
    )

    def wrapped_init(*args):
        return initialize(*args)

    def wrapped_interval(*args):
        return advance(*args)

    def logdensity(x):
        return -jnp.sum(x**2) / 2

    build = slice_fsm.build_chain if chain else slice_fsm.build_kernel
    direct = build(init_fn=initialize, interval=advance)
    wrapped = build(init_fn=wrapped_init, interval=wrapped_interval)
    arguments = (logdensity, direction_proposal())
    if chain:
        arguments += (4,)
    key = jax.random.key(34)
    state = init(jnp.array([0.3, -0.7]), logdensity)
    expected = jax.jit(lambda key, state: direct(key, state, *arguments))(key, state)
    actual = jax.jit(lambda key, state: wrapped(key, state, *arguments))(key, state)
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.mark.parametrize(
    "builder",
    [slice_fsm.build_doubling_kernel, slice_fsm.build_stepping_out_kernel],
)
def test_chain_without_nested_sampling(builder):
    def logdensity(position):
        return -sum(jnp.square(x).sum() for x in jax.tree.leaves(position)) / 2

    positions = {"x": jax.random.normal(jax.random.key(1), (4096, 2))}
    states = jax.vmap(partial(init, logdensity_fn=logdensity))(positions)
    kernel = partial(
        slice_fsm.build_chain(
            init_fn=(
                slice_fsm.init_doubling
                if builder is slice_fsm.build_doubling_kernel
                else slice_fsm.init_stepping_out
            ),
            interval=builder,
        ),
        logdensity_fn=logdensity,
        proposal_generator=direction_proposal(),
        num_inner_steps=8,
    )
    keys = jax.random.split(jax.random.key(2), 4096)
    result, info = jax.jit(jax.vmap(kernel))(keys, states)
    assert jnp.all(info.is_accepted)
    assert info.num_shrink.shape == (4096, 8)
    assert jnp.all(info.num_shrink >= 1)
    np.testing.assert_allclose(result.position["x"].mean(axis=0), 0, atol=0.06)
    np.testing.assert_allclose(result.position["x"].var(axis=0), 1, atol=0.1)
    scalar = jax.jit(kernel)
    for i in (0, 7, 15):
        expected = scalar(keys[i], jax.tree.map(lambda x: x[i], states))
        actual = jax.tree.map(lambda x: x[i], (result, info))
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
