from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.mcmc import slice_fsm
from blackjax.mcmc.slice import direction_proposal, init


@pytest.mark.parametrize("doubling", [False, True])
def test_scheduled_chain_matches_chain(doubling):
    initialize = slice_fsm.init_doubling if doubling else slice_fsm.init_stepping_out
    interval = (
        slice_fsm.build_doubling_kernel
        if doubling
        else slice_fsm.build_stepping_out_kernel
    )

    def logdensity(position):
        return -jnp.sum(position**2) / 2

    key = jax.random.key(34)
    keys = jax.random.split(key, 4)
    state = init(jnp.array([0.3, -0.7]), logdensity)
    generate = direction_proposal()
    chain = slice_fsm.build_chain(init_fn=initialize, interval=interval)
    scheduled = slice_fsm.build_scheduled_chain(
        lambda _, k, p, w: initialize(k, p, w, 10), interval
    )

    def proposal(index, position, logdensity_fn):
        proposal_key, _ = jax.random.split(keys[index])
        return generate(proposal_key, position, logdensity_fn)

    expected = jax.jit(lambda k, p: chain(k, p, logdensity, generate, 4))(key, state)
    actual = jax.jit(lambda k, p: scheduled(k, p, logdensity, proposal))(keys, state)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_scheduled_chain():
    def logdensity(position):
        return -jnp.sum(position**2) / 2

    caps = jnp.array([1, 4, 8, 2])
    directions = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]])

    def initialize(index, key, state, width):
        return slice_fsm.init_stepping_out(key, state, width, caps[index])

    def proposal(index, position, logdensity_fn):
        return lambda t: (init(position + directions[index] * t, logdensity_fn), True)

    chain = slice_fsm.build_scheduled_chain(
        initialize, slice_fsm.build_stepping_out_kernel, max_shrinkage=100
    )
    keys = jax.random.split(jax.random.key(15), (16, 4))
    states = jax.vmap(partial(init, logdensity_fn=logdensity))(
        jax.random.normal(jax.random.key(16), (16, 2))
    )
    run = jax.jit(jax.vmap(lambda k, p: chain(k, p, logdensity, proposal)))
    actual = run(keys, states)
    expected, infos = states, []
    for index in range(4):
        one = slice_fsm.build_scheduled_chain(
            lambda _, k, p, w: initialize(index, k, p, w),
            slice_fsm.build_stepping_out_kernel,
            max_shrinkage=100,
        )
        step = jax.jit(
            jax.vmap(
                lambda k, p: one(
                    k, p, logdensity, lambda _, x, fn: proposal(index, x, fn)
                )
            )
        )
        expected, info = step(keys[:, index : index + 1], expected)
        infos.append(info)
    info = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *infos)
    for x, y in zip(
        jax.tree.leaves(actual), jax.tree.leaves((expected, info)), strict=True
    ):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
    assert jnp.all(actual[1].is_accepted)
    assert jnp.all(actual[1].bracket_right - actual[1].bracket_left <= caps)


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


@pytest.mark.parametrize("doubling", [False, True])
def test_batched_chain_keeps_constants_unbatched(doubling):
    # A batched lax.cond around the likelihood would broadcast this constant
    # across lanes, repeating its work in every lane on every tick.
    table = jnp.linspace(0.0, 1.0, 1013)

    def logdensity(x):
        return -jnp.sum(x**2) / 2 + 0.0 * jnp.sum(jnp.cos(table))

    chain = slice_fsm.build_chain(
        max_shrinkage=100,
        init_fn=slice_fsm.init_doubling if doubling else slice_fsm.init_stepping_out,
        interval=(
            slice_fsm.build_doubling_kernel
            if doubling
            else slice_fsm.build_stepping_out_kernel
        ),
    )
    states = jax.vmap(partial(init, logdensity_fn=logdensity))(jnp.ones((16, 2)))
    keys = jax.random.split(jax.random.key(3), 16)
    jaxpr = jax.make_jaxpr(
        jax.vmap(lambda k, s: chain(k, s, logdensity, direction_proposal(), 4))
    )(keys, states)
    assert "[16,1013]" not in str(jaxpr)
