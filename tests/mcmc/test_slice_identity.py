from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.mcmc import slice_fsm
from blackjax.mcmc.slice import SliceInfo, build_kernel as build_slice_kernel
from blackjax.mcmc.slice import stepping_out
from blackjax.util import thin_kernel


class Particle(NamedTuple):
    position: object
    logdensity: object
    direction: object
    coordinate: object


@pytest.mark.parametrize("budget", [0, 1, 4, 10])
@pytest.mark.parametrize("random_direction", [False, True])
def test_matched_stepping_out_streams(budget, random_direction):
    def logdensity(x):
        return -jnp.sum(x**2) / 2

    def proposal(key, position, logdensity_fn):
        direction = (
            jax.random.normal(key, position.shape)
            if random_direction
            else jnp.full_like(position, 0.7)
        )

        def evaluate(t):
            x = position + t * direction
            return Particle(x, logdensity_fn(x), direction, t), jnp.asarray(True)

        return evaluate

    sync = partial(
        thin_kernel(build_slice_kernel(stepping_out, budget, 100), 1),
        logdensity_fn=logdensity,
        proposal_generator=proposal,
        width=0.7,
    )
    asynchronous = partial(
        slice_fsm.build_chain(
            budget, 0.7, 100, interval=slice_fsm.build_stepping_out_kernel
        ),
        logdensity_fn=logdensity,
        proposal_generator=proposal,
        num_inner_steps=1,
    )
    keys = jax.random.split(jax.random.key(1), 16)
    states = jax.vmap(lambda x: Particle(x, logdensity(x), jnp.zeros_like(x), 0.0))(
        jnp.linspace(-2, 2, 32).reshape(16, 2)
    )
    expected = jax.jit(jax.vmap(sync))(keys, states)
    actual = jax.jit(jax.vmap(asynchronous))(keys, states)
    assert isinstance(actual[1], SliceInfo)
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(
        jax.tree.leaves((actual[0].direction, actual[0].coordinate, actual[1])),
        jax.tree.leaves((expected[0].direction, expected[0].coordinate, expected[1])),
    ):
        np.testing.assert_array_equal(x, y)
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    for x, y in zip(jax.tree.leaves(actual[0]), jax.tree.leaves(expected[0])):
        np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-14)
    x, y = np.asarray(actual[0].position), np.asarray(expected[0].position)
    print(
        dict(
            budget=budget,
            random_direction=random_direction,
            identical_components=int(np.sum(x == y)),
            components=x.size,
            maximum_difference=float(np.max(np.abs(x - y))),
        )
    )
