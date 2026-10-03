from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackjax.mcmc import slice_chain
from blackjax.mcmc.slice import direction_proposal, init


@pytest.mark.parametrize(
    "builder",
    [slice_chain.build_doubling_kernel, slice_chain.build_stepping_out_kernel],
)
def test_chain_without_nested_sampling(builder):
    def logdensity(position):
        return -sum(jnp.square(x).sum() for x in jax.tree.leaves(position)) / 2

    positions = {"x": jax.random.normal(jax.random.key(1), (4096, 2))}
    states = jax.vmap(partial(init, logdensity_fn=logdensity))(positions)
    kernel = partial(
        builder(),
        logdensity_fn=logdensity,
        proposal_generator=direction_proposal(),
        num_inner_steps=8,
    )
    keys = jax.random.split(jax.random.key(2), 4096)
    result, info = jax.jit(jax.vmap(kernel))(keys, states)
    assert jnp.all(info.is_accepted)
    assert jnp.all(info.num_shrink >= 8)
    np.testing.assert_allclose(result.position["x"].mean(axis=0), 0, atol=0.06)
    np.testing.assert_allclose(result.position["x"].var(axis=0), 1, atol=0.1)
    scalar = jax.jit(kernel)
    for i in (0, 7, 15):
        expected = scalar(keys[i], jax.tree.map(lambda x: x[i], states))
        actual = jax.tree.map(lambda x: x[i], (result, info))
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)
