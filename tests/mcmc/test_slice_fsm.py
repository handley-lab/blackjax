"""Independent scalar references and batching tests for asynchronous slices."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from blackjax.mcmc import slice_fsm as fsm
from blackjax.mcmc.slice import SliceState


def normal(x):
    return -jnp.square(x).sum() / 2


def disconnected(x):
    return jnp.where((jnp.abs(x) < 0.4) | (jnp.abs(x - 3) < 0.6), 0.0, -jnp.inf)


def evaluate(logdensity):
    return lambda x: (SliceState(x, logdensity(x)), jnp.asarray(True))


line = (lambda key, x, index: jnp.ones_like(x), lambda x, v, t: x + v * t)


def neal_accept(t, left, right, width, inside):
    """Figure 6, deliberately independent of the implementation's transitions."""
    separated = False
    while right - left > 1.1 * width:
        mid = (left + right) / 2
        if (0 < mid <= t) or (t < mid <= 0):
            separated = True
        if t < mid:
            right = mid
        else:
            left = mid
        if separated and not inside(left) and not inside(right):
            return False
    return True


def reference(key, x, logdensity, strategy, width, budget, num_steps):
    """Ordinary Python loops for Neal's Figures 3--6, with the same random draws."""
    expansions = shrinks = rejected_checks = 0
    for _ in range(num_steps):
        key, _, slice_key = random.split(key, 3)
        if strategy is fsm.stepping_out:
            slice_key, budget_key = random.split(slice_key)
        slice_key, level_key, bracket_key = random.split(slice_key, 3)
        dtype = x.dtype
        level = float(logdensity(x) + jnp.log(random.uniform(level_key, dtype=dtype)))
        left = -width * float(random.uniform(bracket_key, dtype=dtype))
        right = left + width
        inside = lambda t: float(logdensity(x + t)) >= level
        if strategy is fsm.stepping_out:
            j = int(jnp.floor(budget * random.uniform(budget_key)))
            k = budget - 1 - j
            while j > 0 and inside(left):
                left -= width
                j -= 1
                expansions += 1
            while k > 0 and inside(right):
                right += width
                k -= 1
                expansions += 1
        else:
            k = budget
            while k > 0 and (inside(left) or inside(right)):
                slice_key, side_key = random.split(slice_key)
                if bool(random.bernoulli(side_key)):
                    left -= right - left
                else:
                    right += right - left
                k -= 1
                expansions += 1
        lo, hi = left, right
        while True:
            slice_key, subkey = random.split(slice_key)
            t = lo + float(random.uniform(subkey, dtype=dtype)) * (hi - lo)
            shrinks += 1
            accepted = inside(t)
            if accepted and strategy is fsm.doubling:
                accepted = neal_accept(t, left, right, width, inside)
                rejected_checks += not accepted
            if accepted:
                x = x + t
                break
            if t < 0:
                lo = t
            else:
                hi = t
    return x, expansions, shrinks, rejected_checks


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
@pytest.mark.parametrize("budget", [0, 1, 4, 8])
@pytest.mark.parametrize("logdensity", [normal, disconnected])
def test_neal_reference(strategy, budget, logdensity):
    kernel = jax.jit(
        fsm.build_kernel(evaluate(logdensity), *line, strategy, max_expansions=budget)
    )
    x = jnp.asarray(0.0)
    for seed in range(12):
        key = random.key(seed)
        expected, expansions, shrinks, _ = reference(
            key, x, logdensity, strategy, 1.0, budget, 3
        )
        actual, info = kernel(key, evaluate(logdensity)(x)[0], 3)
        np.testing.assert_allclose(actual.position, expected, rtol=2e-5, atol=2e-6)
        assert info.num_expansions == expansions
        assert info.num_shrink == shrinks


def test_doubling_rejects_density_valid_trial():
    inside = lambda t: abs(t) < 0.4 or abs(t - 3) < 0.6
    assert inside(3.0)
    assert not neal_accept(3.0, -0.25, 3.75, 1.0, inside)
    start, advance = fsm.doubling(1.0, 4)
    particle = SliceState(jnp.asarray(0.0), jnp.asarray(0.0))
    s, d = start(random.key(0), particle)
    s = s._replace(
        phase=2,
        left=jnp.asarray(-0.25),
        right=jnp.asarray(3.75),
        t=jnp.asarray(3.0),
        level=jnp.asarray(-1.0),
    )
    d = d._replace(left=s.left, right=s.right)
    trial = SliceState(jnp.asarray(3.0), jnp.asarray(0.0))
    s, d = advance(s, d, trial, jnp.asarray(True))
    probes = []
    while int(s.phase) in (4, 5):
        probes.append(float(s.t))
        probe = SliceState(s.t, disconnected(s.t))
        s, d = advance(s, d, probe, disconnected(s.t) >= s.level)
    assert s.phase == 2  # Rejected and returned to shrinkage, not accepted.
    assert s.right == 3.0
    assert len(probes) > 0


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
def test_scalar_vmap_and_permuted_chains(strategy):
    kernel = fsm.build_kernel(evaluate(normal), *line, strategy)
    keys = random.split(random.key(72), 24)
    states = jax.vmap(lambda x: evaluate(normal)(x)[0])(jnp.linspace(-8, 8, 24))
    counts = jnp.arange(24) % 5
    batched = jax.jit(jax.vmap(kernel))
    actual = batched(keys, states, counts)
    scalar = jax.jit(kernel)
    expected = [
        scalar(keys[i], jax.tree.map(lambda x: x[i], states), counts[i])
        for i in range(24)
    ]
    expected = jax.tree.map(lambda *x: jnp.stack(x), *expected)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6),
        actual,
        expected,
    )
    reverse = batched(keys[::-1], jax.tree.map(lambda x: x[::-1], states), counts[::-1])
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(a, b[::-1]), actual, reverse
    )


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
def test_auxiliary_state_and_constraint(strategy):
    class Particle(NamedTuple):
        position: object
        logdensity: object
        loglikelihood: object

    def evaluate(x):
        return Particle(x, normal(x), x**3 + 10), jnp.abs(x) < 0.25

    kernel = jax.jit(jax.vmap(fsm.build_kernel(evaluate, *line, strategy)))
    x = jnp.zeros(64)
    states = jax.vmap(lambda x: evaluate(x)[0])(x)
    result, info = kernel(random.split(random.key(51), 64), states, jnp.full(64, 5))
    assert jnp.all(jnp.abs(result.position) < 0.25)
    np.testing.assert_allclose(result.loglikelihood, result.position**3 + 10)
    np.testing.assert_allclose(result.logdensity, -result.position**2 / 2)
    assert jnp.all(info.num_shrink >= 5)


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
@pytest.mark.parametrize("logdensity", [normal, disconnected])
def test_stationarity(strategy, logdensity):
    n = 8192
    rng = np.random.default_rng(196)
    if logdensity is normal:
        x = jnp.asarray(rng.normal(size=n))
    else:
        right = rng.random(n) < 0.6
        x = jnp.asarray(
            np.where(right, 3 + rng.uniform(-0.6, 0.6, n), rng.uniform(-0.4, 0.4, n))
        )
    states = jax.vmap(lambda x: evaluate(logdensity)(x)[0])(x)
    kernel = jax.jit(
        jax.vmap(
            fsm.build_kernel(evaluate(logdensity), *line, strategy),
            in_axes=(0, 0, None),
        )
    )
    result, _ = kernel(random.split(random.key(982), n), states, 8)
    result = np.asarray(result.position)
    if logdensity is normal:
        assert abs(result.mean()) < 0.04
        assert abs(result.var() - 1) < 0.06
    else:
        right = result > 1
        assert abs(right.mean() - 0.6) < 0.025
        assert abs(result[right].mean() - 3) < 0.03
        assert abs(result[~right].mean()) < 0.03


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
def test_mixed_topology_and_overlapping_blocks(strategy):
    n = 2048
    rng = np.random.default_rng(531)
    sphere = rng.normal(size=(n, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    x = {
        "abc": jnp.asarray(rng.normal(size=(n, 3))),
        "angle": jnp.asarray(rng.uniform(0, 2 * np.pi, n)),
        "sphere": jnp.asarray(sphere),
    }
    masks = jnp.array([[1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 0]])

    def generate(key, x, index):
        a, b, c = random.split(key, 3)
        group = index % 4
        return {
            "abc": random.normal(a, (3,)) * masks[group],
            "angle": random.normal(b) * (group == 2),
            "sphere": random.normal(c) * (group == 2),
        }

    def move(x, v, t):
        angle = v["sphere"] * t
        c, s = jnp.cos(angle), jnp.sin(angle)
        a, b, z = x["sphere"]
        return {
            "abc": x["abc"] + v["abc"] * t,
            "angle": (x["angle"] + v["angle"] * t) % (2 * jnp.pi),
            "sphere": jnp.array([c * a - s * b, s * a + c * b, z]),
        }

    def evaluate(x):
        return SliceState(x, normal(x["abc"])), jnp.asarray(True)

    states = jax.vmap(lambda x: evaluate(x)[0])(x)
    kernel = jax.jit(
        jax.vmap(
            fsm.build_kernel(evaluate, generate, move, strategy, max_expansions=4),
            in_axes=(0, 0, None),
        )
    )
    result, info = kernel(random.split(random.key(921), n), states, 12)
    assert jnp.all(info.num_shrink >= 12)
    np.testing.assert_allclose(
        jnp.linalg.norm(result.position["sphere"], axis=1), 1, atol=2e-6
    )
    assert jnp.max(jnp.abs(jnp.mean(result.position["abc"], axis=0))) < 0.09
    assert jnp.max(jnp.abs(jnp.var(result.position["abc"], axis=0) - 1)) < 0.12
    assert abs(jnp.mean(jnp.cos(result.position["angle"]))) < 0.06
    assert abs(jnp.mean(jnp.sin(result.position["angle"]))) < 0.06
    assert jnp.max(jnp.abs(jnp.mean(result.position["sphere"], axis=0))) < 0.06


@pytest.mark.parametrize("strategy", [fsm.stepping_out, fsm.doubling])
def test_one_evaluation_round_per_tick(strategy):
    from jax.custom_batching import custom_vmap

    rounds = []

    @custom_vmap
    def target(x):
        return evaluate(normal)(x)

    @target.def_vmap
    def batched_target(axis_size, in_batched, x):
        jax.debug.callback(lambda x: rounds.append(np.asarray(x)), x)
        result = jax.vmap(evaluate(normal))(x)
        return result, jax.tree.map(lambda _: True, result)

    kernel = fsm.build_kernel(target, *line, strategy)
    keys = random.split(random.key(182), 16)
    states = jax.vmap(lambda x: evaluate(normal)(x)[0])(jnp.linspace(-10, 10, 16))
    counts = jnp.arange(16) % 5 + 1
    result, info = jax.jit(jax.vmap(kernel))(keys, states, counts)
    jax.block_until_ready(result)
    jax.effects_barrier()
    assert len(rounds) == int(jnp.max(info.num_evaluations))
    assert np.ptp(np.asarray(info.num_evaluations)) > 0
    # Each lane follows its own scalar evaluation sequence, including later moves.
    for i in (0, 7, 15):
        calls = []

        def scalar_target(x):
            jax.debug.callback(lambda x: calls.append(float(x)), x)
            return evaluate(normal)(x)

        scalar = fsm.build_kernel(scalar_target, *line, strategy)
        scalar_result = jax.jit(scalar)(
            keys[i], jax.tree.map(lambda x: x[i], states), counts[i]
        )
        jax.block_until_ready(scalar_result)
        jax.effects_barrier()
        actual = np.array(rounds)[: len(calls), i]
        np.testing.assert_allclose(actual, calls, rtol=2e-5, atol=2e-6)
