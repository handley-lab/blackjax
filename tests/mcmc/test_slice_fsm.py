"""Independent scalar references and batching tests for asynchronous slices."""

from collections import namedtuple
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


def line(evaluate):
    def proposal_generator(key, state):
        return state.position

    def slice_fn(t, proposal_state):
        return evaluate(proposal_state + t)

    return proposal_generator, slice_fn


class SliceInfo(NamedTuple):
    num_evaluations: object
    num_expansions: object
    num_shrink: object


def state_type(strategy):
    state = {
        fsm.build_stepping_out_kernel: fsm.SteppingOutState,
        fsm.build_doubling_kernel: fsm.DoublingState,
    }[strategy]
    return namedtuple(
        "ChainState",
        (*state._fields, "index", *SliceInfo._fields),
        defaults=(*state._field_defaults.values(), 0, 0, 0, 0),
    )


def build_sample(
    proposal_generator, slice_fn, interval, *, width=1.0, max_expansions=10
):
    init_move = {
        fsm.build_stepping_out_kernel: fsm.init_stepping_out,
        fsm.build_doubling_kernel: fsm.init_doubling,
    }[interval]

    def start_move(rng_key, state):
        rng_key, move_key = random.split(rng_key)
        slice_key, proposal_key = random.split(move_key)
        state = init_move(slice_key, state, width, max_expansions)
        return rng_key, state, proposal_generator(proposal_key, state)

    def sample(rng_key, state, num_steps=1):
        def body(carry):
            rng_key, state, proposal_state = carry
            rng_key, step_key = random.split(rng_key)
            step = interval(lambda t: slice_fn(t, proposal_state), width)
            state, info = step(step_key, state)
            state = state._replace(
                index=state.index + info.is_accepted,
                num_evaluations=state.num_evaluations + info.num_evaluations,
                num_expansions=state.num_expansions + info.num_expansions,
                num_shrink=state.num_shrink + info.num_shrink,
            )
            return jax.lax.cond(
                info.is_accepted & (state.index < num_steps),
                lambda carry: start_move(carry[0], carry[1]),
                lambda carry: carry,
                (rng_key, state, proposal_state),
            )

        state = state._replace(
            index=0, num_evaluations=0, num_expansions=0, num_shrink=0
        )
        carry = start_move(rng_key, state)
        _, state, _ = jax.lax.while_loop(
            lambda carry: carry[1].index < num_steps, body, carry
        )
        info = SliceInfo(state.num_evaluations, state.num_expansions, state.num_shrink)
        return state, info

    return sample


def neal_accept(t, left, right, width, inside, probe=None):
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
        if separated:
            if probe is not None:
                probe()
                probe()
            if not inside(left) and not inside(right):
                return False
    return True


def reference(key, x, logdensity, strategy, width, budget, num_steps):
    """Ordinary Python loops for Neal's Figures 3--6, with the same random draws."""
    expansions = shrinks = rejected_checks = 0

    def tick_key():
        nonlocal key
        key, subkey = random.split(key)
        return subkey

    for _ in range(num_steps):
        move_key = tick_key()
        slice_key, _ = random.split(move_key)
        if strategy is fsm.build_stepping_out_kernel:
            level_key, bracket_key, budget_key = random.split(slice_key, 3)
        else:
            level_key, bracket_key = random.split(slice_key)
        dtype = x.dtype
        level = float(logdensity(x) + jnp.log(random.uniform(level_key, dtype=dtype)))
        left = -width * float(random.uniform(bracket_key, dtype=dtype))
        right = left + width
        inside = lambda t: float(logdensity(x + t)) >= level
        if strategy is fsm.build_stepping_out_kernel:
            j = int(jnp.floor(budget * random.uniform(budget_key)))
            k = budget - 1 - j
            while j > 0:
                tick_key()
                if not inside(left):
                    break
                left -= width
                j -= 1
                expansions += 1
            while k > 0:
                tick_key()
                if not inside(right):
                    break
                right += width
                k -= 1
                expansions += 1
        else:
            k = budget
            if k > 0:
                tick_key()
                side_key = tick_key()
            while k > 0 and (inside(left) or inside(right)):
                if bool(random.bernoulli(side_key)):
                    left -= right - left
                else:
                    right += right - left
                k -= 1
                expansions += 1
                side_key = tick_key()
        lo, hi = left, right
        while True:
            subkey = tick_key()
            t = lo + float(random.uniform(subkey, dtype=dtype)) * (hi - lo)
            shrinks += 1
            accepted = inside(t)
            if accepted and strategy is fsm.build_doubling_kernel:
                accepted = neal_accept(t, left, right, width, inside, tick_key)
                rejected_checks += not accepted
            if accepted:
                x = x + t
                break
            if t < 0:
                lo = t
            else:
                hi = t
    return x, expansions, shrinks, rejected_checks


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
@pytest.mark.parametrize("budget", [0, 1, 4, 8])
@pytest.mark.parametrize("logdensity", [normal, disconnected])
def test_neal_reference(strategy, budget, logdensity):
    kernel = jax.jit(
        build_sample(*line(evaluate(logdensity)), strategy, max_expansions=budget)
    )
    x = jnp.asarray(0.0)
    for seed in range(12):
        key = random.key(seed)
        expected, expansions, shrinks, _ = reference(
            key, x, logdensity, strategy, 1.0, budget, 3
        )
        actual, info = kernel(key, state_type(strategy)(x, logdensity(x)), 3)
        np.testing.assert_allclose(actual.position, expected, rtol=2e-5, atol=2e-6)
        assert info.num_expansions == expansions
        assert info.num_shrink == shrinks


def test_doubling_rejects_density_valid_trial():
    inside = lambda t: abs(t) < 0.4 or abs(t - 3) < 0.6
    assert inside(3.0)
    assert not neal_accept(3.0, -0.25, 3.75, 1.0, inside)
    advance = fsm.build_doubling_kernel(evaluate(disconnected), 1.0)
    particle = fsm.DoublingState(jnp.asarray(0.0), jnp.asarray(0.0))
    s = fsm.init_doubling(random.key(0), particle, 1.0, 4)
    for seed in range(100):
        key = random.key(seed)
        t = -0.25 + 4 * random.uniform(key, dtype=s.left.dtype)
        if inside(t) and not neal_accept(t, -0.25, 3.75, 1.0, inside):
            break
    else:
        pytest.fail("No reverse-construction rejection in the fixture keys")
    s = s._replace(
        phase=2,
        left=jnp.asarray(-0.25),
        right=jnp.asarray(3.75),
        level=jnp.asarray(-1.0),
    )
    s = s._replace(expanded_left=s.left, expanded_right=s.right)
    s, _ = advance(key, s)
    probes = []
    while int(s.phase) in (fsm._CHECK_LEFT, fsm._CHECK_RIGHT):
        probes.append(
            float(s.check_left if int(s.phase) == fsm._CHECK_LEFT else s.check_right)
        )
        key, subkey = random.split(key)
        s, _ = advance(subkey, s)
    assert s.phase == 2  # Rejected and returned to shrinkage, not accepted.
    assert s.right == t
    assert len(probes) > 0


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
def test_scalar_vmap_and_permuted_chains(strategy):
    kernel = build_sample(*line(evaluate(normal)), strategy)
    keys = random.split(random.key(72), 24)
    states = jax.vmap(lambda x: state_type(strategy)(x, normal(x)))(
        jnp.linspace(-8, 8, 24)
    )
    counts = jnp.arange(24) % 5
    batched = jax.jit(jax.vmap(kernel))
    actual = batched(keys, states, counts)
    scalar = jax.jit(kernel)
    expected = [
        scalar(keys[i], jax.tree.map(lambda x: x[i], states), counts[i])
        for i in range(24)
    ]
    expected = jax.tree.map(lambda *x: jnp.stack(x), *expected)

    def arrays(tree):
        return jax.tree.map(
            lambda x: (
                random.key_data(x)
                if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
                else x
            ),
            tree,
        )

    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6),
        arrays(actual),
        arrays(expected),
    )
    reverse = batched(keys[::-1], jax.tree.map(lambda x: x[::-1], states), counts[::-1])
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(a, b[::-1]),
        arrays(actual),
        arrays(reverse),
    )


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
def test_proposal_generated_once_per_move(strategy):
    calls = []

    def proposal_generator(key, state):
        direction = random.normal(key)
        jax.debug.callback(lambda x: calls.append(float(x)), direction)
        return state.position, direction, state.index

    def slice_fn(t, proposal_state):
        x, direction, index = proposal_state
        position = x + direction * t
        return SliceState(position, normal(position)), index >= 0

    kernel = jax.jit(build_sample(proposal_generator, slice_fn, strategy))
    state = state_type(strategy)(jnp.asarray(0.0), jnp.asarray(0.0))
    state, info = kernel(random.key(32), state, 3)
    jax.block_until_ready(state)
    jax.effects_barrier()
    assert len(calls) == 3
    assert info.num_evaluations > 3


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
def test_successive_calls(strategy):
    kernel = jax.jit(build_sample(*line(evaluate(normal)), strategy))
    state = state_type(strategy)(jnp.asarray(0.0), jnp.asarray(0.0))
    state, _ = kernel(random.key(17), state, 3)
    fresh = state_type(strategy)(state.position, state.logdensity)
    actual, info = kernel(random.key(18), state, 2)
    expected, expected_info = kernel(random.key(18), fresh, 2)
    np.testing.assert_array_equal(actual.position, expected.position)
    np.testing.assert_array_equal(actual.logdensity, expected.logdensity)
    np.testing.assert_array_equal(info, expected_info)
    assert actual.index == 2


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
def test_auxiliary_state_and_constraint(strategy):
    class Particle(NamedTuple):
        position: object
        logdensity: object
        loglikelihood: object

    def evaluate(x):
        return Particle(x, normal(x), x**3 + 10), jnp.abs(x) < 0.25

    kernel = jax.jit(jax.vmap(build_sample(*line(evaluate), strategy)))
    x = jnp.zeros(64)
    ExtendedState = namedtuple(
        "ExtendedState", (*state_type(strategy)._fields, "loglikelihood")
    )
    states = jax.vmap(
        lambda x: ExtendedState(*state_type(strategy)(x, normal(x)), x**3 + 10)
    )(x)
    result, info = kernel(random.split(random.key(51), 64), states, jnp.full(64, 5))
    assert jnp.all(jnp.abs(result.position) < 0.25)
    np.testing.assert_allclose(result.loglikelihood, result.position**3 + 10)
    np.testing.assert_allclose(result.logdensity, -result.position**2 / 2)
    assert jnp.all(info.num_shrink >= 5)


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
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
    states = jax.vmap(lambda x: state_type(strategy)(x, logdensity(x)))(x)
    kernel = jax.jit(
        jax.vmap(
            build_sample(*line(evaluate(logdensity)), strategy),
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


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
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

    def proposal_generator(key, state):
        direction = generate(key, state.position, state.index)
        return state.position, direction

    def slice_fn(t, proposal_state):
        x, direction = proposal_state
        return evaluate(move(x, direction, t))

    states = jax.vmap(lambda x: state_type(strategy)(x, normal(x["abc"])))(x)
    kernel = jax.jit(
        jax.vmap(
            build_sample(proposal_generator, slice_fn, strategy, max_expansions=4),
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


@pytest.mark.parametrize(
    "strategy", [fsm.build_stepping_out_kernel, fsm.build_doubling_kernel]
)
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

    kernel = build_sample(*line(target), strategy)
    keys = random.split(random.key(182), 16)
    states = jax.vmap(lambda x: state_type(strategy)(x, normal(x)))(
        jnp.linspace(-10, 10, 16)
    )
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

        scalar = build_sample(*line(scalar_target), strategy)
        scalar_result = jax.jit(scalar)(
            keys[i], jax.tree.map(lambda x: x[i], states), counts[i]
        )
        jax.block_until_ready(scalar_result)
        jax.effects_barrier()
        actual = np.array(rounds)[: len(calls), i]
        np.testing.assert_allclose(actual, calls, rtol=2e-5, atol=2e-6)
