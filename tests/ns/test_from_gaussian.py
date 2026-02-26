"""Tests for the Gaussian proposal inner kernel."""

import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest

from blackjax.ns import base, nrs
from blackjax.ns.nrs import (
    GaussianProposalInfo,
    _build_gaussian_inner_kernel,
    count_survivors,
)
from blackjax.smc.tuning.from_particles import (
    particles_covariance_matrix,
    particles_means,
)


def make_init_state_fn(logprior_fn, loglikelihood_fn):
    return functools.partial(
        base.init_state_strategy,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
    )


class GaussianProposalTest(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_rejection_basic(self):
        """Test basic shapes and constraint satisfaction."""
        ndim = 3
        num_live = 50
        num_delete = 5
        num_proposals = 200

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum((x - 1.0) ** 2)

        init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)

        # Create a state with live particles
        key, init_key = jax.random.split(self.key)
        positions = jax.random.normal(init_key, (num_live, ndim))
        state = base.init(positions, jax.vmap(init_state_fn))

        # Get unravel_fn
        from jax.flatten_util import ravel_pytree

        prototype_pos = jax.tree.map(lambda x: x[0], state.particles.position)
        _, unravel_fn = ravel_pytree(prototype_pos)

        # Compute mean and cov
        mean = particles_means(state.particles.position)
        cov = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))

        # Build and run inner kernel
        update_fn = _build_gaussian_inner_kernel(
            init_state_fn, unravel_fn, num_delete, num_proposals, max_rounds=50
        )

        loglikelihood_0 = jnp.sort(state.particles.loglikelihood)[num_delete - 1]

        key, step_key = jax.random.split(key)
        new_particles, info = update_fn(
            step_key, state, loglikelihood_0, mean=mean, cov=cov
        )

        # Check shapes
        chex.assert_shape(new_particles.position, (num_delete, ndim))
        chex.assert_shape(new_particles.loglikelihood, (num_delete,))
        chex.assert_shape(new_particles.logdensity, (num_delete,))
        chex.assert_shape(new_particles.loglikelihood_birth, (num_delete,))

        # With 200 proposals, 50 rounds, and moderate problem, should succeed
        self.assertTrue(bool(info.success), "Should succeed with these settings")

        # Check constraint satisfaction (all above threshold)
        self.assertTrue(
            jnp.all(new_particles.loglikelihood > loglikelihood_0),
            "All new particles should satisfy likelihood constraint",
        )

        # Check info types and invariants
        self.assertIsInstance(info, GaussianProposalInfo)
        self.assertGreaterEqual(int(info.num_rounds), 1)
        self.assertEqual(
            int(info.num_proposals_total),
            int(info.num_rounds) * num_proposals,
            "Total proposals should be num_rounds * num_proposals",
        )
        self.assertLessEqual(
            int(info.num_accepted_total),
            int(info.num_proposals_total),
            "Accepted cannot exceed total proposals",
        )

    def test_rejection_uniform_weights(self):
        """When q = π (proposal matches prior), all accepted proposals should survive."""
        ndim = 2
        num_live = 100
        num_delete = 3
        num_proposals = 500

        # Use Gaussian prior, constant likelihood, and match proposal to prior.
        # With constant likelihood, log_weights = logdensity - logpdf_q
        # = (logprior + loglikelihood) - logpdf_q = logprior + 0 - logpdf_q.
        # Since proposal q = prior, log_weights = const for all proposals.
        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return jnp.array(0.0)

        init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.normal(init_key, (num_live, ndim))
        state = base.init(positions, jax.vmap(init_state_fn))

        from jax.flatten_util import ravel_pytree

        prototype_pos = jax.tree.map(lambda x: x[0], state.particles.position)
        _, unravel_fn = ravel_pytree(prototype_pos)

        # Use standard normal as proposal (matches prior)
        mean = jnp.zeros(ndim)
        cov = jnp.eye(ndim)

        update_fn = _build_gaussian_inner_kernel(
            init_state_fn, unravel_fn, num_delete, num_proposals, max_rounds=10
        )

        # Use -inf threshold so ALL proposals pass the likelihood constraint.
        # This makes acceptance deterministic (no randomness in acceptance count).
        loglikelihood_0 = jnp.array(-jnp.inf)

        key, step_key = jax.random.split(key)
        new_particles, info = update_fn(
            step_key, state, loglikelihood_0, mean=mean, cov=cov
        )

        # Should succeed (all 500 proposals accepted, and weights are uniform)
        self.assertTrue(bool(info.success), "Should find enough survivors")

        # When q = prior, log-weights are constant (log π - log q = const).
        # So log_w_max = log_w for all, and log(u) < 0 always holds.
        # This means ALL accepted proposals survive — no rescaling attrition.
        # With -inf threshold and 500 proposals, must finish in exactly 1 round.
        self.assertEqual(
            int(info.num_rounds),
            1,
            "Uniform weights should need only 1 round",
        )
        # All 500 proposals should be accepted (none filtered by likelihood)
        self.assertEqual(
            int(info.num_accepted_total),
            num_proposals,
            "All proposals should be accepted with -inf threshold",
        )
        # Uniform weights means no attrition: all accepted survive
        self.assertEqual(
            int(info.num_survivors),
            num_proposals,
            "With uniform weights, all accepted proposals should survive",
        )

    def test_failure_mode_produces_duplicates(self):
        """In failure mode (not enough survivors), fill_value=0 produces duplicates.

        When success=False, jnp.nonzero with fill_value=0 fills remaining
        survivor indices with 0, so multiple rows point to buffer[0].
        This verifies the documented failure semantics.
        """
        ndim = 2
        num_live = 20
        num_delete = 10

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            return -100.0 * jnp.sum((x - 10.0) ** 2)

        init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.normal(init_key, (num_live, ndim))
        state = base.init(positions, jax.vmap(init_state_fn))

        from jax.flatten_util import ravel_pytree

        prototype_pos = jax.tree.map(lambda x: x[0], state.particles.position)
        _, unravel_fn = ravel_pytree(prototype_pos)

        mean = particles_means(state.particles.position)
        cov = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))

        # 2 proposals, 1 round, need 10 survivors — guaranteed failure
        update_fn = _build_gaussian_inner_kernel(
            init_state_fn, unravel_fn, num_delete, num_proposals=2, max_rounds=1
        )

        loglikelihood_0 = state.particles.loglikelihood.max()

        key, step_key = jax.random.split(key)
        new_particles, info = update_fn(
            step_key, state, loglikelihood_0, mean=mean, cov=cov
        )

        self.assertFalse(bool(info.success))

        # With fill_value=0, unfilled survivor slots all point to buffer[0].
        # So multiple returned rows should be exactly equal to row 0.
        pos = new_particles.position
        eq_row0 = jnp.all(pos == pos[0], axis=-1)
        num_eq_row0 = int(eq_row0.sum())
        self.assertGreaterEqual(
            num_eq_row0,
            2,
            f"Expected multiple rows equal to row 0 from fill_value=0, got {num_eq_row0}",
        )

    def test_failure_mode(self):
        """With very restrictive settings, should report failure gracefully."""
        ndim = 2
        num_live = 20
        num_delete = 5

        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()

        def loglikelihood_fn(x):
            # Very peaked likelihood — hard to satisfy
            return -100.0 * jnp.sum((x - 10.0) ** 2)

        init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.normal(init_key, (num_live, ndim))
        state = base.init(positions, jax.vmap(init_state_fn))

        from jax.flatten_util import ravel_pytree

        prototype_pos = jax.tree.map(lambda x: x[0], state.particles.position)
        _, unravel_fn = ravel_pytree(prototype_pos)

        mean = particles_means(state.particles.position)
        cov = jnp.atleast_2d(particles_covariance_matrix(state.particles.position))

        # Very few proposals, only 1 round — likely to fail
        update_fn = _build_gaussian_inner_kernel(
            init_state_fn, unravel_fn, num_delete, num_proposals=2, max_rounds=1
        )

        # Use a very high threshold
        loglikelihood_0 = state.particles.loglikelihood.max()

        key, step_key = jax.random.split(key)
        new_particles, info = update_fn(
            step_key, state, loglikelihood_0, mean=mean, cov=cov
        )

        # Should report failure (very unlikely to get 5 survivors from 2 proposals)
        self.assertIsInstance(info, GaussianProposalInfo)
        chex.assert_shape(new_particles.position, (num_delete, ndim))
        self.assertFalse(bool(info.success), "Should fail with only 2 proposals and 1 round")
        self.assertEqual(int(info.num_rounds), 1, "Should exhaust max_rounds=1")
        self.assertEqual(
            int(info.num_proposals_total), 2, "Should have drawn exactly 2 proposals"
        )

    def test_wmax_rescaling_deterministic(self):
        """Deterministic test of the stored-uniform rescaling logic.

        Directly tests the _count_survivors function used inside the kernel
        with known log-weights, log-uniforms, and log_w_max. Verifies:
        1. Increasing w_max causes previously surviving proposals to be discarded
        2. The result depends on the stored uniforms (reusing vs redrawing differs)
        """
        # Scenario: 4 proposals with different weights
        # Proposal A: low weight, generous uniform (should survive initially)
        # Proposal B: medium weight (survives)
        # Proposal C: high weight (survives)
        # Proposal D: invalid (NEG_INF weight)
        log_weights = jnp.array([1.0, 3.0, 5.0, -jnp.inf])
        log_uniforms = jnp.array([-0.5, -1.0, -0.1, -0.5])

        # Phase 1: w_max = 3.0 (only B and C known so far)
        # Survival condition: log_u < log_w - log_w_max
        # A: -0.5 < 1.0 - 3.0 = -2.0 → False (A doesn't survive)
        # B: -1.0 < 3.0 - 3.0 = 0.0 → True
        # C: -0.1 < 5.0 - 3.0 = 2.0 → True
        # D: invalid → False
        surviving_phase1 = count_survivors(log_weights, log_uniforms, jnp.array(3.0))
        self.assertEqual(
            int(surviving_phase1.sum()), 2, "Phase 1: B and C should survive"
        )
        self.assertFalse(bool(surviving_phase1[0]), "A should not survive at w_max=3")
        self.assertTrue(bool(surviving_phase1[1]), "B should survive at w_max=3")
        self.assertTrue(bool(surviving_phase1[2]), "C should survive at w_max=3")
        self.assertFalse(bool(surviving_phase1[3]), "D (invalid) should not survive")

        # Phase 2: w_max increases to 4.0
        # Condition: log_u < log_w - log_w_max
        # A: -0.5 < 1.0 - 4.0 = -3.0? No → False
        # B: -1.0 < 3.0 - 4.0 = -1.0? No (strict <) → False (B lost!)
        # C: -0.1 < 5.0 - 4.0 = 1.0? Yes → True
        # Increasing w_max from 3 to 4 causes B to be discarded (rescaling!)
        surviving_phase2 = count_survivors(log_weights, log_uniforms, jnp.array(4.0))
        self.assertEqual(
            int(surviving_phase2.sum()), 1, "Phase 2: only C survives"
        )
        self.assertFalse(bool(surviving_phase2[1]), "B discarded after w_max increase")
        self.assertTrue(bool(surviving_phase2[2]), "C survives at w_max=4")

        # Phase 3: w_max = 5.0 → C is marginal
        # C: -0.1 < 5.0 - 5.0 = 0.0? Yes → True
        surviving_phase3 = count_survivors(log_weights, log_uniforms, jnp.array(5.0))
        self.assertEqual(int(surviving_phase3.sum()), 1, "C survives at w_max=5")

        # Phase 4: w_max = 5.2 → C's margin breaks
        # C: -0.1 < 5.0 - 5.2 = -0.2? No (-0.1 > -0.2) → False
        surviving_phase4 = count_survivors(log_weights, log_uniforms, jnp.array(5.2))
        self.assertEqual(int(surviving_phase4.sum()), 0, "No survivors at w_max=5.2")

        # KEY PROPERTY: outcome depends on the STORED uniform, not just weights.
        # If B had log_uniform = -2.0 instead of -1.0, B would survive w_max=4:
        # B: -2.0 < 3.0 - 4.0 = -1.0? Yes → True (would survive!)
        alt_log_uniforms = jnp.array([-0.5, -2.0, -0.1, -0.5])
        surviving_alt = count_survivors(log_weights, alt_log_uniforms, jnp.array(4.0))
        self.assertTrue(
            bool(surviving_alt[1]),
            "B survives w_max=4 with more generous stored uniform",
        )
        self.assertEqual(
            int(surviving_alt.sum()),
            2,
            "With different uniforms, both B and C survive w_max=4",
        )

        # Original B (log_u=-1.0) does not survive w_max=4, but
        # alternate B (log_u=-2.0) does. Same weights, same w_max,
        # different stored uniform → different survival outcome.
        # This proves rescaling correctness depends on stored uniforms.

    def test_wmax_rescaling_integration(self):
        """Integration test that w_max rescaling occurs in a real kernel run.

        Verifies that with variable weights, some accepted proposals are
        discarded by the rejection test (num_survivors < num_accepted_total).
        """
        ndim = 4
        num_live = 50
        num_delete = 5
        num_proposals = 3

        half_width = 5.0

        def logprior_fn(x):
            in_box = jnp.all(jnp.abs(x) <= half_width)
            return jnp.where(in_box, -ndim * jnp.log(2 * half_width), -jnp.inf)

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        init_state_fn = make_init_state_fn(logprior_fn, loglikelihood_fn)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.uniform(
            init_key, (num_live, ndim), minval=-half_width, maxval=half_width
        )
        state = base.init(positions, jax.vmap(init_state_fn))

        from jax.flatten_util import ravel_pytree

        prototype_pos = jax.tree.map(lambda x: x[0], state.particles.position)
        _, unravel_fn = ravel_pytree(prototype_pos)

        mean = jnp.zeros(ndim)
        cov = 0.5 * jnp.eye(ndim)

        update_fn = _build_gaussian_inner_kernel(
            init_state_fn, unravel_fn, num_delete, num_proposals, max_rounds=100
        )

        loglikelihood_0 = jnp.array(-jnp.inf)

        key, step_key = jax.random.split(key)
        new_particles, info = update_fn(
            step_key, state, loglikelihood_0, mean=mean, cov=cov
        )

        self.assertTrue(bool(info.success), "Should succeed with 100 rounds")
        self.assertGreater(int(info.num_rounds), 1)
        self.assertTrue(jnp.isfinite(info.log_w_max))
        self.assertGreaterEqual(
            int(info.num_survivors),
            num_delete,
            "Should have at least num_delete survivors",
        )
        self.assertLess(
            int(info.num_survivors),
            int(info.num_accepted_total),
            "Variable weights should cause some accepted proposals to be "
            "discarded by rescaling (num_survivors < num_accepted_total)",
        )


class NRSIntegrationTest(chex.TestCase):
    """Integration test for the full NRS pipeline."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(123)

    def test_nrs_single_step(self):
        """Test that NRS can initialize and take a single step."""
        ndim = 2
        num_live = 50
        num_delete = 2

        def logprior_fn(x):
            return jnp.where(
                jnp.all(jnp.abs(x) <= 5.0), -jnp.log(10.0) * ndim, -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.uniform(init_key, (num_live, ndim), minval=-5.0, maxval=5.0)

        algorithm = nrs.as_top_level_api(
            logprior_fn,
            loglikelihood_fn,
            prototype_position=positions[0],
            num_delete=num_delete,
            num_proposals=500,
            max_rounds=20,
        )

        state = algorithm.init(positions)

        # Check initial state
        self.assertIsNotNone(state.inner_kernel_params)
        self.assertIn("mean", state.inner_kernel_params)
        self.assertIn("cov", state.inner_kernel_params)

        # Take a step
        key, step_key = jax.random.split(key)
        new_state, info = algorithm.step(step_key, state)

        # Check output structure
        chex.assert_shape(new_state.particles.position, (num_live, ndim))
        chex.assert_shape(info.particles.loglikelihood, (num_delete,))

    def test_nrs_multi_step(self):
        """Test multi-step NRS: dead threshold should increase monotonically."""
        ndim = 2
        num_live = 50
        num_delete = 2
        num_steps = 10

        def logprior_fn(x):
            return jnp.where(
                jnp.all(jnp.abs(x) <= 5.0), -jnp.log(10.0) * ndim, -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.uniform(
            init_key, (num_live, ndim), minval=-5.0, maxval=5.0
        )

        algorithm = nrs.as_top_level_api(
            logprior_fn,
            loglikelihood_fn,
            prototype_position=positions[0],
            num_delete=num_delete,
            num_proposals=500,
            max_rounds=20,
        )

        state = algorithm.init(positions)
        dead_thresholds = []

        for _ in range(num_steps):
            key, step_key = jax.random.split(key)
            state, info = algorithm.step(step_key, state)
            # The dead particles' max loglikelihood is the threshold for that step
            dead_thresholds.append(float(info.particles.loglikelihood.max()))

        # Dead threshold should be monotonically non-decreasing
        for i in range(1, len(dead_thresholds)):
            self.assertGreaterEqual(
                dead_thresholds[i],
                dead_thresholds[i - 1],
                f"Dead threshold decreased: step {i} ({dead_thresholds[i]:.4f})"
                f" < step {i-1} ({dead_thresholds[i-1]:.4f})",
            )

        # After 10 steps, minimum live loglikelihood should have increased
        final_min_ll = float(state.particles.loglikelihood.min())
        self.assertGreater(
            final_min_ll,
            dead_thresholds[0],
            "Minimum live loglikelihood should exceed initial dead threshold",
        )


    def test_nrs_multi_step_consistency(self):
        """Verify that consecutive steps produce valid, finite results."""
        ndim = 2
        num_live = 30
        num_delete = 1

        def logprior_fn(x):
            return jnp.where(
                jnp.all(jnp.abs(x) <= 5.0), -jnp.log(10.0) * ndim, -jnp.inf
            )

        def loglikelihood_fn(x):
            return -0.5 * jnp.sum(x**2)

        key, init_key = jax.random.split(self.key)
        positions = jax.random.uniform(
            init_key, (num_live, ndim), minval=-5.0, maxval=5.0
        )

        algorithm = nrs.as_top_level_api(
            logprior_fn,
            loglikelihood_fn,
            prototype_position=positions[0],
            num_delete=num_delete,
            num_proposals=200,
            max_rounds=10,
        )
        state = algorithm.init(positions)

        # First step
        key, step_key = jax.random.split(key)
        state1, info1 = algorithm.step(step_key, state)
        chex.assert_shape(state1.particles.position, (num_live, ndim))

        # Second step — should produce valid output
        key, step_key = jax.random.split(key)
        state2, info2 = algorithm.step(step_key, state1)
        chex.assert_shape(state2.particles.position, (num_live, ndim))

        # Both steps should return finite positions
        self.assertTrue(
            jnp.all(jnp.isfinite(state1.particles.position)),
            "First step should produce finite positions",
        )
        self.assertTrue(
            jnp.all(jnp.isfinite(state2.particles.position)),
            "Second step should produce finite positions",
        )

        # inner_kernel_params should be finite and correctly shaped
        mean2 = state2.inner_kernel_params["mean"]
        cov2 = state2.inner_kernel_params["cov"]
        self.assertTrue(jnp.all(jnp.isfinite(mean2)), "Mean should be finite")
        self.assertTrue(jnp.all(jnp.isfinite(cov2)), "Cov should be finite")
        chex.assert_shape(mean2, (ndim,))
        chex.assert_shape(cov2, (ndim, ndim))


if __name__ == "__main__":
    absltest.main()
