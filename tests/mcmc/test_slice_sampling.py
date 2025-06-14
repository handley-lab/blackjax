"""Test the Hit-and-Run Slice Sampling algorithm."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

import blackjax
from blackjax.mcmc import ss


class SliceSamplingCoreTest(chex.TestCase):
    """Test core slice sampling functionality."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def logdensity_normal(self, x):
        """Standard multivariate normal."""
        return stats.norm.logpdf(x).sum()

    def logdensity_constrained(self, x):
        """Constrained to positive values."""
        return jnp.where(jnp.all(x > 0), stats.norm.logpdf(x).sum(), -jnp.inf)

    def test_slice_state_structure(self):
        """Test SliceState structure and initialization."""
        position = jnp.array([1.0, -0.5, 2.0])
        state = ss.init(position, self.logdensity_normal)
        
        # Check structure
        self.assertIsInstance(state, ss.SliceState)
        chex.assert_trees_all_close(state.position, position)
        
        # Check logdensity is computed correctly
        expected_logdens = self.logdensity_normal(position)
        chex.assert_trees_all_close(state.logdensity, expected_logdens)
        
        # Check default logslice
        self.assertEqual(state.logslice, jnp.inf)

    def test_slice_info_structure(self):
        """Test SliceInfo structure."""
        info = ss.SliceInfo(
            constraint=jnp.array([1.0, 2.0]),
            l_steps=3,
            r_steps=5,
            s_steps=7,
            evals=15,
            d=jnp.array([0.5, -0.2])
        )
        
        chex.assert_shape(info.constraint, (2,))
        self.assertEqual(info.l_steps, 3)
        self.assertEqual(info.r_steps, 5)
        self.assertEqual(info.s_steps, 7)
        self.assertEqual(info.evals, 15)
        chex.assert_shape(info.d, (2,))

    def test_vertical_slice(self):
        """Test vertical slice height sampling."""
        position = jnp.array([0.0])
        state = ss.init(position, self.logdensity_normal)
        
        # Test multiple samples
        n_samples = 1000
        keys = jax.random.split(self.key, n_samples)
        
        new_states, infos = jax.vmap(ss.vertical_slice, in_axes=(0, None))(keys, state)
        
        # Heights should be below current logdensity
        logdens_at_pos = self.logdensity_normal(position)
        self.assertTrue(jnp.all(new_states.logslice <= logdens_at_pos))
        
        # Mean should be approximately logdens - 1 (E[log(U)] = -1)
        mean_height = jnp.mean(new_states.logslice)
        expected_mean = logdens_at_pos - 1.0
        chex.assert_trees_all_close(mean_height, expected_mean, atol=0.1)
        
        # Check info structure
        self.assertTrue(jnp.all(infos.evals == 0))  # Vertical slice doesn't eval logdensity

    @parameterized.parameters([1, 2, 5])
    def test_slice_sampling_dimensions(self, ndim):
        """Test slice sampling in different dimensions."""
        position = jnp.zeros(ndim)
        state = ss.init(position, self.logdensity_normal)
        
        # Test with simple direction and stepper
        direction = jax.random.normal(self.key, (ndim,))
        direction = direction / jnp.linalg.norm(direction)
        
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        def dummy_constraint_fn(x):
            return jnp.array([])
        
        new_state, info = kernel(
            self.key, state, self.logdensity_normal, direction,
            dummy_constraint_fn, jnp.array([]), jnp.array([])
        )
        
        chex.assert_shape(new_state.position, (ndim,))
        self.assertIsInstance(new_state.logdensity, (float, jax.Array))
        self.assertIsInstance(info, ss.SliceInfo)

    def test_1d_slice_sampling(self):
        """Test 1D slice sampling (edge case for JAX shapes)."""
        position = jnp.array(0.5)  # 1D scalar
        state = ss.init(position, lambda x: -0.5 * x**2)
        
        direction = jnp.array(1.0)  # 1D direction
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        def dummy_constraint_fn(x):
            return jnp.array([])
        
        new_state, info = kernel(
            self.key, state, lambda x: -0.5 * x**2, direction,
            dummy_constraint_fn, jnp.array([]), jnp.array([])
        )
        
        # Check it runs without shape errors
        self.assertIsInstance(new_state.logdensity, (float, jax.Array))
        self.assertIsInstance(info.evals, (int, jax.Array))

    def test_default_stepper_fn(self):
        """Test default stepper function."""
        x = jnp.array([1.0, 2.0, -1.5])
        d = jnp.array([0.5, -0.3, 0.8])
        t = 2.5
        
        result = ss.default_stepper_fn(x, d, t)
        expected = x + t * d
        
        chex.assert_trees_all_close(result, expected)

    def test_stepper_fn_with_pytrees(self):
        """Test stepper function with PyTree inputs."""
        x = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([-0.5])}
        d = {"a": jnp.array([0.3, -0.2]), "b": jnp.array([0.7])}
        t = 1.5
        
        result = ss.default_stepper_fn(x, d, t)
        
        chex.assert_trees_all_close(result["a"], x["a"] + t * d["a"])
        chex.assert_trees_all_close(result["b"], x["b"] + t * d["b"])


class SliceSamplingConstraintsTest(chex.TestCase):
    """Test slice sampling with constraints."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(123)

    def test_constrained_sampling(self):
        """Test slice sampling respects constraints."""
        # Start in valid region (x > 0)
        position = jnp.array([1.0, 2.0])
        
        def constrained_logdens(x):
            return jnp.where(jnp.all(x > 0), -0.5 * jnp.sum(x**2), -jnp.inf)
        
        state = ss.init(position, constrained_logdens)
        direction = jnp.array([1.0, -0.5])  # Could lead outside valid region
        
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        # Test with constraint function
        def constraint_fn(x):
            return x  # Return position values to check > 0
        
        constraint_thresholds = jnp.array([0.0, 0.0])  # Must be > 0
        strict_flags = jnp.array([True, True])  # Strict inequality
        
        new_state, info = kernel(
            self.key, state, constrained_logdens, direction,
            constraint_fn, constraint_thresholds, strict_flags
        )
        
        # Should remain in valid region
        self.assertTrue(jnp.all(new_state.position > 0))
        self.assertFalse(jnp.isneginf(new_state.logdensity))

    def test_constraint_evaluation_ordering(self):
        """Test that constraints are evaluated correctly."""
        position = jnp.array([0.5])
        
        def logdens(x):
            return -0.5 * x**2
        
        state = ss.init(position, logdens)
        direction = jnp.array([1.0])
        
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        # Constraint that evaluates a simple function
        def constraint_fn(x):
            return jnp.array([x[0]**2])  # Square of position
        
        constraint_threshold = jnp.array([0.25])  # x^2 > 0.25, so |x| > 0.5
        strict_flag = jnp.array([True])
        
        new_state, info = kernel(
            self.key, state, logdens, direction,
            constraint_fn, constraint_threshold, strict_flag
        )
        
        # Check constraint is satisfied
        constraint_val = constraint_fn(new_state.position)
        self.assertTrue(jnp.all(constraint_val > constraint_threshold))

    def test_multiple_constraints(self):
        """Test multiple constraints simultaneously."""
        position = jnp.array([1.0, 1.5])
        
        def logdens(x):
            return -0.5 * jnp.sum(x**2)
        
        state = ss.init(position, logdens)
        direction = jnp.array([0.7, -0.3])
        
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        def constraint_fn(x):
            return jnp.array([x[0], x[1], jnp.sum(x)])  # Multiple constraints
        
        constraints = jnp.array([0.2, 0.1, 1.0])  # x[0] > 0.2, x[1] > 0.1, sum > 1.0
        strict = jnp.array([True, True, False])  # Mixed strict/non-strict
        
        new_state, info = kernel(
            self.key, state, logdens, direction,
            constraint_fn, constraints, strict
        )
        
        # Check all constraints are satisfied
        constraint_vals = constraint_fn(new_state.position)
        self.assertTrue(constraint_vals[0] > constraints[0])  # Strict
        self.assertTrue(constraint_vals[1] > constraints[1])  # Strict  
        self.assertTrue(constraint_vals[2] >= constraints[2])  # Non-strict


class HitAndRunSliceSamplingTest(chex.TestCase):
    """Test Hit-and-Run Slice Sampling functionality."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(456)

    def logdensity_normal(self, x):
        return stats.norm.logpdf(x).sum()

    def test_direction_generation_from_covariance(self):
        """Test direction generation from covariance matrix."""
        ndim = 3
        cov = jnp.array([[2.0, 0.5, 0.0],
                         [0.5, 1.5, -0.3],
                         [0.0, -0.3, 1.0]])
        
        direction = ss.sample_direction_from_covariance(self.key, cov)
        
        chex.assert_shape(direction, (ndim,))
        
        # Check Mahalanobis normalization
        invcov = jnp.linalg.inv(cov)
        mahal_norm = jnp.sqrt(jnp.einsum("i,ij,j", direction, invcov, direction))
        chex.assert_trees_all_close(mahal_norm, 1.0, atol=1e-6)

    def test_direction_generation_identity_covariance(self):
        """Test direction generation with identity covariance."""
        ndim = 4
        cov = jnp.eye(ndim)
        
        direction = ss.sample_direction_from_covariance(self.key, cov)
        
        chex.assert_shape(direction, (ndim,))
        
        # With identity covariance, should be unit normalized
        euclidean_norm = jnp.linalg.norm(direction)
        chex.assert_trees_all_close(euclidean_norm, 1.0, atol=1e-6)

    def test_hrss_kernel_construction(self):
        """Test HRSS kernel construction."""
        def direction_fn(rng_key):
            return jax.random.normal(rng_key, (2,))
        
        kernel = ss.build_hrss_kernel(direction_fn, ss.default_stepper_fn)
        
        self.assertTrue(callable(kernel))
        
        # Test kernel execution
        position = jnp.array([0.0, 1.0])
        state = ss.init(position, self.logdensity_normal)
        
        new_state, info = kernel(self.key, state, self.logdensity_normal)
        
        chex.assert_shape(new_state.position, (2,))
        self.assertIsInstance(info, ss.SliceInfo)

    def test_hrss_top_level_api(self):
        """Test hit-and-run slice sampling top-level API."""
        ndim = 2
        cov = jnp.eye(ndim) * 1.5
        
        algorithm = ss.hrss_as_top_level_api(self.logdensity_normal, cov)
        
        # Check it's a proper SamplingAlgorithm
        self.assertIsInstance(algorithm, blackjax.base.SamplingAlgorithm)
        self.assertTrue(hasattr(algorithm, "init"))
        self.assertTrue(hasattr(algorithm, "step"))
        
        # Test initialization
        position = jnp.array([1.0, -0.5])
        state = algorithm.init(position)
        
        self.assertIsInstance(state, ss.SliceState)
        chex.assert_trees_all_close(state.position, position)
        
        # Test step
        new_state, info = algorithm.step(self.key, state)
        
        chex.assert_shape(new_state.position, (ndim,))
        self.assertIsInstance(info, ss.SliceInfo)

    def test_hrss_1d_case(self):
        """Test HRSS with 1D problem."""
        cov = jnp.array([[1.0]])  # 1x1 covariance matrix
        
        def logdens_1d(x):
            return -0.5 * x**2
        
        algorithm = ss.hrss_as_top_level_api(logdens_1d, cov)
        
        position = jnp.array([0.5])
        state = algorithm.init(position)
        
        new_state, info = algorithm.step(self.key, state)
        
        chex.assert_shape(new_state.position, (1,))
        self.assertIsInstance(new_state.logdensity, (float, jax.Array))


class SliceSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests for slice sampling."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(789)

    def test_slice_sampling_mean_estimation(self):
        """Test that HRSS correctly estimates mean of target distribution."""
        # Target: standard normal, should have mean ≈ 0
        def logdens(x):
            return stats.norm.logpdf(x).sum()
        
        cov = jnp.eye(1)
        algorithm = ss.hrss_as_top_level_api(logdens, cov)
        
        # Run short chain
        n_samples = 200  # Modest for testing
        position = jnp.array([0.0])
        state = algorithm.init(position)
        
        samples = []
        keys = jax.random.split(self.key, n_samples)
        
        for i, sample_key in enumerate(keys):
            state, info = algorithm.step(sample_key, state)
            if i >= 50:  # Skip some burn-in
                samples.append(state.position[0])
        
        samples = jnp.array(samples)
        
        # Basic sanity checks
        self.assertFalse(jnp.any(jnp.isnan(samples)))
        self.assertFalse(jnp.any(jnp.isinf(samples)))
        
        # Statistical checks (very loose for small sample size)
        sample_mean = jnp.mean(samples)
        sample_std = jnp.std(samples)
        
        # Mean should be reasonable
        self.assertLess(jnp.abs(sample_mean), 0.5)  # Loose bound
        
        # Standard deviation should be positive and reasonable
        self.assertGreater(sample_std, 0.1)
        self.assertLess(sample_std, 3.0)

    def test_slice_sampling_multimodal(self):
        """Test slice sampling on multimodal distribution."""
        def logdens_bimodal(x):
            # Mixture of two Gaussians at -2 and +2
            mode1 = stats.norm.logpdf(x - 2.0)
            mode2 = stats.norm.logpdf(x + 2.0)
            return jnp.logaddexp(mode1, mode2).sum()
        
        cov = jnp.eye(1) * 4.0  # Wider proposals for multimodal
        algorithm = ss.hrss_as_top_level_api(logdens_bimodal, cov)
        
        # Run chain
        n_samples = 100
        position = jnp.array([1.0])  # Start near one mode
        state = algorithm.init(position)
        
        samples = []
        keys = jax.random.split(self.key, n_samples)
        
        for sample_key in keys:
            state, info = algorithm.step(sample_key, state)
            samples.append(state.position[0])
        
        samples = jnp.array(samples)
        
        # Check basic properties
        self.assertFalse(jnp.any(jnp.isnan(samples)))
        sample_range = jnp.max(samples) - jnp.min(samples)
        self.assertGreater(sample_range, 1.0)  # Should explore reasonable range

    def test_slice_info_diagnostics(self):
        """Test that slice info provides useful diagnostics."""
        def logdens(x):
            return -0.5 * jnp.sum(x**2)
        
        cov = jnp.eye(2)
        algorithm = ss.hrss_as_top_level_api(logdens, cov)
        
        position = jnp.array([0.0, 0.0])
        state = algorithm.init(position)
        
        # Collect diagnostics from multiple steps
        infos = []
        keys = jax.random.split(self.key, 20)
        
        for sample_key in keys:
            state, info = algorithm.step(sample_key, state)
            infos.append(info)
        
        # Check diagnostic fields
        l_steps = jnp.array([info.l_steps for info in infos])
        r_steps = jnp.array([info.r_steps for info in infos])
        s_steps = jnp.array([info.s_steps for info in infos])
        evals = jnp.array([info.evals for info in infos])
        
        # All should be non-negative
        self.assertTrue(jnp.all(l_steps >= 0))
        self.assertTrue(jnp.all(r_steps >= 0))
        self.assertTrue(jnp.all(s_steps >= 0))
        self.assertTrue(jnp.all(evals >= 0))
        
        # Total evaluations should be sum of expansion + shrinking
        expected_evals = l_steps + r_steps + s_steps
        chex.assert_trees_all_close(evals, expected_evals)
        
        # Direction vectors should be present
        directions = jnp.array([info.d for info in infos])
        chex.assert_shape(directions, (20, 2))
        self.assertFalse(jnp.any(jnp.isnan(directions)))


class SliceSamplingEdgeCasesTest(chex.TestCase):
    """Test edge cases and robustness."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(101112)

    def test_zero_covariance_matrix(self):
        """Test behavior with singular covariance matrix."""
        # This should handle gracefully or raise informative error
        cov = jnp.zeros((2, 2))
        
        # JAX's linalg.inv will produce NaN/Inf for singular matrices
        # rather than raising an error, so check for that
        try:
            direction = ss.sample_direction_from_covariance(self.key, cov)
            # If it doesn't raise, check for NaN/Inf
            self.assertTrue(jnp.isnan(direction).any() or jnp.isinf(direction).any())
        except (ValueError, RuntimeError):
            # This is also acceptable behavior
            pass

    def test_very_peaked_distribution(self):
        """Test with very peaked/narrow distribution."""
        def logdens_peaked(x):
            return -100.0 * jnp.sum(x**2)  # Very narrow
        
        cov = jnp.eye(1) * 0.01  # Small proposals
        algorithm = ss.hrss_as_top_level_api(logdens_peaked, cov)
        
        position = jnp.array([0.01])
        state = algorithm.init(position)
        
        # Should handle without numerical issues
        new_state, info = algorithm.step(self.key, state)
        
        self.assertFalse(jnp.isnan(new_state.logdensity))
        self.assertFalse(jnp.isinf(new_state.logdensity))

    def test_large_step_proposals(self):
        """Test with very large step proposals."""
        def logdens(x):
            return -0.5 * jnp.sum(x**2)
        
        cov = jnp.eye(1) * 100.0  # Very large proposals
        algorithm = ss.hrss_as_top_level_api(logdens, cov)
        
        position = jnp.array([0.0])
        state = algorithm.init(position)
        
        # Should still work (though possibly inefficient)
        new_state, info = algorithm.step(self.key, state)
        
        self.assertFalse(jnp.isnan(new_state.position).any())
        self.assertGreater(info.evals, 0)  # Should do some work

    def test_empty_constraint_arrays(self):
        """Test with empty constraint arrays."""
        position = jnp.array([1.0])
        state = ss.init(position, lambda x: -0.5 * x**2)
        direction = jnp.array([1.0])
        
        kernel = ss.build_kernel(ss.default_stepper_fn)
        
        def empty_constraint_fn(x):
            return jnp.array([])
        
        # Should handle empty constraints gracefully
        new_state, info = kernel(
            self.key, state, lambda x: -0.5 * x**2, direction,
            empty_constraint_fn, jnp.array([]), jnp.array([])
        )
        
        self.assertIsInstance(new_state, ss.SliceState)
        chex.assert_shape(info.constraint, (0,))


if __name__ == "__main__":
    absltest.main()