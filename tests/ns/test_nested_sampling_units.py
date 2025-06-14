"""Unit tests for nested sampling components."""
import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.ns import base, nss, utils


class NSStateTest(chex.TestCase):
    """Test NSState data structure."""

    def test_ns_state_creation(self):
        """Test NSState creation."""
        particles = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        loglik = jnp.array([0.1, 0.2])
        loglik_birth = jnp.array([-jnp.inf, 0.05])
        logprior = jnp.array([-1.0, -1.1])
        pid = jnp.array([0, 1])
        logX = -2.0
        logZ = -5.0
        logZ_live = -3.0
        inner_kernel_params = {}
        
        state = base.NSState(
            particles=particles,
            loglikelihood=loglik,
            loglikelihood_birth=loglik_birth,
            logprior=logprior,
            pid=pid,
            logX=logX,
            logZ=logZ,
            logZ_live=logZ_live,
            inner_kernel_params=inner_kernel_params
        )
        
        chex.assert_trees_all_close(state.particles, particles)
        chex.assert_trees_all_close(state.loglikelihood, loglik)
        chex.assert_trees_all_close(state.loglikelihood_birth, loglik_birth)
        chex.assert_trees_all_close(state.logprior, logprior)
        chex.assert_trees_all_close(state.pid, pid)
        self.assertEqual(state.logX, logX)
        self.assertEqual(state.logZ, logZ)
        self.assertEqual(state.logZ_live, logZ_live)
        self.assertEqual(state.inner_kernel_params, inner_kernel_params)

    def test_ns_state_replace(self):
        """Test NSState _replace method."""
        state = base.NSState(
            particles=jnp.array([[1.0], [2.0]]),
            loglikelihood=jnp.array([0.1, 0.2]),
            loglikelihood_birth=jnp.array([-jnp.inf, 0.05]),
            logprior=jnp.array([-1.0, -1.1]),
            pid=jnp.array([0, 1]),
            logX=-2.0,
            logZ=-5.0,
            logZ_live=-3.0,
            inner_kernel_params={}
        )
        
        new_logZ = -4.5
        new_state = state._replace(logZ=new_logZ)
        
        self.assertEqual(new_state.logZ, new_logZ)
        self.assertEqual(new_state.logZ_live, -3.0)  # Unchanged
        self.assertEqual(new_state.logX, -2.0)  # Unchanged
        chex.assert_trees_all_close(new_state.particles, state.particles)


class NSInfoTest(chex.TestCase):
    """Test NSInfo data structure."""

    def test_ns_info_creation(self):
        """Test NSInfo creation."""
        particles = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        loglik = jnp.array([0.1, 0.2])
        loglik_birth = jnp.array([-jnp.inf, 0.05])
        logprior = jnp.array([-1.0, -1.1])
        kernel_info = {"test": "value"}
        
        info = base.NSInfo(
            particles=particles,
            loglikelihood=loglik,
            loglikelihood_birth=loglik_birth,
            logprior=logprior,
            inner_kernel_info=kernel_info
        )
        
        chex.assert_trees_all_close(info.particles, particles)
        chex.assert_trees_all_close(info.loglikelihood, loglik)
        chex.assert_trees_all_close(info.loglikelihood_birth, loglik_birth)
        chex.assert_trees_all_close(info.logprior, logprior)
        self.assertEqual(info.inner_kernel_info, kernel_info)


class InitFunctionTest(chex.TestCase):
    """Test NS initialization function."""

    def setUp(self):
        super().setUp()
        self.logprior_fn = lambda x: -0.5 * jnp.sum(x**2)
        self.loglik_fn = lambda x: -jnp.sum(x**2)

    @parameterized.parameters([10, 50, 100])
    def test_init_particle_count(self, num_live):
        """Test initialization with different numbers of live points."""
        particles = jax.random.normal(jax.random.key(42), (num_live, 2))
        
        state = base.init(particles, self.logprior_fn, self.loglik_fn)
        
        chex.assert_shape(state.particles, (num_live, 2))
        chex.assert_shape(state.loglikelihood, (num_live,))
        chex.assert_shape(state.logprior, (num_live,))
        chex.assert_shape(state.pid, (num_live,))

    def test_init_1d_particles(self):
        """Test initialization with 1D particles."""
        num_live = 20
        particles = jax.random.normal(jax.random.key(42), (num_live,))
        
        def logprior_1d(x):
            return -0.5 * x**2
        
        def loglik_1d(x):
            return -x**2
        
        state = base.init(particles, logprior_1d, loglik_1d)
        
        chex.assert_shape(state.particles, (num_live,))
        chex.assert_shape(state.loglikelihood, (num_live,))
        chex.assert_shape(state.logprior, (num_live,))

    def test_init_computes_correct_values(self):
        """Test that init computes loglikelihood and logprior correctly."""
        particles = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        
        state = base.init(particles, self.logprior_fn, self.loglik_fn)
        
        # Check computed values match manual computation
        expected_logprior = jax.vmap(self.logprior_fn)(particles)
        expected_loglik = jax.vmap(self.loglik_fn)(particles)
        
        chex.assert_trees_all_close(state.logprior, expected_logprior)
        chex.assert_trees_all_close(state.loglikelihood, expected_loglik)

    def test_init_particle_ids_unique(self):
        """Test that particle IDs are unique."""
        num_live = 15
        particles = jax.random.normal(jax.random.key(42), (num_live, 3))
        
        state = base.init(particles, self.logprior_fn, self.loglik_fn)
        
        unique_ids = jnp.unique(state.pid)
        self.assertEqual(len(unique_ids), num_live)

    def test_init_with_pytree_particles(self):
        """Test initialization with PyTree particles."""
        num_live = 10
        particles = {
            "x": jax.random.normal(jax.random.key(42), (num_live, 2)),
            "y": jax.random.normal(jax.random.key(43), (num_live,))
        }
        
        def logprior_pytree(p):
            return -0.5 * (jnp.sum(p["x"]**2) + p["y"]**2)
        
        def loglik_pytree(p):
            return -(jnp.sum(p["x"]**2) + p["y"]**2)
        
        state = base.init(particles, logprior_pytree, loglik_pytree)
        
        chex.assert_shape(state.particles["x"], (num_live, 2))
        chex.assert_shape(state.particles["y"], (num_live,))
        chex.assert_shape(state.loglikelihood, (num_live,))


class DeleteFunctionTest(chex.TestCase):
    """Test particle deletion function."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def create_test_state(self, num_live=20):
        """Helper to create test state."""
        particles = jax.random.normal(self.key, (num_live, 2))
        logprior_fn = lambda x: -0.5 * jnp.sum(x**2)
        loglik_fn = lambda x: -jnp.sum(x**2)
        return base.init(particles, logprior_fn, loglik_fn)

    @parameterized.parameters([1, 3, 5, 10])
    def test_delete_fn_shapes(self, num_delete):
        """Test delete function returns correct shapes."""
        state = self.create_test_state(num_live=20)
        
        dead_idx, target_idx, start_idx = base.delete_fn(self.key, state, num_delete)
        
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))
        chex.assert_shape(start_idx, (num_delete,))

    def test_delete_fn_selects_worst(self):
        """Test that delete function selects worst particles."""
        state = self.create_test_state(num_live=20)
        num_delete = 3
        
        dead_idx, _, _ = base.delete_fn(self.key, state, num_delete)
        
        # Should select particles with lowest likelihood
        worst_indices = jnp.argsort(state.loglikelihood)[:num_delete]
        selected_indices = jnp.sort(dead_idx)
        expected_indices = jnp.sort(worst_indices)
        
        chex.assert_trees_all_close(selected_indices, expected_indices)

    def test_delete_fn_valid_indices(self):
        """Test that delete function returns valid indices."""
        num_live = 15
        state = self.create_test_state(num_live=num_live)
        num_delete = 4
        
        dead_idx, target_idx, start_idx = base.delete_fn(self.key, state, num_delete)
        
        # All indices should be valid
        self.assertTrue(jnp.all(dead_idx >= 0))
        self.assertTrue(jnp.all(dead_idx < num_live))
        self.assertTrue(jnp.all(target_idx >= 0))
        self.assertTrue(jnp.all(target_idx < num_live))
        self.assertTrue(jnp.all(start_idx >= 0))
        self.assertTrue(jnp.all(start_idx < num_live))

    def test_delete_fn_no_duplicates(self):
        """Test that delete function doesn't return duplicate indices."""
        state = self.create_test_state(num_live=20)
        num_delete = 5
        
        dead_idx, target_idx, start_idx = base.delete_fn(self.key, state, num_delete)
        
        # Dead indices should be unique
        self.assertEqual(len(jnp.unique(dead_idx)), num_delete)


class NSKernelExecutionTest(chex.TestCase):
    """Test full NS kernel execution."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(999)

    def test_kernel_full_execution(self):
        """Test full NS kernel execution workflow."""
        # Create a simple mock inner kernel
        def mock_inner_kernel(rng_key, inner_state, logprior_fn, loglik_fn, loglik_0, params):
            # Simple random walk that respects the likelihood constraint
            pos = inner_state.position
            new_pos = pos + jax.random.normal(rng_key, pos.shape) * 0.1
            new_loglik = loglik_fn(new_pos)
            new_logprior = logprior_fn(new_pos)
            
            # Accept if likelihood is above threshold, otherwise return original
            accept = new_loglik >= loglik_0
            final_pos = jnp.where(accept, new_pos, pos)
            final_loglik = jnp.where(accept, new_loglik, inner_state.loglikelihood)
            final_logprior = jnp.where(accept, new_logprior, inner_state.logprior)
            
            new_inner_state = base.PartitionedState(final_pos, final_logprior, final_loglik)
            return new_inner_state, {"accepted": accept}

        # Set up test functions
        def logprior_fn(x):
            return -0.5 * jnp.sum(x**2)
        
        def loglik_fn(x):
            return -jnp.sum(x**2)
        
        # Create initial state
        num_live = 10
        particles = jax.random.normal(self.key, (num_live, 2)) * 0.5
        state = base.init(particles, logprior_fn, loglik_fn)
        
        # Build kernel with delete function
        def delete_fn(rng_key, state):
            # Delete 1 worst particle
            dead_idx = jnp.array([jnp.argmin(state.loglikelihood)])
            target_idx = jnp.array([0])  # Replace with first particle
            start_idx = jnp.array([0])   # Start from first particle
            return dead_idx, target_idx, start_idx
        
        kernel = base.build_kernel(logprior_fn, loglik_fn, delete_fn, mock_inner_kernel)
        
        # Execute kernel
        new_state, info = kernel(self.key, state)
        
        # Check that state is updated correctly
        self.assertIsInstance(new_state, base.NSState)
        self.assertIsInstance(info, base.NSInfo)
        
        # Should still have same number of particles
        chex.assert_shape(new_state.particles, (num_live, 2))
        chex.assert_shape(new_state.loglikelihood, (num_live,))
        
        # Evidence should be updated
        self.assertNotEqual(new_state.logZ, state.logZ)
        
        # Info should contain dead particle information
        chex.assert_shape(info.particles, (1, 2))  # 1 dead particle
        chex.assert_shape(info.loglikelihood, (1,))


class RuntimeInfoUpdateTest(chex.TestCase):
    """Test runtime info update functions."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(555)

    def test_update_ns_runtime_info(self):
        """Test update_ns_runtime_info function."""
        # Test data
        logX = -2.0
        logZ = -5.0
        loglikelihood = jnp.array([-1.0, -1.5, -2.0, -2.5])  # Live points
        dead_loglikelihood = jnp.array([-3.0, -3.2])  # Dead points
        
        new_logX, new_logZ, new_logZ_live = base.update_ns_runtime_info(
            logX, logZ, loglikelihood, dead_loglikelihood
        )
        
        # Check types and finiteness
        self.assertIsInstance(new_logX, (float, jax.Array))
        self.assertIsInstance(new_logZ, (float, jax.Array))
        self.assertIsInstance(new_logZ_live, (float, jax.Array))
        
        self.assertFalse(jnp.isnan(new_logX))
        self.assertFalse(jnp.isnan(new_logZ))
        self.assertFalse(jnp.isnan(new_logZ_live))
        
        # Evidence should increase (or at least not decrease significantly)
        self.assertGreaterEqual(new_logZ, logZ - 1e-10)
        
        # LogX should decrease (volume shrinking)
        self.assertLess(new_logX, logX)

    def test_update_ns_runtime_info_single_particle(self):
        """Test runtime update with single particle deletion."""
        logX = -1.0
        logZ = -10.0
        loglikelihood = jnp.array([-2.0, -2.5, -3.0])
        dead_loglikelihood = jnp.array([-4.0])  # Single deletion
        
        new_logX, new_logZ, new_logZ_live = base.update_ns_runtime_info(
            logX, logZ, loglikelihood, dead_loglikelihood
        )
        
        # Should work with single particle
        self.assertFalse(jnp.isnan(new_logX))
        self.assertFalse(jnp.isnan(new_logZ))
        self.assertFalse(jnp.isnan(new_logZ_live))


class PartitionedStateTest(chex.TestCase):
    """Test PartitionedState and PartitionedInfo structures."""

    def test_new_state_and_info(self):
        """Test new_state_and_info function."""
        position = jnp.array([1.0, 2.0])
        logprior = -1.5
        loglikelihood = -2.0
        info = {"test": "value"}
        
        state, returned_info = base.new_state_and_info(
            position, logprior, loglikelihood, info
        )
        
        # Check PartitionedState
        self.assertIsInstance(state, base.PartitionedState)
        chex.assert_trees_all_close(state.position, position)
        self.assertEqual(state.logprior, logprior)
        self.assertEqual(state.loglikelihood, loglikelihood)
        
        # Check PartitionedInfo
        self.assertIsInstance(returned_info, base.PartitionedInfo)
        chex.assert_trees_all_close(returned_info.position, position)
        self.assertEqual(returned_info.logprior, logprior)
        self.assertEqual(returned_info.loglikelihood, loglikelihood)
        self.assertEqual(returned_info.info, info)


class UtilityFunctionsTest(chex.TestCase):
    """Test utility functions."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(123)

    def create_mock_info(self, n_dead=30):
        """Helper to create mock NSInfo."""
        # Increasing likelihood sequence
        loglik = jnp.linspace(-5, -1, n_dead)
        loglik_birth = jnp.concatenate([
            jnp.array([-jnp.inf]),
            loglik[:-1] - 0.1
        ])
        
        return base.NSInfo(
            particles=jnp.zeros((n_dead, 2)),
            loglikelihood=loglik,
            loglikelihood_birth=loglik_birth,
            logprior=jnp.zeros(n_dead),
            inner_kernel_info={}
        )

    def test_get_first_row_array(self):
        """Test get_first_row with arrays."""
        x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        result = utils.get_first_row(x)
        expected = jnp.array([1, 2, 3])
        
        chex.assert_trees_all_close(result, expected)

    def test_get_first_row_pytree(self):
        """Test get_first_row with PyTree."""
        x = {
            "a": jnp.array([[1, 2], [3, 4], [5, 6]]),
            "b": jnp.array([10, 20, 30])
        }
        
        result = utils.get_first_row(x)
        
        chex.assert_trees_all_close(result["a"], jnp.array([1, 2]))
        self.assertEqual(result["b"], 10)

    def test_compute_num_live_shape(self):
        """Test compute_num_live returns correct shape."""
        mock_info = self.create_mock_info(n_dead=25)
        
        num_live = utils.compute_num_live(mock_info)
        
        chex.assert_shape(num_live, (25,))

    def test_compute_num_live_values(self):
        """Test compute_num_live returns reasonable values."""
        mock_info = self.create_mock_info(n_dead=20)
        
        num_live = utils.compute_num_live(mock_info)
        
        # Should be positive
        self.assertTrue(jnp.all(num_live >= 1))
        # Should be reasonable (not too large)
        self.assertTrue(jnp.all(num_live <= 1000))
        # Should not be NaN
        self.assertFalse(jnp.any(jnp.isnan(num_live)))

    def test_logX_shapes(self):
        """Test logX returns correct shapes."""
        mock_info = self.create_mock_info(n_dead=15)
        n_samples = 10
        
        logX_seq, logdX_seq = utils.logX(self.key, mock_info, shape=n_samples)
        
        chex.assert_shape(logX_seq, (15, n_samples))
        chex.assert_shape(logdX_seq, (15, n_samples))

    def test_logX_monotonicity(self):
        """Test that logX is decreasing."""
        mock_info = self.create_mock_info(n_dead=10)
        n_samples = 5
        
        logX_seq, _ = utils.logX(self.key, mock_info, shape=n_samples)
        
        # Each column should be decreasing
        for i in range(n_samples):
            differences = logX_seq[1:, i] - logX_seq[:-1, i]
            self.assertTrue(jnp.all(differences <= 1e-12))  # Allow for numerical precision

    def test_log_weights_shapes(self):
        """Test log_weights returns correct shape."""
        mock_info = self.create_mock_info(n_dead=12)
        n_samples = 8
        
        log_weights = utils.log_weights(self.key, mock_info, shape=n_samples)
        
        chex.assert_shape(log_weights, (12, n_samples))

    def test_log_weights_finite(self):
        """Test that most log_weights are finite."""
        mock_info = self.create_mock_info(n_dead=20)
        n_samples = 5
        
        log_weights = utils.log_weights(self.key, mock_info, shape=n_samples)
        
        # Most weights should be finite
        finite_fraction = jnp.mean(jnp.isfinite(log_weights))
        self.assertGreater(finite_fraction, 0.3)  # At least 30% should be finite

    def test_ess_properties(self):
        """Test ESS computation properties."""
        mock_info = self.create_mock_info(n_dead=30)
        
        ess = utils.ess(self.key, mock_info)
        
        # ESS should be positive and finite
        self.assertGreater(ess, 0.0)
        self.assertFalse(jnp.isnan(ess))
        self.assertFalse(jnp.isinf(ess))
        # ESS should not exceed number of samples
        self.assertLessEqual(ess, 30)

    def test_log1mexp_values(self):
        """Test log1mexp utility function."""
        # Test values where we know the expected result
        x = jnp.array([-0.1, -1.0, -2.0, -10.0])
        
        result = utils.log1mexp(x)
        
        # Should all be finite and negative (since log(1-exp(x)) < 0 for x < 0)
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        # For large negative x, log(1-exp(x)) ≈ log(1) = 0
        self.assertAlmostEqual(result[-1], 0.0, places=3)  # Less strict for numerical precision

    def test_log1mexp_edge_cases(self):
        """Test log1mexp edge cases."""
        # Test near the transition point
        x_transition = jnp.array([-0.6931472])  # Approximately -log(2)
        
        result = utils.log1mexp(x_transition)
        
        self.assertTrue(jnp.isfinite(result))
        self.assertLess(result, 0.0)


class NSSAdvancedTest(chex.TestCase):
    """Test NSS advanced functionality and missing coverage."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(777)

    def test_nss_top_level_api(self):
        """Test NSS as_top_level_api function."""
        def logprior_fn(x):
            return -0.5 * jnp.sum(x**2)
        
        def loglik_fn(x):
            return -jnp.sum(x**2)
        
        num_live = 20
        
        # Test the top-level API
        algorithm = nss.as_top_level_api(
            logprior_fn,
            loglik_fn,
            5  # num_inner_steps
        )
        
        # Should return a SamplingAlgorithm
        self.assertTrue(hasattr(algorithm, "init"))
        self.assertTrue(hasattr(algorithm, "step"))
        self.assertTrue(callable(algorithm.init))
        self.assertTrue(callable(algorithm.step))
        
        # Test initialization - NSS uses adaptive.init which needs different signature
        particles = jax.random.normal(self.key, (num_live, 2))
        state = algorithm.init(particles)
        
        self.assertIsInstance(state, base.NSState)
        chex.assert_shape(state.particles, (num_live, 2))

    def test_nss_inner_kernel_execution(self):
        """Test NSS inner kernel execution by building a full kernel."""
        def logprior_fn(x):
            return -0.5 * jnp.sum(x**2)
        
        def loglik_fn(x):
            return -jnp.sum(x**2)
        
        # Build NSS kernel
        kernel = nss.build_kernel(logprior_fn, loglik_fn, num_inner_steps=2)
        
        # Create initial state with proper inner_kernel_params
        num_live = 5
        particles = jax.random.normal(self.key, (num_live, 2)) * 0.3
        state = base.init(particles, logprior_fn, loglik_fn)
        # NSS needs covariance params
        cov_params = nss.compute_covariance_from_particles(state, None, {})
        state = state._replace(inner_kernel_params=cov_params)
        
        # Execute kernel - this tests the inner kernel execution paths
        new_state, info = kernel(self.key, state)
        
        # Check that state is updated correctly
        self.assertIsInstance(new_state, base.NSState)
        self.assertIsInstance(info, base.NSInfo)
        
        # Should still have same number of particles
        chex.assert_shape(new_state.particles, (num_live, 2))
        chex.assert_shape(new_state.loglikelihood, (num_live,))
        
        # Evidence should be updated
        self.assertNotEqual(new_state.logZ, state.logZ)

    def test_nss_compute_covariance_edge_cases(self):
        """Test covariance computation edge cases."""
        # Test with very few particles
        num_live = 3
        particles = jnp.array([[1.0], [2.0], [3.0]])  # 1D particles
        
        def logprior_fn(x):
            return -0.5 * x**2
        
        def loglik_fn(x):
            return -x**2
        
        state = base.init(particles, logprior_fn, loglik_fn)
        
        # Should handle small number of particles
        params = nss.compute_covariance_from_particles(state, None, {})
        
        self.assertIn("cov", params)
        cov = params["cov"]
        
        # Should not be NaN or infinite
        self.assertFalse(jnp.isnan(cov).any())
        self.assertFalse(jnp.isinf(cov).any())

    def test_nss_direction_sampling_edge_cases(self):
        """Test direction sampling edge cases."""
        # Test with nearly singular covariance
        cov = jnp.array([[1e-6, 0.0], [0.0, 1e-6]])
        params = {"cov": cov}
        
        direction = nss.sample_direction_from_covariance(self.key, params)
        
        chex.assert_shape(direction, (2,))
        # Should be finite even with small covariance
        self.assertTrue(jnp.all(jnp.isfinite(direction)))


class MissingUtilityFunctionsTest(chex.TestCase):
    """Test utility functions that were missed in coverage."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(888)

    def test_missing_utility_functions(self):
        """Test utility functions that weren't covered."""
        # Create test data that would exercise missing lines
        
        # Test with edge case data - use float dtypes
        mock_info = base.NSInfo(
            particles=jnp.zeros((5, 1)),
            loglikelihood=jnp.array([-10.0, -5.0, -3.0, -2.0, -1.0]),  # Wide range, float
            loglikelihood_birth=jnp.array([-jnp.inf, -15.0, -8.0, -4.0, -2.5]),
            logprior=jnp.zeros(5),
            inner_kernel_info={}
        )
        
        # Test functions that might have missing coverage
        num_live = utils.compute_num_live(mock_info)
        self.assertTrue(jnp.all(num_live >= 1))
        
        # Test with different shapes
        logX_seq, logdX_seq = utils.logX(self.key, mock_info, shape=3)
        chex.assert_shape(logX_seq, (5, 3))
        
        # Test log_weights with edge cases
        log_weights = utils.log_weights(self.key, mock_info, shape=2)
        chex.assert_shape(log_weights, (5, 2))

    def test_repeat_kernel_decorator(self):
        """Test repeat_kernel decorator function."""
        # Simple mock kernel
        @utils.repeat_kernel(3)
        def mock_kernel(rng_key, state, *args):
            # Just update position slightly
            new_pos = state["position"] + jax.random.normal(rng_key, state["position"].shape) * 0.01
            new_state = state.copy()
            new_state["position"] = new_pos
            return new_state, {"step": 1}
        
        initial_state = {"position": jnp.array([1.0, 2.0])}
        
        # Test decorated kernel
        final_state, infos = mock_kernel(self.key, initial_state)
        
        # Should have run 3 times (scan packs infos into dict structure)
        self.assertIsInstance(final_state, dict)
        self.assertIn("position", final_state)
        chex.assert_shape(final_state["position"], (2,))
        
        # Info structure depends on how scan handles the dict
        self.assertIsInstance(infos, dict)
        self.assertIn("step", infos)
        chex.assert_shape(infos["step"], (3,))  # 3 steps recorded


class CompleteCoverageTest(chex.TestCase):
    """Tests to achieve 100% coverage on remaining uncovered lines."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_combine_dead_info(self):
        """Test combine_dead_info function (utils.py lines 233-247)."""
        # Create mock dead info list and live state
        dead1 = base.NSInfo(
            particles=jnp.array([[1.0], [2.0]]),
            loglikelihood=jnp.array([0.1, 0.2]),
            loglikelihood_birth=jnp.array([-jnp.inf, 0.05]),
            logprior=jnp.array([-1.0, -1.1]),
            inner_kernel_info={"test": "value1"}
        )
        
        dead2 = base.NSInfo(
            particles=jnp.array([[3.0], [4.0]]),
            loglikelihood=jnp.array([0.3, 0.4]),
            loglikelihood_birth=jnp.array([0.25, 0.35]),
            logprior=jnp.array([-1.2, -1.3]),
            inner_kernel_info={"test": "value2"}
        )
        
        # Mock live state
        live = base.NSState(
            particles=jnp.array([[5.0], [6.0]]),
            loglikelihood=jnp.array([0.5, 0.6]),
            loglikelihood_birth=jnp.array([0.45, 0.55]),
            logprior=jnp.array([-1.4, -1.5]),
            pid=jnp.array([4, 5]),
            logX=-2.0,
            logZ=-5.0,
            logZ_live=-3.0,
            inner_kernel_params={}
        )
        
        combined = utils.combine_dead_info([dead1, dead2], live)
        
        # Should combine all dead + live particles
        expected_total = 2 + 2 + 2  # dead1 + dead2 + live
        chex.assert_shape(combined.particles, (expected_total, 1))
        chex.assert_shape(combined.loglikelihood, (expected_total,))

    def test_sample_particles(self):
        """Test sample_particles function (utils.py lines 303-311)."""
        # Create mock dead info
        mock_info = base.NSInfo(
            particles=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
            loglikelihood=jnp.array([0.1, 0.3, 0.2, 0.4]),
            loglikelihood_birth=jnp.array([-jnp.inf, 0.05, 0.15, 0.25]),
            logprior=jnp.array([-1.0, -1.1, -1.2, -1.3]),
            inner_kernel_info={}
        )
        
        # Sample particles
        sampled = utils.sample_particles(self.key, mock_info, shape=6)
        
        chex.assert_shape(sampled, (6, 1))

    def test_uniform_prior_function(self):
        """Test uniform_prior function (utils.py lines 381-398)."""
        bounds = {"x": (-2.0, 2.0), "y": (0.0, 1.0)}
        num_live = 10
        
        particles, logprior_fn = utils.uniform_prior(self.key, bounds, num_live)
        
        # Check particles structure
        self.assertIn("x", particles)
        self.assertIn("y", particles)
        chex.assert_shape(particles["x"], (num_live,))
        chex.assert_shape(particles["y"], (num_live,))
        
        # Check bounds are respected
        self.assertTrue(jnp.all(particles["x"] >= -2.0))
        self.assertTrue(jnp.all(particles["x"] <= 2.0))
        self.assertTrue(jnp.all(particles["y"] >= 0.0))
        self.assertTrue(jnp.all(particles["y"] <= 1.0))
        
        # Test logprior function
        test_params = {"x": 0.5, "y": 0.3}
        logprior_val = logprior_fn(test_params)
        self.assertIsInstance(logprior_val, (float, jax.Array))
        self.assertTrue(jnp.isfinite(logprior_val))

    def test_adaptive_kernel_missing_lines(self):
        """Test adaptive kernel missing coverage (adaptive.py lines 148-154)."""
        from blackjax.ns import adaptive
        
        def logprior_fn(x):
            return -0.5 * jnp.sum(x**2)
        
        def loglik_fn(x):
            return -jnp.sum(x**2)
            
        def mock_inner_kernel(rng_key, inner_state, logprior_fn, loglik_fn, loglik_0, params):
            # Mock that always accepts
            new_inner_state = base.PartitionedState(
                inner_state.position + 0.01,
                inner_state.logprior,
                inner_state.loglikelihood + 0.1
            )
            return new_inner_state, {"accepted": True}
        
        def mock_update_fn(state, info, params):
            # This should exercise the missing lines in adaptive.py
            return {"updated": True, "cov": jnp.eye(2)}
        
        # Create state
        num_live = 5
        particles = jax.random.normal(self.key, (num_live, 2)) * 0.1
        state = base.init(particles, logprior_fn, loglik_fn)
        
        # Build adaptive kernel
        delete_fn = functools.partial(base.delete_fn, num_delete=1)
        kernel = adaptive.build_kernel(
            logprior_fn, loglik_fn, delete_fn, mock_inner_kernel, mock_update_fn
        )
        
        # Execute to test missing lines
        new_state, info = kernel(self.key, state)
        
        self.assertIsInstance(new_state, base.NSState)
        # Check that update function was called (adaptive logic)
        self.assertIn("updated", new_state.inner_kernel_params)


if __name__ == "__main__":
    absltest.main()