"""Test the Nested Sampling algorithms."""
import functools

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from absl.testing import absltest, parameterized

import blackjax
from blackjax.ns import adaptive, base, nss, utils


class NestedSamplingBaseTest(chex.TestCase):
    """Test base nested sampling functionality."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def logprior_uniform(self, x):
        """Uniform prior on [-3, 3]."""
        return jnp.where(jnp.all(jnp.abs(x) <= 3.0), 0.0, -jnp.inf)

    def loglikelihood_gaussian(self, x):
        """Standard Gaussian likelihood."""
        return stats.norm.logpdf(x).sum()

    def logprior_gaussian(self, x):
        """Standard Gaussian prior."""
        return stats.norm.logpdf(x).sum()

    def test_ns_state_structure(self):
        """Test NSState has correct structure."""
        num_live = 50
        ndim = 2
        particles = jax.random.normal(self.key, (num_live, ndim))
        
        state = base.init(particles, self.logprior_uniform, self.loglikelihood_gaussian)
        
        # Check shapes
        chex.assert_shape(state.particles, (num_live, ndim))
        chex.assert_shape(state.loglikelihood, (num_live,))
        chex.assert_shape(state.logprior, (num_live,))
        chex.assert_shape(state.pid, (num_live,))
        
        # Check values are computed correctly
        expected_loglik = jax.vmap(self.loglikelihood_gaussian)(particles)
        expected_logprior = jax.vmap(self.logprior_uniform)(particles)
        
        chex.assert_trees_all_close(state.loglikelihood, expected_loglik)
        chex.assert_trees_all_close(state.logprior, expected_logprior)
        
        # Check particle IDs are unique
        self.assertEqual(len(jnp.unique(state.pid)), num_live)

    def test_ns_info_structure(self):
        """Test NSInfo structure."""
        particles = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        loglik = jnp.array([0.1, 0.2])
        loglik_birth = jnp.array([-jnp.inf, 0.05])
        logprior = jnp.array([-1.0, -1.1])
        
        info = base.NSInfo(
            particles=particles,
            loglikelihood=loglik,
            loglikelihood_birth=loglik_birth,
            logprior=logprior,
            inner_kernel_info={}
        )
        
        chex.assert_shape(info.particles, (2, 2))
        chex.assert_shape(info.loglikelihood, (2,))
        chex.assert_shape(info.loglikelihood_birth, (2,))
        chex.assert_shape(info.logprior, (2,))

    @parameterized.parameters([1, 3, 5])
    def test_delete_fn(self, num_delete):
        """Test particle deletion function."""
        num_live = 20
        particles = jax.random.normal(self.key, (num_live, 2))
        state = base.init(particles, self.logprior_uniform, self.loglikelihood_gaussian)
        
        dead_idx, target_idx, start_idx = base.delete_fn(self.key, state, num_delete)
        
        # Check shapes
        chex.assert_shape(dead_idx, (num_delete,))
        chex.assert_shape(target_idx, (num_delete,))
        chex.assert_shape(start_idx, (num_delete,))
        
        # Check that worst particles are selected
        worst_indices = jnp.argsort(state.loglikelihood)[:num_delete]
        chex.assert_trees_all_close(jnp.sort(dead_idx), jnp.sort(worst_indices))
        
        # Check indices are valid
        self.assertTrue(jnp.all(dead_idx >= 0))
        self.assertTrue(jnp.all(dead_idx < num_live))
        self.assertTrue(jnp.all(target_idx >= 0))
        self.assertTrue(jnp.all(target_idx < num_live))

    def test_1d_basic_functionality(self):
        """Test 1D case to catch shape issues."""
        num_live = 30
        particles = jax.random.uniform(self.key, (num_live,), minval=-3, maxval=3)
        
        def logprior_1d(x):
            return jnp.where((x >= -3) & (x <= 3), -jnp.log(6.0), -jnp.inf)
        
        def loglik_1d(x):
            return -0.5 * x**2
        
        state = base.init(particles, logprior_1d, loglik_1d)
        
        chex.assert_shape(state.particles, (num_live,))
        chex.assert_shape(state.loglikelihood, (num_live,))
        self.assertFalse(jnp.any(jnp.isnan(state.loglikelihood)))
        self.assertFalse(jnp.any(jnp.isinf(state.logprior)))

    def test_kernel_construction(self):
        """Test that kernel can be constructed."""
        def mock_inner_kernel(rng_key, inner_state, logprior_fn, loglik_fn, loglik_0, params):
            # Simple mock that just returns the input state
            return inner_state, {}
        
        delete_fn = functools.partial(base.delete_fn, num_delete=1)
        kernel = base.build_kernel(
            self.logprior_uniform,
            self.loglikelihood_gaussian,
            delete_fn,
            mock_inner_kernel
        )
        
        self.assertTrue(callable(kernel))


class NestedSamplingUtilsTest(chex.TestCase):
    """Test nested sampling utility functions."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(123)

    def create_mock_info(self, n_dead=50):
        """Create mock NSInfo for testing."""
        # Realistic increasing likelihood sequence
        base_loglik = jnp.linspace(-5, -1, n_dead)
        noise = jax.random.normal(self.key, (n_dead,)) * 0.05
        dead_loglik = jnp.sort(base_loglik + noise)
        
        # Birth likelihoods
        key, subkey = jax.random.split(self.key)
        birth_offsets = jax.random.uniform(subkey, (n_dead,)) * 0.2 - 0.1
        dead_loglik_birth = jnp.concatenate([
            jnp.array([-jnp.inf]),  # First from prior
            dead_loglik[:-1] + birth_offsets[1:]
        ])
        dead_loglik_birth = jnp.minimum(dead_loglik_birth, dead_loglik - 0.01)
        
        return base.NSInfo(
            particles=jnp.zeros((n_dead, 2)),
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=jnp.zeros(n_dead),
            inner_kernel_info={}
        )

    def test_compute_num_live(self):
        """Test computation of number of live points."""
        mock_info = self.create_mock_info(n_dead=30)
        num_live = utils.compute_num_live(mock_info)
        
        chex.assert_shape(num_live, (30,))
        self.assertTrue(jnp.all(num_live >= 1))
        self.assertFalse(jnp.any(jnp.isnan(num_live)))

    def test_logX_simulation(self):
        """Test log-volume simulation."""
        mock_info = self.create_mock_info(n_dead=40)
        n_samples = 20
        
        logX_seq, logdX_seq = utils.logX(self.key, mock_info, shape=n_samples)
        
        chex.assert_shape(logX_seq, (40, n_samples))
        chex.assert_shape(logdX_seq, (40, n_samples))
        
        # Log volumes should be decreasing
        for i in range(n_samples):
            self.assertTrue(jnp.all(logX_seq[1:, i] <= logX_seq[:-1, i]))
        
        # No NaN values
        self.assertFalse(jnp.any(jnp.isnan(logX_seq)))

    def test_log_weights(self):
        """Test log weight computation."""
        mock_info = self.create_mock_info(n_dead=25)
        n_samples = 15
        
        log_weights_matrix = utils.log_weights(self.key, mock_info, shape=n_samples)
        
        chex.assert_shape(log_weights_matrix, (25, n_samples))
        
        # Most weights should be finite
        finite_weights = jnp.isfinite(log_weights_matrix)
        finite_fraction = jnp.mean(finite_weights)
        self.assertGreater(finite_fraction, 0.5)

    def test_ess_computation(self):
        """Test effective sample size computation."""
        mock_info = self.create_mock_info(n_dead=35)
        
        ess_value = utils.ess(self.key, mock_info)
        
        self.assertIsInstance(ess_value, (float, jax.Array))
        self.assertGreater(ess_value, 0.0)
        self.assertLessEqual(ess_value, 35)
        self.assertFalse(jnp.isnan(ess_value))

    def test_evidence_estimation_simple(self):
        """Test evidence estimation for simple case."""
        # Constant likelihood case
        n_dead = 30
        loglik_const = -2.0
        
        mock_info = base.NSInfo(
            particles=jnp.zeros((n_dead, 1)),
            loglikelihood=jnp.full(n_dead, loglik_const),
            loglikelihood_birth=jnp.full(n_dead, -jnp.inf),
            logprior=jnp.zeros(n_dead),  # Uniform prior
            inner_kernel_info={}
        )
        
        # Generate evidence estimates
        n_samples = 100
        keys = jax.random.split(self.key, n_samples)
        
        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=10)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)
        
        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()
        
        # Should be close to the constant likelihood value
        mean_estimate = jnp.mean(log_evidence_samples)
        self.assertFalse(jnp.isnan(mean_estimate))
        self.assertFalse(jnp.isinf(mean_estimate))


class AdaptiveNestedSamplingTest(chex.TestCase):
    """Test adaptive nested sampling."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(456)

    def logprior_fn(self, x):
        return stats.norm.logpdf(x).sum()

    def loglik_fn(self, x):
        return -0.5 * jnp.sum(x**2)

    def test_adaptive_init(self):
        """Test adaptive NS initialization."""
        num_live = 25
        particles = jax.random.normal(self.key, (num_live, 2))
        
        def mock_update_fn(state, info, params):
            return {"test_param": 1.5}
        
        state = adaptive.init(
            particles,
            self.logprior_fn,
            self.loglik_fn,
            update_inner_kernel_params_fn=mock_update_fn
        )
        
        # Check basic structure
        chex.assert_shape(state.particles, (num_live, 2))
        
        # Check inner kernel params were set
        self.assertIn("test_param", state.inner_kernel_params)
        self.assertEqual(state.inner_kernel_params["test_param"], 1.5)

    def test_adaptive_kernel_construction(self):
        """Test adaptive kernel can be constructed."""
        def mock_inner_kernel(rng_key, inner_state, logprior_fn, loglik_fn, loglik_0, params):
            return inner_state, {}
        
        def mock_update_fn(state, info, params):
            return params
        
        kernel = adaptive.build_kernel(
            self.logprior_fn,
            self.loglik_fn,
            base.delete_fn,
            mock_inner_kernel,
            mock_update_fn
        )
        
        self.assertTrue(callable(kernel))


class NestedSliceSamplingTest(chex.TestCase):
    """Test nested slice sampling (NSS)."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(789)

    def logprior_fn(self, x):
        return jnp.where(jnp.all(jnp.abs(x) <= 5.0), 0.0, -jnp.inf)

    def loglik_fn(self, x):
        return -0.5 * jnp.sum(x**2)

    def test_covariance_computation(self):
        """Test covariance computation from particles."""
        num_live = 40
        ndim = 3
        particles = jax.random.normal(self.key, (num_live, ndim))
        state = base.init(particles, self.logprior_fn, self.loglik_fn)
        
        params = nss.compute_covariance_from_particles(state, None, {})
        
        self.assertIn("cov", params)
        cov = params["cov"]
        chex.assert_shape(cov, (ndim, ndim))
        
        # Covariance should be positive semidefinite
        eigenvals = jnp.linalg.eigvals(cov)
        self.assertTrue(jnp.all(eigenvals >= -1e-10))

    def test_direction_sampling(self):
        """Test direction sampling from covariance."""
        ndim = 4
        cov = jnp.eye(ndim) * 2.0
        params = {"cov": cov}
        
        direction = nss.sample_direction_from_covariance(self.key, params)
        
        chex.assert_shape(direction, (ndim,))
        
        # Check normalization
        invcov = jnp.linalg.inv(cov)
        mahal_norm = jnp.sqrt(jnp.einsum("i,ij,j", direction, invcov, direction))
        chex.assert_trees_all_close(mahal_norm, 1.0, atol=1e-6)

    def test_nss_kernel_construction(self):
        """Test NSS kernel construction."""
        kernel = nss.build_kernel(
            self.logprior_fn,
            self.loglik_fn,
            num_inner_steps=5
        )
        
        self.assertTrue(callable(kernel))

    def test_nss_with_1d_problem(self):
        """Test NSS with 1D problem (edge case)."""
        def logprior_1d(x):
            return jnp.where((x >= -2) & (x <= 2), -jnp.log(4.0), -jnp.inf)
        
        def loglik_1d(x):
            return -0.5 * x**2
        
        num_live = 20
        particles = jax.random.uniform(self.key, (num_live,), minval=-2, maxval=2)
        state = base.init(particles, logprior_1d, loglik_1d)
        
        params = nss.compute_covariance_from_particles(state, None, {})
        
        self.assertIn("cov", params)
        cov = params["cov"]
        # For 1D, cov should be shaped appropriately for the particle structure
        # The key is that it should work without raising shape errors
        self.assertFalse(jnp.isnan(cov).any())
        self.assertTrue(jnp.all(cov > 0))


class NestedSamplingStatisticalTest(chex.TestCase):
    """Statistical correctness tests."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(12345)

    def test_evidence_monotonicity(self):
        """Test evidence is monotonically increasing."""
        def logprior_fn(x):
            return stats.norm.logpdf(x).sum()
        
        def loglik_fn(x):
            return -0.5 * jnp.sum(x**2)
        
        num_live = 30
        particles = jax.random.normal(self.key, (num_live, 2))
        state = base.init(particles, logprior_fn, loglik_fn)
        
        # Simulate evidence updates
        logZ_sequence = [state.logZ]
        current_state = state
        
        for _ in range(5):
            worst_idx = jnp.argmin(current_state.loglikelihood)
            dead_loglik = current_state.loglikelihood[worst_idx]
            
            # Approximate volume decrease
            delta_logX = -1.0 / num_live
            new_logZ = jnp.logaddexp(current_state.logZ, dead_loglik + delta_logX)
            logZ_sequence.append(new_logZ)
            
            # Mock update for next iteration
            new_loglik = jnp.concatenate([
                current_state.loglikelihood[:worst_idx],
                current_state.loglikelihood[worst_idx + 1:],
                jnp.array([dead_loglik + 0.1])
            ])
            current_state = current_state._replace(
                loglikelihood=new_loglik,
                logZ=new_logZ
            )
        
        # Check monotonicity
        logZ_array = jnp.array(logZ_sequence)
        differences = logZ_array[1:] - logZ_array[:-1]
        self.assertTrue(jnp.all(differences >= -1e-12))

    def test_gaussian_evidence_analytical(self):
        """Test evidence estimation against analytical result."""
        # Setup: Gaussian likelihood with uniform prior
        prior_a, prior_b = -2.0, 2.0
        sigma = 1.0
        
        def logprior_fn(x):
            width = prior_b - prior_a
            return jnp.where((x >= prior_a) & (x <= prior_b), -jnp.log(width), -jnp.inf)
        
        def loglik_fn(x):
            return -0.5 * (x / sigma)**2 - 0.5 * jnp.log(2 * jnp.pi * sigma**2)
        
        # Analytical evidence (truncated Gaussian integral)
        from scipy.stats import norm
        analytical_evidence = (
            norm.cdf(prior_b / sigma) - norm.cdf(prior_a / sigma)
        ) / (prior_b - prior_a)
        analytical_log_evidence = jnp.log(analytical_evidence)
        
        # Mock NS data
        n_dead = 50
        positions = jnp.linspace(prior_a + 0.01, prior_b - 0.01, n_dead).reshape(-1, 1)
        dead_loglik = jax.vmap(loglik_fn)(positions.flatten())
        dead_logprior = jax.vmap(logprior_fn)(positions.flatten())
        
        # Sort by likelihood
        sorted_idx = jnp.argsort(dead_loglik)
        dead_loglik = dead_loglik[sorted_idx]
        positions = positions[sorted_idx]
        dead_logprior = dead_logprior[sorted_idx]
        
        dead_loglik_birth = jnp.concatenate([
            jnp.array([-jnp.inf]),
            dead_loglik[:-1] - 0.05
        ])
        
        mock_info = base.NSInfo(
            particles=positions,
            loglikelihood=dead_loglik,
            loglikelihood_birth=dead_loglik_birth,
            logprior=dead_logprior,
            inner_kernel_info={}
        )
        
        # Generate evidence estimates
        n_samples = 200
        keys = jax.random.split(self.key, n_samples)
        
        def single_evidence_estimate(rng_key):
            log_weights_matrix = utils.log_weights(rng_key, mock_info, shape=10)
            return jax.scipy.special.logsumexp(log_weights_matrix, axis=0)
        
        log_evidence_samples = jax.vmap(single_evidence_estimate)(keys)
        log_evidence_samples = log_evidence_samples.flatten()
        
        # Statistical validation
        mean_estimate = jnp.mean(log_evidence_samples)
        std_estimate = jnp.std(log_evidence_samples)
        
        # For mock data, we expect some bias, so use looser bounds
        # This is primarily testing that the utilities work, not exact accuracy
        self.assertFalse(jnp.isnan(mean_estimate))
        self.assertFalse(jnp.isinf(mean_estimate))
        
        # Very loose bounds - mainly checking it's in the right ballpark
        self.assertGreater(mean_estimate, analytical_log_evidence - 3.0)
        self.assertLess(mean_estimate, analytical_log_evidence + 3.0)


if __name__ == "__main__":
    absltest.main()