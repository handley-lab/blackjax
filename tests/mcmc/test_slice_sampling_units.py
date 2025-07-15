"""Unit tests for slice sampling components."""
import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from blackjax.mcmc import ss


class SliceStateTest(chex.TestCase):
    """Test SliceState data structure."""

    def test_slice_state_creation(self):
        """Test SliceState creation and default values."""
        position = jnp.array([1.0, 2.0])
        logdensity = -3.5

        # Test with default logslice
        state = ss.SliceState(position, logdensity)
        chex.assert_trees_all_close(state.position, position)
        self.assertEqual(state.logdensity, logdensity)
        self.assertEqual(state.logslice, jnp.inf)

        # Test with explicit logslice
        logslice = -1.2
        state = ss.SliceState(position, logdensity, logslice)
        self.assertEqual(state.logslice, logslice)

    def test_slice_state_replace(self):
        """Test SliceState _replace method."""
        state = ss.SliceState(jnp.array([1.0]), -2.0, -5.0)

        new_state = state._replace(logslice=-3.0)
        self.assertEqual(new_state.logslice, -3.0)
        self.assertEqual(new_state.logdensity, -2.0)  # Unchanged
        chex.assert_trees_all_close(new_state.position, state.position)


class SliceInfoTest(chex.TestCase):
    """Test SliceInfo data structure."""

    def test_slice_info_creation(self):
        """Test SliceInfo creation and default values."""
        # Test with defaults
        info = ss.SliceInfo()
        chex.assert_shape(info.constraint, (0,))
        self.assertEqual(info.l_steps, 0)
        self.assertEqual(info.r_steps, 0)
        self.assertEqual(info.s_steps, 0)
        self.assertEqual(info.evals, 0)
        self.assertIsNone(info.d)

        # Test with explicit values
        constraint = jnp.array([1.0, 2.0])
        direction = jnp.array([0.5, -0.3])
        info = ss.SliceInfo(
            constraint=constraint,
            l_steps=3,
            r_steps=5,
            s_steps=7,
            evals=15,
            d=direction,
        )
        chex.assert_trees_all_close(info.constraint, constraint)
        self.assertEqual(info.l_steps, 3)
        self.assertEqual(info.r_steps, 5)
        self.assertEqual(info.s_steps, 7)
        self.assertEqual(info.evals, 15)
        chex.assert_trees_all_close(info.d, direction)


class InitFunctionTest(chex.TestCase):
    """Test slice sampling initialization."""

    def setUp(self):
        super().setUp()
        self.logdensity_fn = lambda x: -0.5 * jnp.sum(x**2)

    @parameterized.parameters(
        [
            (jnp.array([0.0]),),
            (jnp.array([1.5, -2.0]),),
            (jnp.array([[1.0, 2.0], [3.0, 4.0]]),),
        ]
    )
    def test_init_shapes(self, position):
        """Test init with different position shapes."""
        state = ss.init(position, self.logdensity_fn)

        chex.assert_trees_all_close(state.position, position)
        expected_logdens = self.logdensity_fn(position)
        chex.assert_trees_all_close(state.logdensity, expected_logdens)
        self.assertEqual(state.logslice, jnp.inf)

    def test_init_with_pytree(self):
        """Test init with PyTree position."""
        position = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}

        def logdens_pytree(x):
            return -0.5 * (jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2))

        state = ss.init(position, logdens_pytree)

        chex.assert_trees_all_close(state.position, position)
        expected_logdens = logdens_pytree(position)
        self.assertEqual(state.logdensity, expected_logdens)


class VerticalSliceTest(chex.TestCase):
    """Test vertical slice function."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(42)

    def test_vertical_slice_height_bounds(self):
        """Test that slice height is always below current logdensity."""
        position = jnp.array([0.0])
        logdensity = -1.5
        state = ss.SliceState(position, logdensity)

        # Test multiple samples
        keys = jax.random.split(self.key, 100)
        new_states, infos = jax.vmap(ss.vertical_slice, in_axes=(0, None))(keys, state)

        # All slice heights should be <= logdensity
        self.assertTrue(jnp.all(new_states.logslice <= logdensity))

        # Info should have zero evaluations (vertical slice doesn't eval logdensity)
        self.assertTrue(jnp.all(infos.evals == 0))

    def test_vertical_slice_deterministic_bound(self):
        """Test that slice height has correct statistical properties."""
        position = jnp.array([0.0])
        logdensity = -2.0
        state = ss.SliceState(position, logdensity)

        # Generate many samples
        n_samples = 5000
        keys = jax.random.split(self.key, n_samples)
        new_states, _ = jax.vmap(ss.vertical_slice, in_axes=(0, None))(keys, state)

        # Mean of log(U) where U ~ Uniform(0,1) is -1
        mean_height = jnp.mean(new_states.logslice)
        expected_mean = logdensity - 1.0

        # Should be close to expected mean (loose tolerance for finite sample)
        self.assertAlmostEqual(mean_height, expected_mean, delta=0.1)

    def test_vertical_slice_preserves_position(self):
        """Test that vertical slice preserves position and logdensity."""
        position = jnp.array([1.5, -0.5])
        logdensity = -3.2
        state = ss.SliceState(position, logdensity)

        new_state, info = ss.vertical_slice(self.key, state)

        chex.assert_trees_all_close(new_state.position, position)
        self.assertEqual(new_state.logdensity, logdensity)
        self.assertNotEqual(new_state.logslice, jnp.inf)  # Should be updated


class StepperFunctionTest(chex.TestCase):
    """Test stepper function."""

    def test_default_stepper_array(self):
        """Test default stepper with arrays."""
        x = jnp.array([1.0, 2.0])
        d = jnp.array([0.5, -0.3])
        t = 2.5

        result = ss.default_stepper_fn(x, d, t)
        expected = x + t * d

        chex.assert_trees_all_close(result, expected)

    def test_default_stepper_scalar(self):
        """Test default stepper with scalars."""
        x = 3.0
        d = -1.2
        t = 0.8

        result = ss.default_stepper_fn(x, d, t)
        expected = x + t * d

        self.assertEqual(result, expected)

    def test_default_stepper_pytree(self):
        """Test default stepper with PyTree."""
        x = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
        d = {"a": jnp.array([0.1, -0.2]), "b": jnp.array([0.5])}
        t = 1.5

        result = ss.default_stepper_fn(x, d, t)

        chex.assert_trees_all_close(result["a"], x["a"] + t * d["a"])
        chex.assert_trees_all_close(result["b"], x["b"] + t * d["b"])

    def test_stepper_zero_step(self):
        """Test stepper with zero step size."""
        x = jnp.array([1.0, 2.0, 3.0])
        d = jnp.array([10.0, -5.0, 2.0])
        t = 0.0

        result = ss.default_stepper_fn(x, d, t)
        chex.assert_trees_all_close(result, x)


class DirectionSamplingTest(chex.TestCase):
    """Test direction sampling functions."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(123)

    def test_sample_direction_identity_covariance(self):
        """Test direction sampling with identity covariance."""
        ndim = 3
        cov = jnp.eye(ndim)

        direction = ss.sample_direction_from_covariance(self.key, cov)

        chex.assert_shape(direction, (ndim,))

        # With identity covariance, should be unit normalized
        norm = jnp.linalg.norm(direction)
        chex.assert_trees_all_close(norm, 1.0, atol=1e-6)

    def test_sample_direction_scaled_covariance(self):
        """Test direction sampling with scaled covariance."""
        ndim = 2
        scale = 4.0
        cov = jnp.eye(ndim) * scale

        direction = ss.sample_direction_from_covariance(self.key, cov)

        chex.assert_shape(direction, (ndim,))

        # Check Mahalanobis normalization
        invcov = jnp.linalg.inv(cov)
        mahal_norm = jnp.sqrt(jnp.einsum("i,ij,j", direction, invcov, direction))
        chex.assert_trees_all_close(mahal_norm, 1.0, atol=1e-6)

    def test_sample_direction_general_covariance(self):
        """Test direction sampling with general covariance matrix."""
        cov = jnp.array([[2.0, 0.5], [0.5, 1.0]])

        direction = ss.sample_direction_from_covariance(self.key, cov)

        chex.assert_shape(direction, (2,))

        # Check Mahalanobis normalization
        invcov = jnp.linalg.inv(cov)
        mahal_norm = jnp.sqrt(jnp.einsum("i,ij,j", direction, invcov, direction))
        chex.assert_trees_all_close(mahal_norm, 1.0, atol=1e-6)

    def test_sample_direction_1d(self):
        """Test direction sampling for 1D case."""
        cov = jnp.array([[2.0]])

        direction = ss.sample_direction_from_covariance(self.key, cov)

        chex.assert_shape(direction, (1,))

        # Check Mahalanobis normalization (should be 1)
        invcov = jnp.linalg.inv(cov)
        mahal_norm = jnp.sqrt(jnp.einsum("i,ij,j", direction, invcov, direction))
        chex.assert_trees_all_close(mahal_norm, 1.0, atol=1e-6)

    def test_sample_direction_multiple_samples(self):
        """Test that multiple direction samples are different."""
        cov = jnp.eye(2)
        keys = jax.random.split(self.key, 10)

        directions = jax.vmap(ss.sample_direction_from_covariance, in_axes=(0, None))(
            keys, cov
        )

        chex.assert_shape(directions, (10, 2))

        # All should be unit normalized
        norms = jnp.linalg.norm(directions, axis=1)
        chex.assert_trees_all_close(norms, jnp.ones(10), atol=1e-6)

        # Should not all be the same
        std_of_directions = jnp.std(directions, axis=0)
        self.assertTrue(jnp.all(std_of_directions > 0.1))  # Some variation expected


class HorizontalSliceTest(chex.TestCase):
    """Test horizontal slice function directly."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.key(456)

    def test_horizontal_slice_basic(self):
        """Test horizontal slice basic functionality."""
        position = jnp.array([0.5])
        logdensity = -0.5 * position**2
        logslice = -2.0
        state = ss.SliceState(position, logdensity, logslice)

        direction = jnp.array([1.0])

        def logdens_fn(x):
            return jnp.sum(-0.5 * x**2)

        def constraint_fn(x):
            return jnp.array([], dtype=jnp.float32)

        new_state, info = ss.horizontal_slice(
            self.key,
            state,
            direction,
            ss.default_stepper_fn,
            logdens_fn,
            constraint_fn,
            jnp.array([], dtype=jnp.float32),
            jnp.array([], dtype=bool),
        )

        self.assertIsInstance(new_state, ss.SliceState)
        self.assertIsInstance(info, ss.SliceInfo)
        self.assertGreater(info.evals, 0)  # Should have done some evaluations

    def test_horizontal_slice_with_constraints(self):
        """Test horizontal slice with constraints."""
        position = jnp.array([1.0])
        state = ss.SliceState(position, -0.5, -1.0)
        direction = jnp.array([1.0])

        def logdens_fn(x):
            return jnp.sum(-0.5 * x**2)

        def constraint_fn(x):
            return jnp.array([x[0]])  # Must be positive

        constraint_thresholds = jnp.array([0.0])
        strict_flags = jnp.array([True])

        new_state, info = ss.horizontal_slice(
            self.key,
            state,
            direction,
            ss.default_stepper_fn,
            logdens_fn,
            constraint_fn,
            constraint_thresholds,
            strict_flags,
        )

        # Should satisfy constraints
        self.assertTrue(jnp.all(new_state.position > 0))
        self.assertGreater(info.l_steps + info.r_steps + info.s_steps, 0)

    def test_horizontal_slice_info_completeness(self):
        """Test that horizontal slice returns complete info."""
        position = jnp.array([0.0])
        state = ss.SliceState(position, 0.0, -1.0)
        direction = jnp.array([1.0])

        def logdens_fn(x):
            return jnp.sum(-(x**2))

        def constraint_fn(x):
            return jnp.array([x[0] ** 2])

        new_state, info = ss.horizontal_slice(
            self.key,
            state,
            direction,
            ss.default_stepper_fn,
            logdens_fn,
            constraint_fn,
            jnp.array([0.1]),
            jnp.array([True]),
        )

        # Check all info fields are populated
        self.assertIsInstance(info.l_steps, (int, jax.Array))
        self.assertIsInstance(info.r_steps, (int, jax.Array))
        self.assertIsInstance(info.s_steps, (int, jax.Array))
        self.assertIsInstance(info.evals, (int, jax.Array))
        chex.assert_shape(info.constraint, (1,))

        # Total evaluations should equal sum of steps
        self.assertEqual(info.evals, info.l_steps + info.r_steps + info.s_steps)


class KernelBuildingTest(chex.TestCase):
    """Test kernel building functions."""

    def test_build_kernel_callable(self):
        """Test that build_kernel returns a callable."""

        def simple_stepper(x, d, t):
            return x + t * d

        kernel = ss.build_kernel(simple_stepper)
        self.assertTrue(callable(kernel))

    def test_build_hrss_kernel_callable(self):
        """Test that build_hrss_kernel returns a callable."""

        def direction_fn(rng_key):
            return jax.random.normal(rng_key, (2,))

        def simple_stepper(x, d, t):
            return x + t * d

        kernel = ss.build_hrss_kernel(direction_fn, simple_stepper)
        self.assertTrue(callable(kernel))

    def test_hrss_top_level_api_structure(self):
        """Test top-level API returns correct structure."""

        def simple_logdens(x):
            return -0.5 * jnp.sum(x**2)

        cov = jnp.eye(2)
        algorithm = ss.hrss_as_top_level_api(simple_logdens, cov)

        # Should have init and step methods
        self.assertTrue(hasattr(algorithm, "init"))
        self.assertTrue(hasattr(algorithm, "step"))
        self.assertTrue(callable(algorithm.init))
        self.assertTrue(callable(algorithm.step))


if __name__ == "__main__":
    absltest.main()
