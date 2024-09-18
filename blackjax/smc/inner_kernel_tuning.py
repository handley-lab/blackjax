from typing import Callable, Dict, NamedTuple, Tuple

from blackjax.base import SamplingAlgorithm
from blackjax.smc.base import SMCInfo, SMCState
from blackjax.types import ArrayTree, PRNGKey


class StateWithParameterOverride(NamedTuple):
    """
    Stores both the sampling status and also a dictionary
    that contains an dictionary with parameter names as key
    and (n_particles, *) arrays as meanings. The latter
    represent a parameter per chain for the next mutation step.
    """

    sampler_state: ArrayTree
    parameter_override: Dict[str, ArrayTree]


def init(alg_init_fn, position, initial_parameter_value):
    return StateWithParameterOverride(alg_init_fn(position), initial_parameter_value)


def build_kernel(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    mcmc_parameter_update_fn: Callable[[SMCState, SMCInfo], Dict[str, ArrayTree]],
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> Callable:
    """In the context of an SMC sampler (whose step_fn returning state has a .particles attribute), there's an inner
    MCMC that is used to perturbate/update each of the particles. This adaptation tunes some parameter of that MCMC,
    based on particles. The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair).
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_step_fn:
        The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
        mcmc_step_fn(rng_key, state, tempered_logposterior_fn, **mcmc_parameter_update_fn())
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the inner kernel in i+1 iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.
    """

    def kernel(
        rng_key: PRNGKey, state: StateWithParameterOverride, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        step_fn = smc_algorithm(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=mcmc_step_fn,
            mcmc_init_fn=mcmc_init_fn,
            mcmc_parameters=state.parameter_override,
            resampling_fn=resampling_fn,
            num_mcmc_steps=num_mcmc_steps,
            **extra_parameters,
        ).step
        new_state, info = step_fn(rng_key, state.sampler_state, **extra_step_parameters)
        new_parameter_override = mcmc_parameter_update_fn(new_state, info)
        return StateWithParameterOverride(new_state, new_parameter_override), info

    return kernel


def as_top_level_api(
    smc_algorithm,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_step_fn: Callable,
    mcmc_init_fn: Callable,
    resampling_fn: Callable,
    mcmc_parameter_update_fn: Callable[[SMCState, SMCInfo], Dict[str, ArrayTree]],
    initial_parameter_value,
    num_mcmc_steps: int = 10,
    **extra_parameters,
) -> SamplingAlgorithm:
    """In the context of an SMC sampler (whose step_fn returning state
    has a .particles attribute), there's an inner MCMC that is used
    to perturbate/update each of the particles. This adaptation tunes some
    parameter of that MCMC, based on particles.
    The parameter type must be a valid JAX type.

    Parameters
    ----------
    smc_algorithm
        Either blackjax.adaptive_tempered_smc or blackjax.tempered_smc (or any other implementation of
        a sampling algorithm that returns an SMCState and SMCInfo pair). See blackjax.smc_family
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given position.
    mcmc_step_fn
        The transition kernel, should take as parameters the dictionary output of mcmc_parameter_update_fn.
    mcmc_init_fn
        A callable that initializes the inner kernel
    mcmc_parameter_update_fn
        A callable that takes the SMCState and SMCInfo at step i and constructs a parameter to be used by the
        inner kernel in i+1 iteration.
    initial_parameter_value
        Parameter to be used by the mcmc_factory before the first iteration.
    extra_parameters:
        parameters to be used for the creation of the smc_algorithm.

    Returns
    -------
    A ``SamplingAlgorithm``.

    """


    kernel = build_kernel(
        smc_algorithm,
        logprior_fn,
        loglikelihood_fn,
        mcmc_step_fn,
        mcmc_init_fn,
        resampling_fn,
        mcmc_parameter_update_fn,
        num_mcmc_steps,
        **extra_parameters,
    )

    def init_fn(position, rng_key=None):
        del rng_key
        return init(smc_algorithm.init, position, initial_parameter_value)

    def step_fn(
        rng_key: PRNGKey, state, **extra_step_parameters
    ) -> Tuple[StateWithParameterOverride, SMCInfo]:
        return kernel(rng_key, state, **extra_step_parameters)

    return SamplingAlgorithm(init_fn, step_fn)
