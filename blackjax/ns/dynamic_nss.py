# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamic Nested Slice Sampling (Dynamic NSS).

A thin wrapper around :mod:`blackjax.ns.nss` that makes *dynamic* Nested
Sampling — replacing ``num_delete > 1`` live points per step — a first-class
API entry point.  The underlying machinery (Hit-and-Run Slice Sampling inner
kernel, covariance-based direction proposals, adaptive parameter updates) is
identical to :func:`blackjax.nss`; only the default regime and documentation
differ.

In standard (static) NS, ``num_delete=1``: the single worst live point is
removed and replaced at every step.  In *dynamic* NS, ``num_delete`` is set to
a significant fraction of the live population (a common default is
``num_delete = num_live // 2``), which improves parallelism and makes the
algorithm more efficient on modern hardware.
"""

from typing import Callable

from blackjax import SamplingAlgorithm
from blackjax.mcmc.ss import sample_direction_from_covariance
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.base import init_state_strategy
from blackjax.ns.from_mcmc import update_with_mcmc_take_last
from blackjax.ns.nss import (
    as_top_level_api as _nss_as_top_level_api,
)
from blackjax.ns.nss import (
    build_kernel,  # noqa: F401 — re-exported for generate_top_level_api_from
)
from blackjax.ns.nss import (
    default_stepper_fn,
)
from blackjax.ns.nss import (
    init,  # noqa: F401 — re-exported for generate_top_level_api_from
)
from blackjax.ns.nss import (
    update_inner_kernel_params,  # noqa: F401 — re-exported
)

__all__ = [
    "as_top_level_api",
    "build_kernel",
    "init",
    "update_inner_kernel_params",
]

def as_top_level_api(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_delete: int,
    num_inner_steps: int,
    stepper_fn: Callable = default_stepper_fn,
    generate_slice_direction_fn: Callable = sample_direction_from_covariance,
    init_state_strategy_fn: Callable = init_state_strategy,
    update_inner_kernel_params_fn: Callable = update_inner_kernel_params,
    delete_fn: Callable = default_delete_fn,
    update_strategy: Callable = update_with_mcmc_take_last,
    max_steps: int = 10,
    max_shrinkage: int = 100,
) -> SamplingAlgorithm:
    """Creates a Dynamic Nested Slice Sampling (Dynamic NSS) algorithm.

    This is a convenience wrapper around :func:`blackjax.nss` that highlights
the *dynamic* regime where multiple live points (``num_delete > 1``) are
removed and replaced at every step.  All inner-kernel logic (Hit-and-Run
Slice Sampling with adaptive covariance directions) is identical to
:func:`blackjax.nss`.

    A typical choice is ``num_delete = num_live // 2``, which removes half of
the live population per step and maximises GPU utilisation via the ``vmap``
over the batch that is already built into the inner kernel.

    Parameters
    ----------
    logprior_fn
        A function that computes the log-prior probability of a single particle.
    loglikelihood_fn
        A function that computes the log-likelihood of a single particle.
    num_delete
        The number of live points to remove and replace at each NS step.
        Unlike :func:`blackjax.nss`, this argument is **required** here because
        it is the defining characteristic of the dynamic regime.  A good
        default for most problems is ``num_live // 2``.
    num_inner_steps
        The number of Hit-and-Run Slice Sampling steps used to generate each
        replacement live point.  Should be a multiple of the parameter
dimension.
    stepper_fn
        The stepper function ``(x, direction, t) -> (x_new, is_accepted)`` for
        the HRSS kernel.  Defaults to the standard linear stepper.
    generate_slice_direction_fn
        A function ``(rng_key, position, **kwargs) -> direction_pytree`` that
        generates a normalised direction for HRSS.  Defaults to
        :func:`~blackjax.mcmc.ss.sample_direction_from_covariance`.
    init_state_strategy_fn
        A function to initialise :class:`~blackjax.ns.base.StateWithLogLikelihood`
        from a position.  Defaults to
        :func:`~blackjax.ns.base.init_state_strategy`.
    update_inner_kernel_params_fn
        Function that updates inner-kernel parameters (covariance) from the
        current particle population.  Defaults to
        :func:`~blackjax.ns.nss.update_inner_kernel_params`.
    delete_fn
        Function that selects which live points to remove.  Defaults to
        :func:`~blackjax.ns.base.delete_fn`.
    update_strategy
        Strategy for updating live points via MCMC.  Defaults to
        :func:`~blackjax.ns.from_mcmc.update_with_mcmc_take_last`.
    max_steps
        Maximum number of stepping-out steps in the HRSS interval expansion.
        Defaults to 10.
    max_shrinkage
        Maximum number of shrinkage steps in HRSS.  Defaults to 100.

    Returns
    -------
    SamplingAlgorithm
        A :class:`~blackjax.base.SamplingAlgorithm` with ``init`` and ``step``
        functions for the configured Dynamic Nested Slice Sampler.
    """
    return _nss_as_top_level_api(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
        stepper_fn=stepper_fn,
        generate_slice_direction_fn=generate_slice_direction_fn,
        init_state_strategy_fn=init_state_strategy_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        delete_fn=delete_fn,
        update_strategy=update_strategy,
        max_steps=max_steps,
        max_shrinkage=max_shrinkage,
    )
