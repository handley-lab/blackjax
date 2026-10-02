blackjax.mcmc.diffusions
========================

.. py:module:: blackjax.mcmc.diffusions

.. autoapi-nested-parse::

   Solvers for Langevin diffusions.



Functions
---------

.. autoapisummary::

   blackjax.mcmc.diffusions.overdamped_langevin


Module Contents
---------------

.. py:function:: overdamped_langevin(logdensity_grad_fn)

   Euler solver for overdamped Langevin diffusion.

   :param logdensity_grad_fn: A function that returns a ``(logdensity, logdensity_grad)`` tuple given
                              a position and optional batch arguments.

   :rtype: A ``one_step`` function that advances the diffusion by one Euler step.


