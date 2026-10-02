blackjax.mcmc.termination
=========================

.. py:module:: blackjax.mcmc.termination


Classes
-------

.. autoapisummary::

   blackjax.mcmc.termination.IterativeUTurnState


Functions
---------

.. autoapisummary::

   blackjax.mcmc.termination.iterative_uturn_numpyro


Module Contents
---------------

.. py:class:: IterativeUTurnState



   .. py:attribute:: momentum
      :type:  blackjax.types.Array


   .. py:attribute:: momentum_sum
      :type:  blackjax.types.Array


   .. py:attribute:: idx_min
      :type:  int


   .. py:attribute:: idx_max
      :type:  int


.. py:function:: iterative_uturn_numpyro(is_turning: blackjax.mcmc.metrics.CheckTurning)

   Numpyro style dynamic U-Turn criterion.

   :param is_turning: A function that checks whether a trajectory is turning back on itself,
                      given the left momentum, right momentum, and summed momentum.

   :returns: * A tuple of ``(new_state, update_criterion_state, is_iterative_turning)``
             * *functions that together implement the iterative U-turn criterion.*


