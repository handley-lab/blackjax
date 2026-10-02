blackjax.adaptation.laps_burn_in
================================

.. py:module:: blackjax.adaptation.laps_burn_in


Classes
-------

.. autoapisummary::

   blackjax.adaptation.laps_burn_in.History
   blackjax.adaptation.laps_burn_in.AdaptationState
   blackjax.adaptation.laps_burn_in.Adaptation


Functions
---------

.. autoapisummary::

   blackjax.adaptation.laps_burn_in.no_nans
   blackjax.adaptation.laps_burn_in.nan_reject
   blackjax.adaptation.laps_burn_in.build_kernel
   blackjax.adaptation.laps_burn_in.initialize
   blackjax.adaptation.laps_burn_in.update_history
   blackjax.adaptation.laps_burn_in.update_history_scalar
   blackjax.adaptation.laps_burn_in.contract_history
   blackjax.adaptation.laps_burn_in.equipartition_diagonal
   blackjax.adaptation.laps_burn_in.equipartition_fullrank
   blackjax.adaptation.laps_burn_in.equipartition_diagonal_loss
   blackjax.adaptation.laps_burn_in.equipartition_fullrank_loss


Module Contents
---------------

.. py:function:: no_nans(a)

.. py:function:: nan_reject(nonans, old, new)

   Equivalent to
   return new if nonans else old


.. py:function:: build_kernel(logdensity_fn, ndims, microcanonical=True)

   MCLMC kernel (with nan rejection)


.. py:function:: initialize(rng_key, logdensity_fn, microcanonical, sample_init, num_chains, mesh, superchain_size)

   initialize the chains based on the equipartition of the initial condition.
   We initialize the velocity along grad log p if E_ii > 1 and along -grad log p if E_ii < 1.


.. py:function:: update_history(new_vals, history)

.. py:function:: update_history_scalar(new_val, history)

.. py:function:: contract_history(theta, weights)

.. py:class:: History



   .. py:attribute:: observables
      :type:  blackjax.types.Array


   .. py:attribute:: stopping
      :type:  blackjax.types.Array


   .. py:attribute:: weights
      :type:  blackjax.types.Array


.. py:class:: AdaptationState



   .. py:attribute:: L
      :type:  float


   .. py:attribute:: inverse_mass_matrix
      :type:  Any


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: step_count
      :type:  int


   .. py:attribute:: EEVPD
      :type:  float


   .. py:attribute:: EEVPD_wanted
      :type:  float


   .. py:attribute:: history
      :type:  Any


.. py:function:: equipartition_diagonal(state)

   Ei = E_ensemble (- grad log p_i x_i ). Ei is 1 if we have converged.
   equipartition_loss = average over parameters (Ei)


.. py:function:: equipartition_fullrank(state, rng_key)

   loss = Tr[(1 - E)^T (1 - E)] / d^2
   where Eij = <xi gj> is the equipartition patrix.
   Loss is computed with the Hutchinson's trick.


.. py:function:: equipartition_diagonal_loss(Eii)

.. py:function:: equipartition_fullrank_loss(delta_z)

.. py:class:: Adaptation(ndims, microcanonical, alpha=1.0, C=0.1, r_end=0.01, bias_type=0, save_num=10, observables=lambda x: 0.0, observables_for_bias=lambda x: x, contract=lambda x: 0.0)

   .. py:attribute:: ndims


   .. py:attribute:: alpha
      :value: 1.0



   .. py:attribute:: C
      :value: 0.1



   .. py:attribute:: r_end
      :value: 0.01



   .. py:attribute:: observables


   .. py:attribute:: observables_for_bias


   .. py:attribute:: contract


   .. py:attribute:: bias_type
      :value: 0



   .. py:attribute:: save_num
      :value: 10



   .. py:attribute:: norm_factor


   .. py:attribute:: initial_state


   .. py:method:: summary_statistics_fn(state, info, rng_key)


   .. py:method:: update(adaptation_state, Etheta)


   .. py:method:: while_cond(info, counter)

      determine if we want to switch to adjustment



