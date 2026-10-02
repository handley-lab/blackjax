blackjax.adaptation.laps
========================

.. py:module:: blackjax.adaptation.laps


Classes
-------

.. autoapisummary::

   blackjax.adaptation.laps.AdaptationState
   blackjax.adaptation.laps.Adaptation


Functions
---------

.. autoapisummary::

   blackjax.adaptation.laps.bias
   blackjax.adaptation.laps.while_steps_num
   blackjax.adaptation.laps.laps


Module Contents
---------------

.. py:class:: AdaptationState



   .. py:attribute:: steps_per_sample
      :type:  float


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: stepsize_adaptation_state
      :type:  Any


   .. py:attribute:: iteration
      :type:  int


.. py:class:: Adaptation(adaptation_state, num_adaptation_samples, steps_per_sample=15, acc_prob_target=0.8, observables=lambda x: 0.0, observables_for_bias=lambda x: 0.0, contract=lambda x: 0.0)

   .. py:attribute:: num_adaptation_samples


   .. py:attribute:: observables


   .. py:attribute:: observables_for_bias


   .. py:attribute:: contract

      amount of tuning in the adjusted phase before fixing params
      steps_per_sample: number of steps per sample
      acc_prob_target: target acceptance probability
      observables: function to compute observables, for diagnostics
      observables_for_bias: function to compute observables for bias, for diagnostics
      contract: function to contract observables, for diagnostics

      :type: num_adaptation_samples


   .. py:attribute:: epsadap_update


   .. py:attribute:: initial_state


   .. py:method:: summary_statistics_fn(state, info, rng_key)


   .. py:method:: update(adaptation_state, Etheta)


.. py:function:: bias(model)

   should be transfered to benchmarks/


.. py:function:: while_steps_num(cond)

.. py:function:: laps(logdensity_fn, sample_init, ndims, num_steps1, num_steps2, num_chains, mesh, rng_key, microcanonical=True, alpha=1.9, save_frac=0.2, C=0.1, early_stop=True, r_end=0.01, bias_type=3, diagonal_preconditioning=True, integrator_coefficients=None, steps_per_sample=15, acc_prob=None, observables_for_bias=lambda x: x, all_chains_info=None, diagnostics=True, contract=lambda x: 0.0, superchain_size=1)

   model: the target density object
   num_steps1: number of steps in the first phase
   num_steps2: number of steps in the second phase
   num_chains: number of chains
   mesh: the mesh object, used for distributing the computation across cpus and nodes
   rng_key: the random key
   alpha: L = sqrt{d} * alpha * variances
   save_frac: the fraction of samples used to estimate the fluctuation in the first phase
   C: constant in stage 1 that determines step size (eq (9) of EMAUS paper)
   early_stop: whether to stop the first phase early
   r_end
   diagonal_preconditioning: whether to use diagonal preconditioning
   integrator_coefficients: the coefficients of the integrator
   steps_per_sample: the number of steps per sample
   acc_prob: the acceptance probability
   observables: the observables (for diagnostic use)
   all_chains_info: summary statistics calculated and stored for all chain at each iteration so it can be memory intensive
   diagnostics: whether to return diagnostics


