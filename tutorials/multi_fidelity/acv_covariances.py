r"""
Delta-based covariance formulas for Approximate Control Variates
=========================================================================
The following lists the formula needed to use ACV to compute vector-valued statistics comprised of means, variances or both. The formulas use the expressions for :math:`\mat{V}, \mat{W}, \mat{B}` in :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multioutput_monte_carlo.py`.

These covariancesdepend on how samples are allocated to the sets :math:`\rvset_\alpha,\rvset_\alpha^*`, which we call the sample allocation :math:`\mathcal{A}`. Specifically, :math:`\mathcal{A}` specifies: the number of samples in the sets :math:`\rvset_\alpha,\rvset_\alpha^*, \forall \alpha`, denoted by :math:`N_\alpha` and :math:`N_{\alpha^*}`, respectively; the number of samples in the intersections of pairs of sets, that is :math:`N_{\alpha\cap\beta} =|\rvset_\alpha \cap \rvset_\beta|`, :math:`N_{\alpha^*\cap\beta} =|\rvset_\alpha^* \cap \rvset_\beta|`, :math:`N_{\alpha^*\cap\beta^*} =|\rvset_\alpha^* \cap \rvset_\beta^*|`; and the number of samples in the union of pairs of sets :math:`N_{\alpha\cup\beta} = |\rvset_\alpha\cup \rvset_\beta|` and similarly :math:`N_{\alpha^*\cup\beta}`, :math:`N_{\alpha^*\cup\beta^*}`.

Mean
----
.. math:: \covar{\mat{\Delta}_\alpha}{\mat{\Delta}_\beta} = F_{\alpha\beta}\covar{f_\alpha}{f_\beta}\in\reals^{K\times K}


.. math::     F_{\alpha\beta} = \frac{N_{\alpha^*\cap \beta^*}}{N_{\alpha^*}N_{\beta^*}} - \frac{N_{\alpha^*\cap \beta}}{N_{\alpha^*}N_{\beta}} - \frac{N_{\alpha\cap \beta^*}}{N_{\alpha}N_{\beta^*}} + \frac{N_{\alpha\cap \beta}}{N_{\alpha}N_{\beta}}


.. math:: \covar{\mat{Q}_0}{\mat{\Delta}_\alpha} = G_{\alpha}\covar{f_0}{f_\alpha}\in\reals^{K\times K}


.. math::    G_{\alpha} = \frac{N_{0\cap \alpha^*}}{N_{0}N_{\alpha^*}} -  \frac{N_{0\cap \alpha}}{N_{0}N_{\alpha}}


Variance
--------

.. math:: \covar{\mat{\Delta}_\alpha}{\mat{\Delta}_\beta} = F_{\alpha\beta}\mat{W}_{\alpha\beta}+H_{\alpha\beta}\mat{V}_{\alpha\beta}\in\reals^{K^2\times K^2}

.. math:: H_{\alpha\beta} &= \frac{N_{\alpha^*\cap \beta^*}(N_{\alpha^*\cap \beta^*}-1)}{N_{\alpha^*}N_{\beta^*}(N_{\alpha^*}-1)(N_{\beta^*}-1)} - \frac{N_{\alpha^*\cap \beta}(N_{\alpha^*\cap \beta}-1)}{N_{\alpha^*}N_{\beta}(N_{\alpha^*}-1)(N_{\beta}-1)} \\& \qquad\quad- \frac{N_{\alpha\cap \beta^*}(N_{\alpha\cap \beta^*}-1)}{N_{\alpha}N_{\beta^*}(N_{\alpha}-1)(N_{\beta^*}-1)}  + \frac{N_{\alpha\cap \beta}(N_{\alpha\cap \beta}-1)}{N_{\alpha}N_{\beta}(N_{\alpha}-1)(N_{\beta}-1)}

.. math:: \covar{\mat{Q}_0}{\mat{\Delta}_\alpha} = J_\alpha\mat{V}_{0\alpha}+G_{\alpha}\mat{W}_{0\alpha}\in\reals^{K^2\times K^2}


.. math::    J_{\alpha} = \frac{N_{0\cap \alpha^*}(N_{0\cap \alpha^*}-1)}{N_{0}N_{\alpha^*}(N_{0}-1)(N_{\alpha^*}-1)} - \frac{N_{0\cap \alpha}(N_{0\cap \alpha}-1)}{N_{0}N_{\alpha}(N_{0}-1)(N_{\alpha}-1)}

Mean and Variance
-----------------

.. math::  \covar{\mat{\Delta}_\alpha}{\mat{\Delta}_\beta}(\rvset_\text{ACV}) = \begin{bmatrix}\covar{\Delta^\mu_\alpha}{\Delta^\mu_\beta} & \covar{\Delta^{\mu}_\alpha}{\Delta^{\Sigma}_\beta}\\\covar{\Delta^{\Sigma}_\beta}{\Delta^{\mu}_\alpha} & \covar{\Delta^{\Sigma}_\alpha}{\Delta^{\Sigma}_\beta}\end{bmatrix}\in\reals^{(K+K^2)\times (K+K^2)}

.. math:: \covar{\mat{Q}_0}{\mat{\Delta}_\alpha}(\rvset_\text{ACV}) = \begin{bmatrix}\covar{Q^\mu_0}{\Delta^\mu_\alpha} & \covar{Q^{\mu}_0}{\Delta^{\Sigma}_\alpha} \\ \covar{Q^{\Sigma}_0}{\Delta^{\mu}_\alpha} & \covar{Q^{\Sigma}_0}{\Delta^{\Sigma}_\alpha}\end{bmatrix} \in\reals^{(K+K^2)\times (K+K^2)}.


.. math::    \covar{\Delta^{\mu}_\alpha}{\Delta^{\Sigma}_\beta} = F_{\alpha\beta}\mat{B}_{\alpha\beta}\in\reals^{K\times (K+K^2)}


.. math::    \covar{Q^{\mu}_0}{\Delta^{\Sigma}_\alpha}=G_\alpha\mat{B}_{0\alpha}\in\reals^{K\times (K+K^2)}
"""
