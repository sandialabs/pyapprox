r"""
Recursive Approximate Control Variate Monte Carlo
=======================================
This tutorial builds upon :ref:`sphx_glr_auto_examples_plot_approximate_control_variate_monte_carlo.py` and demonstrates that multi-level Monte Carlo and multi-fidelity Monte Carlo are both approximate control variate techniques and how this understanding can be used to improve their efficiency.


Multi-level Monte Carlo (MLMC)
------------------------------

Total cost is

.. math::

   C_{\mathrm{tot}}=\sum_{l=1}^L C_lr_lN_1
   
Variance of estimator is

.. math::
  
   \var{Q_L}=\sum_{l=1}^L \var{Y_l}r_lN_1
   
Treating :math:`r_l` as a continuous variable the variance of the MLMC estimator is minimized for a fixed budget :math:`C` by setting

.. math::

   N_l=r_lN_1=\sqrt{\var{Y_l}/C_l}
   
Choose L so that

.. math::
   
   \left(\mean{Q_L}-\mean{Q}\right)^2<\frac{1}{2}\epsilon^2
   
Choose :math:`N_l` so total variance

.. math::
   \var{Q_L}<\frac{1}{2}\epsilon^2
"""

#%%
#
#Multi-fidelity Monte Carlo (MFMC)
#---------------------------------
#
#.. math::
#   
#   r_i=\left(\frac{C_1(\rho^2_{1i}-\rho^2_{1i+1})}{C_i(1-\rho^2_{12})}\right)^{\frac{1}{2}}
#   
#Let :math:`C=(C_1\cdots C_L)^T r=(r_1\cdots r_L)^T` then
#
#.. math::
#
#   N_1=\frac{C_{\mathrm{tot}}}{C^Tr} & & N_i=r_iN_1\\
#
#  
#The control variate weights are
#
#.. math::
#   
#   \alpha_i=\frac{\rho_{1i}\sigma_1}{\sigma_i}

#%%
#References
#^^^^^^^^^^
#.. [PWGSIAM2016] `B. Peherstorfer, K. Willcox, M. Gunzburger, Optimal model management for multifidelity Monte Carlo estimation, SIAM J. Sci. Comput. 38 (2016) 59 A3163–A3194. <https://doi.org/10.1137/15M1046472>`_
#
#.. [CGSTCVS2011] `K.A. Cliffe, M.B. Giles, R. Scheichl, A.L. Teckentrup, Multilevel Monte Carlo methods and applications to elliptic PDEs with random coefficients, Comput. Vis. Sci. 14 (2011) <https://doi.org/10.1007/s00791-011-0160-x>`_
#
#.. [GOR2008] `M.B. Giles, Multilevel Monte Carlo path simulation, Oper. Res. 56 (2008) 607–617. <https://doi.org/10.1287/opre.1070.0496>`_
#
#.. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification, Journal of Computational Physics, In press, (2020) <https://doi.org/10.1016/j.jcp.2020.109257>`_
