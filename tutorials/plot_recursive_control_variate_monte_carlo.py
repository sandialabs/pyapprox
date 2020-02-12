r"""
Recursive Approximate Control Variate Monte Carlo
=================================================
This tutorial builds upon :ref:`sphx_glr_auto_tutorials_plot_approximate_control_variate_monte_carlo.py` and demonstrates that multi-level Monte Carlo and multi-fidelity Monte Carlo are both approximate control variate techniques and how this understanding can be used to improve their efficiency.


Multi-level Monte Carlo (MLMC)
------------------------------
The multi-level (MLMC) estimator based on :math:`M+1` models :math:`f_0,\ldots,f_M` ordered by decreasing fidelity (note typically MLMC literature reverses this order) is given by

.. math:: Q_0^\mathrm{ML}=\mean{f_M}+\sum_{\alpha=1}^M\mean{f_{\alpha-1}-f_\alpha}

Similarly to ACV we approximate each expectation using Monte Carlo sampling such that

.. math::Q_{0,N,\V{r}}^\mathrm{ML}=\sum_{n=1}^{\hat{r}_MN}f_M^{(n)}(\hat{\mathcal{Z}}_{M})+\sum_{\alpha=1}^M\left(\sum_{n=1}^{\hat{r}_{\alpha-1} N}\left(f_{\alpha-1}^{(n)}(\hat{\mathcal{Z}}_{\alpha-1})-f_\alpha^{(n)}(\hat{\mathcal{Z}}_{\alpha-1})\right)\right)

The 2 model MLMC estimator is

.. math::

   Q_{0,N,\V{r}}^\mathrm{ML}  &=Q_{1,\mathcal{Z}_{1,2}}+\left(Q_{0,\mathcal{Z}_{0,2}}-Q_{1,\mathcal{Z}_{1,1}}\right)\\
   &=\sum_{n=1}^{\hat{r}_1N}f_1^{(n)}(\hat{\mathcal{Z}}_1)+\sum_{n=1}^{N}\left(f_{0}^{(n)}(\hat{\mathcal{Z}}_0)-f_1^{(n)}(\hat{\mathcal{Z}}_0)\right)


By rearranging terms it is clear that this is just a control variate estimator

.. math::

   Q_{0,N,\V{r}}^\mathrm{ML}&=Q_{0,\mathcal{Z}_{0,2}}-\left(Q_{1,\mathcal{Z}_{1,1}}-Q_{1,\mathcal{Z}_{1,2}}\right)\\
  &=\sum_{n=1}^{N}f_0^{(n)}(\mathcal{Z}_{0,2}) -\left(\sum_{n=1}^{N}f_1^{(n)}(\mathcal{Z}_{1,1})-\sum_{n=1}^{\hat{r}_1N}f_{1}^{(n)}(\mathcal{Z}_{1,2})\right)\\
  &=\sum_{n=1}^{N}f_0^{(n)}(\mathcal{Z}_{0,2}) -\left(\sum_{n=1}^{N}f_1^{(n)}(\mathcal{Z}_{1,1})-\mu_{1,N,r_1}\right)

is just an approximate control variate estimator with the control variate weight :math:`\eta=-1` and :math:`\mathcal{Z}_{0,2}=\mathcal{Z}_{1,1}=\hat{\mathcal{Z}}_0`, :math:`\mathcal{Z}_{1,2}=\hat{\mathcal{Z}}_1`, and :math:`\mathcal{Z}_{0,1}=\emptyset`, . But unlike before the values of the low-fidelity model obtained at the samples used by the high-fidelity model as well are not used to approxixmate the low-fidelity mean :math:`\mu_{1,N,r_1}` so  in the ACV notation :math:`r_1=\hat{r}_1+N`

When using more than two models, MLMC just introduces more expectations of differences. For three models we have

.. math:: Q_{0,N,\V{r}}^\mathrm{ML}=Q_{2,\mathcal{Z}_{2,2}}+\left(Q_{1,\mathcal{Z}_{1,2}}-Q_{2,\mathcal{Z}_{2,1}}\right)+\left(Q_{0,\mathcal{Z}_{0,2}}-Q_{1,\mathcal{Z}_{1,1}}\right)

Which we can then rearrange into a control variate estimator

.. math:: Q_{0,N,\V{r}}^\mathrm{ML}=Q_{0,\mathcal{Z}_{0,2}} - \left(Q_{1,\mathcal{Z}_{1,1}}-Q_{1,\mathcal{Z}_{1,2}}\right)-\left(Q_{2,\mathcal{Z}_{2,1}}-Q_{2,\mathcal{Z}_{2,2}}\right)

and by inductive reasoning we get the M model ACV version of the MLMC estimator

.. math:: Q_{0,N,\V{r}^\mathrm{ML}}=Q_{0,\mathcal{Z}_{0,1}} +\sum_{\alpha=1}^M\eta_\alpha\left(Q_{\alpha,\mathcal{Z}_{\alpha-1,1}}-\mu_{\alpha,\mathcal{Z}_{\alpha,2}}\right)

where :math:`\eta_\alpha=-1,\forall\alpha` and :math:`\mathcal{Z}_{\alpha,1}=\mathcal{Z}_{\alpha-1,2}`, and :math:`\mu_{\alpha,\mathcal{Z}_{\alpha,2}}=Q_{\alpha,\mathcal{Z}_{\alpha,2}}`
 
.. list-table::

   * - .. figure:: ../figures/mlmc.png
          :width: 100%
          :align: center

          MLMC sampling strategy

     - .. figure:: ../figures/acv_is.png
          :width: 100%
          :align: center

          ACV sampling strategy


.. math::  R^2 = - \eta_1^2 \tau_{1}^2 - 2 \eta_1 \rho_{1} \tau_{1} - \eta_M^2 \frac{\tau_{M}}{\hat{r}_{M}} - \sum_{1=2}^M \frac{1}{\hat{r}_{i-1}}
                                                          \left( \eta_i^2 \tau_{i}^2 + \tau_{i-1}^2 \tau_{i-1}^2 - 2 \eta_i \eta_{i-1} \rho_{i,i-1} \tau_{i} \tau_{i-1} \right),

where  :math:`\tau_\alpha=\left(\frac{\var{Q_\alpha}}{\var{Q_0}}\right)^{\frac{1}{2}}`. Recall that and :math:`\hat{r}_\alpha=\lvert\mathcal{Z}_{\alpha,2}\rvert/N` is the ratio of the cardinality of the sets :math:`\mathcal{Z}_{\alpha,2}` and :math:`\mathcal{Z}_{0,2}`. Thus we can see that the variance reduction is bounded by the CV estimator using the lowest fidelity model with the highest correlation with :math:`f_0`

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
