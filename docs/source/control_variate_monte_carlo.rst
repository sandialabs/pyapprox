Control Variate Monte Carlo
===========================

.. math::

   \hat{Q}=N^{-1}\sum_{n=1}^N Q^{(n)}

.. math::
   :nowrap:
   
   \begin{align}
   \mean{\left(\hat{Q}-\mean{Q}\right)^2}&=\mean{\left(\hat{Q}-\mean{\hat{Q}}+\mean{\hat{Q}}-\mean{Q}\right)^2}\\
   &=\mean{\left(\hat{Q}-\mean{\hat{Q}}\right)^2}+\mean{\left(\mean{\hat{Q}}-\mean{Q}\right)^2}+\mean{2\left(\hat{Q}-\mean{\hat{Q}}\right)\left(\mean{\hat{Q}}-\mean{Q}\right)}\\
   &=\var{\hat{Q}}+\left(\mean{\hat{Q}}-\mean{Q}\right)^2
   \end{align}
   
We have used that :math:`\hat{Q}` is an unbiased estimator, i.e. :math:`\mean{\hat{Q}}=\mean{Q}` so the third term on the second line is zero. Now using

.. math::

   \var{\hat{Q}}=\var{N^{-1}\sum_{n=1}^N Q^{(n)}}=N^{-1}\sum_{n=1}^N \var{Q^{(n)}}=N^{-1}\var{Q}

yields

.. math::

   \mean{\left(\hat{Q}-\mean{Q}\right)^2}=\underbrace{N^{-1}\var{Q}}_{I}+\underbrace{\left(\mean{\hat{Q}}-\mean{Q}\right)^2}_{II}

I : stochastic error
II : deterministic bias

Control-variate Monte Carlo (CVMC)
++++++++++++++++++++++++++++++++++

.. math::

   \tilde{Q}_{\V{\alpha}}^{\text{CV}} = \tilde{Q}_{\V{\alpha}} + \gamma \left( \tilde{Q}_{\V{\kappa}} - \mu_{\V{\kappa}} \right) 

.. math::
   
   \var{\tilde{Q}_{\V{\alpha}}^{\text{CV}}}=(1-\frac{r_\V{\kappa}-r_\V{\alpha}}{r_\V{\kappa}r_\V{\alpha}}\rho^2)\var{\tilde{Q}_{\V{\alpha}}}

Multi-level Monte Carlo (MLMC)
++++++++++++++++++++++++++++++

Total cost is

.. math::

   C_{\mathrm{tot}}=\sum_{l=1}^L C_lr_lN_1
   
Variance of estimator is

.. math::
  
   \var{\hat{Q}_L}=\sum_{l=1}^L \var{Y_l}r_lN_1
   
Treating :math:`r_l` as a continuous variable the variance of the MLMC estimator is minimized for a fixed budget :math:`C` by setting

.. math::

   N_l=r_lN_1=\sqrt{\var{Y_l}/C_l}
   
Choose L so that

.. math::
   
   \left(\mean{\hat{Q}_L}-\mean{Q}\right)^2<\frac{1}{2}\epsilon^2
   
Choose :math:`N_l` so total variance

.. math::
   \var{\hat{Q}_L}<\frac{1}{2}\epsilon^2

Multi-fidelity Monte Carlo (MFMC)
+++++++++++++++++++++++++++++++++

.. math::
   
   r_i=\left(\frac{C_1(\rho^2_{1i}-\rho^2_{1i+1})}{C_i(1-\rho^2_{12})}\right)^{\frac{1}{2}}
   
Let :math:`C=(C_1\cdots C_L)^T r=(r_1\cdots r_L)^T` then

.. math::
   :nowrap:
   
   \begin{align} N_1=\frac{C_{\mathrm{tot}}}{C^Tr} & & N_i=r_iN_1\end{align}
  
The control variate weights are

.. math::
   
   \alpha_i=\frac{\rho_{1i}\sigma_1}{\sigma_i}
