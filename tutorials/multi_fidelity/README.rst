Multi-Fidelity Methods
----------------------

Multi-fidelity methods utilize an ensemble of models, enriching a small number of high-fidelity simulations with larger numbers of simulations from models of varying prediction accuracy and reduced cost, to enable greater exploration and resolution of uncertainty while maintaining deterministic prediction accuracy. The effectiveness of multi-fidelity approaches depends on the ability to identify and exploit relationships among models within the ensemble.

The relationships among models within a model ensemble vary greatly, and most existing approaches focus on exploiting a specific type of structure for a presumed model sequence. For example, [KOB2000]_, [LGIJUQ2014]_, [NGXSISC2014]_, [TJWGSIAMUQ2015]_ build surrogate approximations that leverage a 1D hierarchy of models of increasing fidelity, with varying physics and/or numerical discretizations. While Multi-index collocation [HNTTCMAME2016]_ leverage a multi-dimensional hierarchy controlled my two or more numerical discretization hyper-parameters. Similary [CGSTCVS2011]_, [GOR2008]_ exploit a 1D hierarchy of models to estimate statistics such as mean and variance using Monte Carlo methods.

This gallery of tutorials discusses the most popular multi-fidelity methods for quantifying uncertainty in complex models.

.. [NGXSISC2014] `Narayan, A. and Gittelson, C. and Xiu, D. A Stochastic Collocation Algorithm with Multifidelity Models. SIAM Journal on Scientific Computing 36(2), A495-A521, 2014. <https://doi.org/10.1137/130929461>`_
		 
.. [LGIJUQ2014]	`L. Le Gratiet and J. Garnier Recursive co-kriging model for design of computer experiments with multiple levels of fidelity. International Journal for Uncertainty Quantification, 4(5), 365--386, 2014 <http://dx.doi.org/10.1615/Int.J.UncertaintyQuantification.2014006914>`_
		
.. [KOB2000] `M. C. Kennedy and A. O'Hagan. Predicting the Output from a Complex Computer Code When Fast Approximations Are Available. Biometrika, 87(1), 1-13, 2000. <http://www.jstor.org/stable/2673557>`_

.. [HNTTCMAME2016] `A. Haji-Ali, F. Nobile, L. Tamellini, and R. Tempone. Multi-index stochastic collocation for random pdes. Computer Methods in Applied Mechanics and Engineering, 306:95 – 122, 2016. <http://www. sciencedirect.com/science/article/pii/S0045782516301141, doi:10.1016/j.cma.2016.03.029>`_

.. [TJWGSIAMUQ2015] `A. Teckentrup, P. Jantsch, C. Webster, and M. Gunzburger. A multilevel stochastic collocation method for partial differential equations with random input data. SIAM/ASA Journal on Uncertainty Quantification, 3(1):1046-1074, 2015. <https://doi.org/10.1137/140969002>1`_

