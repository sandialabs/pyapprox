*************************
Multi-Fidelity Surrogates
*************************

Multi-fidelity methods utilize an ensemble of models, enriching a small number of high-fidelity simulations with larger numbers of simulations from models of varying prediction accuracy and reduced cost, to enable greater exploration and resolution of uncertainty while maintaining deterministic prediction accuracy. 
This gallery of tutorials discusses the most popular multi-fidelity methods for constructing surrogates

The effectiveness of multi-fidelity approaches depends on the ability to identify and exploit relationships among models within the ensemble. The relationships among models within a model ensemble vary greatly, and most existing approaches focus on exploiting a specific type of structure for a presumed model sequence. For example, [KOB2000]_, [LGIJUQ2014]_, [NGXSISC2014]_, [TJWGSIAMUQ2015]_ build surrogate approximations that leverage a 1D hierarchy of models of increasing fidelity, with varying physics and/or numerical discretizations. While Multi-index collocation [HNTTCMAME2016]_ leverage a multi-dimensional hierarchy controlled my two or more numerical discretization hyper-parameters.

.. [NGXSISC2014] `Narayan, A. and Gittelson, C. and Xiu, D. A Stochastic Collocation Algorithm with Multifidelity Models. SIAM Journal on Scientific Computing 36(2), A495-A521, 2014. <https://doi.org/10.1137/130929461>`_
		 