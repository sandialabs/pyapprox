Multi-Fidelity Monte Carlo Methods
----------------------------------

Multi-fidelity methods utilize an ensemble of models, enriching a small number of high-fidelity simulations with larger numbers of simulations from models of varying prediction accuracy and reduced cost, to enable greater exploration and resolution of uncertainty while maintaining deterministic prediction accuracy. The effectiveness of multi-fidelity approaches depends on the ability to identify and exploit relationships among models within the ensemble.

The relationships among models within a model ensemble vary greatly, and most existing approaches focus on exploiting a specific type of structure for a presumed model sequence. For example, [KOB2000]_, [LGIJUQ2014]_, [NGXSISC2014]_ build approximations that leverage a hierarchy of models of increasing fidelity, with varying physics and/or numerical discretizations. Multi-level [GOR2008]_, [TJWGSIAMUQ2015]_ and multi-index [HNTTCMAME2016]_ also leverage a set of models of increasing fidelity, with the additional assumption that a model sequence forms a convergent hierarchy.

This gallery of tutorials discusses the most popular multi-fidelity methods for quantifying uncertainty in complex models.

References
^^^^^^^^^^
.. [NGXSISC2014] `Narayan, A. and Gittelson, C. and Xiu, D. A Stochastic Collocation Algorithm with Multifidelity Models. SIAM Journal on Scientific Computing 36(2), A495-A521, 2014. <https://doi.org/10.1137/130929461>`_
		 
.. [LGIJUQ2014]	`L. Le Gratiet and J. Garnier Recursive co-kriging model for design of computer experiments with multiple levels of fidelity. International Journal for Uncertainty Quantification, 4(5), 365--386, 2014 <http://dx.doi.org/10.1615/Int.J.UncertaintyQuantification.2014006914>`_
		
.. [KOB2000] `M. C. Kennedy and A. O'Hagan. Predicting the Output from a Complex Computer Code When Fast Approximations Are Available. Biometrika, 87(1), 1-13, 2000. <http://www.jstor.org/stable/2673557>`_
