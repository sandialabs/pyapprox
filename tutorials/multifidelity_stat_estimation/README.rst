*************************************
Multi-Fidelity Statistical Estimation
*************************************
Multi-fidelity methods utilize an ensemble of models, enriching a small number of high-fidelity simulations with larger numbers of simulations from models of varying prediction accuracy and reduced cost, to enable greater exploration and resolution of uncertainty while maintaining deterministic prediction accuracy. 
This gallery of tutorials discusses the most popular multi-fidelity methods for estimating statistics quantifying the uncertainty in the predictions of a model

The effectiveness of multi-fidelity approaches depends on the ability to identify and exploit relationships among models within the ensemble. [CGSTCVS2011]_, [GOR2008]_ exploit a 1D hierarchy of models to estimate statistics such as mean and variance using Monte Carlo methods. In constrast, [GGEJJCP2020]_, [SUSIAMUQ2020]_ can be used with an ordered ensemble of low-fidelity models.

.. [NGXSISC2014] `Narayan, A. and Gittelson, C. and Xiu, D. A Stochastic Collocation Algorithm with Multifidelity Models. SIAM Journal on Scientific Computing 36(2), A495-A521, 2014. <https://doi.org/10.1137/130929461>`_
		 




