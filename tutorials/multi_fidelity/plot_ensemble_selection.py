r"""
Model Ensemble Selection
========================
For many applications a large number of model fidelities are available. However, it may not be beneficial to use all the lower-fidelity models because if a lower-fidelity model is poorly correlated with the other models it can increase the covariance of the ACV estimator. In some extreme cases, the covariance can be worse than the variance of a single fidelity MC estimator that solely uses the high-fidelity data. Consequently, in practice it is important to choose the subset of models that produces the smallest estimator covariance.

The following tutorial shows how to choose the best subset of models.
"""
