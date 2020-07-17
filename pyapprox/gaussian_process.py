#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
class GaussianProcess(GaussianProcessRegressor):       
    def fit(self,train_samples,train_values):
        """
        A light weight wrapper of sklearn GaussianProcessRegressor.fit
        function. See sklearn documentation for more info. This wrapper
        is needed because sklearn stores a unique sample in each row
        of a samples matrix whereas pyapprox uses the transpose.

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples at which to evaluate the GP. Sklearn requires the
            transpose of this matrix, i.e a matrix with size (nsamples,nvars)
        """

        return super().fit(train_samples.T,train_values)

    def __call__(self, samples, return_std=False, return_cov=False):
        """
        A light weight wrapper of sklearn GaussianProcessRegressor.predict
        function. See sklearn documentation for more info. This wrapper
        is needed because sklearn stores a unique sample in each row
        of a samples matrix whereas pyapprox uses the transpose.

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples at which to evaluate the GP. Sklearn requires the
            transpose of this matrix, i.e a matrix with size (nsamples,nvars)
        """
        return self.predict(samples.T,return_std,return_cov)

    


