import torch
from torch.distributions import MultivariateNormal
from typing import Tuple
from scipy import stats

from pyapprox.expdesign.low_discrepancy_sequences import halton_sequence
from pyapprox.variables.transforms import IndependentMarginalsVariable

from pyapprox.surrogates.autogp._torch_wrappers import inv, eye
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform)
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp._torch_wrappers import (
    diag, full)
from pyapprox.surrogates.autogp.kernels import Kernel


class InducingSamples():
    def __init__(self, nvars, ninducing_samples, inducing_variable=None):
        self.nvars = nvars
        self.ninducing_samples = ninducing_samples
        self.inducing_variable, inducing_samples, inducing_sample_bounds = (
            self._init_inducing_samples(inducing_variable))
        self._inducing_samples = HyperParameter(
            "inducing_samples", self.nvars*self.ninducing_samples,
            inducing_samples.flatten(), inducing_sample_bounds,
            IdentityHyperParameterTransform())
        self.hyp_list = HyperParameterList([self._inducing_samples])

    def _init_inducing_samples(self, inducing_variable):
        if inducing_variable is None:
            inducing_variable = IndependentMarginalsVariable(
                [stats.uniform(-1, 2)]*self.nvars)
        if not inducing_variable.is_bounded_continuous_variable():
            raise ValueError("unbounded variables currently not supported")
        inducing_samples = halton_sequence(
            self.nvars, self.ninducing_samples)
        if inducing_samples.shape != (self.nvars, self.ninducing_samples):
            raise ValueError("inducing_samples shape is incorrect")
        inducing_sample_bounds = inducing_variable.get_statistics(
            "interval", 0.)
        return inducing_samples, inducing_sample_bounds


class InducingGaussianProcess(ExactGaussianProcess):
    def __init__(self, nvars: int,
                 kernel: Kernel,
                 inducing_samples: InducingSamples,
                 kernel_reg: float = 0,
                 var_trans=None,
                 values_trans=None):
        super().__init__(nvars, kernel, kernel_reg, var_trans,
                         values_trans)
        self.inducing_samples = inducing_samples

    def _evaluate_posterior(self, Z, return_std):
        K_XU = self._KXU()
        K_UU = self._KUU()

        K_UU_inv = inv(K_UU)
        Lambda = K_UU_inv + K_UU_inv@K_XU.T@K_XU@K_UU_inv/self.noise**2
        Lambda_inv = inv(Lambda)
        m = Lambda_inv@K_UU_inv@K_XU.T@self.y/self.noise**2

        K_ZU = self.compute_kernel_matrix(Z, self.inducing_x_mu)
        K_ZZ = self.compute_kernel_matrix(Z, Z)

        mu = K_ZU@K_UU_inv@m
        sigma = (K_ZZ - K_ZU@K_UU_inv@K_ZU.T +
                 K_ZU@K_UU_inv@Lambda_inv@K_UU_inv@K_ZU.T)
        if not return_std:
            return mu
        return mu,  diag(sigma)[:, None]

    def _K_XU(self) -> Tuple:
        kmat = self.kernel(
            self.canonical_train_samples, self.inducing_samples)
        kmat = kmat + diag(full((kmat.shape[0], 1), float(self.kernel_reg)))
        return kmat

    def _K_UU(self) -> Tuple:
        kmat = self.kernel(
            self.inducing_samples, self.inducing_samples)
        kmat = kmat + diag(full((kmat.shape[0], 1), float(self.kernel_reg)))
        return kmat

    def _neg_log_likelihood(self, active_opt_params):
        self.kernel.hyp_list.set_active_opt_params(active_opt_params)
        K_XX = self._training_kernel_matrix()
        K_XU = self._K_XU()
        K_UU = self._K_UU()
        K_UU_inv = inv(K_UU)
        Sigma = K_XU @ K_UU_inv @ K_XU.T + self.noise**2*eye(K_XU.shape[0])
        p_y = MultivariateNormal(
            self.canonical_train_values.squeeze()*0, covariance_matrix=Sigma)
        mll = p_y.log_prob(self.canonical_train_values.squeeze())
        # add a regularization term to regularize variance
        Q_XX = K_XX - K_XU @ K_UU_inv @ K_XU.T
        mll -= 1/(2 * self.noise**2) * torch.trace(Q_XX)
        return -mll
