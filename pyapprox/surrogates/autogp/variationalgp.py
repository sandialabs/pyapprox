from torch.distributions import MultivariateNormal
from typing import Tuple
from scipy import stats
import numpy as np

from pyapprox.expdesign.low_discrepancy_sequences import halton_sequence
from pyapprox.variables.transforms import IndependentMarginalsVariable

from pyapprox.surrogates.autogp._torch_wrappers import (
    inv, eye, multidot, trace)
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform,
    LogHyperParameterTransform)
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp._torch_wrappers import (
    diag, full)
from pyapprox.surrogates.autogp.kernels import Kernel, SumKernel


class InducingSamples():
    def __init__(self, nvars, ninducing_samples, inducing_variable=None,
                 inducing_samples=None, noise=None):
        self.nvars = nvars
        self.ninducing_samples = ninducing_samples
        (self.inducing_variable, self.init_inducing_samples,
         inducing_sample_bounds) = self._init_inducing_samples(
             inducing_variable, inducing_samples)
        self._inducing_samples = HyperParameter(
            "inducing_samples", self.nvars*self.ninducing_samples,
            self.init_inducing_samples.flatten(),
            inducing_sample_bounds.flatten(),
            IdentityHyperParameterTransform())
        if noise is None:
            noise = HyperParameter(
                'noise', 1, 1e-2, (1e-15, 1e3), LogHyperParameterTransform())
        self._noise = noise
        self.hyp_list = HyperParameterList(
            [self._noise, self._inducing_samples])

    def _init_inducing_samples(self, inducing_variable, inducing_samples):
        if inducing_variable is None:
            inducing_variable = IndependentMarginalsVariable(
                [stats.uniform(-1, 2)]*(self.nvars*self.ninducing_samples))
        if not inducing_variable.is_bounded_continuous_variable():
            raise ValueError("unbounded variables currently not supported")
        if inducing_samples is None:
            inducing_samples = halton_sequence(
                self.nvars, self.ninducing_samples)
        if inducing_samples.shape != (self.nvars, self.ninducing_samples):
            raise ValueError("inducing_samples shape is incorrect")
        inducing_sample_bounds = inducing_variable.get_statistics(
            "interval", 1.)
        return inducing_variable, inducing_samples, inducing_sample_bounds

    def get_samples(self):
        return self._inducing_samples.get_values().reshape(
            self.nvars, self.ninducing_samples)

    def get_noise(self):
        return self._noise.get_values()[0]

    def __repr__(self):
        return "{0}(ninducing_samples={1}, noise={2})".format(
            self.__class__.__name__, self.ninducing_samples, self._noise)


class InducingGaussianProcess(ExactGaussianProcess):
    r"""
    The Titsias report states that obtained :math:`\sigma^2` will be equal to
    the estimated “actual” noise plus a “correction” term that is the
    average squared error associated with the prediction of the training
    latent values f from the inducing variables :math:`f_m`. Thus, the
    variational lower bound naturally prefers to set :math:`\sigma^2`
    larger than the “actual” noise in a way that is proportional to the
    inaccuracy of the approximation
    """
    def __init__(self, nvars: int,
                 kernel: Kernel,
                 inducing_samples: InducingSamples,
                 kernel_reg: float = 0,
                 var_trans=None,
                 values_trans=None):
        super().__init__(nvars, kernel, kernel_reg, var_trans,
                         values_trans)

        if isinstance(kernel, SumKernel):
            # TODO check that sumkernel is return when using
            # constantkernel*kernel + white_noise
            # and all permutations of order
            msg = "Do not use kernel with noise with inducing samples. "
            msg += "Noise will be estimated as part of the variational "
            msg += "inference procedure"
            raise ValueError(msg)

        self.inducing_samples = inducing_samples
        self.hyp_list += self.inducing_samples.hyp_list

    def _evaluate_posterior(self, Z, return_std):
        noise = self.inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()

        K_UU_inv = inv(K_UU)
        Lambda = K_UU_inv + multidot((
            K_UU_inv, K_XU.T, K_XU, K_UU_inv/noise**2))
        Lambda_inv = inv(Lambda)
        m = multidot((Lambda_inv, K_UU_inv, K_XU.T,
                      self.canonical_train_values.squeeze()/noise**2))

        K_ZU = self.kernel(
            Z, self.inducing_samples.get_samples())
        K_ZZ = self.kernel(Z, Z)

        mu = multidot((K_ZU, K_UU_inv, m))

        if not return_std:
            return mu

        sigma = (K_ZZ - multidot((K_ZU, K_UU_inv, K_ZU.T)) +
                 multidot((K_ZU, K_UU_inv, Lambda_inv, K_UU_inv, K_ZU.T)))
        return mu[:, None],  diag(sigma)[:, None]

    def _K_XU(self) -> Tuple:
        kmat = self.kernel(
            self.canonical_train_samples, self.inducing_samples.get_samples())
        return kmat

    def _K_UU(self) -> Tuple:
        inducing_samples = self.inducing_samples.get_samples()
        kmat = self.kernel(inducing_samples, inducing_samples)
        kmat = kmat + eye(kmat.shape[0])*float(self.kernel_reg)
        return kmat

    def _training_kernel_matrix(self):
        # there is no need for K_XX to be regularized because it is not
        # inverted. K_UU must be regularized
        return self.kernel(self.canonical_train_samples)

    def _get_random_optimizer_initial_guess(self, bounds):
        # do not randomize guess for inducing samples as they need to be well
        # spaced
        guess = np.random.uniform(bounds[:, 0], bounds[:, 1])
        if self.hyp_list.hyper_params[-1].name != "inducing_samples":
            msg = "This funct6ion assumes inducing samples is the last"
            msg += "hyperparameter"
            raise RuntimeError(msg)
        hyp = self.hyp_list.hyper_params[-1]
        init_samples = self.inducing_samples.init_inducing_samples.flatten()
        active_opt_inducing_samples = hyp.transform.to_opt_space(
            init_samples[hyp._active_indices])
        if active_opt_inducing_samples.shape[0] > 0:
            guess[-active_opt_inducing_samples.shape[0]:] = (
                active_opt_inducing_samples)
        return guess

    def _neg_log_likelihood(self, active_opt_params):
        self.hyp_list.set_active_opt_params(active_opt_params)
        noise = self.inducing_samples.get_noise()
        K_XX = self._training_kernel_matrix()
        K_XU = self._K_XU()
        K_UU = self._K_UU()
        K_UU_inv = inv(K_UU)
        Sigma = multidot((K_XU, K_UU_inv, K_XU.T)) + noise**2*eye(
            K_XU.shape[0])
        zeros = full((self.canonical_train_values.shape[0],), 0.)
        # if the following line throws a ValueError it is likely
        # because self.noise is to small. If so adjust noise bounds
        p_y = MultivariateNormal(zeros, covariance_matrix=Sigma)
        mll = p_y.log_prob(self.canonical_train_values[:, 0])
        # add a regularization term to regularize variance
        Q_XX = K_XX - multidot((K_XU, K_UU_inv, K_XU.T))
        mll -= 1/(2*noise**2) * trace(Q_XX)
        return -mll

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self.hyp_list._short_repr())
