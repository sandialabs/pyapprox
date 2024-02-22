from torch.distributions import MultivariateNormal
from typing import Tuple
from scipy import stats
import numpy as np

from pyapprox.expdesign.low_discrepancy_sequences import halton_sequence
from pyapprox.variables.transforms import IndependentMarginalsVariable

from pyapprox.surrogates.autogp._torch_wrappers import (
    inv, eye, multidot, trace, sqrt, cholesky, solve_triangular, asarray,
    log, repeat)
from pyapprox.surrogates.autogp.hyperparameter import (
    HyperParameter, HyperParameterList, IdentityHyperParameterTransform,
    LogHyperParameterTransform)
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.autogp._torch_wrappers import (
    diag, full)
from pyapprox.surrogates.autogp.kernels import Kernel, SumKernel


def _log_prob_gaussian_with_noisy_nystrom_covariance(
        noise_std, L_UU, K_XU, values):
    N, M = K_XU.shape
    Delta = solve_triangular(L_UU, K_XU.T)/noise_std
    Omega = eye(M) + Delta@Delta.T
    L_Omega = cholesky(Omega)
    log_det = 2*log(L_Omega.diag()).sum()+2*N*log(noise_std)
    gamma = solve_triangular(L_Omega, Delta @ values)
    log_pdf = -0.5*(N*np.log(2*np.pi)+log_det+(values.T@values -
                    gamma.T@gamma)/noise_std**2)
    return log_pdf

# see Alvarez Efficient Multioutput Gaussian Processes through Variational Inducing Kernels for details how to generaize from noise covariance sigma^2I to \Sigma


class InducingSamples():
    def __init__(self, nvars, ninducing_samples, inducing_variable=None,
                 inducing_samples=None, inducing_sample_bounds=None,
                 noise=None):
        # inducing bounds and inducing samples must be in the canonical gp
        # space e.g. the one defined by gp.var_trans
        self.nvars = nvars
        self.ninducing_samples = ninducing_samples
        (self.inducing_variable, self.init_inducing_samples,
         inducing_sample_bounds) = self._init_inducing_samples(
             inducing_variable, inducing_samples, inducing_sample_bounds)
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

    def _init_inducing_samples(self, inducing_variable, inducing_samples,
                               inducing_sample_bounds):
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

        msg = "inducing_sample_bounds has the wrong shape"
        if inducing_sample_bounds is None:
            inducing_sample_bounds = inducing_variable.get_statistics(
                "interval", 1.)
        else:
            inducing_sample_bounds = asarray(inducing_sample_bounds)
            if inducing_sample_bounds.ndim == 1:
                if inducing_sample_bounds.shape[0] != 2:
                    raise ValueError(msg)
                inducing_sample_bounds = repeat(
                    inducing_sample_bounds, self.ninducing_samples).reshape(
                        self.ninducing_samples, 2)
        if (inducing_sample_bounds.shape !=
                (self.nvars*self.ninducing_samples, 2)):
            raise ValueError(msg)
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
        # return self.kernel(self.canonical_train_samples)
        msg = "This function should never be called because we only need "
        msg += "the diagonal of the training matrix"
        raise RuntimeError(msg)

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
        noise_std = self.inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()
        # if the following line throws a ValueError it is likely
        # because self.noise is to small. If so adjust noise bounds
        L_UU = cholesky(K_UU)
        mll = _log_prob_gaussian_with_noisy_nystrom_covariance(
            noise_std, L_UU, K_XU, self.canonical_train_values)
        # add a regularization term to regularize variance noting that
        # trace of matrix sum is sum of traces
        K_XX_diag = self.kernel.diag(self.canonical_train_samples)
        tmp = solve_triangular(L_UU, K_XU.T)
        K_tilde_trace = K_XX_diag.sum() - trace(multidot((tmp.T, tmp)))
        mll -= 1/(2*noise_std**2) * K_tilde_trace
        return -mll

    def _evaluate_posterior(self, Z, return_std):
        noise_std = self.inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()

        K_UU_inv = inv(K_UU)
        # Titsias 2009 Equation (6) B = Kuu_inv*A(Kuu_inv)
        # A is s Equation (11) in Vanderwilk 2020
        # which depends on \Sigma defined below Equation (10) Titsias
        # which we call Lambda below
        Lambda = K_UU_inv + multidot((
            K_UU_inv, K_XU.T, K_XU, K_UU_inv/noise_std**2))
        Lambda_inv = inv(Lambda)
        m = multidot((Lambda_inv, K_UU_inv, K_XU.T,
                      self.canonical_train_values.squeeze()/noise_std**2))

        #TODO replace lamnda inv with use of cholesky factors

        K_ZU = self.kernel(
            Z, self.inducing_samples.get_samples())
        K_ZZ = self.kernel(Z, Z)

        # Equation (6) in Titsias 2009 or
        # Equation (11) in Vanderwilk 2020
        mu = multidot((K_ZU, K_UU_inv, m))

        if not return_std:
            return mu

        # The following is from Equation (6) in Titsias 2009 and
        # Equation (11) in Vanderwilk 2020 where Lambda^{-1} = S
        sigma = (K_ZZ - multidot((K_ZU, K_UU_inv, K_ZU.T)) +
                 multidot((K_ZU, K_UU_inv, Lambda_inv, K_UU_inv, K_ZU.T)))
        return mu[:, None],  sqrt(diag(sigma))[:, None]
        # return mu[:, None],  (diag(sigma))[:, None]
