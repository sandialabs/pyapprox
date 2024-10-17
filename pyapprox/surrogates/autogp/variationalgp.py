from typing import Tuple

from scipy import stats
import numpy as np

# TODO remove torch and switch to LinAlgMixin

from pyapprox.util.hyperparameter import (
    HyperParameter,
    IdentityHyperParameterTransform,
    LogHyperParameterTransform,
    HyperParameterList,
)
from pyapprox.util.linearalgebra.numpylinalg import (
    NumpyLinAlgMixin,
    LinAlgMixin,
)
from pyapprox.expdesign.low_discrepancy_sequences import halton_sequence
from pyapprox.variables.transforms import IndependentMarginalsVariable
from pyapprox.surrogates.autogp.exactgp import ExactGaussianProcess
from pyapprox.surrogates.kernels.kernels import SumKernel, Kernel


def _log_prob_gaussian_with_noisy_nystrom_covariance(
    noise_std, L_UU, K_XU, values, bkd
):
    N, M = K_XU.shape
    Delta = bkd.solve_triangular(L_UU, K_XU.T) / noise_std
    Omega = bkd.eye(M) + Delta @ Delta.T
    L_Omega = bkd.cholesky(Omega)
    log_det = 2 * bkd.log(
        bkd.get_diagonal(L_Omega)
    ).sum() + 2 * N * bkd.log(bkd.atleast1d(noise_std))
    gamma = bkd.solve_triangular(L_Omega, Delta @ values)
    log_pdf = -0.5 * (
        N * np.log(2 * np.pi)
        + log_det
        + (values.T @ values - gamma.T @ gamma) / noise_std**2
    )
    return log_pdf


# see Alvarez Efficient Multioutput Gaussian Processes through Variational Inducing Kernels for details how to generaize from noise covariance sigma^2I to \Sigma


class InducingSamples:
    def __init__(
        self,
        nvars: int,
        ninducing_samples: int,
        inducing_variable: IndependentMarginalsVariable = None,
        inducing_samples=None,
        inducing_sample_bounds=None,
        noise: float = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        # inducing bounds and inducing samples must be in the canonical gp
        # space e.g. the one defined by gp.var_trans
        self._bkd = backend
        self.nvars = nvars
        self.ninducing_samples = ninducing_samples
        (
            self.inducing_variable,
            self.init_inducing_samples,
            inducing_sample_bounds,
        ) = self._init_inducing_samples(
            inducing_variable, inducing_samples, inducing_sample_bounds
        )
        self._inducing_samples = HyperParameter(
            "inducing_samples",
            self.nvars * self.ninducing_samples,
            self.init_inducing_samples.flatten(),
            inducing_sample_bounds.flatten(),
            IdentityHyperParameterTransform(backend=self._bkd),
        )
        if noise is None:
            noise = HyperParameter(
                "noise",
                1,
                1e-2,
                (1e-15, 1e3),
                LogHyperParameterTransform(backend=self._bkd),
            )
        self._noise = noise
        self.hyp_list = HyperParameterList(
            [self._noise, self._inducing_samples]
        )

    def _init_inducing_samples(
        self, inducing_variable, inducing_samples, inducing_sample_bounds
    ):
        if inducing_variable is None:
            inducing_variable = IndependentMarginalsVariable(
                [stats.uniform(-1, 2)] * (self.nvars * self.ninducing_samples)
            )
        if not inducing_variable.is_bounded_continuous_variable():
            raise ValueError("unbounded variables currently not supported")
        if inducing_samples is None:
            inducing_samples = halton_sequence(
                self.nvars, self.ninducing_samples
            )
        if inducing_samples.shape != (self.nvars, self.ninducing_samples):
            raise ValueError("inducing_samples shape is incorrect")

        msg = "inducing_sample_bounds has the wrong shape"
        if inducing_sample_bounds is None:
            inducing_sample_bounds = inducing_variable.get_statistics(
                "interval", 1.0
            )
        else:
            inducing_sample_bounds = inducing_sample_bounds
            if inducing_sample_bounds.ndim == 1:
                if inducing_sample_bounds.shape[0] != 2:
                    raise ValueError(msg)
                inducing_sample_bounds = self._bkd.repeat(
                    inducing_sample_bounds, self.ninducing_samples
                ).reshape(self.ninducing_samples, 2)
        if inducing_sample_bounds.shape != (
            self.nvars * self.ninducing_samples,
            2,
        ):
            raise ValueError(msg)
        return inducing_variable, inducing_samples, inducing_sample_bounds

    def get_samples(self):
        return self._inducing_samples.get_values().reshape(
            self.nvars, self.ninducing_samples
        )

    def get_noise(self):
        return self._noise.get_values()[0]

    def __repr__(self):
        return "{0}(ninducing_samples={1}, noise={2})".format(
            self.__class__.__name__, self.ninducing_samples, self._noise
        )


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

    def __init__(
        self,
        nvars: int,
        kernel: Kernel,
        inducing_samples,
        kernel_reg: float = 0,
    ):
        super().__init__(nvars, kernel, None, kernel_reg)
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
        self.set_optimizer()

    def _K_XU(self) -> Tuple:
        kmat = self.kernel(
            self._ctrain_samples, self.inducing_samples.get_samples()
        )
        return kmat

    def _K_UU(self) -> Tuple:
        inducing_samples = self.inducing_samples.get_samples()
        kmat = self.kernel(inducing_samples, inducing_samples)
        kmat = kmat + self._bkd.eye(kmat.shape[0]) * float(self.kernel_reg)
        return kmat

    def _training_kernel_matrix(self):
        # there is no need for K_XX to be regularized because it is not
        # inverted. K_UU must be regularized
        # return self.kernel(self._ctrain_samples)
        msg = "This function should never be called because we only need "
        msg += "the diagonal of the training matrix"
        raise RuntimeError(msg)

    def _neg_log_likelihood(self, active_opt_params):
        self.hyp_list.set_active_opt_params(active_opt_params)
        noise_std = self.inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()
        # if the following line throws a ValueError it is likely
        # because self.noise is to small. If so adjust noise bounds
        L_UU = self._bkd.cholesky(K_UU)
        mll = _log_prob_gaussian_with_noisy_nystrom_covariance(
            noise_std, L_UU, K_XU, self._ctrain_values, self._bkd
        )
        # add a regularization term to regularize variance noting that
        # trace of matrix sum is sum of traces
        K_XX_diag = self.kernel.diag(self._ctrain_samples)
        tmp = self._bkd.solve_triangular(L_UU, K_XU.T)
        K_tilde_trace = K_XX_diag.sum() - self._bkd.trace(
            self._bkd.multidot((tmp.T, tmp))
        )
        mll -= 1 / (2 * noise_std**2) * K_tilde_trace
        return -mll[:, 0]

    def _evaluate_canonical_posterior(self, Z, return_std):
        noise_std = self.inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()

        K_UU_inv = self._bkd.inv(K_UU)
        # Titsias 2009 Equation (6) B = Kuu_inv*A(Kuu_inv)
        # A is s Equation (11) in Vanderwilk 2020
        # which depends on \Sigma defined below Equation (10) Titsias
        # which we call Lambda below
        Lambda = K_UU_inv + self._bkd.multidot(
            (K_UU_inv, K_XU.T, K_XU, K_UU_inv / noise_std**2)
        )
        Lambda_inv = self._bkd.inv(Lambda)
        m = self._bkd.multidot(
            (
                Lambda_inv,
                K_UU_inv,
                K_XU.T,
                self._ctrain_values.squeeze() / noise_std**2,
            )
        )

        # TODO replace lamnda inv with use of cholesky factors

        K_ZU = self.kernel(Z, self.inducing_samples.get_samples())
        K_ZZ = self.kernel(Z, Z)

        # Equation (6) in Titsias 2009 or
        # Equation (11) in Vanderwilk 2020
        mu = self._bkd.multidot((K_ZU, K_UU_inv, m))

        if not return_std:
            return mu, None

        # The following is from Equation (6) in Titsias 2009 and
        # Equation (11) in Vanderwilk 2020 where Lambda^{-1} = S
        sigma = (
            K_ZZ
            - self._bkd.multidot((K_ZU, K_UU_inv, K_ZU.T))
            + self._bkd.multidot(
                (K_ZU, K_UU_inv, Lambda_inv, K_UU_inv, K_ZU.T)
            )
        )
        return (
            mu[:, None],
            self._bkd.sqrt(self._bkd.get_diagonal(sigma))[:, None],
        )
        # return mu[:, None],  (diag(sigma))[:, None]
