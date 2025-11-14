from typing import Tuple

from scipy import stats
import numpy as np

# TODO remove torch and switch to BackendMixin

from pyapprox.util.hyperparameter import (
    HyperParameter,
    IdentityHyperParameterTransform,
    LogHyperParameterTransform,
    HyperParameterList,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.variables.transforms import IndependentMarginalsVariable
from pyapprox.surrogates.gaussianprocess.exactgp import ExactGaussianProcess
from pyapprox.surrogates.kernels.kernels import SumKernel, Kernel
from pyapprox.expdesign.sequences import HaltonSequence


def _log_prob_gaussian_with_noisy_nystrom_covariance(
    noise_std: float,
    L_UU: Array,
    K_XU: Array,
    values: Array,
    bkd: BackendMixin,
) -> float:
    """
    Compute the log probability of a Gaussian distribution with a noisy Nyström covariance matrix.

    This function uses the Nyström approximation to compute the log probability
    of the observed data under a Gaussian Process model. The covariance matrix
    is approximated using inducing points, and noise is added to the covariance
    matrix for numerical stability.

    Parameters
    ----------
    noise_std : float
        Standard deviation of the noise.
    L_UU : Array
        Cholesky decomposition of the covariance matrix of the inducing points
        (K_UU).
    K_XU : Array
        Covariance matrix between data points (X) and inducing points (U).
    values : Array
        Observed data values (y).
    bkd : BackendMixin
        Backend for numerical operations (e.g., NumPy, JAX, TensorFlow).

    Returns
    -------
    float
        Log probability of the data under the Nyström approximation.
    """
    N, M = K_XU.shape
    # Step 1: Compute Delta
    # ----------------------------------------
    # Delta = K_UU^{-1} @ K_XU.T / noise_std
    # Delta is the scaled projection of the data points onto the inducing points.
    # This uses the Cholesky decomposition of K_UU (L_UU) for efficient computation.
    Delta = bkd.solve_triangular(L_UU, K_XU.T) / noise_std

    # Step 2: Compute Omega
    # ----------------------------------------
    # Omega = I_M + Delta @ Delta.T
    # Omega is the covariance matrix in the projected space of the inducing points.
    # The identity matrix (I_M) represents the prior covariance of the inducing points,
    # and Delta @ Delta.T represents the contribution from the data points.
    Omega = bkd.eye(M) + Delta @ Delta.T

    # Step 3: Compute Cholesky decomposition of Omega
    # ----------------------------------------
    # L_Omega = Cholesky(Omega)
    # The Cholesky decomposition of Omega is used to compute the log determinant
    # and solve linear systems efficiently.
    L_Omega = bkd.cholesky(Omega)

    # Step 4: Compute log determinant
    # ----------------------------------------
    # log_det = log |Omega| + log |noise_std^2|
    # The log determinant of the covariance matrix is computed as:
    # log |Omega| = 2 * sum(log(diagonal(L_Omega)))
    # log |noise_std^2| = 2 * N * log(noise_std)
    log_det = 2 * bkd.log(bkd.get_diagonal(L_Omega)).sum() + 2 * N * bkd.log(
        bkd.atleast1d(bkd.asarray(noise_std))
    )
    # Step 5: Compute gamma
    # ----------------------------------------
    # gamma = Omega^{-1} @ Delta @ values
    # Gamma is the projection of the observed data values (y) onto the inducing points.
    # This uses the Cholesky decomposition of Omega (L_Omega) for efficient computation.
    gamma = bkd.solve_triangular(L_Omega, Delta @ values)

    # Step 6: Compute log probability
    # ----------------------------------------
    # log_pdf = -0.5 * [N * log(2 * pi) + log_det + quadratic term]
    # The log probability of the data is computed as:
    # - The normalization constant: N * log(2 * pi)
    # - The log determinant of the covariance matrix: log_det
    # - The quadratic term: (y^T @ y - gamma^T @ gamma) / noise_std^2
    log_pdf = -0.5 * (
        N * np.log(2 * np.pi)
        + log_det
        + (values.T @ values - gamma.T @ gamma) / noise_std**2
    )
    return log_pdf


# see Alvarez Efficient Multioutput Gaussian Processes through Variational
# Inducing Kernels for details how to generaize from noise covariance sigma^2I
# to \Sigma


class InducingSamples:
    def __init__(
        self,
        nvars: int,
        ninducing_samples: int,
        backend: BackendMixin,
        inducing_variable: IndependentMarginalsVariable = None,
        inducing_samples=None,
        inducing_sample_bounds=None,
        noise_std: float = 1e-2,
        noise_std_bounds: Tuple = (1e-6, 1.0),
    ):
        """
        Class to manage inducing samples and noise for Variational
        Gaussian Processes.

        Parameters
        ----------
        nvars : int
            Number of input variables.
        ninducing_samples : int
            Number of inducing samples.
            backend : BackendMixin
            Backend for numerical operations.
        inducing_variable : IndependentMarginalsVariable, optional
            Variable defining the distribution of inducing samples.
        inducing_samples : Array, optional
            Initial inducing samples.
        inducing_sample_bounds : Array, optional
            Bounds for inducing samples.
        noise_std : float, optional
            Initial noise stdev.
        noise_std : tuple, optional
            Lower and upper bounds on noise stdev.
        """

        # inducing bounds and inducing samples must be in the canonical gp
        # space e.g. the one defined by gp.var_trans
        self._bkd = backend
        self._nvars = nvars
        self._ninducing_samples = ninducing_samples
        (
            self._inducing_variable,
            self._init_inducing_samples,
            inducing_sample_bounds,
        ) = self._init_inducing_samples(
            inducing_variable, inducing_samples, inducing_sample_bounds
        )
        self._inducing_samples = HyperParameter(
            "inducing_samples",
            self._nvars * self._ninducing_samples,
            self._init_inducing_samples.flatten(),
            inducing_sample_bounds.flatten(),
            IdentityHyperParameterTransform(backend=self._bkd),
            backend=self._bkd,
        )
        noise = HyperParameter(
            "noise",
            1,
            noise_std,
            noise_std_bounds,
            LogHyperParameterTransform(backend=self._bkd),
            backend=self._bkd,
        )
        if not self._bkd.bkd_equal(self._bkd, noise._bkd):
            raise ValueError("noise._bkd and backend are different")
        self._noise = noise
        self._hyp_list = HyperParameterList(
            [self._noise, self._inducing_samples]
        )

    def hyp_list(self) -> HyperParameterList:
        """
        Get the list of hyperparameters.

        Returns
        -------
        HyperParameterList
            List of hyperparameters for inducing samples and noise.
        """
        return self._hyp_list

    def _init_inducing_samples(
        self,
        inducing_variable: IndependentMarginalsVariable,
        inducing_samples: Array,
        inducing_sample_bounds: Array,
    ) -> Tuple[IndependentMarginalsVariable, Array, Array]:
        """
        Initialize inducing samples and bounds.

        Returns
        -------
        inducing_variable : IndependentMarginalsVariable
            Variable defining the distribution of inducing samples.
        inducing_samples : Array
            Initial inducing samples.
        inducing_sample_bounds : Array
            Bounds for inducing samples.
        """
        if inducing_variable is None:
            inducing_variable = IndependentMarginalsVariable(
                [stats.uniform(-1, 2)]
                * (self._nvars * self._ninducing_samples)
            )
        if not inducing_variable.is_bounded_continuous_variable():
            raise ValueError("unbounded variables currently not supported")
        if inducing_samples is None:
            seq = HaltonSequence(self._nvars)
            inducing_samples = seq.rvs(self._ninducing_samples)
        if inducing_samples.shape != (self._nvars, self._ninducing_samples):
            raise ValueError("inducing_samples shape is incorrect")

        if inducing_sample_bounds is None:
            inducing_sample_bounds = inducing_variable.interval(1.0)
        else:
            if inducing_sample_bounds.ndim == 1:
                if inducing_sample_bounds.shape[0] != 2:
                    msg = "inducing_sample_bounds has the wrong shape {0}".format(
                        inducing_sample_bounds.shape
                    )
                    raise ValueError(msg)
                inducing_sample_bounds = self._bkd.tile(
                    inducing_sample_bounds,
                    (self._nvars * self._ninducing_samples,),
                ).reshape(self._ninducing_samples * self._nvars, 2)
        if inducing_sample_bounds.shape != (
            self._nvars * self._ninducing_samples,
            2,
        ):
            raise ValueError(
                "inducing_sample_bounds has shape {0} "
                "but should be {1}".format(
                    inducing_sample_bounds.shape,
                    (self._nvars * self._ninducing_samples, 2),
                )
            )
        return inducing_variable, inducing_samples, inducing_sample_bounds

    def get_samples(self) -> Array:
        """
        Get the inducing samples.

        Returns
        -------
        np.ndarray
            Inducing samples reshaped to (nvars, ninducing_samples).
        """
        return self._inducing_samples.get_values().reshape(
            self._nvars, self._ninducing_samples
        )

    def get_noise(self) -> float:
        """
        Get the noise value.

        Returns
        -------
        float
            Noise value.
        """
        return self._noise.get_values()[0]

    def __repr__(self) -> str:
        return "{0}(ninducing_samples={1}, noise={2})".format(
            self.__class__.__name__, self._ninducing_samples, self._noise
        )


class InducingGaussianProcess(ExactGaussianProcess):
    r"""
    Class for implementing an Inducing Gaussian Process (IGP) with
    variational inference.

    This class extends the ExactGaussianProcess class and uses inducing points
    to approximate the posterior distribution. It supports variational
    inference for estimating hyperparameters and noise.

    Notes
    -----
    The Titsias report states that the obtained :math:`\sigma^2` will be equal
    to the estimated “actual” noise plus a “correction” term that is the average
    squared error associated with the prediction of the training latent values
    :math:`f` from the inducing variables :math:`f_m`. Thus, the variational
    lower bound naturally prefers to set :math:`\sigma^2` larger than the
    "actual"
    noise in a way that is proportional to the inaccuracy of the approximation.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    kernel : Kernel
        Kernel function used for covariance computations.
    inducing_samples : InducingSamples
        Object managing inducing points and noise hyperparameters.
    kernel_reg : float, optional
        Regularization parameter for the kernel matrix (default: 0).
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

        self._inducing_samples = inducing_samples
        self._hyp_list += self._inducing_samples.hyp_list()
        self.set_optimizer(self.default_optimizer())

    def _set_coef(self):
        """
        Override base class to avoid errors.
        """
        return

    def analytical_neg_log_like_jacobian_implemented(self) -> bool:
        """
        Check if the analytical Jacobian of the negative log-likelihood is
        implemented.

        Returns
        -------
        bool
            False, as the analytical Jacobian is not implemented.
        """
        return False

    def _K_XU(self) -> Array:
        """
        Compute the covariance matrix between training samples and inducing
        samples.

        Returns
        -------
        Tuple
            Covariance matrix between training samples and inducing samples.
        """
        kmat = self._kernel(
            self._ctrain_samples, self._inducing_samples.get_samples()
        )
        return kmat

    def _K_UU(self) -> Array:
        """
        Compute the covariance matrix between inducing samples.

        Returns
        -------
        Tuple
            Covariance matrix between inducing samples, with regularization applied.
        """
        inducing_samples = self._inducing_samples.get_samples()
        kmat = self._kernel(inducing_samples, inducing_samples)
        kmat = kmat + self._bkd.eye(kmat.shape[0]) * float(self._kernel_reg)
        return kmat

    def _training_kernel_matrix(self):
        # there is no need for K_XX to be regularized because it is not
        # inverted. K_UU must be regularized
        # return self._kernel(self._ctrain_samples)
        msg = "This function should never be called because we only need "
        msg += "the diagonal of the training matrix"
        raise RuntimeError(msg)

    def _neg_log_like(self, active_opt_params: Array):
        """
        Compute the negative log-likelihood for variational inference.

        Parameters
        ----------
        active_opt_params : np.ndarray
            Active optimization parameters.

        Returns
        -------
        np.ndarray
            Negative log-likelihood value.

        Notes
        -----
        If the noise is too small, this function may throw a ValueError.
        Adjust the noise bounds if necessary.
        """
        self._hyp_list.set_active_opt_params(active_opt_params)
        noise_std = self._inducing_samples.get_noise()
        K_XU = self._K_XU()
        K_UU = self._K_UU()
        # if the following line throws a ValueError it is likely
        # because self._noise is to small. If so adjust noise bounds
        L_UU = self._bkd.cholesky(K_UU)
        mll = _log_prob_gaussian_with_noisy_nystrom_covariance(
            noise_std, L_UU, K_XU, self._ctrain_values, self._bkd
        )
        # add a regularization term to regularize variance noting that
        # trace of matrix sum is sum of traces
        K_XX_diag = self._kernel.diag(self._ctrain_samples)
        tmp = self._bkd.solve_triangular(L_UU, K_XU.T)
        K_tilde_trace = K_XX_diag.sum() - self._bkd.trace(
            self._bkd.multidot((tmp.T, tmp))
        )
        mll -= 1 / (2 * noise_std**2) * K_tilde_trace
        return -mll[:, 0]

    def _evaluate_canonical_posterior(
        self, Z: Array, return_std: bool
    ) -> Tuple[Array, float]:
        """
        Evaluate the canonical posterior distribution.

        Parameters
        ----------
        Z : np.ndarray
            Test points where the posterior is evaluated.
        return_std : bool
            Whether to return the standard deviation of the posterior.

        Returns
        -------
        Tuple
            Posterior mean and standard deviation (if return_std is True).
        """
        noise_std = self._inducing_samples.get_noise()
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

        K_ZU = self._kernel(Z, self._inducing_samples.get_samples())
        K_ZZ = self._kernel(Z, Z)

        # Equation (6) in Titsias 2009 or
        # Equation (11) in Vanderwilk 2020
        mu = self._bkd.multidot((K_ZU, K_UU_inv, m))

        if not return_std:
            return mu[:, None], None

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

    def inducing_samples(self) -> InducingSamples:
        """
        Return the inducing samples object.

        Return
        ------
        inducing_samples; InducingSamples
            The inducing samples object
        """
        return self._inducing_samples

    def _set_training_data(self, train_samples: Array, train_values: Array):
        if train_values.shape != (train_samples.shape[1], 1):
            raise ValueError(
                "training_values must have shape "
                f"{(train_samples.shape[1], 1)}"
            )
        super()._set_training_data(train_samples, train_values)
