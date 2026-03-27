"""
IID Gaussian noise kernel for modeling independent observation noise.

This module provides the IIDGaussianNoise kernel, which models independent
and identically distributed Gaussian noise in Gaussian process regression.
"""

from typing import Tuple

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList, LogHyperParameter


class IIDGaussianNoise(Kernel[Array]):
    """
    IID Gaussian noise kernel for modeling independent observation noise.

    This kernel represents uncorrelated noise:
    - K(X, X) = σ² * I (identity matrix scaled by noise variance)
    - K(X, X') = 0 for X ≠ X' (no correlation between different points)

    This is essential for Gaussian process regression where observations
    are assumed to be noisy: y = f(x) + ε, where ε ~ N(0, σ²I).

    Parameters
    ----------
    noise_variance : float
        The noise variance σ².
    variance_bounds : Tuple[float, float]
        Bounds for the noise variance parameter.
    bkd : Backend
        Backend for numerical computations.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).

    Examples
    --------
    >>> from pyapprox.surrogates.kernels import MaternKernel, IIDGaussianNoise
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> noise = IIDGaussianNoise(0.1, (0.01, 1.0), bkd)
    >>> gp_kernel = matern + noise  # GP kernel with observation noise
    """

    def __init__(
        self,
        noise_variance: float,
        variance_bounds: Tuple[float, float],
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        """
        Initialize the IIDGaussianNoise kernel.

        Parameters
        ----------
        noise_variance : float
            The noise variance σ².
        variance_bounds : Tuple[float, float]
            Bounds for the noise variance parameter.
        bkd : Backend[Array]
            Backend for numerical computations.
        fixed : bool, optional
            Whether the hyperparameter is fixed (default is False).
        """
        super().__init__(bkd)

        # Use LogHyperParameter to ensure noise variance is positive
        self._log_noise_variance = LogHyperParameter(
            "noise_variance",
            1,  # Scalar parameter
            [noise_variance],
            variance_bounds,
            bkd=self._bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._log_noise_variance])

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        return self._hyp_list

    def nvars(self) -> int:
        """
        Return the number of input variables.

        For IIDGaussianNoise, this is inferred from the input data shape
        and not stored as an attribute.

        Returns
        -------
        nvars : int
            Returns 0 as IIDGaussianNoise doesn't depend on input dimensions.
        """
        # IIDGaussianNoise doesn't have spatial dependence, so nvars is ambiguous.
        # We return 0 to indicate it works with any number of dimensions.
        return 0

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        For IIDGaussianNoise, the diagonal is a vector of noise variance values.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix, shape (n,).
        """
        n = X1.shape[1]
        noise_variance = self._log_noise_variance.exp_values()[0]
        return self._bkd.full((n,), noise_variance)

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the IID Gaussian noise kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array, optional
            Input data, shape (nvars, n2). If None, uses X1.

        Returns
        -------
        K : Array
            IID Gaussian noise kernel matrix.
            - If X2 is None or X2 is X1: diagonal matrix σ²*I, shape (n1, n1)
            - If X2 is provided and different from X1: zeros, shape (n1, n2)
              (no correlation between different inputs)
        """
        n1 = X1.shape[1]
        noise_variance = self._log_noise_variance.exp_values()[0]

        if X2 is None or X2 is X1:
            # Self-covariance: return diagonal matrix
            return self._bkd.eye(n1) * noise_variance
        else:
            # Cross-covariance: return zeros (no correlation)
            n2 = X2.shape[1]
            return self._bkd.zeros((n1, n2))

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of IID Gaussian noise kernel w.r.t. inputs.

        Since the IID Gaussian noise kernel has no spatial dependence,
        the Jacobian is zero everywhere.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array
            Input data, shape (nvars, n2).

        Returns
        -------
        jac : Array
            Jacobian, shape (n1, n2, nvars). All zeros.
        """
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        nvars = X1.shape[0]

        return self._bkd.zeros((n1, n2, nvars))

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of IID Gaussian noise kernel w.r.t. hyperparameters.

        For IIDGaussianNoise with log-parameterization:
        K = σ² * I (diagonal matrix)
        log_σ² = log(σ²)
        dK/d(log_σ²) = dK/dσ² * dσ²/d(log_σ²) = I * σ² = σ² * I

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, 1).
            - Diagonal entries: σ² (noise variance)
            - Off-diagonal entries: 0
        """
        n = samples.shape[1]
        noise_variance = self._log_noise_variance.exp_values()[0]

        # Vectorized: create diagonal matrix and add parameter dimension
        # jac[i, j, 0] = noise_variance * δ_{ij}
        jac = self._bkd.eye(n)[:, :, None] * noise_variance

        return jac

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters.

        For IIDGaussianNoise with log-parameterization:
        K = σ² * I
        θ = log(σ²)
        ∂K/∂θ = σ² * I
        ∂²K/∂θ² = σ² * I

        Since there's only one parameter, HVP is just the Hessian times the scalar
        direction.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).
        direction : Array
            Direction vector, shape (1,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, 1).
            hvp[:, :, 0] = ∂²K/∂θ² * direction[0]
        """
        n = X1.shape[1]
        noise_variance = self._log_noise_variance.exp_values()[0]

        # Vectorized: create diagonal matrix and scale by noise_variance * direction[0]
        hvp = self._bkd.eye(n)[:, :, None] * (noise_variance * direction[0])

        return hvp

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute HVP of IID Gaussian noise kernel w.r.t. first argument.

        Since IIDGaussianNoise has no spatial dependence, the Hessian is zero.
        Therefore, the HVP is also zero.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector for HVP

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            HVP (all zeros since kernel has no spatial dependence)
        """
        nvars = X1.shape[0]
        n1 = X1.shape[1]
        n2 = X2.shape[1]

        return self._bkd.zeros((n1, n2, nvars))
