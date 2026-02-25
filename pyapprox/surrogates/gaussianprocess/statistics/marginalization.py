"""
Marginalized Gaussian Process for dimension reduction.

This module implements the MarginalizedGP class which reduces a d-dimensional GP
to a lower-dimensional GP by integrating out selected variables. This is useful for:

1. **Visualization**: Plotting the effect of one or two variables
2. **Sensitivity analysis**: Computing main effects V_i = Var[f̃(z_i)]
3. **Dimension reduction**: Creating interpretable lower-dimensional representations

Key Result
----------
If f(z) is a GP, then f̃(z_p) = E_{z_~p}[f(z) | z_p] is also a GP with:

- **Scaled kernel**: C̃(z_p, z'_p) = u_~p · C_p(z_p, z'_p)
- **Modified correlations**: τ̃(z_p) = t_p(z_p) ⊙ τ_~p

where:
- p is the set of active (kept) dimensions
- ~p is the set of marginalized (integrated out) dimensions
- u_~p = ∏_{k∈~p} u_k is the product of 1D u values for marginalized dims
- τ_~p = ∏_{k∈~p} τ_k (element-wise) is the product of 1D τ vectors for marg. dims
- t_p(z_p) is the correlation vector using only the active dimensions
- C_p(z_p, z'_p) = ∏_{k∈p} C_k(z_{p,k}, z'_{p,k}) is the product kernel over active dims

Mathematical Reference
---------------------
See docs/plans/gp_integration/04_marginalization.qmd for derivations.
"""

from typing import Generic, Sequence, List
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.surrogates.gaussianprocess.statistics.integrals import (
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.integrals_1d import (
    compute_tau_1d,
    compute_u_1d,
)


class MarginalizedGP(Generic[Array]):
    """
    A marginalized Gaussian Process reduced to a subset of dimensions.

    Given a d-dimensional GP f(z), this class represents the lower-dimensional GP:
        f̃(z_p) = E_{z_~p}[f(z) | z_p]

    which is the conditional expectation with respect to all variables NOT in
    the active set p.

    The marginalized GP has:
    - Mean: m̃*(z_p) = (t_p(z_p) ⊙ τ_~p)^T A^{-1} y
    - Variance: C̃*(z_p, z_p) = u_~p · C_p(z_p, z_p) - (t_p(z_p) ⊙ τ_~p)^T A^{-1} (t_p(z_p) ⊙ τ_~p)

    where:
    - p is the set of active dimensions (kept)
    - ~p is the set of marginalized dimensions (integrated out)
    - ⊙ denotes element-wise (Hadamard) product
    - t_p(z_p) = ∏_{k∈p} t_k(z_{p,k}) is the product of 1D correlations for active dims
    - C_p(z_p, z_p) = ∏_{k∈p} C_k(z_{p,k}, z_{p,k}) = 1 for normalized kernels

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process with a separable (product) kernel.
    integral_calculator : SeparableKernelIntegralCalculator[Array]
        Calculator for kernel integrals, which provides access to 1D kernels,
        quadrature rules, and training data.
    active_dims : Sequence[int]
        The dimensions to keep. All other dimensions are integrated out.
        Examples:
        - [0] -> 1D marginalized GP (keep only dimension 0)
        - [0, 1] -> 2D marginalized GP (keep dimensions 0 and 1)
        - [0, 2, 3] -> 3D marginalized GP (keep dimensions 0, 2, 3)

    Raises
    ------
    ValueError
        If active_dims is empty, has duplicates, or contains invalid indices.
    RuntimeError
        If the GP is not fitted.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
    >>> from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
    >>> from pyapprox.surrogates.gaussianprocess.statistics import (
    ...     SeparableKernelIntegralCalculator,
    ... )
    >>> from pyapprox.surrogates.sparsegrids.basis_factory import (
    ...     create_basis_factories,
    ... )
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    >>> k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    >>> k3 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    >>> kernel = k1 * k2 * k3
    >>> gp = ExactGaussianProcess(kernel, 3, bkd)
    >>> X = bkd.array(np.random.rand(3, 10) * 2 - 1)
    >>> y = bkd.array(np.random.rand(10, 1))
    >>> gp.fit(X, y)
    >>>
    >>> marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(3)]
    >>> factories = create_basis_factories(marginals, bkd, "gauss")
    >>> bases = [f.create_basis() for f in factories]
    >>> for b in bases:
    ...     b.set_nterms(20)
    >>>
    >>> calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    >>>
    >>> # 1D marginalized GP (keep only dimension 0)
    >>> marg_gp_1d = MarginalizedGP(gp, calc, active_dims=[0])
    >>> z_1d = bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
    >>> mean_1d = marg_gp_1d.predict_mean(z_1d)
    >>>
    >>> # 2D marginalized GP (keep dimensions 0 and 2)
    >>> marg_gp_2d = MarginalizedGP(gp, calc, active_dims=[0, 2])
    >>> z_2d = bkd.array([[-0.5, 0.0], [0.5, 0.5]])  # Shape (2, 2)
    >>> mean_2d = marg_gp_2d.predict_mean(z_2d)
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        integral_calculator: SeparableKernelIntegralCalculator[Array],
        active_dims: Sequence[int],
    ) -> None:
        """Initialize the MarginalizedGP."""
        # Validate GP is fitted
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before creating marginalized GP. "
                "Call gp.fit(X_train, y_train) first."
            )

        nvars = len(integral_calculator._kernels_1d)

        # Validate active_dims
        active_dims_list = list(active_dims)
        if len(active_dims_list) == 0:
            raise ValueError("active_dims must not be empty")

        if len(active_dims_list) != len(set(active_dims_list)):
            raise ValueError(
                f"active_dims contains duplicates: {active_dims_list}"
            )

        for dim in active_dims_list:
            if dim < 0 or dim >= nvars:
                raise ValueError(
                    f"active_dims contains invalid index {dim}. "
                    f"Must be in range [0, {nvars})."
                )

        # Sort active_dims to ensure consistent ordering
        self._active_dims: List[int] = sorted(active_dims_list)
        self._marginalized_dims: List[int] = [
            d for d in range(nvars) if d not in self._active_dims
        ]

        self._gp = gp
        self._calc = integral_calculator
        self._bkd = integral_calculator.bkd()
        self._nvars = nvars

        # Get training data
        self._train_samples = gp.data().X()  # Shape: (nvars, N)
        self._n_train = gp.data().n_samples()
        self._y_train = gp.data().y()  # Shape: (N, nqoi)

        # Get cholesky factor for efficient solving
        self._cholesky = gp.cholesky()

        # Precompute τ_~p and u_~p (products over marginalized dimensions)
        self._tau_not_p = self._compute_tau_not_p()
        self._u_not_p = self._compute_u_not_p()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def active_dims(self) -> List[int]:
        """Return the list of active (kept) dimension indices."""
        return self._active_dims.copy()

    def marginalized_dims(self) -> List[int]:
        """Return the list of marginalized (integrated out) dimension indices."""
        return self._marginalized_dims.copy()

    def ndims(self) -> int:
        """Return the number of active dimensions in the marginalized GP."""
        return len(self._active_dims)

    def nvars_original(self) -> int:
        """Return the number of variables in the original GP."""
        return self._nvars

    def u_not_p(self) -> Array:
        """
        Return u_~p, the product of u_k for all marginalized dimensions k ∈ ~p.

        This is the scaling factor for the prior kernel of the marginalized GP.

        Returns
        -------
        Array
            Scalar value u_~p = ∏_{k∈~p} u_k.
        """
        return self._u_not_p

    def tau_not_p(self) -> Array:
        """
        Return τ_~p, the element-wise product of τ_k for all marginalized dims.

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        return self._tau_not_p

    def _compute_tau_not_p(self) -> Array:
        """
        Compute τ_~p = ∏_{k∈~p} τ_k (element-wise product over marginalized dims).

        This is the partial product of 1D tau vectors for the marginalized
        dimensions. Used to compute marginalized correlation vectors.

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        bkd = self._bkd
        tau_not_p = bkd.ones((self._n_train,))

        for dim in self._marginalized_dims:
            # Compute 1D tau for this marginalized dimension
            tau_1d = compute_tau_1d(
                self._calc._quad_samples[dim],
                self._calc._quad_weights[dim],
                self._calc._get_train_samples_1d(dim),
                self._calc._get_kernel_callable(dim),
                bkd
            )
            tau_not_p = tau_not_p * tau_1d

        return tau_not_p

    def _compute_u_not_p(self) -> Array:
        """
        Compute u_~p = ∏_{k∈~p} u_k (product over marginalized dimensions).

        This is the scaling factor for the prior variance of the marginalized GP.

        Returns
        -------
        Array
            Scalar value u_~p.
        """
        bkd = self._bkd
        u_not_p = bkd.asarray(1.0)

        for dim in self._marginalized_dims:
            # Compute 1D u for this marginalized dimension
            u_1d = compute_u_1d(
                self._calc._quad_samples[dim],
                self._calc._quad_weights[dim],
                self._calc._get_kernel_callable(dim),
                bkd
            )
            u_not_p = u_not_p * u_1d

        return u_not_p

    def _compute_t_p(self, z_p: Array) -> Array:
        """
        Compute t_p(z_p), the product kernel correlation over active dimensions.

        t_p(z_p)_j = ∏_{k∈p} C_k(z_{p,k}, x_k^{(j)})

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).
            Row i corresponds to active_dims[i].

        Returns
        -------
        Array
            Correlation matrix, shape (n_test, N) where N is number of training pts.
        """
        bkd = self._bkd
        n_test = z_p.shape[1]

        # Initialize to ones (will multiply in each dimension's contribution)
        t_p = bkd.ones((n_test, self._n_train))

        for idx, dim in enumerate(self._active_dims):
            # Get 1D kernel for this active dimension
            kernel_1d = self._calc._get_kernel_callable(dim)

            # Get training samples in this dimension: shape (1, N)
            train_samples_1d = self._calc._get_train_samples_1d(dim)

            # Get test samples in this dimension: shape (1, n_test)
            z_dim = z_p[idx:idx+1, :]

            # Compute kernel for this dimension: shape (n_test, N)
            t_dim = kernel_1d(z_dim, train_samples_1d)

            # Multiply into the product
            t_p = t_p * t_dim

        return t_p

    def _compute_tau_tilde(self, z_p: Array) -> Array:
        """
        Compute τ̃(z_p) = t_p(z_p) ⊙ τ_~p.

        This is the marginalized correlation vector.

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).

        Returns
        -------
        Array
            Marginalized correlation, shape (n_test, N).
        """
        # t_p shape: (n_test, N)
        t_p = self._compute_t_p(z_p)

        # τ_~p shape: (N,) -> broadcast to (n_test, N)
        # Element-wise product along the N dimension
        # Type annotation needed because __getitem__ returns Any in ArrayProtocol
        tau_tilde: Array = t_p * self._tau_not_p[None, :]

        return tau_tilde

    def _compute_C_p(self, z_p: Array) -> Array:
        """
        Compute C_p(z_p, z_p), the product kernel diagonal over active dims.

        C_p(z_p, z_p) = ∏_{k∈p} C_k(z_{p,k}, z_{p,k})

        For normalized kernels (where C_k(x, x) = 1), this returns all 1s.
        However, we compute it explicitly to handle potential non-normalized kernels.

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).

        Returns
        -------
        Array
            Kernel diagonal values, shape (n_test,).
        """
        bkd = self._bkd
        n_test = z_p.shape[1]

        # Initialize to ones
        C_p_diag = bkd.ones((n_test,))

        for idx, dim in enumerate(self._active_dims):
            # Get 1D kernel for this active dimension
            kernel_1d = self._calc._get_kernel_callable(dim)

            # Get test samples in this dimension: shape (1, n_test)
            z_dim = z_p[idx:idx+1, :]

            # Compute kernel diagonal: C_k(z, z) for each test point
            # kernel returns (n_test, n_test), we need the diagonal
            K_dim = kernel_1d(z_dim, z_dim)
            C_dim_diag = bkd.diag(K_dim)

            # Multiply into the product
            C_p_diag = C_p_diag * C_dim_diag

        return C_p_diag

    def predict_mean(self, z_p: Array) -> Array:
        """
        Predict the marginalized GP mean at test points z_p.

        m̃*(z_p) = (t_p(z_p) ⊙ τ_~p)^T A^{-1} y = τ̃(z_p)^T α

        where α = A^{-1} y is precomputed by the GP.

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).
            Row i corresponds to active_dims[i].
            Uses 2D format per array shape conventions.

        Returns
        -------
        Array
            Marginalized mean at test points, shape (n_test,).

        Examples
        --------
        >>> # For 1D marginalized GP (active_dims=[0])
        >>> z_1d = bkd.array([[-0.5, 0.0, 0.5]])  # Shape (1, 3)
        >>> mean = marg_gp.predict_mean(z_1d)
        >>>
        >>> # For 2D marginalized GP (active_dims=[0, 1])
        >>> z_2d = bkd.array([[-0.5, 0.0], [0.5, 0.5]])  # Shape (2, 2)
        >>> mean = marg_gp.predict_mean(z_2d)
        """
        self._validate_input_shape(z_p)

        # τ̃(z_p) shape: (n_test, N)
        tau_tilde = self._compute_tau_tilde(z_p)

        # α = A^{-1} y, shape: (nqoi, N) - need to transpose for multiplication
        alpha = self._gp.alpha()

        # m̃* = τ̃ @ αᵀ, shape: (n_test, nqoi)
        mean = tau_tilde @ alpha.T

        # Flatten if nqoi=1
        if mean.shape[1] == 1:
            mean = self._bkd.reshape(mean, (-1,))

        return mean

    def predict_variance(self, z_p: Array) -> Array:
        """
        Predict the marginalized GP variance at test points z_p.

        C̃*(z_p, z_p) = u_~p · C_p(z_p, z_p) - τ̃(z_p)^T A^{-1} τ̃(z_p)
                      = u_~p · C_p(z_p, z_p) - ||L^{-1} τ̃(z_p)||²

        where L is the Cholesky factor of A (A = L L^T).

        For normalized kernels, C_p(z_p, z_p) = 1, so this simplifies to:
        C̃*(z_p, z_p) = u_~p - ||L^{-1} τ̃(z_p)||²

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).
            Row i corresponds to active_dims[i].

        Returns
        -------
        Array
            Marginalized variance at test points, shape (n_test,).
            Always non-negative and bounded by u_~p · C_p(z_p, z_p).

        Examples
        --------
        >>> z_test = bkd.array([[-0.5, 0.0, 0.5]])  # 3 test points, 1D
        >>> var = marg_gp.predict_variance(z_test)
        >>> print(var.shape)  # (3,)
        >>> # Variance is bounded by prior variance
        >>> assert (var <= marg_gp.u_not_p() + 1e-10).all()
        """
        self._validate_input_shape(z_p)
        bkd = self._bkd

        # τ̃(z_p) shape: (n_test, N)
        tau_tilde = self._compute_tau_tilde(z_p)

        # C_p(z_p, z_p) shape: (n_test,) - kernel diagonal over active dims
        C_p_diag = self._compute_C_p(z_p)

        # Compute τ̃^T A^{-1} τ̃ for each test point
        # Using Cholesky: A^{-1} = L^{-T} L^{-1}, so τ̃^T A^{-1} τ̃ = ||L^{-1} τ̃||²
        # tau_tilde.T shape: (N, n_test)
        # Solve L v = tau_tilde.T -> v shape: (N, n_test)
        L = self._cholesky.factor()
        v = bkd.solve_triangular(L, tau_tilde.T, lower=True)

        # ||v||² along the N dimension -> shape (n_test,)
        quadratic_form: Array = bkd.sum(v * v, axis=0)

        # C̃*(z_p, z_p) = u_~p · C_p(z_p, z_p) - τ̃^T A^{-1} τ̃
        variance = self._u_not_p * C_p_diag - quadratic_form

        # Ensure non-negative (numerical issues may cause small negatives)
        variance = bkd.maximum(variance, bkd.asarray(0.0))

        return variance

    def predict(self, z_p: Array) -> tuple[Array, Array]:
        """
        Predict both mean and variance at test points.

        This is more efficient than calling predict_mean and predict_variance
        separately when both are needed, as it reuses the τ̃ computation.

        Parameters
        ----------
        z_p : Array
            Test points in the active dimensions, shape (ndims, n_test).

        Returns
        -------
        mean : Array
            Marginalized mean at test points, shape (n_test,).
        variance : Array
            Marginalized variance at test points, shape (n_test,).

        Examples
        --------
        >>> z_test = bkd.array([[-0.5, 0.0, 0.5]])
        >>> mean, var = marg_gp.predict(z_test)
        """
        self._validate_input_shape(z_p)
        bkd = self._bkd

        # Compute τ̃(z_p) once for both mean and variance
        tau_tilde = self._compute_tau_tilde(z_p)

        # Mean: m̃* = τ̃ @ αᵀ (alpha shape is (nqoi, N))
        alpha = self._gp.alpha()
        mean = tau_tilde @ alpha.T
        if mean.shape[1] == 1:
            mean = bkd.reshape(mean, (-1,))

        # Variance: u_~p · C_p - τ̃^T A^{-1} τ̃
        C_p_diag = self._compute_C_p(z_p)
        L = self._cholesky.factor()
        v = bkd.solve_triangular(L, tau_tilde.T, lower=True)
        quadratic_form: Array = bkd.sum(v * v, axis=0)
        variance = self._u_not_p * C_p_diag - quadratic_form
        variance = bkd.maximum(variance, bkd.asarray(0.0))

        return mean, variance

    def _validate_input_shape(self, z_p: Array) -> None:
        """
        Validate that input z_p has correct shape (ndims, n_test).

        Parameters
        ----------
        z_p : Array
            Test points to validate.

        Raises
        ------
        ValueError
            If z_p has incorrect shape.
        """
        expected_ndims = len(self._active_dims)
        if z_p.ndim != 2:
            raise ValueError(
                f"z_p must be 2D with shape (ndims, n_test), got {z_p.ndim}D"
            )
        if z_p.shape[0] != expected_ndims:
            raise ValueError(
                f"z_p must have {expected_ndims} rows (one per active dim), "
                f"got {z_p.shape[0]}. Active dims are {self._active_dims}."
            )
