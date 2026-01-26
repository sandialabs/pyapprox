"""
Separable kernel integral calculator for GP statistics.

This module provides the SeparableKernelIntegralCalculator class which computes
the multidimensional kernel integrals needed for GP statistics by factoring
them into products of 1D integrals.

For a product kernel C(x, z) = prod_k C_k(x_k, z_k), the multidimensional
integrals factor as:
    tau = prod_k tau_k
    P = prod_k P_k (Hadamard product)
    u = prod_k u_k

where tau_k, P_k, u_k are computed using 1D quadrature rules matched to
the marginal distributions.

Usage
-----
Users should create quadrature rules using the sparse grid infrastructure:

    from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
        create_basis_factories,
    )
    from pyapprox.typing.probability.univariate import UniformMarginal

    bkd = NumpyBkd()
    marginals = [UniformMarginal(-1.0, 1.0, bkd), UniformMarginal(-1.0, 1.0, bkd)]
    basis_factories = create_basis_factories(marginals, bkd, "gauss")

    # Create bases and get quadrature rules
    bases = [f.create_basis() for f in basis_factories]
    for b in bases:
        b.set_nterms(nquad_points)

    calc = SeparableKernelIntegralCalculator(gp, bases, bkd=bkd)
"""

from typing import Generic, List, Callable, Optional
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.validation import (
    validate_separable_kernel,
    validate_zero_mean,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.integrals_1d import (
    compute_tau_1d,
    compute_P_1d,
    compute_u_1d,
    compute_nu_1d,
    compute_lambda_1d,
    compute_Pi_1d,
    compute_xi1_1d,
    compute_Gamma_1d,
    compute_conditional_P_1d,
)
from pyapprox.typing.surrogates.kernels.composition import SeparableProductKernel
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.surrogates.affine.protocols.quadrature import (
    QuadratureRuleStatefulProtocol,
)
from pyapprox.typing.probability.protocols.distribution import MarginalProtocol


def _extract_1d_kernels(kernel: KernelProtocol[Array]) -> List[KernelProtocol[Array]]:
    """
    Extract 1D kernel components from a separable product kernel.

    Parameters
    ----------
    kernel : KernelProtocol[Array]
        The SeparableProductKernel or single 1D kernel.

    Returns
    -------
    kernels : List[KernelProtocol[Array]]
        List of 1D kernel components.
    """
    if isinstance(kernel, SeparableProductKernel):
        # Extract 1D kernels from SeparableProductKernel
        return [kernel.get_kernel_1d(dim) for dim in range(kernel.nvars())]
    elif kernel.nvars() == 1:
        # Single 1D kernel
        return [kernel]
    else:
        raise TypeError(
            f"Cannot extract 1D kernels from {type(kernel).__name__}. "
            f"Use SeparableProductKernel for multi-dimensional inputs."
        )


class SeparableKernelIntegralCalculator(Generic[Array]):
    """
    Calculator for kernel integrals needed for GP statistics.

    This class computes the multidimensional kernel integrals (tau, P, u, etc.)
    by factoring them into products of 1D integrals. This factorization is
    only valid for separable (product) kernels.

    The integrals are computed using numerical quadrature rules that the user
    provides. The quadrature rules should be created using the sparse grid
    infrastructure, which handles domain transformations correctly.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process with a separable (product) kernel.
    quadrature_bases : List[QuadratureRuleStatefulProtocol[Array]]
        List of 1D quadrature bases (e.g., LagrangeBasis1D from sparse grids),
        one per input dimension. Each basis must have `set_nterms()` called
        before passing to this constructor. The quadrature rules must return
        points in the physical domain with weights normalized for the
        probability measure.
    marginals : List[MarginalProtocol[Array]]
        List of 1D marginal distributions, one per input dimension.
        These define the probability measure for integration and are used
        by GaussianProcessEnsemble for Monte Carlo sampling.
    bkd : Backend[Array], optional
        Backend for numerical operations. If None, uses GP's backend.

    Raises
    ------
    TypeError
        If the kernel is not separable (product structure).
    ValueError
        If the GP does not use zero mean.
    RuntimeError
        If the GP has not been fitted.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
    >>> from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
    >>> from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    ...     create_basis_factories,
    ... )
    >>> from pyapprox.typing.probability.univariate import UniformMarginal
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    >>> k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    >>> kernel = k1 * k2
    >>> gp = ExactGaussianProcess(kernel, 2, bkd)
    >>> X = bkd.array(np.random.rand(2, 10) * 2 - 1)
    >>> y = bkd.array(np.random.rand(10, 1))
    >>> gp.fit(X, y)
    >>>
    >>> # Create quadrature bases using sparse grid infrastructure
    >>> marginals = [UniformMarginal(-1.0, 1.0, bkd), UniformMarginal(-1.0, 1.0, bkd)]
    >>> factories = create_basis_factories(marginals, bkd, "gauss")
    >>> bases = [f.create_basis() for f in factories]
    >>> for b in bases:
    ...     b.set_nterms(20)  # Number of quadrature points
    >>>
    >>> calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    >>> tau = calc.tau()  # Shape: (10,)
    >>> P = calc.P()      # Shape: (10, 10)
    >>> u = calc.u()      # Scalar
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        quadrature_bases: List[QuadratureRuleStatefulProtocol[Array]],
        marginals: List[MarginalProtocol[Array]],
        bkd: Optional[Backend[Array]] = None,
    ):
        # Validate GP is fitted
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before computing statistics. "
                "Call gp.fit(X_train, y_train) first."
            )

        # Validate kernel is separable
        validate_separable_kernel(gp.kernel())

        # Validate GP uses zero mean
        validate_zero_mean(gp)

        # Validate number of quadrature bases matches nvars
        nvars = gp.nvars()
        if len(quadrature_bases) != nvars:
            raise ValueError(
                f"Number of quadrature bases ({len(quadrature_bases)}) must match "
                f"number of input variables ({nvars})."
            )

        # Validate number of marginals matches nvars
        if len(marginals) != nvars:
            raise ValueError(
                f"Number of marginals ({len(marginals)}) must match "
                f"number of input variables ({nvars})."
            )

        self._gp = gp
        self._quadrature_bases = quadrature_bases
        self._marginals = marginals

        # Use GP's backend if not provided
        if bkd is None:
            bkd = gp.bkd()
        self._bkd = bkd

        # Extract 1D kernel components
        self._kernels_1d = _extract_1d_kernels(gp.kernel())

        # Validate we have the right number of 1D kernels
        if len(self._kernels_1d) != nvars:
            raise ValueError(
                f"Product kernel has {len(self._kernels_1d)} 1D components, "
                f"but GP has {nvars} input variables. Each dimension needs "
                f"exactly one 1D kernel component."
            )

        # Get training data
        self._train_samples = gp.data().X()  # Shape: (nvars, N)
        self._n_train = gp.data().n_samples()

        # Get quadrature rules from the bases (points in physical domain)
        self._quad_samples: List[Array] = []
        self._quad_weights: List[Array] = []
        for dim in range(nvars):
            basis = quadrature_bases[dim]
            samples, weights = basis.quadrature_rule()
            # samples shape: (1, nquad), weights shape: (nquad, 1)
            self._quad_samples.append(samples)
            # Flatten weights to (nquad,)
            self._quad_weights.append(bkd.reshape(weights, (-1,)))

        # Cache for computed integrals
        self._cache: dict = {}

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def marginals(self) -> List[MarginalProtocol[Array]]:
        """Return the list of marginal distributions."""
        return self._marginals

    def _get_kernel_callable(self, dim: int) -> Callable[[Array, Array], Array]:
        """
        Get a callable wrapper for the 1D kernel at dimension dim.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Callable[[Array, Array], Array]
            Function that evaluates the 1D kernel.
        """
        kernel = self._kernels_1d[dim]
        return lambda x1, x2: kernel(x1, x2)

    def _get_train_samples_1d(self, dim: int) -> Array:
        """
        Get training samples for dimension dim.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Array
            Training samples for dimension dim, shape (1, N).
        """
        return self._train_samples[dim:dim+1, :]

    def tau(self) -> Array:
        """
        Compute the tau vector for E[f] computation.

        tau_i = integral C(x, x^(i)) rho(x) dx

        For separable kernels: tau = prod_k tau_k (element-wise product).

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        if 'tau' in self._cache:
            return self._cache['tau']

        nvars = len(self._kernels_1d)

        # Compute 1D tau for each dimension and take product
        tau = self._bkd.ones((self._n_train,))
        for dim in range(nvars):
            tau_1d = compute_tau_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_train_samples_1d(dim),
                self._get_kernel_callable(dim),
                self._bkd
            )
            tau = tau * tau_1d

        self._cache['tau'] = tau
        return tau

    def P(self) -> Array:
        """
        Compute the P matrix for Var[f] computation.

        P_ij = integral C(x, x^(i)) C(x, x^(j)) rho(x) dx

        For separable kernels: P = prod_k P_k (Hadamard product).

        Returns
        -------
        Array
            Shape (N, N) where N is the number of training points.
            This matrix is symmetric positive semi-definite.
        """
        if 'P' in self._cache:
            return self._cache['P']

        nvars = len(self._kernels_1d)

        # Compute 1D P for each dimension and take Hadamard product
        P = self._bkd.ones((self._n_train, self._n_train))
        for dim in range(nvars):
            P_1d = compute_P_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_train_samples_1d(dim),
                self._get_kernel_callable(dim),
                self._bkd
            )
            P = P * P_1d

        self._cache['P'] = P
        return P

    def u(self) -> Array:
        """
        Compute the u scalar for prior variance.

        u = integral integral C(x, z) rho(x) rho(z) dx dz

        For separable kernels: u = prod_k u_k.

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        if 'u' in self._cache:
            return self._cache['u']

        nvars = len(self._kernels_1d)

        # Compute 1D u for each dimension and take product
        u = self._bkd.asarray(1.0)
        for dim in range(nvars):
            u_1d = compute_u_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_kernel_callable(dim),
                self._bkd
            )
            u = u * u_1d

        self._cache['u'] = u
        return u

    def nu(self) -> Array:
        """
        Compute the nu scalar for Var[gamma] computation.

        nu = integral integral C(x, z)^2 rho(x) rho(z) dx dz

        For separable kernels: nu = prod_k nu_k.

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        if 'nu' in self._cache:
            return self._cache['nu']

        nvars = len(self._kernels_1d)

        # Compute 1D nu for each dimension and take product
        nu = self._bkd.asarray(1.0)
        for dim in range(nvars):
            nu_1d = compute_nu_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_kernel_callable(dim),
                self._bkd
            )
            nu = nu * nu_1d

        self._cache['nu'] = nu
        return nu

    def lambda_vec(self) -> Array:
        """
        Compute the lambda vector for Var[gamma] computation.

        lambda_i = integral C(z, z) C(z, x^(i)) rho(z) dz

        For separable kernels: lambda = prod_k lambda_k.

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        if 'lambda_vec' in self._cache:
            return self._cache['lambda_vec']

        nvars = len(self._kernels_1d)

        # Compute 1D lambda for each dimension and take product
        lambda_vec = self._bkd.ones((self._n_train,))
        for dim in range(nvars):
            lambda_1d = compute_lambda_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_train_samples_1d(dim),
                self._get_kernel_callable(dim),
                self._bkd
            )
            lambda_vec = lambda_vec * lambda_1d

        self._cache['lambda_vec'] = lambda_vec
        return lambda_vec

    def Pi(self) -> Array:
        """
        Compute the Pi matrix for Var[gamma] computation.

        Pi_ij = integral integral C(x, x^(i)) C(x, z) C(z, x^(j)) rho(x) rho(z) dx dz

        For separable kernels: Pi = prod_k Pi_k (Hadamard product).

        Returns
        -------
        Array
            Shape (N, N) where N is the number of training points.
        """
        if 'Pi' in self._cache:
            return self._cache['Pi']

        nvars = len(self._kernels_1d)

        # Compute 1D Pi for each dimension and take Hadamard product
        Pi = self._bkd.ones((self._n_train, self._n_train))
        for dim in range(nvars):
            Pi_1d = compute_Pi_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_train_samples_1d(dim),
                self._get_kernel_callable(dim),
                self._bkd
            )
            Pi = Pi * Pi_1d

        self._cache['Pi'] = Pi
        return Pi

    def xi1(self) -> Array:
        """
        Compute the xi1 scalar for Var[gamma] computation.

        xi1 = integral integral integral C(w, x) C(w, z) rho(w) rho(x) rho(z) dw dx dz

        For separable kernels: xi1 = prod_k xi1_k.

        Returns
        -------
        Array
            Scalar (0-dimensional or shape () array).
        """
        if 'xi1' in self._cache:
            return self._cache['xi1']

        nvars = len(self._kernels_1d)

        # Compute 1D xi1 for each dimension and take product
        xi1 = self._bkd.asarray(1.0)
        for dim in range(nvars):
            xi1_1d = compute_xi1_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_kernel_callable(dim),
                self._bkd
            )
            xi1 = xi1 * xi1_1d

        self._cache['xi1'] = xi1
        return xi1

    def Gamma(self) -> Array:
        """
        Compute the Gamma vector for Var[gamma] computation.

        Gamma_i = integral integral C(x^(i), z) C(z, v) rho(z) rho(v) dz dv

        For separable kernels: Gamma = prod_k Gamma_k (element-wise product).

        This integral is needed for the correct vartheta_2 formula in
        variance_of_variance().

        Returns
        -------
        Array
            Shape (N,) where N is the number of training points.
        """
        if 'Gamma' in self._cache:
            return self._cache['Gamma']

        nvars = len(self._kernels_1d)

        # Compute 1D Gamma for each dimension and take product
        Gamma = self._bkd.ones((self._n_train,))
        for dim in range(nvars):
            Gamma_1d = compute_Gamma_1d(
                self._quad_samples[dim],
                self._quad_weights[dim],
                self._get_train_samples_1d(dim),
                self._get_kernel_callable(dim),
                self._bkd
            )
            Gamma = Gamma * Gamma_1d

        self._cache['Gamma'] = Gamma
        return Gamma

    def conditional_P(self, index: Array) -> Array:
        """
        Compute conditional P matrix for sensitivity analysis.

        For index set p (dimensions where index[k] = 1):
        - Dimensions k ∈ p (conditioned on): use standard P_{k,ij}
        - Dimensions k ∉ p (integrated out): use P̃_{k,ij} = τ_{k,i} τ_{k,j}

        Mathematical Background
        -----------------------
        For sensitivity analysis, we need to compute conditional variances
        where some dimensions are conditioned on and others are integrated out.

        The standard P matrix uses a single integration point:
        P_{k,ij} = ∫ C_k(x, x^(i)) C_k(x, x^(j)) ρ(x) dx

        The conditional P̃ matrix uses two independent integration points:
        P̃_{k,ij} = ∫∫ C_k(x, x^(i)) C_k(z, x^(j)) ρ(x) ρ(z) dx dz
                 = τ_{k,i} · τ_{k,j}

        The key insight is that P̃ factors into an outer product τ τᵀ because
        the two integration variables x and z are independent.

        Mathematical reference: docs/plans/gp-stats-phase4-sensitivity-bugfix.md

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
            index[k] = 1: dimension k is CONDITIONED ON (use standard P_k)
            index[k] = 0: dimension k is INTEGRATED OUT (use P̃_k = τ_k τ_k^T)

        Returns
        -------
        Array
            Conditional P matrix, shape (N, N).

        Raises
        ------
        ValueError
            If index length does not match number of dimensions.

        Examples
        --------
        # Main effect of variable 0 (only z_0 conditioned):
        index = bkd.asarray([1.0, 0.0, 0.0])  # nvars=3
        P_p = calc.conditional_P(index)

        # Total effect of variable 0 (all except z_0 conditioned):
        index = bkd.asarray([0.0, 1.0, 1.0])  # nvars=3
        P_p = calc.conditional_P(index)
        """
        nvars = len(self._kernels_1d)
        index_np = self._bkd.to_numpy(index).flatten()

        if len(index_np) != nvars:
            raise ValueError(
                f"index must have length {nvars} (one per dimension), "
                f"got length {len(index_np)}"
            )

        P_p = self._bkd.ones((self._n_train, self._n_train))
        for dim in range(nvars):
            if index_np[dim] == 1:
                # Conditioned on: use standard P (single integration point)
                P_1d = compute_P_1d(
                    self._quad_samples[dim],
                    self._quad_weights[dim],
                    self._get_train_samples_1d(dim),
                    self._get_kernel_callable(dim),
                    self._bkd
                )
            else:
                # Integrated out: use outer product P̃ = τ τᵀ
                P_1d = compute_conditional_P_1d(
                    self._quad_samples[dim],
                    self._quad_weights[dim],
                    self._get_train_samples_1d(dim),
                    self._get_kernel_callable(dim),
                    self._bkd
                )
            P_p = P_p * P_1d

        return P_p

    def conditional_u(self, index: Array) -> Array:
        """
        Compute conditional u scalar for sensitivity analysis.

        For index set p (dimensions where index[k] = 1):
        - Dimensions k ∈ p (conditioned on): contribute factor 1
        - Dimensions k ∉ p (integrated out): contribute factor u_k

        u_p = ∏_{k ∉ p} u_k

        Mathematical Background
        -----------------------
        When dimension k is conditioned on (fixed), we're evaluating:
        ∫ C_k(z_k, z_k) ρ_k(z_k) dz_k

        For correlation kernels (normalized so C_k(x, x) = 1):
        ∫ 1 · ρ_k(z_k) dz_k = 1

        When dimension k is integrated out, we use the standard u_k:
        u_k = ∫∫ C_k(x, z) ρ_k(x) ρ_k(z) dx dz

        Mathematical reference: docs/plans/gp-stats-phase4-sensitivity-bugfix.md

        Parameters
        ----------
        index : Array
            Binary vector of shape (nvars,).
            index[k] = 1: dimension k is CONDITIONED ON (factor = 1)
            index[k] = 0: dimension k is INTEGRATED OUT (factor = u_k)

        Returns
        -------
        Array
            Conditional u scalar.

        Raises
        ------
        ValueError
            If index length does not match number of dimensions.

        Examples
        --------
        # All conditioned: u_p = 1
        index = bkd.asarray([1.0, 1.0, 1.0])
        u_p = calc.conditional_u(index)  # Returns 1.0

        # None conditioned: u_p = u (full prior variance)
        index = bkd.asarray([0.0, 0.0, 0.0])
        u_p = calc.conditional_u(index)  # Returns u = ∏_k u_k
        """
        nvars = len(self._kernels_1d)
        index_np = self._bkd.to_numpy(index).flatten()

        if len(index_np) != nvars:
            raise ValueError(
                f"index must have length {nvars} (one per dimension), "
                f"got length {len(index_np)}"
            )

        u_p = self._bkd.asarray(1.0)
        for dim in range(nvars):
            if index_np[dim] == 0:
                # Integrated out: multiply by u_k
                u_1d = compute_u_1d(
                    self._quad_samples[dim],
                    self._quad_weights[dim],
                    self._get_kernel_callable(dim),
                    self._bkd
                )
                u_p = u_p * u_1d
            # If index[k] = 1 (conditioned on): factor is 1, no multiplication needed

        return u_p
