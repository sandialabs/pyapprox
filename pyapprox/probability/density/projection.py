"""Density coefficient fitting via L2 projection."""

from typing import Generic, Optional

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.probability.density._fitters import LinearDensityFitter
from pyapprox.probability.density.kernel_density_basis import (
    KernelDensityBasis,
)
from pyapprox.probability.density.protocols import DensityBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ProjectionDensityFitter(Generic[Array]):
    """Fit density coefficients via L2 projection: M*d = b.

    Given quadrature data (y_values, weights), computes the projection
    coefficients d such that f_Y(y) = sum_j d_j phi_j(y) minimizes
    the L2 error.

    Parameters
    ----------
    basis : DensityBasisProtocol[Array]
        The density basis providing evaluate() and mass_matrix().

    Raises
    ------
    TypeError
        If basis does not satisfy DensityBasisProtocol.
    """

    def __init__(self, basis: DensityBasisProtocol[Array]) -> None:
        if not isinstance(basis, DensityBasisProtocol):
            raise TypeError(
                f"basis must satisfy DensityBasisProtocol, got {type(basis).__name__}"
            )
        self._basis = basis
        self._bkd = basis.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> DensityBasisProtocol[Array]:
        """Return the density basis."""
        return self._basis

    def fit(self, y_values: Array, weights: Array) -> Array:
        """Fit density coefficients via L2 projection.

        Solves M*d = b where:
        - M_ij = int phi_i(y) phi_j(y) dy  (mass matrix)
        - b_i = sum_q w_q phi_i(y_q)       (weighted evaluation)

        Parameters
        ----------
        y_values : Array
            Quadrature points in y-space. Shape: (1, nquad).
        weights : Array
            Quadrature weights. Shape: (nquad,).

        Returns
        -------
        Array
            Density coefficients d. Shape: (nbasis,).
        """
        bkd = self._bkd
        Phi = self._basis.evaluate(y_values)  # (nbasis, nquad)
        b = bkd.dot(Phi, weights)  # (nbasis,)
        M = self._basis.mass_matrix()  # (nbasis, nbasis)
        fitter = LinearDensityFitter(bkd)
        return fitter.fit(M, b)  # (nbasis,)

    def ise_criterion(self, y_values: Array, weights: Array) -> Array:
        """Compute ISE criterion: b^T M^{-1} b.

        Used for kernel hyperparameter optimization:
        theta* = argmax_theta b(theta)^T M(theta)^{-1} b(theta)

        Parameters
        ----------
        y_values : Array
            Quadrature points in y-space. Shape: (1, nquad).
        weights : Array
            Quadrature weights. Shape: (nquad,).

        Returns
        -------
        Array
            Scalar ISE criterion value.
        """
        bkd = self._bkd
        Phi = self._basis.evaluate(y_values)  # (nbasis, nquad)
        b = bkd.dot(Phi, weights)  # (nbasis,)
        M = self._basis.mass_matrix()  # (nbasis, nbasis)
        M_inv_b = bkd.solve(M, b)  # (nbasis,)
        return bkd.sum(b * M_inv_b)


class _NegativeISELoss(Generic[Array]):
    """Negative ISE criterion as an objective for optimizer binding.

    Satisfies FunctionProtocol: nvars = nactive hyperparameters, nqoi = 1.
    __call__ takes params shape (nvars, 1), returns shape (1, 1).
    """

    def __init__(
        self,
        basis: KernelDensityBasis[Array],
        y_values: Array,
        weights: Array,
    ) -> None:
        self._basis = basis
        self._bkd = basis.bkd()
        self._hyp_list = basis.hyp_list()
        self._y_values = y_values
        self._weights = weights
        self._fitter = ProjectionDensityFitter(basis)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._hyp_list.nactive_params()

    def nqoi(self) -> int:
        return 1

    def __call__(self, params: Array) -> Array:
        bkd = self._bkd
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]
        self._hyp_list.set_active_values(params)
        ise = self._fitter.ise_criterion(self._y_values, self._weights)
        return bkd.reshape(-ise, (1, 1))


class ISEOptimizingFitter(Generic[Array]):
    """Optimize kernel hyperparameters via ISE criterion, then fit.

    Finds theta* = argmax b(theta)^T M(theta)^{-1} b(theta), then solves
    M(theta*) * d = b(theta*) for density coefficients.

    Parameters
    ----------
    basis : KernelDensityBasis[Array]
        Kernel density basis whose hyperparameters will be optimized.
    optimizer : BindableOptimizerProtocol[Array], optional
        Optimizer to use. If None, defaults to ChainedOptimizer
        (differential evolution + trust-constr).

    Raises
    ------
    TypeError
        If basis is not a KernelDensityBasis.
    """

    def __init__(
        self,
        basis: KernelDensityBasis[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        if not isinstance(basis, KernelDensityBasis):
            raise TypeError(
                f"basis must be KernelDensityBasis, got {type(basis).__name__}"
            )
        self._basis = basis
        self._bkd = basis.bkd()
        self._optimizer = optimizer

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> KernelDensityBasis[Array]:
        """Return the kernel density basis."""
        return self._basis

    def fit(self, y_values: Array, weights: Array) -> Array:
        """Optimize hyperparameters via ISE, then fit density coefficients.

        Parameters
        ----------
        y_values : Array
            Quadrature points in y-space. Shape: (1, nquad).
        weights : Array
            Quadrature weights. Shape: (nquad,).

        Returns
        -------
        Array
            Density coefficients d. Shape: (nbasis,).
        """
        bkd = self._bkd
        basis = self._basis
        hyp_list = basis.hyp_list()

        loss = _NegativeISELoss(basis, y_values, weights)

        bounds = hyp_list.get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            # TODO: Is lazy import necessary here
            from pyapprox.optimization.minimize.chained.chained_optimizer import (
                ChainedOptimizer,
            )
            from pyapprox.optimization.minimize.scipy.diffevol import (
                ScipyDifferentialEvolutionOptimizer,
            )
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )

            optimizer = ChainedOptimizer(
                ScipyDifferentialEvolutionOptimizer(maxiter=50),
                ScipyTrustConstrOptimizer(verbosity=0, maxiter=200),
            )

        optimizer.bind(loss, bounds)

        init_guess = hyp_list.get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = bkd.reshape(init_guess, (-1, 1))

        result = optimizer.minimize(init_guess)

        optimal_params = result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        hyp_list.set_active_values(optimal_params)

        fitter = ProjectionDensityFitter(basis)
        return fitter.fit(y_values, weights)

# TODO: Why is __all__ in this file
__all__ = ["ProjectionDensityFitter", "ISEOptimizingFitter"]
