"""Fitter for variational inference via ELBO optimization.

Encapsulates the optimize-and-push pattern: extract bounds from the
variational distribution's hyp_list, bind an optimizer, minimize,
and push optimal params back into the distribution.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.inverse.variational.elbo import ELBOObjective


class VIFitResult(Generic[Array]):
    """Result from variational inference fitting.

    All attributes are accessed via methods per CLAUDE.md convention.

    Parameters
    ----------
    neg_elbo : float
        Negative ELBO at the optimal parameters.
    initial_params : Array
        Variational parameters before optimization, shape ``(nvars, 1)``.
    optimized_params : Array
        Variational parameters after optimization, shape ``(nvars, 1)``.
    optimization_result : ScipyOptimizerResultWrapper[Array]
        The raw optimization result.
    """

    def __init__(
        self,
        neg_elbo: float,
        initial_params: Array,
        optimized_params: Array,
        optimization_result: ScipyOptimizerResultWrapper[Array],
    ) -> None:
        self._neg_elbo = neg_elbo
        self._initial_params = initial_params
        self._optimized_params = optimized_params
        self._opt_result = optimization_result

    def neg_elbo(self) -> float:
        """Return negative ELBO at the optimum."""
        return self._neg_elbo

    def initial_params(self) -> Array:
        """Return variational parameters before optimization."""
        return self._initial_params

    def optimized_params(self) -> Array:
        """Return variational parameters after optimization."""
        return self._optimized_params

    def optimization_result(self) -> ScipyOptimizerResultWrapper[Array]:
        """Return the raw optimization result."""
        return self._opt_result

    def __repr__(self) -> str:
        return (
            f"VIFitResult(neg_elbo={self._neg_elbo:.4f}, "
            f"success={self._opt_result.success()})"
        )


class VariationalFitter(Generic[Array]):
    """Fitter for variational inference via ELBO optimization.

    Encapsulates the optimize-and-push pattern for variational
    distributions. Accepts a pre-constructed ELBOObjective and
    minimizes it using a configurable optimizer.

    The variational distribution held by the ELBO is modified in-place
    (parameters pushed via ``elbo(result.optima())``), so the caller's
    reference to the var_dist reflects the fitted state after ``fit()``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Pre-configured optimizer template. If None, uses a default
        ScipyTrustConstrOptimizer with maxiter=500 and gtol=1e-8.
        The optimizer is cloned via ``.copy()`` for each ``fit()`` call
        to avoid shared state.

    Examples
    --------
    >>> fitter = VariationalFitter(bkd)
    >>> result = fitter.fit(elbo)
    >>> print(result.neg_elbo())

    >>> # With custom optimizer
    >>> opt = ScipyTrustConstrOptimizer(maxiter=1000, gtol=1e-10)
    >>> fitter = VariationalFitter(bkd, optimizer=opt)
    >>> result = fitter.fit(elbo, init_guess=my_guess)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def fit(
        self,
        elbo: ELBOObjective[Array],
        init_guess: Optional[Array] = None,
    ) -> VIFitResult[Array]:
        """Minimize the ELBO and push optimal params into the distribution.

        Parameters
        ----------
        elbo : ELBOObjective[Array]
            The ELBO objective to minimize.
        init_guess : Optional[Array]
            Initial guess for variational parameters, shape
            ``(nvars, 1)``. If None, uses zeros.

        Returns
        -------
        VIFitResult[Array]
            Result containing negative ELBO, parameters, and
            optimization metadata.
        """
        if init_guess is None:
            init_guess = self._bkd.zeros((elbo.nvars(), 1))
        initial_params = self._bkd.array(init_guess)

        bounds = elbo.bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.scipy.trust_constr import (
                ScipyTrustConstrOptimizer,
            )
            optimizer = ScipyTrustConstrOptimizer(
                maxiter=500, gtol=1e-8, verbosity=0,
            )

        optimizer.bind(elbo, bounds)
        opt_result = optimizer.minimize(init_guess)

        # Push optimal params into the variational distribution
        elbo(opt_result.optima())

        return VIFitResult(
            neg_elbo=opt_result.fun(),
            initial_params=initial_params,
            optimized_params=opt_result.optima(),
            optimization_result=opt_result,
        )
