"""MSE-based gradient fitter for SUPN.

Default optimizer chains Adam warm-starting with trust-region Newton-CG,
matching the two-phase training procedure recommended in Morrow et al.
(2025, Section 4).  When ROL (Rapid Optimization Library) is available,
the trust-region phase uses ROL for faster convergence; otherwise it
falls back to scipy's trust-constr.
"""

from typing import Generic, Optional

from pyapprox.optimization.minimize.adam.adam_optimizer import (
    AdamOptimizer,
)
from pyapprox.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.surrogates.supn.fitters.results import SUPNFitterResult
from pyapprox.surrogates.supn.losses import SUPNMSELoss
from pyapprox.surrogates.supn.supn import SUPN
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available


def supn_paper_rol_parameter_list() -> "pyrol.ParameterList":
    """Return ROL parameters matching Morrow et al. (2025, Section 4).

    Uses a secant (L-BFGS) preconditioner for the truncated-CG
    trust-region subproblem and a large initial trust-region radius.

    Returns
    -------
    pyrol.ParameterList
        ROL parameter list ready to pass to ``ROLOptimizer(parameters=...)``.

    Examples
    --------
    >>> from pyapprox.surrogates.supn.fitters import (
    ...     SUPNMSEFitter, supn_paper_rol_parameter_list,
    ... )
    >>> from pyapprox.optimization.minimize.rol.rol_optimizer import (
    ...     ROLOptimizer,
    ... )
    >>> params = supn_paper_rol_parameter_list()
    >>> rol = ROLOptimizer(verbosity=0, parameters=params)
    >>> fitter = SUPNMSEFitter(bkd)
    >>> fitter.set_optimizer(
    ...     ChainedOptimizer(AdamOptimizer(lr=1e-3, maxiter=500), rol)
    ... )
    """
    import pyrol

    params = pyrol.ParameterList()

    params["General"] = pyrol.ParameterList()
    params["General"]["Output Level"] = 1
    params["General"]["secant"] = pyrol.ParameterList()
    params["General"]["secant"]["Use as Preconditioner"] = True

    params["Status Test"] = pyrol.ParameterList()
    params["Status Test"]["Iteration Limit"] = 500
    params["Status Test"]["Gradient Tolerance"] = 5e-5
    params["Status Test"]["Step Tolerance"] = 5e-5

    params["Step"] = pyrol.ParameterList()
    params["Step"]["Trust Region"] = pyrol.ParameterList()
    params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
    params["Step"]["Trust Region"]["Initial Radius"] = 100.0
    params["Step"]["Trust Region"]["Radius Growing Threshold"] = 0.5
    params["Step"]["Trust Region"][
        "Radius Shrinking Rate (Negative rho)"
    ] = 0.01
    params["Step"]["Trust Region"][
        "Radius Shrinking Rate (Positive rho)"
    ] = 0.01

    return params


class SUPNMSEFitter(Generic[Array]):
    """MSE-based fitter for SUPN surrogates.

    Minimizes L(theta) = (1/2K) ||f_theta(X) - Y||^2 using gradient-based
    optimization with analytical gradient and Hessian-vector product.

    Default optimizer chains Adam warm-starting (500 iterations) with
    trust-region Newton-CG, matching the two-phase training procedure
    in Morrow et al. (2025, Section 4).  When ROL is available the
    trust-region phase uses ROL; otherwise it falls back to scipy's
    trust-constr.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> fitter = SUPNMSEFitter(bkd)
    >>> fitter.set_optimizer(ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8))
    >>> result = fitter.fit(supn, samples, values)
    >>> fitted_supn = result.surrogate()
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def set_optimizer(
        self, optimizer: BindableOptimizerProtocol[Array]
    ) -> None:
        """Set custom optimizer.

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            Configured optimizer. Cloned during fit() to avoid shared state.
        """
        self._optimizer = optimizer

    def optimizer(self) -> Optional[BindableOptimizerProtocol[Array]]:
        """Return current optimizer (or None if using default)."""
        return self._optimizer

    def fit(
        self,
        surrogate: SUPN[Array],
        samples: Array,
        values: Array,
        bounds: Optional[Array] = None,
    ) -> SUPNFitterResult[Array]:
        """Fit SUPN to data using gradient-based optimization.

        Parameters
        ----------
        surrogate : SUPN[Array]
            The SUPN to fit (provides initial parameters).
        samples : Array
            Training samples. Shape: (nvars, nsamples)
        values : Array
            Training values. Shape: (nqoi, nsamples) or (nsamples,)
        bounds : Array, optional
            Parameter bounds. Shape: (nparams, 2).
            If None, uses unbounded.

        Returns
        -------
        SUPNFitterResult[Array]
            Result containing fitted surrogate and diagnostics.
        """
        if not isinstance(surrogate, SUPN):
            raise TypeError(
                f"SUPNMSEFitter only works with SUPN, "
                f"got {type(surrogate).__name__}"
            )

        if values.ndim == 1:
            values = self._bkd.reshape(values, (1, -1))

        nvars = surrogate.nvars()
        nqoi = surrogate.nqoi()
        nsamples = samples.shape[1]

        if samples.shape[0] != nvars:
            raise ValueError(
                f"samples has {samples.shape[0]} variables, "
                f"surrogate expects {nvars}"
            )
        if values.shape[0] != nqoi:
            raise ValueError(
                f"values has {values.shape[0]} QoIs, surrogate has {nqoi}"
            )
        if values.shape[1] != nsamples:
            raise ValueError(
                f"values has {values.shape[1]} samples, "
                f"samples has {nsamples}"
            )

        nparams = surrogate.nparams()

        # Create loss function (caches basis matrix)
        loss = SUPNMSELoss(surrogate, samples, values, self._bkd)

        # Get or create optimizer (clone if user-provided)
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            adam = AdamOptimizer[Array](lr=1e-3, maxiter=500)
            if package_available("pyrol"):
                from pyapprox.optimization.minimize.rol.rol_optimizer import (
                    ROLOptimizer,
                )

                trust: BindableOptimizerProtocol[Array] = (
                    ROLOptimizer[Array](verbosity=0)
                )
            else:
                trust = ScipyTrustConstrOptimizer[Array](
                    verbosity=0, maxiter=1000
                )
            optimizer = ChainedOptimizer(adam, trust)

        # Get bounds (default to unbounded)
        if bounds is None:
            bounds = self._default_bounds(nparams)

        # Bind optimizer to loss and bounds
        optimizer.bind(loss, bounds)

        # Get initial guess from current parameters
        init_guess = surrogate._flatten_params()
        if init_guess.ndim == 1:
            init_guess = self._bkd.reshape(init_guess, (-1, 1))

        # Run optimization
        result = optimizer.minimize(init_guess)

        # Extract optimal parameters and create fitted surrogate
        optimal_params = result.optima()
        if optimal_params.ndim == 2:
            optimal_params = optimal_params[:, 0]
        fitted_surrogate = surrogate.with_params(optimal_params)

        # Compute final loss
        final_loss = self._bkd.to_float(loss(optimal_params)[0, 0])

        return SUPNFitterResult(
            surrogate=fitted_surrogate,
            optimizer_result=result,
            final_loss=final_loss,
        )

    def _default_bounds(self, nparams: int) -> Array:
        """Create default unbounded bounds."""
        import numpy as np

        bounds = np.full((nparams, 2), [-np.inf, np.inf])
        return self._bkd.asarray(bounds)

    def __repr__(self) -> str:
        opt_str = (
            type(self._optimizer).__name__
            if self._optimizer is not None
            else "ChainedOptimizer(Adam+TrustRegion)"
        )
        return f"SUPNMSEFitter(optimizer={opt_str})"
