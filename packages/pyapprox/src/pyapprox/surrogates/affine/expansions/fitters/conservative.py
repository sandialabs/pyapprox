"""Conservative fitters for basis expansions.

Conservative fitters adjust the constant term of a surrogate to ensure that
the risk measure of the surrogate is at least as large as the risk measure
of the training data. This provides a conservative approximation useful for
robust optimization and uncertainty quantification.

The conservative pattern requires risk measures that are:
- Positively homogeneous: R[t*X] = t*R[X] for t > 0
- Translation equivariant: R[X + c] = R[X] + c for constant c

This allows adjusting the constant coefficient to achieve conservativeness.
"""

from typing import Any, Generic, Optional

from pyapprox.probability.risk import (
    AverageValueAtRisk,
    SafetyMarginRiskMeasure,
)
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.quantile import (
    QuantileFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ConservativeLstSqFitter(Generic[Array]):
    """Conservative least squares fitter using Safety Margin Risk Measure.

    Fits via least squares, then adjusts constant term to ensure
    risk_measure(surrogate) >= risk_measure(data).

    The safety margin risk measure is: R[X] = E[X] + strength * std[X]

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    strength : float
        Strength parameter for safety margin. Higher = more conservative.
    rcond : float, optional
        Cutoff for small singular values in least squares. Default: machine precision.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        strength: float,
        rcond: Optional[float] = None,
    ):
        self._bkd = bkd
        self._strength = strength
        self._base_fitter = LeastSquaresFitter(bkd, rcond=rcond)
        self._risk_measure = SafetyMarginRiskMeasure(bkd, strength)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def strength(self) -> float:
        """Return strength parameter."""
        return self._strength

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit with conservative adjustment.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion with conservative adjustment.

        Raises
        ------
        ValueError
            If nqoi > 1.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        # Validate single QoI
        if values.shape[0] != 1:
            raise ValueError(
                f"ConservativeLstSqFitter only supports nqoi=1, got {values.shape[0]}"
            )

        # 1. Fit via base fitter
        result = self._base_fitter.fit(expansion, samples, values)
        params = bkd.copy(result.params())  # (nterms, 1)

        # 2. Get basis matrix for residual computation
        Phi = expansion.basis_matrix(samples)  # (nsamples, nterms)

        # 3. Zero out constant term (index 0)
        params[0, :] = 0.0

        # 4. Compute residuals
        # values.T is (nsamples, 1), Phi @ params is (nsamples, 1)
        residuals = values.T - bkd.dot(Phi, params)  # (nsamples, 1)

        # 5. Set constant coefficient to risk measure of residuals
        self._risk_measure.set_samples(residuals.T)  # (1, nsamples)
        params[0, 0] = self._risk_measure()

        # 6. Create fitted expansion
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )


class ConservativeQuantileFitter(Generic[Array]):
    """Conservative quantile regression fitter using Average Value at Risk.

    Fits via quantile regression, then adjusts constant term to ensure
    risk_measure(surrogate) >= risk_measure(data).

    The Average Value at Risk (CVaR) at level beta is the expected value
    of X given X >= VaR_beta.

    Only supports nqoi=1 (single quantity of interest).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    quantile : float
        Target quantile tau in [0, 1]. Default: 0.5 (median).
        Also sets the beta level for AVaR.
    options : dict, optional
        Options passed to scipy.optimize.linprog.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        quantile: float = 0.5,
        options: Optional[dict[str, Any]] = None,
    ):
        if not 0 <= quantile < 1:
            raise ValueError(f"quantile must be in [0, 1), got {quantile}")
        self._bkd = bkd
        self._quantile = quantile
        self._base_fitter = QuantileFitter(bkd, quantile, options)
        self._risk_measure = AverageValueAtRisk(bkd, quantile)

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def quantile(self) -> float:
        """Return target quantile."""
        return self._quantile

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> DirectSolverResult[Array, BasisExpansionProtocol[Array]]:
        """Fit with conservative adjustment.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (1, nsamples) or (nsamples,).
            Only nqoi=1 supported.

        Returns
        -------
        DirectSolverResult
            Result containing fitted expansion with conservative adjustment.

        Raises
        ------
        ValueError
            If nqoi > 1.
        """
        bkd = self._bkd

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        # Validate single QoI
        if values.shape[0] != 1:
            raise ValueError(
                f"ConservativeQuantileFitter only supports nqoi=1, "
                f"got {values.shape[0]}"
            )

        # 1. Fit via base fitter
        result = self._base_fitter.fit(expansion, samples, values)
        params = bkd.copy(result.params())  # (nterms, 1)

        # 2. Get basis matrix for residual computation
        Phi = expansion.basis_matrix(samples)  # (nsamples, nterms)

        # 3. Zero out constant term (index 0)
        params[0, :] = 0.0

        # 4. Compute residuals
        # values.T is (nsamples, 1), Phi @ params is (nsamples, 1)
        residuals = values.T - bkd.dot(Phi, params)  # (nsamples, 1)

        # 5. Set constant coefficient to risk measure of residuals
        self._risk_measure.set_samples(residuals.T)  # (1, nsamples)
        params[0, 0] = self._risk_measure()

        # 6. Create fitted expansion
        fitted_expansion = expansion.with_params(params)

        return DirectSolverResult(
            surrogate=fitted_expansion,
            params=params,
        )
