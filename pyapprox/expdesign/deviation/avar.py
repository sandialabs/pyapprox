"""
AVaR deviation measure for prediction OED.

Computes AVaR[qoi | obs] - E[qoi | obs].
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.deviation.base import DeviationMeasure
from pyapprox.expdesign.statistics.avar import SampleAverageSmoothedAVaR


class AVaRDeviationMeasure(DeviationMeasure[Array], Generic[Array]):
    """
    AVaR deviation of QoI prediction.

    Computes:
        AVaRDev[qoi | obs] = AVaR_alpha[qoi | obs] - E[qoi | obs]

    where AVaR (Average Value at Risk, also known as CVaR or Expected Shortfall)
    is the expected value of outcomes exceeding the alpha quantile.

    The expectations are taken over the posterior (likelihood-weighted prior).

    Parameters
    ----------
    npred : int
        Number of prediction QoIs.
    alpha : float
        Risk level in (0, 1). AVaR_alpha is the expected value given that
        outcomes exceed the alpha quantile.
    bkd : Backend[Array]
        Computational backend.
    delta : float, optional
        Smoothing parameter for AVaR. Larger values give more accurate
        estimates. Default is 100.
    """

    def __init__(
        self,
        npred: int,
        alpha: float,
        bkd: Backend[Array],
        delta: float = 100,
    ) -> None:
        super().__init__(npred, bkd)
        self._alpha = alpha
        self._delta = delta
        self._smoothed_avar = SampleAverageSmoothedAVaR(alpha, bkd, delta)

    def _evaluate(self, design_weights: Array) -> Array:
        """
        Compute AVaR deviation.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            AVaR deviation values. Shape: (1, npred * nouter)
        """
        # Compute evidence
        evidences = self._evidence(design_weights).T  # (nouter, 1)

        # Normalized quad-weighted likelihoods as posterior weights
        # Shape: (ninner, nouter)
        normalized_like = (
            self._evidence.quad_weighted_like_vals / evidences[:, 0]
        )

        # Compute mean for each (qoi, outer) pair
        mean = self._first_moment(normalized_like)  # (npred, nouter)

        # Compute AVaR for each (qoi, outer) pair
        # AVaR requires weights summing to 1, which normalized_like provides
        nouter = self.nouter()
        npred = self._npred

        avar_vals = []
        for qq in range(npred):
            outer_vals = []
            for oo in range(nouter):
                # Get posterior weights for this outer sample
                # _evaluate_single expects (1, nsamples) for both
                weights = normalized_like[:, oo:oo + 1].T  # (1, ninner)
                values = self._qoi_vals[:, qq:qq + 1].T  # (1, ninner)

                # Compute AVaR using smoothed estimator
                avar = self._smoothed_avar._evaluate_single(values, weights)
                outer_vals.append(avar)

            avar_vals.append(self._bkd.hstack(outer_vals))

        avar_matrix = self._bkd.stack(avar_vals, axis=0)  # (npred, nouter)

        # Return AVaR deviation, flattened
        deviation = avar_matrix - mean
        return self._bkd.flatten(deviation)[None, :]

    def _jacobian(self, design_weights: Array) -> Array:
        """
        Compute Jacobian of AVaR deviation using autodiff.

        The AVaR projection algorithm does not have an easily derivable
        analytical Jacobian, so we use automatic differentiation when
        available.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nvars)
        """
        if hasattr(self._bkd, "jacobian"):
            # Autodiff returns shape (1, npred*nouter, nobs, 1)
            # Need to reshape to (npred*nouter, nobs)
            jac = self._bkd.jacobian(self._evaluate, design_weights)
            return self._bkd.reshape(jac, (self._npred * self.nouter(), self.nvars()))

        raise NotImplementedError(
            "AVaRDeviationMeasure jacobian requires autodiff support. "
            "Use PyTorch backend."
        )

    def label(self) -> str:
        """Return a short label for plotting."""
        return "AVaRDev"
