"""
Standard deviation measure for prediction OED.

Computes sqrt(Var[qoi | obs]) = sqrt(E[qoi^2 | obs] - E[qoi | obs]^2).
"""

from typing import Generic

from pyapprox.expdesign.deviation.base import DeviationMeasure
from pyapprox.util.backends.protocols import Array


class StandardDeviationMeasure(DeviationMeasure[Array], Generic[Array]):
    """
    Standard deviation of QoI prediction.

    Computes:
        StdDev[qoi | obs] = sqrt(E[qoi^2 | obs] - E[qoi | obs]^2)

    where expectations are taken over the posterior (likelihood-weighted prior).
    """

    def _second_moment(self, quad_weighted_like_vals: Array) -> Array:
        """
        Compute second moment E[qoi^2 | obs].

        Parameters
        ----------
        quad_weighted_like_vals : Array
            Quadrature-weighted likelihoods. Shape: (ninner, nouter)

        Returns
        -------
        Array
            Second moment. Shape: (npred, nouter)
        """
        # E[qoi_q^2 | obs_o] = sum_i qoi[i, q]^2 * like[i, o] * quad_weight[i]
        return self._bkd.einsum("iq,io->qo", self._qoi_vals**2, quad_weighted_like_vals)

    def _second_moment_jac(self, quad_weighted_like_vals_jac: Array) -> Array:
        """
        Compute Jacobian of second moment w.r.t. design weights.

        Parameters
        ----------
        quad_weighted_like_vals_jac : Array
            Jacobian of quad-weighted likelihoods. Shape: (ninner, nouter, nvars)

        Returns
        -------
        Array
            Jacobian of second moment. Shape: (npred, nouter, nvars)
        """
        return self._bkd.einsum(
            "iq,iod->qod", self._qoi_vals**2, quad_weighted_like_vals_jac
        )

    def _evaluate(self, design_weights: Array) -> Array:
        """
        Compute standard deviation.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Standard deviation values. Shape: (1, npred * nouter)
        """
        # Compute evidence
        evidences = self._evidence(design_weights).T  # (nouter, 1)

        # Normalized quad-weighted likelihoods
        normalized_like = self._evidence.quad_weighted_like_vals / evidences[:, 0]

        # Compute variance = E[qoi^2] - E[qoi]^2
        first_mom = self._first_moment(normalized_like)
        second_mom = self._second_moment(normalized_like)
        variance = second_mom - first_mom**2

        # Avoid small negative values due to numerical precision
        variance = self._bkd.maximum(variance, self._bkd.full(variance.shape, 1e-16))

        # Return standard deviation, flattened
        stdev = self._bkd.sqrt(variance)
        return self._bkd.flatten(stdev)[None, :]

    def _jacobian(self, design_weights: Array) -> Array:
        """
        Compute Jacobian of standard deviation.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nvars)
        """
        # Get current values
        values = self._evaluate(design_weights)  # (1, npred * nouter)

        # Compute evidence and its jacobian
        evidences = self._evidence(design_weights).T  # (nouter, 1)
        evidences_jac = self._evidence.jacobian(design_weights)  # (nouter, nvars)

        # Jacobian of quad-weighted likelihood
        like_jac = self._evidence.quad_weighted_likelihood_jacobian(
            design_weights
        )  # (ninner, nouter, nvars)

        # Normalized quantities
        normalized_like = self._evidence.quad_weighted_like_vals / evidences[:, 0]

        # Compute moments
        first_mom = self._first_moment(normalized_like)  # (npred, nouter)
        self._second_moment(normalized_like)  # (npred, nouter)

        # Jacobian of normalized quad-weighted likelihood
        # d/dw (like / evidence) = (d like / dw) / evidence - like * (d evidence / dw) /
        # evidence^2
        normalized_like_jac = (
            like_jac / evidences[None, :, 0, None]
            - self._evidence.quad_weighted_like_vals[:, :, None]
            * evidences_jac[None, :, :]
            / evidences[None, :, 0, None] ** 2
        )

        # Compute Jacobians of moments
        first_mom_jac = self._first_moment_jac(
            normalized_like_jac
        )  # (npred, nouter, nvars)
        second_mom_jac = self._second_moment_jac(
            normalized_like_jac
        )  # (npred, nouter, nvars)

        # Jacobian of variance = d(second_mom)/dw - 2 * first_mom * d(first_mom)/dw
        variance_jac = second_mom_jac - 2.0 * first_mom[:, :, None] * first_mom_jac

        # Reshape to (npred * nouter, nvars)
        variance_jac = self._bkd.reshape(
            variance_jac, (self._npred * self.nouter(), self.nvars())
        )

        # Jacobian of sqrt(variance) = d(var)/dw / (2 * sqrt(var))
        sqrt_jac = variance_jac / (2.0 * values.T)

        return sqrt_jac

    def label(self) -> str:
        """Return a short label for plotting."""
        return "StdDev"
