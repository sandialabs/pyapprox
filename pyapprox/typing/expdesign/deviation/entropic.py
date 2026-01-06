"""
Entropic deviation measure for prediction OED.

Computes entropic_risk[qoi | obs] - E[qoi | obs].
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.deviation.base import DeviationMeasure


class EntropicDeviationMeasure(DeviationMeasure[Array], Generic[Array]):
    """
    Entropic deviation of QoI prediction.

    Computes:
        EntropicDev[qoi | obs] = (1/alpha) * log(E[exp(alpha * qoi) | obs]) - E[qoi | obs]

    where expectations are taken over the posterior (likelihood-weighted prior).

    Parameters
    ----------
    npred : int
        Number of prediction QoIs.
    alpha : float
        Risk aversion parameter. Must be > 0.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self, npred: int, alpha: float, bkd: Backend[Array]
    ) -> None:
        super().__init__(npred, bkd)
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float) -> None:
        """Set the risk aversion parameter."""
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self._alpha = alpha

    def _entropic_risk(self, normalized_like: Array) -> Array:
        """
        Compute entropic risk E_entropic[qoi | obs].

        Parameters
        ----------
        normalized_like : Array
            Normalized (by evidence) quad-weighted likelihoods.
            Shape: (ninner, nouter)

        Returns
        -------
        Array
            Entropic risk. Shape: (npred, nouter)
        """
        # entropic[q, o] = (1/alpha) * log(sum_i exp(alpha * qoi[i,q]) * like[i,o])
        # qoi_vals: (ninner, npred), normalized_like: (ninner, nouter)
        exp_alpha_qoi = self._bkd.exp(self._alpha * self._qoi_vals)
        expectation = self._bkd.einsum("iq,io->qo", exp_alpha_qoi, normalized_like)
        return self._bkd.log(expectation) / self._alpha

    def _entropic_risk_jac(
        self, normalized_like: Array, normalized_like_jac: Array
    ) -> Array:
        """
        Compute Jacobian of entropic risk w.r.t. design weights.

        Parameters
        ----------
        normalized_like : Array
            Normalized quad-weighted likelihoods. Shape: (ninner, nouter)
        normalized_like_jac : Array
            Jacobian of normalized likelihoods. Shape: (ninner, nouter, nvars)

        Returns
        -------
        Array
            Jacobian of entropic risk. Shape: (npred, nouter, nvars)
        """
        # E_exp[q,o] = sum_i exp(alpha * qoi[i,q]) * like[i,o]
        # d/dw log(E_exp) / alpha = (1 / (alpha * E_exp)) * d/dw E_exp
        # d/dw E_exp = sum_i exp(alpha * qoi[i,q]) * d/dw like[i,o]

        exp_alpha_qoi = self._bkd.exp(self._alpha * self._qoi_vals)

        # E_exp[q, o] = sum_i exp(alpha * qoi[i,q]) * like[i,o]
        expectation = self._bkd.einsum("iq,io->qo", exp_alpha_qoi, normalized_like)

        # d/dw E_exp[q, o, d] = sum_i exp(alpha * qoi[i,q]) * like_jac[i,o,d]
        d_expectation = self._bkd.einsum(
            "iq,iod->qod", exp_alpha_qoi, normalized_like_jac
        )

        # d/dw entropic = d_expectation / (alpha * expectation)
        return d_expectation / (self._alpha * expectation[:, :, None])

    def _evaluate(self, design_weights: Array) -> Array:
        """
        Compute entropic deviation.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Entropic deviation values. Shape: (1, npred * nouter)
        """
        # Compute evidence
        evidences = self._evidence(design_weights).T  # (nouter, 1)

        # Normalized quad-weighted likelihoods
        normalized_like = (
            self._evidence.quad_weighted_like_vals / evidences[:, 0]
        )

        # Compute entropic risk and mean
        entropic = self._entropic_risk(normalized_like)  # (npred, nouter)
        mean = self._first_moment(normalized_like)  # (npred, nouter)

        # Return entropic deviation, flattened
        deviation = entropic - mean
        return self._bkd.flatten(deviation)[None, :]

    def _jacobian(self, design_weights: Array) -> Array:
        """
        Compute Jacobian of entropic deviation.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nvars)
        """
        # Compute evidence and its jacobian
        evidences = self._evidence(design_weights).T  # (nouter, 1)
        evidences_jac = self._evidence.jacobian(design_weights)  # (nouter, nvars)

        # Jacobian of quad-weighted likelihood
        like_jac = self._evidence.quad_weighted_likelihood_jacobian(
            design_weights
        )  # (ninner, nouter, nvars)

        # Normalized quantities
        normalized_like = (
            self._evidence.quad_weighted_like_vals / evidences[:, 0]
        )

        # Jacobian of normalized quad-weighted likelihood
        # d/dw (like / evidence) = (d like / dw) / evidence
        #                        - like * (d evidence / dw) / evidence^2
        normalized_like_jac = (
            like_jac / evidences[None, :, 0, None]
            - self._evidence.quad_weighted_like_vals[:, :, None]
            * evidences_jac[None, :, :]
            / evidences[None, :, 0, None] ** 2
        )

        # Compute Jacobians of entropic risk and mean
        entropic_jac = self._entropic_risk_jac(
            normalized_like, normalized_like_jac
        )  # (npred, nouter, nvars)
        mean_jac = self._first_moment_jac(
            normalized_like_jac
        )  # (npred, nouter, nvars)

        # Jacobian of deviation = d(entropic)/dw - d(mean)/dw
        deviation_jac = entropic_jac - mean_jac

        # Reshape to (npred * nouter, nvars)
        return self._bkd.reshape(
            deviation_jac, (self._npred * self.nouter(), self.nvars())
        )

    def label(self) -> str:
        """Return a short label for plotting."""
        return "Entropic"
