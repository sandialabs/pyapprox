"""
Standard deviation measure for prediction OED.

Computes sqrt(Var[qoi | obs]) = sqrt(E[qoi^2 | obs] - E[qoi | obs]^2).
"""

import os
from typing import Generic

from pyapprox.expdesign.deviation.base import DeviationMeasure
from pyapprox.util.backends.protocols import Array


# Module-level switch between the legacy broadcast+einsum jacobian and the
# fused (numba-parallel, no 3D intermediate) jacobian. Fused is the default
# (validated on the C2 risk-OED sweep to ~5.8e-13 vs legacy, ~8x speedup);
# set PYAPPROX_STDDEV_JACOBIAN=legacy to revert, or override per-instance
# via `set_jacobian_impl()`. The fused path transparently falls back to
# legacy on backends without a fused moment-jacobian kernel.
STDDEV_JACOBIAN_IMPL = os.environ.get("PYAPPROX_STDDEV_JACOBIAN", "fused")
if STDDEV_JACOBIAN_IMPL not in ("legacy", "fused"):
    raise ValueError(
        f"PYAPPROX_STDDEV_JACOBIAN must be 'legacy' or 'fused', "
        f"got {STDDEV_JACOBIAN_IMPL!r}"
    )


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

        Dispatches between the legacy broadcast+einsum path and the fused
        numba path (see module-level ``STDDEV_JACOBIAN_IMPL``). Per-instance
        override is available via ``set_jacobian_impl()``.
        """
        impl = getattr(self, "_jacobian_impl", STDDEV_JACOBIAN_IMPL)
        if impl == "fused" and self._evidence.has_fused_weighted_jacobian():
            return self._jacobian_fused(design_weights)
        return self._jacobian_legacy(design_weights)

    def set_jacobian_impl(self, impl: str) -> None:
        """Override the jacobian implementation for this instance.

        Accepts ``"legacy"``, ``"fused"``, or ``"default"`` (use module-level
        setting). The fused path silently falls back to legacy on backends
        that don't expose a fused moment-jacobian kernel.
        """
        if impl not in ("legacy", "fused", "default"):
            raise ValueError(
                f"impl must be 'legacy', 'fused', or 'default', got {impl!r}"
            )
        if impl == "default":
            if hasattr(self, "_jacobian_impl"):
                del self._jacobian_impl
        else:
            self._jacobian_impl = impl

    def _jacobian_legacy(self, design_weights: Array) -> Array:
        """Broadcast + einsum path (materializes the (ninner, nouter, nobs) jac)."""
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

    def _jacobian_fused(self, design_weights: Array) -> Array:
        """Fused path via ``Evidence.fused_weighted_jacobian``.

        Avoids the ``(ninner, nouter, nobs)`` ``normalized_like_jac``
        intermediate by delegating the two einsums + their subtracted
        evid-jac outer product to a single numba kernel call.
        """
        # Must call _evaluate before accessing cached evidence internals.
        values = self._evaluate(design_weights)  # (1, npred * nouter)

        evidences = self._evidence(design_weights).T  # (nouter, 1)
        normalized_like = self._evidence.quad_weighted_like_vals / evidences[:, 0]
        first_mom = self._first_moment(normalized_like)  # (npred, nouter)

        first_mom_jac, second_mom_jac = self._evidence.fused_weighted_jacobian(
            design_weights, self._qoi_vals, self._qoi_vals ** 2,
        )  # each (npred, nouter, nvars)

        variance_jac = second_mom_jac - 2.0 * first_mom[:, :, None] * first_mom_jac
        variance_jac = self._bkd.reshape(
            variance_jac, (self._npred * self.nouter(), self.nvars())
        )
        return variance_jac / (2.0 * values.T)

    def label(self) -> str:
        """Return a short label for plotting."""
        return "StdDev"
