"""
Base class for prediction deviation measures.

Deviation measures quantify the spread or uncertainty of QoI predictions
conditional on observed data.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional

from pyapprox.expdesign.evidence.evidence import Evidence
from pyapprox.util.backends.protocols import Array, Backend


class DeviationMeasure(ABC, Generic[Array]):
    """
    Abstract base class for prediction deviation measures.

    A deviation measure quantifies the uncertainty/spread of a quantity of
    interest (QoI) prediction conditional on observed data and design weights.

    The deviation is computed as a function of:
    - QoI values at inner loop (prior) samples: qoi_vals[i, q]
    - Evidence (marginal likelihood) for each outer sample
    - Quadrature-weighted likelihood values

    Parameters
    ----------
    npred : int
        Number of prediction QoIs.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, npred: int, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._npred = npred
        self._evidence: Optional[Evidence[Array]] = None
        self._qoi_vals: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def npred(self) -> int:
        """Number of prediction QoIs."""
        return self._npred

    def ninner(self) -> int:
        """Number of inner loop samples."""
        if self._evidence is None:
            raise RuntimeError("Must call set_evidence first")
        return self._evidence.ninner()

    def nouter(self) -> int:
        """Number of outer loop samples."""
        if self._evidence is None:
            raise RuntimeError("Must call set_evidence first")
        return self._evidence.nouter()

    def nvars(self) -> int:
        """Number of design variables (same as nobs)."""
        if self._evidence is None:
            raise RuntimeError("Must call set_evidence first")
        return self._evidence._loglike.nobs()

    def set_evidence(self, evidence: Evidence[Array]) -> None:
        """
        Set the evidence computation object.

        Parameters
        ----------
        evidence : Evidence[Array]
            Evidence object that computes marginal likelihood.
        """
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an instance of Evidence")
        self._evidence = evidence

    def set_qoi_data(self, qoi_vals: Array) -> None:
        """
        Set the QoI values at inner loop samples.

        Parameters
        ----------
        qoi_vals : Array
            QoI values at inner loop samples. Shape: (ninner, npred)
        """
        if self._evidence is None:
            raise RuntimeError("Must call set_evidence first")
        expected_shape = (self.ninner(), self._npred)
        if qoi_vals.shape != expected_shape:
            raise ValueError(
                f"qoi_vals must have shape {expected_shape}, got {qoi_vals.shape}"
            )
        self._qoi_vals = qoi_vals

    def _validate_setup(self) -> None:
        """Validate that evidence and qoi_data have been set."""
        if self._evidence is None:
            raise RuntimeError("Must call set_evidence first")
        if self._qoi_vals is None:
            raise RuntimeError("Must call set_qoi_data first")

    def _first_moment(self, quad_weighted_like_vals: Array) -> Array:
        """
        Compute first moment E[qoi | obs].

        Parameters
        ----------
        quad_weighted_like_vals : Array
            Quadrature-weighted likelihoods. Shape: (ninner, nouter)

        Returns
        -------
        Array
            First moment. Shape: (npred, nouter)
        """
        # E[qoi_q | obs_o] = sum_i qoi[i, q] * like[i, o] * quad_weight[i]
        # qoi_vals: (ninner, npred), quad_weighted_like: (ninner, nouter)
        # Result: (npred, nouter)
        return self._bkd.einsum("iq,io->qo", self._qoi_vals, quad_weighted_like_vals)

    def _first_moment_jac(self, quad_weighted_like_vals_jac: Array) -> Array:
        """
        Compute Jacobian of first moment w.r.t. design weights.

        Parameters
        ----------
        quad_weighted_like_vals_jac : Array
            Jacobian of quad-weighted likelihoods. Shape: (ninner, nouter, nvars)

        Returns
        -------
        Array
            Jacobian of first moment. Shape: (npred, nouter, nvars)
        """
        # d/dw E[qoi | obs] = sum_i qoi[i, q] * d/dw(like[i, o] * quad[i])
        # qoi_vals: (ninner, npred), jac: (ninner, nouter, nvars)
        # Result: (npred, nouter, nvars)
        return self._bkd.einsum(
            "iq,iod->qod", self._qoi_vals, quad_weighted_like_vals_jac
        )

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute deviation measure.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Deviation values. Shape: (1, npred * nouter)
            Flattened as required by the OED objective interface.
        """
        self._validate_setup()
        return self._evaluate(design_weights)

    @abstractmethod
    def _evaluate(self, design_weights: Array) -> Array:
        """
        Evaluate the deviation measure.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Deviation values. Shape: (1, npred * nouter)
        """
        raise NotImplementedError

    def jacobian(self, design_weights: Array) -> Array:
        """
        Compute Jacobian of deviation measure.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nvars)
        """
        self._validate_setup()
        return self._jacobian(design_weights)

    @abstractmethod
    def _jacobian(self, design_weights: Array) -> Array:
        """
        Compute Jacobian of deviation measure.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nvars)
        """
        raise NotImplementedError

    def label(self) -> str:
        """Return a short label for plotting."""
        return self.__class__.__name__
