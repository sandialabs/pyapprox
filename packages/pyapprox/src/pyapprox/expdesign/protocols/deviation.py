"""
Protocols for prediction OED deviation measures.

Deviation measures quantify uncertainty in predictions given observations,
such as standard deviation, entropic deviation, or AVaR deviation.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DeviationMeasureProtocol(Protocol, Generic[Array]):
    """
    Protocol for OED deviation measures.

    Deviation measures compute the uncertainty in QoI predictions given
    observed data. Common measures include:
    - Standard deviation: sqrt(Var[qoi | obs])
    - Entropic deviation: log(E[exp(alpha*qoi)])/alpha - E[qoi]
    - AVaR deviation: AVaR[qoi] - E[qoi]

    The deviation depends on the evidence (marginal likelihood) which
    provides the posterior weights over inner samples.

    Methods
    -------
    bkd()
        Get the computational backend.
    npred()
        Number of prediction QoI locations.
    ninner()
        Number of inner loop samples.
    nouter()
        Number of outer loop samples.
    set_evidence(evidence)
        Set the Evidence object for posterior weights.
    set_qoi_data(qoi_vals, qoi_weights)
        Set QoI values at inner samples and prediction quadrature weights.
    __call__(design_weights)
        Compute deviation for all outer samples and QoI.
    jacobian(design_weights)
        Jacobian of deviation w.r.t. design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def npred(self) -> int:
        """
        Number of prediction QoI locations.

        Returns
        -------
        int
            Number of QoI prediction points.
        """
        ...

    def ninner(self) -> int:
        """
        Number of inner loop samples.

        Returns
        -------
        int
            Number of prior samples for posterior computation.
        """
        ...

    def nouter(self) -> int:
        """
        Number of outer loop samples.

        Returns
        -------
        int
            Number of data realizations.
        """
        ...

    def nvars(self) -> int:
        """
        Number of design variables (observation locations).

        Returns
        -------
        int
            Number of observation locations.
        """
        ...

    def set_qoi_data(self, qoi_vals: Array, qoi_weights: Array) -> None:
        """
        Set QoI values and prediction quadrature weights.

        Parameters
        ----------
        qoi_vals : Array
            QoI values at inner samples. Shape: (ninner, npred)
        qoi_weights : Array
            Prediction quadrature weights. Shape: (npred, 1)
        """
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute deviation for all outer samples and QoI.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Deviation values. Shape: (1, npred * nouter)
            Flattened in order: (qoi_0_outer_0, ..., qoi_0_outer_N, qoi_1_outer_0, ...)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of deviation w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (npred * nouter, nobs)
        """
        ...
