"""
Protocols for prediction-based OED objectives.

Prediction OED minimizes the expected deviation in QoI predictions,
combining deviation measures, risk measures, and noise statistics.
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class PredictionOEDObjectiveProtocol(Protocol, Generic[Array]):
    """
    Protocol for prediction-based OED objective.

    The prediction OED objective computes:
        objective(w) = noise_stat[risk_measure[deviation(qoi | obs, w)]]

    where:
    - deviation: Measures uncertainty in QoI predictions (StdDev, Entropic, AVaR)
    - risk_measure: Aggregates deviations over prediction space
    - noise_stat: Averages over data realizations

    This extends OEDObjectiveProtocol with prediction-specific methods.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of design variables (= nobs).
    nqoi()
        Number of outputs (= 1 for scalar objective).
    nobs()
        Number of observation locations.
    ninner()
        Number of inner (prior) samples.
    nouter()
        Number of outer (observation) samples.
    npred()
        Number of prediction QoI locations.
    __call__(design_weights)
        Evaluate objective at design weights.
    jacobian(design_weights)
        Jacobian of objective w.r.t. design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Number of design variables.

        Returns
        -------
        int
            Number of observation locations (nobs).
        """
        ...

    def nqoi(self) -> int:
        """
        Number of quantities of interest.

        Returns
        -------
        int
            Always 1 for scalar objective.
        """
        ...

    def nobs(self) -> int:
        """
        Number of observation locations.

        Returns
        -------
        int
            Number of candidate observation points.
        """
        ...

    def ninner(self) -> int:
        """
        Number of inner (prior) samples for evidence integration.

        Returns
        -------
        int
            Number of inner samples.
        """
        ...

    def nouter(self) -> int:
        """
        Number of outer (observation) samples for expectation.

        Returns
        -------
        int
            Number of outer samples.
        """
        ...

    def npred(self) -> int:
        """
        Number of prediction QoI locations.

        Returns
        -------
        int
            Number of prediction points.
        """
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate objective at design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Objective value. Shape: (1, 1)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of objective w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nobs)
        """
        ...
