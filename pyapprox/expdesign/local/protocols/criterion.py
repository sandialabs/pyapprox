"""
Protocols for local OED criteria.

Local OED criteria are objective functions that measure the quality of an
experimental design for linear regression models. The design is specified
by a probability measure over candidate design points.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LocalOEDCriterionProtocol(Protocol, Generic[Array]):
    """
    Protocol for local OED criterion functions.

    A criterion takes design weights and returns an objective value to minimize.
    The design weights form a probability measure over candidate design points.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of design variables (= number of candidate design points).
    nqoi()
        Number of quantities of interest (= 1 for scalar criteria).
    __call__(design_weights)
        Evaluate criterion at design weights.
    jacobian(design_weights)
        Jacobian of criterion w.r.t. design weights.
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
            Number of candidate design points.
        """
        ...

    def nqoi(self) -> int:
        """
        Number of quantities of interest.

        Returns
        -------
        int
            Number of outputs. 1 for scalar criteria, >1 for vector criteria
            (e.g., G-optimal returns prediction variances at each point).
        """
        ...

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate criterion at design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights forming probability measure. Shape: (nvars, 1)
            Must satisfy: sum(design_weights) = 1, design_weights >= 0

        Returns
        -------
        Array
            Criterion value. Shape: (nqoi, 1)
        """
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of criterion w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        ...


@runtime_checkable
class LocalOEDCriterionWithHVPProtocol(Protocol, Generic[Array]):
    """
    Protocol for local OED criteria with Hessian-vector product support.

    Extends LocalOEDCriterionProtocol with hvp() method for second-order
    optimization methods.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Number of design variables."""
        ...

    def nqoi(self) -> int:
        """Number of quantities of interest."""
        ...

    def __call__(self, design_weights: Array) -> Array:
        """Evaluate criterion. Shape: (nvars, 1) -> (nqoi, 1)"""
        ...

    def jacobian(self, design_weights: Array) -> Array:
        """Jacobian. Shape: (nvars, 1) -> (nqoi, nvars)"""
        ...

    def hvp(self, design_weights: Array, vec: Array) -> Array:
        """
        Hessian-vector product.

        Only valid when nqoi() == 1.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)
        vec : Array
            Direction vector. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)
        """
        ...
