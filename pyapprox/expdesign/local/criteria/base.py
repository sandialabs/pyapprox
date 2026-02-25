"""
Base class for local OED criteria.

All local OED criteria share common structure:
- They take design weights as input
- They return a scalar (or vector for G-optimal) objective value
- They provide jacobian (and optionally hvp) for optimization
"""

from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.local.protocols import DesignMatricesProtocol


class LocalOEDCriterionBase(ABC, Generic[Array]):
    """
    Base class for local OED criteria.

    Local OED criteria are objective functions that measure the quality of an
    experimental design. They use design matrices M0 and M1 computed from
    the design weights.

    Parameters
    ----------
    design_matrices : DesignMatricesProtocol[Array]
        Object that computes M0, M1 matrices from design weights.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        design_matrices: DesignMatricesProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        self._design_matrices = design_matrices
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """
        Number of design variables.

        Returns
        -------
        int
            Number of candidate design points.
        """
        return self._design_matrices.ndesign_pts()

    def nqoi(self) -> int:
        """
        Number of quantities of interest.

        Returns
        -------
        int
            1 for scalar criteria, >1 for vector criteria (e.g., G-optimal).
        """
        return 1

    def ndesign_vars(self) -> int:
        """
        Number of design variables (basis function dimension).

        Returns
        -------
        int
            Dimension of the regression basis.
        """
        return self._design_matrices.ndesign_vars()

    def is_homoscedastic(self) -> bool:
        """Whether noise is constant across design points."""
        return self._design_matrices.is_homoscedastic()

    def design_matrices(self) -> DesignMatricesProtocol[Array]:
        """Get the design matrices object."""
        return self._design_matrices

    @abstractmethod
    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate criterion at design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nvars, 1)

        Returns
        -------
        Array
            Criterion value. Shape: (nqoi, 1)
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    # NOTE: hvp() is NOT defined here. Following the optional methods convention,
    # hvp should only exist on subclasses that actually implement it.
    # Check with hasattr(criterion, 'hvp') before using.
