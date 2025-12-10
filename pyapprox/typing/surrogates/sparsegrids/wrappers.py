"""Wrapper classes for sparse grid integration with derivative checking.

This module provides wrapper classes that adapt sparse grid surrogates
to the function protocols used by DerivativeChecker.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.sparsegrids.protocols import (
    SparseGridWithDerivativesProtocol,
)


class SparseGridFunction(Generic[Array]):
    """Wrapper for sparse grid as a function with derivatives.

    Adapts a SparseGridWithDerivativesProtocol to the function protocols
    required by DerivativeChecker:
    - FunctionWithJacobianAndHVPProtocol (for nqoi=1)
    - FunctionWithJacobianAndWHVPProtocol (for nqoi>1)

    Parameters
    ----------
    sparse_grid : SparseGridWithDerivativesProtocol[Array]
        Sparse grid surrogate with derivative support.

    Examples
    --------
    >>> from pyapprox.typing.interface.functions.derivative_checks import (
    ...     DerivativeChecker
    ... )
    >>> # Create and set up sparse grid...
    >>> wrapper = SparseGridFunction(sparse_grid)
    >>> checker = DerivativeChecker(wrapper)
    >>> sample = bkd.asarray([[0.3], [0.5]])
    >>> errors = checker.check_derivatives(sample)
    """

    def __init__(self, sparse_grid: SparseGridWithDerivativesProtocol[Array]):
        self._grid = sparse_grid

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._grid.bkd()

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._grid.nvars()

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._grid.nqoi()

    def __call__(self, samples: Array) -> Array:
        """Evaluate the sparse grid at samples.

        Parameters
        ----------
        samples : Array
            Sample points of shape (nvars, nsamples)

        Returns
        -------
        Array
            Values of shape (nsamples, nqoi)
        """
        return self._grid(samples)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample point.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix of shape (nqoi, nvars)
        """
        return self._grid.jacobian(sample)

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product for scalar-valued function.

        Only valid when nqoi=1.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1)
        """
        return self._grid.hvp(sample, vec)

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product for vector-valued function.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)
        weights : Array
            Weights for each QoI of shape (nqoi, 1)

        Returns
        -------
        Array
            Weighted Hessian-vector product of shape (nvars, 1)
        """
        return self._grid.whvp(sample, vec, weights)

    def __repr__(self) -> str:
        return (
            f"SparseGridFunction(nvars={self.nvars()}, nqoi={self.nqoi()})"
        )
