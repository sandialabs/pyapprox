"""CombinationSurrogate — pure evaluation class for sparse grids.

A fitted sparse grid surrogate that evaluates as a weighted sum of
tensor product subspaces using Smolyak combination coefficients.

This class contains NO fitting logic — it is constructed by fitters
and used purely for evaluation, derivatives, and moment computation.
"""

from typing import Generic, List, Optional

from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend


class CombinationSurrogate(Generic[Array]):
    """Sparse grid surrogate: weighted sum of tensor product subspaces.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of physical variables.
    subspaces : List[TensorProductSubspace[Array]]
        Tensor product subspaces (shallow-copied).
    coefs : Array
        Smolyak combination coefficients, shape (nsubspaces,).
    nqoi : int
        Number of quantities of interest.
    indices : Array, optional
        Subspace multi-indices, shape (nvars_index, nsubspaces).
        Stored for use by VarianceChangeIndicator and diagnostics.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        subspaces: List[TensorProductSubspace[Array]],
        coefs: Array,
        nqoi: int,
        indices: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._subspaces = list(subspaces)
        self._coefs = coefs
        self._nqoi = nqoi
        self._indices = indices
        self._setup_derivative_methods()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of physical variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._nqoi

    def nsubspaces(self) -> int:
        """Return the number of subspaces."""
        return len(self._subspaces)

    def subspaces(self) -> List[TensorProductSubspace[Array]]:
        """Return a shallow copy of the subspace list."""
        return list(self._subspaces)

    def coefficients(self) -> Array:
        """Return the Smolyak coefficients."""
        return self._bkd.copy(self._coefs)

    def indices(self) -> Optional[Array]:
        """Return the subspace indices, if stored."""
        if self._indices is not None:
            return self._bkd.copy(self._indices)
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, samples: Array) -> Array:
        """Evaluate sparse grid interpolant.

        Parameters
        ----------
        samples : Array
            Evaluation points, shape (nvars_physical, npoints).

        Returns
        -------
        Array
            Interpolant values, shape (nqoi, npoints).
        """
        npoints = samples.shape[1]
        result = self._bkd.zeros((self._nqoi, npoints))
        for j, subspace in enumerate(self._subspaces):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * subspace(samples)
        return result

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def _setup_derivative_methods(self) -> None:
        """Bind derivative methods based on nqoi."""
        self.jacobian = self._jacobian
        if self._nqoi == 1:
            self.hvp = self._hvp
        elif hasattr(self, "hvp"):
            delattr(self, "hvp")
        self.whvp = self._whvp

    def _jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample point.

        Parameters
        ----------
        sample : Array
            Single evaluation point, shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian, shape (nqoi, nvars).
        """
        jacobian = self._bkd.zeros((self._nqoi, self._nvars))
        for j, subspace in enumerate(self._subspaces):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                jacobian = jacobian + coef * subspace.jacobian(sample)
        return jacobian

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product (nqoi=1 only).

        Parameters
        ----------
        sample : Array
            Single evaluation point, shape (nvars, 1).
        vec : Array
            Direction vector, shape (nvars, 1).

        Returns
        -------
        Array
            HVP result, shape (nvars, 1).
        """
        result = self._bkd.zeros((self._nvars, 1))
        for j, subspace in enumerate(self._subspaces):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * subspace.hvp(sample, vec)
        return result

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product.

        Parameters
        ----------
        sample : Array
            Single evaluation point, shape (nvars, 1).
        vec : Array
            Direction vector, shape (nvars, 1).
        weights : Array
            Weights for each QoI.

        Returns
        -------
        Array
            WHVP result, shape (nvars, 1).
        """
        result = self._bkd.zeros((self._nvars, 1))
        for j, subspace in enumerate(self._subspaces):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * subspace.whvp(sample, vec, weights)
        return result

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def mean(self) -> Array:
        """Compute mean (expected value) via sparse grid quadrature.

        Returns
        -------
        Array
            Mean values, shape (nqoi,).
        """
        return self._compute_moment("integrate")

    def variance(self) -> Array:
        """Compute variance via sparse grid quadrature.

        Returns
        -------
        Array
            Variance values, shape (nqoi,).
        """
        return self._compute_moment("variance")

    def _compute_moment(self, moment: str) -> Array:
        """Compute a moment using Smolyak combination.

        Parameters
        ----------
        moment : str
            Either "integrate" or "variance".

        Returns
        -------
        Array
            Moment values, shape (nqoi,).
        """
        result = self._bkd.zeros((self._nqoi,))
        for j, subspace in enumerate(self._subspaces):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * getattr(subspace, moment)()
        return result

    def __repr__(self) -> str:
        return (
            f"CombinationSurrogate(nvars={self._nvars}, "
            f"nsubspaces={self.nsubspaces()}, nqoi={self._nqoi})"
        )
