"""CombinationSurrogate — pure evaluation class for sparse grids.

A fitted sparse grid surrogate that evaluates as a weighted sum of
tensor product subspaces using Smolyak combination coefficients.

This class contains NO fitting logic — it is constructed by fitters
and used purely for evaluation, derivatives, and moment computation.
"""

from typing import Generic, List, Optional

from pyapprox.interface.functions.derivatives import (
    Derivatives,
    HVPFn,
    JacobianFn,
    WHVPFn,
)
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
        self._sub_jacs: Optional[List[JacobianFn[Array]]] = None
        self._sub_hvps: Optional[List[HVPFn[Array]]] = None
        self._sub_whvps: Optional[List[WHVPFn[Array]]] = None
        self._derivs: Derivatives[Array] = self._build_derivatives()

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

    def _build_derivatives(self) -> Derivatives[Array]:
        """Compose the capability bundle from the subspaces' bundles.

        Each field is available exactly when every subspace declares it
        (hvp additionally requires nqoi == 1); the captured fields are
        aligned index-for-index with the subspace list.
        """
        sub_derivs = [subspace.derivatives() for subspace in self._subspaces]
        jacs = [d.jacobian for d in sub_derivs]
        hvps = [d.hvp for d in sub_derivs]
        whvps = [d.whvp for d in sub_derivs]

        narrowed_jacs = [j for j in jacs if j is not None]
        if len(narrowed_jacs) != len(jacs):
            return Derivatives.none()
        self._sub_jacs = narrowed_jacs

        narrowed_whvps = [w for w in whvps if w is not None]
        if len(narrowed_whvps) == len(whvps):
            self._sub_whvps = narrowed_whvps

        narrowed_hvps = [h for h in hvps if h is not None]
        if self._nqoi == 1 and len(narrowed_hvps) == len(hvps):
            self._sub_hvps = narrowed_hvps

        if self._sub_hvps is not None and self._sub_whvps is not None:
            # hvp AND whvp together is an unusual combination, so the raw
            # constructor is used
            return Derivatives(
                jacobian=self._jacobian, hvp=self._hvp, whvp=self._whvp
            )
        if self._sub_whvps is not None:
            return Derivatives.second_order_weighted(
                jacobian=self._jacobian, whvp=self._whvp
            )
        return Derivatives.first_order(jacobian=self._jacobian)

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle."""
        return self._derivs

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
        if self._sub_jacs is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        jacobian = self._bkd.zeros((self._nqoi, self._nvars))
        for j, sub_jac in enumerate(self._sub_jacs):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                jacobian = jacobian + coef * sub_jac(sample)
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
        if self._sub_hvps is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        result = self._bkd.zeros((self._nvars, 1))
        for j, sub_hvp in enumerate(self._sub_hvps):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_hvp(sample, vec)
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
        if self._sub_whvps is None:
            raise RuntimeError(
                "whvp is unavailable; check derivatives() before calling"
            )
        result = self._bkd.zeros((self._nvars, 1))
        for j, sub_whvp in enumerate(self._sub_whvps):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_whvp(sample, vec, weights)
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
