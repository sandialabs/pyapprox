"""CombinationSurrogate — pure evaluation class for sparse grids.

A fitted sparse grid surrogate that evaluates as a weighted sum of
tensor product subspaces using Smolyak combination coefficients.

This class contains NO fitting logic — it is constructed by fitters
and used purely for evaluation, derivatives, and moment computation.
"""

from typing import Generic, List, Optional

from pyapprox.interface.functions.derivatives import (
    Derivatives,
    HessianBatchFn,
    HessianFn,
    HVPBatchFn,
    HVPFn,
    JacobianBatchFn,
    JacobianFn,
    WHVPBatchFn,
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
        Stored for use by SummedSubspaceVarianceIndicator and diagnostics.
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
        self._sub_hessians: Optional[List[HessianFn[Array]]] = None
        self._sub_jac_batches: Optional[List[JacobianBatchFn[Array]]] = None
        self._sub_hvp_batches: Optional[List[HVPBatchFn[Array]]] = None
        self._sub_whvp_batches: Optional[List[WHVPBatchFn[Array]]] = None
        self._sub_hess_batches: Optional[List[HessianBatchFn[Array]]] = None
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

        Every derivative here is linear in the subspaces, so the
        combination can offer a field exactly when every subspace
        declares it; hessian and hvp additionally require nqoi == 1.
        The captured fields are aligned index-for-index with the
        subspace list.

        ``jvp`` stays ``None`` because the subspaces do not provide it
        either, not by choice here.
        """
        sub_derivs = [subspace.derivatives() for subspace in self._subspaces]
        jacs = [d.jacobian for d in sub_derivs]
        hvps = [d.hvp for d in sub_derivs]
        whvps = [d.whvp for d in sub_derivs]
        hessians = [d.hessian for d in sub_derivs]
        jac_batches = [d.jacobian_batch for d in sub_derivs]
        hvp_batches = [d.hvp_batch for d in sub_derivs]
        whvp_batches = [d.whvp_batch for d in sub_derivs]
        hess_batches = [d.hessian_batch for d in sub_derivs]

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

        narrowed_hessians = [h for h in hessians if h is not None]
        if self._nqoi == 1 and len(narrowed_hessians) == len(hessians):
            self._sub_hessians = narrowed_hessians

        narrowed_jac_batches = [j for j in jac_batches if j is not None]
        if len(narrowed_jac_batches) == len(jac_batches):
            self._sub_jac_batches = narrowed_jac_batches

        narrowed_whvp_batches = [w for w in whvp_batches if w is not None]
        if len(narrowed_whvp_batches) == len(whvp_batches):
            self._sub_whvp_batches = narrowed_whvp_batches

        narrowed_hvp_batches = [h for h in hvp_batches if h is not None]
        if self._nqoi == 1 and len(narrowed_hvp_batches) == len(hvp_batches):
            self._sub_hvp_batches = narrowed_hvp_batches

        narrowed_hess_batches = [h for h in hess_batches if h is not None]
        if self._nqoi == 1 and len(narrowed_hess_batches) == len(hess_batches):
            self._sub_hess_batches = narrowed_hess_batches

        # The named constructors cover the usual combinations; the raw
        # one is used whenever the available set is not one of those.
        if self._sub_whvps is not None and self._sub_hvps is None:
            if self._sub_hessians is None:
                return Derivatives.second_order_weighted(
                    jacobian=self._jacobian,
                    jacobian_batch=(
                        self._jacobian_batch
                        if self._sub_jac_batches is not None
                        else None
                    ),
                    whvp=self._whvp,
                    whvp_batch=(
                        self._whvp_batch
                        if self._sub_whvp_batches is not None
                        else None
                    ),
                )
        return Derivatives(
            jacobian=self._jacobian,
            jacobian_batch=(
                self._jacobian_batch
                if self._sub_jac_batches is not None
                else None
            ),
            hvp=self._hvp if self._sub_hvps is not None else None,
            hvp_batch=(
                self._hvp_batch
                if self._sub_hvp_batches is not None
                else None
            ),
            whvp=self._whvp if self._sub_whvps is not None else None,
            whvp_batch=(
                self._whvp_batch
                if self._sub_whvp_batches is not None
                else None
            ),
            hessian=(
                self._hessian if self._sub_hessians is not None else None
            ),
            hessian_batch=(
                self._hessian_batch
                if self._sub_hess_batches is not None
                else None
            ),
        )

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

    def _hessian(self, sample: Array) -> Array:
        """Compute the Hessian at a single sample point (nqoi=1 only).

        Parameters
        ----------
        sample : Array
            Single evaluation point, shape (nvars, 1).

        Returns
        -------
        Array
            Hessian, shape (nvars, nvars).
        """
        if self._sub_hessians is None:
            raise RuntimeError(
                "hessian is unavailable; check derivatives() before calling"
            )
        hessian = self._bkd.zeros((self._nvars, self._nvars))
        for j, sub_hessian in enumerate(self._sub_hessians):
            hessian = hessian + self._coefs[j] * sub_hessian(sample)
        return hessian

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

    def _jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians at many sample points.

        Parameters
        ----------
        samples : Array
            Evaluation points, shape (nvars, npoints).

        Returns
        -------
        Array
            Jacobians, shape (npoints, nqoi, nvars).
        """
        if self._sub_jac_batches is None:
            raise RuntimeError(
                "jacobian_batch is unavailable; check derivatives() "
                "before calling"
            )
        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nqoi, self._nvars))
        for j, sub_jac in enumerate(self._sub_jac_batches):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_jac(samples)
        return result

    def _hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians at many sample points (nqoi=1 only).

        Parameters
        ----------
        samples : Array
            Evaluation points, shape (nvars, npoints).

        Returns
        -------
        Array
            Hessians, shape (npoints, nvars, nvars).
        """
        if self._sub_hess_batches is None:
            raise RuntimeError(
                "hessian_batch is unavailable; check derivatives() "
                "before calling"
            )
        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nvars, self._nvars))
        for j, sub_hessian in enumerate(self._sub_hess_batches):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_hessian(samples)
        return result

    def _hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute Hessian-vector products at many points (nqoi=1 only).

        Parameters
        ----------
        samples : Array
            Evaluation points, shape (nvars, npoints).
        vecs : Array
            Direction vectors, shape (nvars, npoints).

        Returns
        -------
        Array
            HVP results, shape (npoints, nvars).
        """
        if self._sub_hvp_batches is None:
            raise RuntimeError(
                "hvp_batch is unavailable; check derivatives() before calling"
            )
        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nvars))
        for j, sub_hvp in enumerate(self._sub_hvp_batches):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_hvp(samples, vecs)
        return result

    def _whvp_batch(
        self, samples: Array, vecs: Array, weights: Array
    ) -> Array:
        """Compute weighted Hessian-vector products at many points.

        Parameters
        ----------
        samples : Array
            Evaluation points, shape (nvars, npoints).
        vecs : Array
            Direction vectors, shape (nvars, npoints).
        weights : Array
            Weights for each QoI, shape (nqoi, 1).

        Returns
        -------
        Array
            WHVP results, shape (npoints, nvars).
        """
        if self._sub_whvp_batches is None:
            raise RuntimeError(
                "whvp_batch is unavailable; check derivatives() before calling"
            )
        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nvars))
        for j, sub_whvp in enumerate(self._sub_whvp_batches):
            coef: float = self._coefs[j].item()
            if abs(coef) > 1e-14:
                result = result + coef * sub_whvp(samples, vecs, weights)
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
