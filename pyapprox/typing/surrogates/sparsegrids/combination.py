"""Combination sparse grid base class.

Sparse grids are linear combinations of tensor product subspaces using
the Smolyak combination technique.
"""

from typing import Dict, Generic, List, Optional, Set, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
    IndexGrowthRuleProtocol,
)

from .smolyak import compute_smolyak_coefficients, _index_to_tuple
from .subspace import TensorProductSubspace


class CombinationSparseGrid(Generic[Array]):
    """Base class for combination sparse grids.

    A sparse grid is a weighted sum of tensor product subspaces:
        I_L(f) = sum_{k in K} c_k * I_k(f)

    where K is a downward-closed index set and c_k are Smolyak coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    univariate_bases : List[Basis1DProtocol[Array]]
        Univariate bases for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> bases = [LegendrePolynomial1D(bkd) for _ in range(2)]
    >>> growth = LinearGrowthRule()
    >>> grid = CombinationSparseGrid(bkd, bases, growth)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        univariate_bases: List[Basis1DProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
    ):
        self._bkd = bkd
        self._univariate_bases = univariate_bases
        self._growth_rule = growth_rule
        self._nvars = len(univariate_bases)

        # Subspace storage
        self._subspaces: Dict[Tuple[int, ...], TensorProductSubspace[Array]] = {}
        self._subspace_list: List[TensorProductSubspace[Array]] = []
        self._smolyak_coefficients: Optional[Array] = None

        # Sample tracking
        self._unique_samples: Optional[Array] = None
        self._sample_to_idx: Dict[Tuple[float, ...], int] = {}
        self._values: Optional[Array] = None
        self._nqoi: Optional[int] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsubspaces(self) -> int:
        """Return the number of subspaces."""
        return len(self._subspace_list)

    def get_subspaces(self) -> List[TensorProductSubspace[Array]]:
        """Return list of all subspaces."""
        return list(self._subspace_list)

    def get_smolyak_coefficients(self) -> Array:
        """Return Smolyak combination coefficients."""
        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()
        return self._bkd.copy(self._smolyak_coefficients)

    def get_samples(self) -> Array:
        """Return all unique sample locations."""
        if self._unique_samples is None:
            self._collect_unique_samples()
        return self._bkd.copy(self._unique_samples)

    def nsamples(self) -> int:
        """Return number of unique samples."""
        if self._unique_samples is None:
            self._collect_unique_samples()
        return self._unique_samples.shape[1]

    def set_values(self, values: Array) -> None:
        """Set function values at all unique samples.

        Parameters
        ----------
        values : Array
            Values of shape (nsamples, nqoi)
        """
        if self._unique_samples is None:
            self._collect_unique_samples()

        nsamples = self._unique_samples.shape[1]
        if values.shape[0] != nsamples:
            raise ValueError(
                f"Expected {nsamples} values, got {values.shape[0]}"
            )

        self._values = self._bkd.copy(values)
        self._nqoi = values.shape[1]

        # Distribute values to subspaces
        self._distribute_values_to_subspaces()

    def _add_subspace(self, index: Array) -> TensorProductSubspace[Array]:
        """Add a subspace to the sparse grid.

        Parameters
        ----------
        index : Array
            Multi-index for the subspace, shape (nvars,)

        Returns
        -------
        TensorProductSubspace[Array]
            The newly created subspace
        """
        key = _index_to_tuple(index)
        if key in self._subspaces:
            return self._subspaces[key]

        subspace = TensorProductSubspace(
            self._bkd,
            index,
            self._univariate_bases,
            self._growth_rule,
        )
        self._subspaces[key] = subspace
        self._subspace_list.append(subspace)

        # Invalidate cached data
        self._smolyak_coefficients = None
        self._unique_samples = None

        return subspace

    def _update_smolyak_coefficients(self) -> None:
        """Recompute Smolyak coefficients for current subspaces."""
        if len(self._subspace_list) == 0:
            self._smolyak_coefficients = self._bkd.zeros((0,))
            return

        # Build index array
        indices = self._bkd.zeros(
            (self._nvars, len(self._subspace_list)),
            dtype=self._bkd.int64_dtype()
        )
        for j, subspace in enumerate(self._subspace_list):
            indices[:, j] = subspace.get_index()

        self._smolyak_coefficients = compute_smolyak_coefficients(
            indices, self._bkd
        )

    def _collect_unique_samples(self) -> None:
        """Collect unique samples from all subspaces."""
        if len(self._subspace_list) == 0:
            self._unique_samples = self._bkd.zeros((self._nvars, 0))
            return

        # Collect all samples
        all_samples: List[Array] = []
        for subspace in self._subspace_list:
            all_samples.append(subspace.get_samples())

        combined = self._bkd.hstack(all_samples)

        # Find unique samples
        unique_list: List[List[float]] = []
        self._sample_to_idx = {}

        for j in range(combined.shape[1]):
            sample = combined[:, j]
            key = tuple(float(sample[i]) for i in range(self._nvars))

            if key not in self._sample_to_idx:
                self._sample_to_idx[key] = len(unique_list)
                unique_list.append(list(key))

        # Build unique samples array
        n_unique = len(unique_list)
        self._unique_samples = self._bkd.zeros((self._nvars, n_unique))
        for j, sample in enumerate(unique_list):
            for i in range(self._nvars):
                self._unique_samples[i, j] = sample[i]

    def _distribute_values_to_subspaces(self) -> None:
        """Distribute global values to each subspace."""
        if self._values is None:
            return

        for subspace in self._subspace_list:
            samples = subspace.get_samples()
            nsamples = samples.shape[1]

            subspace_values = self._bkd.zeros((nsamples, self._nqoi))
            for j in range(nsamples):
                key = tuple(
                    float(samples[i, j]) for i in range(self._nvars)
                )
                idx = self._sample_to_idx[key]
                subspace_values[j, :] = self._values[idx, :]

            subspace.set_values(subspace_values)

    def __call__(self, samples: Array) -> Array:
        """Evaluate sparse grid interpolant.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (npoints, nqoi)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()

        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nqoi))

        for j, subspace in enumerate(self._subspace_list):
            coef = float(self._smolyak_coefficients[j])
            if abs(coef) > 1e-14:
                subspace_vals = subspace(samples)
                result = result + coef * subspace_vals

        return result

    def get_subspace_indices(self) -> Array:
        """Return indices of all subspaces.

        Returns
        -------
        Array
            Multi-indices of shape (nvars, nsubspaces)
        """
        if len(self._subspace_list) == 0:
            return self._bkd.zeros(
                (self._nvars, 0), dtype=self._bkd.int64_dtype()
            )

        indices = self._bkd.zeros(
            (self._nvars, len(self._subspace_list)),
            dtype=self._bkd.int64_dtype()
        )
        for j, subspace in enumerate(self._subspace_list):
            indices[:, j] = subspace.get_index()

        return indices

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        if self._nqoi is None:
            raise ValueError("Values not set. Call set_values() first.")
        return self._nqoi

    def jacobian_supported(self) -> bool:
        """Return whether Jacobian computation is supported."""
        return True

    def hvp_supported(self) -> bool:
        """Return whether HVP/WHVP computation is supported."""
        return True

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
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()

        jacobian = self._bkd.zeros((self._nqoi, self._nvars))

        for j, subspace in enumerate(self._subspace_list):
            coef = float(self._smolyak_coefficients[j])
            if abs(coef) > 1e-14:
                subspace_jac = subspace.jacobian(sample)
                jacobian = jacobian + coef * subspace_jac

        return jacobian

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product for scalar-valued function.

        Only valid when nqoi=1. Uses efficient computation without forming
        the full Hessian.

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
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        if self._nqoi != 1:
            raise ValueError(
                f"hvp requires nqoi=1, got nqoi={self._nqoi}. Use whvp instead."
            )

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()

        # Use efficient subspace HVP without forming full Hessian
        result = self._bkd.zeros((self._nvars, 1))

        for j, subspace in enumerate(self._subspace_list):
            coef = float(self._smolyak_coefficients[j])
            if abs(coef) > 1e-14:
                subspace_hvp = subspace.hvp(sample, vec, qoi_idx=0)
                result = result + coef * subspace_hvp

        return result

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product for vector-valued function.

        For vector-valued functions, computes sum_i weights[i] * H_i @ vec
        where H_i is the Hessian of the i-th QoI. Uses efficient computation
        without forming full Hessians.

        Parameters
        ----------
        sample : Array
            Single evaluation point of shape (nvars, 1)
        vec : Array
            Direction vector of shape (nvars, 1)
        weights : Array
            Weights for each QoI. Can be shape (nqoi, 1), (1, nqoi), or (nqoi,).

        Returns
        -------
        Array
            Weighted Hessian-vector product of shape (nvars, 1)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()

        # Use efficient subspace WHVP without forming full Hessians
        result = self._bkd.zeros((self._nvars, 1))

        for j, subspace in enumerate(self._subspace_list):
            coef = float(self._smolyak_coefficients[j])
            if abs(coef) > 1e-14:
                subspace_whvp = subspace.whvp(sample, vec, weights)
                result = result + coef * subspace_whvp

        return result

    def __repr__(self) -> str:
        return (
            f"CombinationSparseGrid(nvars={self._nvars}, "
            f"nsubspaces={self.nsubspaces()}, nsamples={self.nsamples()})"
        )
