"""Combination sparse grid base class.

Sparse grids are linear combinations of tensor product subspaces using
the Smolyak combination technique.
"""

from typing import Dict, Generic, List, Optional, Tuple, Union

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)
from pyapprox.typing.surrogates.affine.indices import IndexGenerator
from pyapprox.typing.surrogates.affine.indices.basis_generator import (
    BasisIndexGenerator,
)

from .smolyak import compute_smolyak_coefficients, _index_to_tuple
from .subspace import TensorProductSubspace
from .basis_factory import BasisFactoryProtocol
from .protocols import SubspaceProtocol
from .validation import (
    validate_backend,
    validate_basis_factories,
    validate_growth_rules,
    validate_piecewise_growth_compatibility,
)


class CombinationSparseGrid(Generic[Array]):
    """Base class for combination sparse grids.

    A sparse grid is a weighted sum of tensor product subspaces:
        I_L(f) = sum_{k in K} c_k * I_k(f)

    where K is a downward-closed index set and c_k are Smolyak coefficients.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    basis_factories : List[BasisFactoryProtocol[Array]]
        Factories for creating univariate bases for each dimension.
        Each factory's create_basis() is called when creating subspaces.
    growth_rules : IndexGrowthRuleProtocol or List[IndexGrowthRuleProtocol]
        Rule(s) mapping level to number of points. If a single rule, it is
        used for all dimensions. If a list, each element applies to the
        corresponding dimension.

    Notes
    -----
    Values have shape (nqoi, nsamples) following CLAUDE.md conventions:
    - Output f(X) (batch): (nqoi, nsamples) - QoIs as rows, samples as columns

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> from pyapprox.typing.surrogates.sparsegrids import PrebuiltBasisFactory
    >>> bkd = NumpyBkd()
    >>> bases = [LegendrePolynomial1D(bkd) for _ in range(2)]
    >>> factories = [PrebuiltBasisFactory(b) for b in bases]
    >>> growth = LinearGrowthRule()
    >>> grid = CombinationSparseGrid(bkd, factories, growth)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis_factories: List[BasisFactoryProtocol[Array]],
        growth_rules: Union[IndexGrowthRuleProtocol, List[IndexGrowthRuleProtocol]],
    ):
        # Runtime protocol validation
        validate_backend(bkd)
        validate_basis_factories(basis_factories)
        validate_growth_rules(growth_rules)
        validate_piecewise_growth_compatibility(basis_factories, growth_rules)

        self._bkd = bkd
        self._basis_factories = basis_factories
        self._growth_rules = growth_rules
        self._nvars = len(basis_factories)

        # Subspace storage
        self._subspaces: Dict[Tuple[int, ...], TensorProductSubspace[Array]] = {}
        self._subspace_list: List[TensorProductSubspace[Array]] = []
        self._smolyak_coefficients: Optional[Array] = None

        # Detect if all factories provide nested rules
        all_nested = all(
            getattr(f, "is_nested", lambda: False)()
            for f in basis_factories
        )

        # Basis index generator for sample deduplication
        self._basis_gen = BasisIndexGenerator(
            bkd, self._nvars, growth_rules, all_nested=all_nested
        )

        # Sample tracking
        self._unique_samples: Optional[Array] = None
        self._values: Optional[Array] = None
        self._nqoi: Optional[int] = None

        # Optional index generator (set by subclasses)
        self._index_gen: Optional[IndexGenerator[Array]] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsubspaces(self) -> int:
        """Return the number of subspaces."""
        return len(self._subspace_list)

    def get_index_generator(self) -> Optional[IndexGenerator[Array]]:
        """Return the index generator if set.

        Returns
        -------
        Optional[IndexGenerator[Array]]
            The index generator used for this sparse grid, or None if
            indices were added manually.
        """
        return self._index_gen

    def get_subspaces(self) -> List[SubspaceProtocol[Array]]:
        """Return list of all subspaces."""
        return list(self._subspace_list)

    def get_smolyak_coefficients(self) -> Array:
        """Return Smolyak combination coefficients."""
        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()
        assert self._smolyak_coefficients is not None
        return self._bkd.copy(self._smolyak_coefficients)

    def get_samples(self) -> Array:
        """Return all unique sample locations."""
        if self._unique_samples is None:
            self._collect_unique_samples()
        assert self._unique_samples is not None
        return self._bkd.copy(self._unique_samples)

    def nsamples(self) -> int:
        """Return number of unique samples."""
        if self._unique_samples is None:
            self._collect_unique_samples()
        assert self._unique_samples is not None
        return int(self._unique_samples.shape[1])

    def set_values(self, values: Array) -> None:
        """Set function values at all unique samples.

        Parameters
        ----------
        values : Array
            Values of shape (nqoi, nsamples).
        """
        if self._unique_samples is None:
            self._collect_unique_samples()
        assert self._unique_samples is not None

        nsamples = self._unique_samples.shape[1]
        if values.shape[1] != nsamples:
            raise ValueError(
                f"Expected {nsamples} samples (columns), got {values.shape[1]}"
            )

        self._values = self._bkd.copy(values)
        self._nqoi = values.shape[0]

        # Distribute values to subspaces
        self._distribute_values_to_subspaces()

        # Bind derivative methods based on nqoi
        self._setup_derivative_methods()

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

        subspace_idx = len(self._subspace_list)
        subspace = TensorProductSubspace(
            self._bkd,
            index,
            self._basis_factories,
            self._growth_rules,
        )
        self._subspaces[key] = subspace
        self._subspace_list.append(subspace)

        # Build index mappings for sample deduplication
        # Pass samples for non-nested path (needed for coordinate-based hashing)
        self._basis_gen._set_unique_subspace_basis_indices(
            index, subspace_idx,
            subspace_samples=subspace.get_samples()
        )

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

    def _adjust_smolyak_coefficients(
        self,
        smolyak_coefs: Array,
        new_index: Array,
        indices: Array,
    ) -> Array:
        """Incrementally update Smolyak coefficients when adding a new index.

        This is more efficient than recomputing all coefficients from scratch.
        The algorithm updates coefficients for all indices that are "neighbors"
        of the new index (differ by at most 1 in each component).

        This matches the legacy implementation in
        pyapprox.surrogates.sparsegrids.combination.AdaptiveCombinationSparseGrid._adjust_smolyak_coefficients().

        Parameters
        ----------
        smolyak_coefs : Array
            Current Smolyak coefficients. Shape: (nindices,)
        new_index : Array
            New index being added. Shape: (nvars,)
        indices : Array
            All indices (including new_index). Shape: (nvars, nindices)

        Returns
        -------
        Array
            Updated Smolyak coefficients.
        """
        new_smolyak_coefs = self._bkd.copy(smolyak_coefs)
        nindices = indices.shape[1]

        for idx in range(nindices):
            diff = new_index - indices[:, idx]
            # Check if this index is a neighbor: all diffs >= 0 and max diff <= 1
            max_diff: float = self._bkd.max(diff).item()
            if self._bkd.all_bool(diff >= 0) and max_diff <= 1:
                # Update coefficient using inclusion-exclusion formula
                sum_diff: float = self._bkd.sum(diff).item()
                new_smolyak_coefs[idx] = new_smolyak_coefs[idx] + (
                    (-1.0) ** sum_diff
                )

        return new_smolyak_coefs

    def _collect_unique_samples(self) -> None:
        """Collect unique samples using precomputed index mappings."""
        if len(self._subspace_list) == 0:
            self._unique_samples = self._bkd.zeros((self._nvars, 0))
            return

        n_unique = self._basis_gen.n_unique_samples()
        self._unique_samples = self._bkd.zeros((self._nvars, n_unique))

        global_idx = 0
        for subspace_idx, subspace in enumerate(self._subspace_list):
            subspace_samples = subspace.get_samples()
            unique_local = self._basis_gen.get_unique_local_indices(subspace_idx)

            if len(unique_local) > 0:
                idx_array = self._bkd.asarray(
                    unique_local, dtype=self._bkd.int64_dtype()
                )
                n_new = len(unique_local)
                self._unique_samples[:, global_idx:global_idx + n_new] = (
                    subspace_samples[:, idx_array]
                )
                global_idx += n_new

    def _distribute_values_to_subspaces(self) -> None:
        """Distribute global values using precomputed indices.

        No per-sample iteration, no .item() calls, preserves autograd.
        """
        if self._values is None:
            return

        for subspace_idx, subspace in enumerate(self._subspace_list):
            global_indices = self._basis_gen.get_subspace_value_indices(
                subspace_idx
            )
            subspace_values = self._values[:, global_indices]
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
            Interpolant values of shape (nqoi, npoints)
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        assert self._nqoi is not None

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()
        assert self._smolyak_coefficients is not None

        npoints = samples.shape[1]
        result = self._bkd.zeros((self._nqoi, npoints))

        for j, subspace in enumerate(self._subspace_list):
            coef: float = self._smolyak_coefficients[j].item()
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

    def _setup_derivative_methods(self) -> None:
        """Bind derivative methods based on nqoi.

        Per CLAUDE.md "Optional Methods Convention", methods that depend
        on runtime state should be bound dynamically so hasattr() works
        for capability discovery.

        DerivativeChecker uses hasattr(obj, 'hvp') to decide whether to
        use hvp (nqoi=1) or whvp (any nqoi).
        """
        # jacobian - always available once values are set
        self.jacobian = self._jacobian

        # hvp - only for scalar functions (nqoi == 1)
        if self._nqoi == 1:
            self.hvp = self._hvp
        elif hasattr(self, "hvp"):
            delattr(self, "hvp")

        # whvp - always available (works for any nqoi)
        self.whvp = self._whvp

    def _jacobian(self, sample: Array) -> Array:
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
        assert self._nqoi is not None

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()
        assert self._smolyak_coefficients is not None

        jacobian = self._bkd.zeros((self._nqoi, self._nvars))

        for j, subspace in enumerate(self._subspace_list):
            coef: float = self._smolyak_coefficients[j].item()
            if abs(coef) > 1e-14:
                subspace_jac = subspace.jacobian(sample)
                jacobian = jacobian + coef * subspace_jac

        return jacobian

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product for scalar-valued function.

        Only valid when nqoi=1. Uses efficient computation without forming
        the full Hessian. This method is bound to self.hvp by
        _setup_derivative_methods() only when nqoi == 1.

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

        if self._smolyak_coefficients is None:
            self._update_smolyak_coefficients()
        assert self._smolyak_coefficients is not None

        # Use efficient subspace HVP without forming full Hessian
        result = self._bkd.zeros((self._nvars, 1))

        for j, subspace in enumerate(self._subspace_list):
            coef: float = self._smolyak_coefficients[j].item()
            if abs(coef) > 1e-14:
                subspace_hvp = subspace.hvp(sample, vec)
                result = result + coef * subspace_hvp

        return result

    def _whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product for vector-valued function.

        For vector-valued functions, computes sum_i weights[i] * H_i @ vec
        where H_i is the Hessian of the i-th QoI. Uses efficient computation
        without forming full Hessians. This method is bound to self.whvp by
        _setup_derivative_methods().

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
        assert self._smolyak_coefficients is not None

        # Use efficient subspace WHVP without forming full Hessians
        result = self._bkd.zeros((self._nvars, 1))

        for j, subspace in enumerate(self._subspace_list):
            coef: float = self._smolyak_coefficients[j].item()
            if abs(coef) > 1e-14:
                subspace_whvp = subspace.whvp(sample, vec, weights)
                result = result + coef * subspace_whvp

        return result

    def mean(self) -> Array:
        """Compute the integral of the interpolant using sparse grid quadrature.

        Computes the Smolyak combination of tensor product integrals:
            I[f] = sum_k c_k * I_k[f]

        where c_k are Smolyak coefficients and I_k[f] is the tensor product
        quadrature integral for subspace k.

        For orthonormal polynomial bases with probability=True (default),
        the quadrature weights sum to 1, so this directly computes E[f].
        For physics measure (probability=False), weights reflect the
        integral of the weight function.

        Returns
        -------
        Array
            Integral values of shape (nqoi,)

        Notes
        -----
        Quadrature weights are NOT normalized here - they come directly
        from the Gauss quadrature rule of each polynomial basis.
        """
        return self._compute_moment("integrate")

    def variance(self) -> Array:
        """Compute the variance of the interpolant using sparse grid quadrature.

        Computes the Smolyak combination of tensor product variances:
            Var[f] = sum_k c_k * Var_k[f]

        where c_k are Smolyak coefficients and Var_k[f] is the tensor product
        quadrature variance for subspace k.

        This matches the legacy implementation in
        pyapprox.surrogates.sparsegrids.combination.CombinationSparseGrid.variance().

        Returns
        -------
        Array
            Variance values of shape (nqoi,)
        """
        return self._compute_moment("variance")

    def _compute_moment(
        self, moment: str, smolyak_coefs: Optional[Array] = None
    ) -> Array:
        """Compute a moment using Smolyak combination of subspace moments.

        This matches the legacy implementation in
        pyapprox.surrogates.sparsegrids.combination.CombinationSparseGrid._compute_moment().

        Parameters
        ----------
        moment : str
            The moment to compute. Either "integrate" (for mean) or "variance".
        smolyak_coefs : Array, optional
            Smolyak coefficients to use. If None, uses the grid's current
            coefficients. This allows computing moments with hypothetical
            coefficient sets (e.g., for variance-based refinement criteria).

        Returns
        -------
        Array
            Moment values of shape (nqoi,).
        """
        if self._values is None:
            raise ValueError("Values not set. Call set_values() first.")
        assert self._nqoi is not None

        if smolyak_coefs is None:
            if self._smolyak_coefficients is None:
                self._update_smolyak_coefficients()
            assert self._smolyak_coefficients is not None
            smolyak_coefs = self._smolyak_coefficients

        result = self._bkd.zeros((self._nqoi,))
        ncoefs = smolyak_coefs.shape[0]

        for j, subspace in enumerate(self._subspace_list):
            if j >= ncoefs:
                break
            coef: float = smolyak_coefs[j].item()
            if abs(coef) > 1e-14:
                subspace_moment = getattr(subspace, moment)()
                result = result + coef * subspace_moment

        return result

    def __repr__(self) -> str:
        return (
            f"CombinationSparseGrid(nvars={self._nvars}, "
            f"nsubspaces={self.nsubspaces()}, nsamples={self.nsamples()})"
        )
