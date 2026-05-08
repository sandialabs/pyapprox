"""
Gaussian factor with variable ID tracking for Bayesian networks.

This module extends GaussianCanonicalForm with variable ID tracking,
enabling scope-aware operations for graphical model inference.
"""

from typing import Generic, List, Tuple

from pyapprox.probability.gaussian import GaussianCanonicalForm
from pyapprox.util.backends.protocols import Array, Backend


class GaussianFactor(Generic[Array]):
    """
    Gaussian factor with variable ID tracking for Bayesian networks.

    Wraps GaussianCanonicalForm and adds:
    - var_ids: identifies which network variables are in scope
    - nvars_per_var: number of dimensions per variable
    - Scope-aware multiply that expands to common variable sets

    Parameters
    ----------
    canonical : GaussianCanonicalForm[Array]
        The underlying canonical form Gaussian.
    var_ids : List[int]
        Variable IDs for each variable in the factor.
        Each ID corresponds to a unique variable in the network.
    nvars_per_var : List[int]
        Number of dimensions per variable.
    bkd : Backend[Array]
        Computational backend.

    Notes
    -----
    The total dimension of the canonical form should equal sum(nvars_per_var).
    Variables are stored in order of var_ids, so the first nvars_per_var[0]
    dimensions correspond to var_ids[0], etc.
    """

    def __init__(
        self,
        canonical: GaussianCanonicalForm[Array],
        var_ids: List[int],
        nvars_per_var: List[int],
        bkd: Backend[Array],
    ):
        self._canonical = canonical
        self._var_ids = list(var_ids)
        self._nvars_per_var = list(nvars_per_var)
        self._bkd = bkd

        # Verify dimensions match
        total_dims = sum(nvars_per_var)
        if total_dims != canonical.nvars():
            raise ValueError(
                f"Dimension mismatch: sum(nvars_per_var)={total_dims} "
                f"!= canonical.nvars()={canonical.nvars()}"
            )

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def canonical(self) -> GaussianCanonicalForm[Array]:
        """Get the underlying canonical form."""
        return self._canonical

    def var_ids(self) -> List[int]:
        """Get the variable IDs in this factor's scope."""
        return self._var_ids.copy()

    def nvars_per_var(self) -> List[int]:
        """Get the number of dimensions per variable."""
        return self._nvars_per_var.copy()

    def total_dims(self) -> int:
        """Get the total number of dimensions."""
        return self._canonical.nvars()

    def scope_size(self) -> int:
        """Get the number of variables in scope."""
        return len(self._var_ids)

    @classmethod
    def from_moments(
        cls,
        mean: Array,
        covariance: Array,
        var_ids: List[int],
        nvars_per_var: List[int],
        bkd: Backend[Array],
    ) -> "GaussianFactor[Array]":
        """
        Create a factor from mean and covariance.

        Parameters
        ----------
        mean : Array
            Mean vector. Shape: (total_dims,) or (total_dims, 1)
        covariance : Array
            Covariance matrix. Shape: (total_dims, total_dims)
        var_ids : List[int]
            Variable IDs for each variable.
        nvars_per_var : List[int]
            Number of dimensions per variable.
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        GaussianFactor
            Factor in canonical form.
        """
        canonical = GaussianCanonicalForm.from_moments(mean, covariance, bkd)
        return cls(canonical, var_ids, nvars_per_var, bkd)

    def to_moments(self) -> Tuple[Array, Array]:
        """
        Convert to mean and covariance form.

        Returns
        -------
        mean : Array
            Mean vector. Shape: (total_dims,)
        covariance : Array
            Covariance matrix. Shape: (total_dims, total_dims)
        """
        return self._canonical.to_moments()

    def _get_var_indices(self, var_id: int) -> List[int]:
        """Get the indices corresponding to a variable ID."""
        if var_id not in self._var_ids:
            raise ValueError(f"Variable {var_id} not in scope {self._var_ids}")

        idx = self._var_ids.index(var_id)
        start = sum(self._nvars_per_var[:idx])
        end = start + self._nvars_per_var[idx]
        return list(range(start, end))

    def expand_scope(
        self,
        target_var_ids: List[int],
        target_nvars_per_var: List[int],
    ) -> "GaussianFactor[Array]":
        """
        Expand this factor to include additional variables.

        New variables are added with zero precision (vacuous information),
        which means they don't affect the product during multiplication.

        Parameters
        ----------
        target_var_ids : List[int]
            Target variable IDs (must include current var_ids).
        target_nvars_per_var : List[int]
            Dimensions for all target variables.

        Returns
        -------
        GaussianFactor
            Factor expanded to target scope.
        """
        # Check that current vars are subset of target
        for var_id in self._var_ids:
            if var_id not in target_var_ids:
                raise ValueError(
                    f"Variable {var_id} not in target scope {target_var_ids}"
                )

        # If same scope, return copy
        if set(self._var_ids) == set(target_var_ids):
            return GaussianFactor(
                self._canonical,
                self._var_ids,
                self._nvars_per_var,
                self._bkd,
            )

        total_target_dims = sum(target_nvars_per_var)

        # Build expanded precision and shift
        new_precision = self._bkd.zeros((total_target_dims, total_target_dims))
        new_shift = self._bkd.zeros((total_target_dims,))

        # Map current variables to target indices
        for i, var_id in enumerate(self._var_ids):
            target_idx = target_var_ids.index(var_id)
            # Source indices in current factor
            src_start = sum(self._nvars_per_var[:i])
            src_end = src_start + self._nvars_per_var[i]
            # Target indices in expanded factor
            tgt_start = sum(target_nvars_per_var[:target_idx])
            tgt_end = tgt_start + target_nvars_per_var[target_idx]

            # Copy shift
            shift_np = self._bkd.to_numpy(self._canonical.shift())
            new_shift_np = self._bkd.to_numpy(new_shift)
            new_shift_np[tgt_start:tgt_end] = shift_np[src_start:src_end]
            new_shift = self._bkd.asarray(new_shift_np)

            # Copy precision blocks
            for j, var_id_j in enumerate(self._var_ids):
                target_idx_j = target_var_ids.index(var_id_j)
                src_start_j = sum(self._nvars_per_var[:j])
                src_end_j = src_start_j + self._nvars_per_var[j]
                tgt_start_j = sum(target_nvars_per_var[:target_idx_j])
                tgt_end_j = tgt_start_j + target_nvars_per_var[target_idx_j]

                prec_np = self._bkd.to_numpy(self._canonical.precision())
                new_prec_np = self._bkd.to_numpy(new_precision)
                new_prec_np[tgt_start:tgt_end, tgt_start_j:tgt_end_j] = prec_np[
                    src_start:src_end, src_start_j:src_end_j
                ]
                new_precision = self._bkd.asarray(new_prec_np)

        new_canonical = GaussianCanonicalForm(
            new_precision,
            new_shift,
            self._canonical.normalization(),
            self._bkd,
        )

        return GaussianFactor(
            new_canonical,
            target_var_ids,
            target_nvars_per_var,
            self._bkd,
        )

    def multiply(self, other: "GaussianFactor[Array]") -> "GaussianFactor[Array]":
        """
        Multiply two factors with scope expansion.

        Both factors are first expanded to the union of their scopes,
        then multiplied in canonical form.

        Parameters
        ----------
        other : GaussianFactor
            Other factor to multiply with.

        Returns
        -------
        GaussianFactor
            Product factor.
        """
        # Get union of variable IDs
        all_var_ids: List[int] = []
        all_nvars_per_var: List[int] = []

        # Start with self's variables
        for i, var_id in enumerate(self._var_ids):
            all_var_ids.append(var_id)
            all_nvars_per_var.append(self._nvars_per_var[i])

        # Add other's variables that aren't in self
        for i, var_id in enumerate(other._var_ids):
            if var_id not in all_var_ids:
                all_var_ids.append(var_id)
                all_nvars_per_var.append(other._nvars_per_var[i])
            else:
                # Check dimension consistency
                idx = all_var_ids.index(var_id)
                if all_nvars_per_var[idx] != other._nvars_per_var[i]:
                    raise ValueError(
                        f"Dimension mismatch for variable {var_id}: "
                        f"{all_nvars_per_var[idx]} vs {other._nvars_per_var[i]}"
                    )

        # Expand both factors to common scope
        self_expanded = self.expand_scope(all_var_ids, all_nvars_per_var)
        other_expanded = other.expand_scope(all_var_ids, all_nvars_per_var)

        # Multiply in canonical form
        product_canonical = self_expanded._canonical.multiply(other_expanded._canonical)

        return GaussianFactor(
            product_canonical,
            all_var_ids,
            all_nvars_per_var,
            self._bkd,
        )

    def marginalize_vars(self, marg_var_ids: List[int]) -> "GaussianFactor[Array]":
        """
        Marginalize out variables by ID.

        Parameters
        ----------
        marg_var_ids : List[int]
            Variable IDs to marginalize out.

        Returns
        -------
        GaussianFactor
            Marginal distribution.
        """
        # Get indices to marginalize
        marg_indices: List[int] = []
        for var_id in marg_var_ids:
            marg_indices.extend(self._get_var_indices(var_id))

        # Get remaining var_ids and nvars_per_var
        remain_var_ids: List[int] = []
        remain_nvars_per_var: List[int] = []
        for i, var_id in enumerate(self._var_ids):
            if var_id not in marg_var_ids:
                remain_var_ids.append(var_id)
                remain_nvars_per_var.append(self._nvars_per_var[i])

        # Marginalize in canonical form
        marg_indices_arr = self._bkd.array(marg_indices)
        marginal_canonical = self._canonical.marginalize(marg_indices_arr)

        return GaussianFactor(
            marginal_canonical,
            remain_var_ids,
            remain_nvars_per_var,
            self._bkd,
        )

    def condition_vars(
        self, fixed_var_ids: List[int], values: Array
    ) -> "GaussianFactor[Array]":
        """
        Condition on variables by ID.

        Parameters
        ----------
        fixed_var_ids : List[int]
            Variable IDs to condition on.
        values : Array
            Values for fixed variables. Shape: (total_fixed_dims,)

        Returns
        -------
        GaussianFactor
            Conditional distribution.
        """
        # Get indices to fix
        fixed_indices: List[int] = []
        for var_id in fixed_var_ids:
            fixed_indices.extend(self._get_var_indices(var_id))

        # Verify value dimensions
        expected_dims = sum(
            self._nvars_per_var[self._var_ids.index(vid)] for vid in fixed_var_ids
        )
        if values.ndim == 2:
            values = self._bkd.flatten(values)
        if len(values) != expected_dims:
            raise ValueError(
                f"Values dimension {len(values)} doesn't match expected {expected_dims}"
            )

        # Get remaining var_ids and nvars_per_var
        remain_var_ids: List[int] = []
        remain_nvars_per_var: List[int] = []
        for i, var_id in enumerate(self._var_ids):
            if var_id not in fixed_var_ids:
                remain_var_ids.append(var_id)
                remain_nvars_per_var.append(self._nvars_per_var[i])

        # Condition in canonical form
        fixed_indices_arr = self._bkd.array(fixed_indices)
        conditional_canonical = self._canonical.condition(fixed_indices_arr, values)

        return GaussianFactor(
            conditional_canonical,
            remain_var_ids,
            remain_nvars_per_var,
            self._bkd,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GaussianFactor(var_ids={self._var_ids}, "
            f"nvars_per_var={self._nvars_per_var})"
        )
