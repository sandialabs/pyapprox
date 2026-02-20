"""Locally adaptive combination sparse grid.

This module provides locally-adaptive sparse grids that refine
individual basis functions rather than entire subspaces.
"""

from typing import Callable, Dict, Generic, List, Optional, Tuple, cast

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    Basis1DProtocol,
)
from pyapprox.typing.surrogates.affine.indices import PriorityQueue

from .index_generator import LocalIndexGenerator
from .refinement import LocalHierarchicalRefinementCriteria


class LocallyAdaptiveCombinationSparseGrid(Generic[Array]):
    """Locally adaptive sparse grid with basis-level refinement.

    Unlike standard adaptive sparse grids that refine entire subspaces,
    this class refines individual basis functions based on their
    hierarchical surplus.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.
    univariate_basis : Basis1DProtocol[Array]
        Univariate basis (same for all dimensions).
    refinement_priority : Callable, optional
        Function(basis_index, basis_value, grid) -> (priority, error).
        Default: LocalHierarchicalRefinementCriteria.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import (
    ...     PiecewiseLinearBasis1D
    ... )
    >>> bkd = NumpyBkd()
    >>> basis = PiecewiseLinearBasis1D(bkd)
    >>> grid = LocallyAdaptiveCombinationSparseGrid(bkd, nvars=2, basis)
    >>> samples = grid.step_samples()
    >>> values = my_function(samples)
    >>> grid.step_values(values)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nvars: int,
        univariate_basis: Basis1DProtocol[Array],
        refinement_priority: Optional[
            Callable[
                [Array, Array, "LocallyAdaptiveCombinationSparseGrid[Array]"],
                Tuple[float, float],
            ]
        ] = None,
    ):
        # TODO: Runtime protocol validation incomplete - univariate_basis
        # parameter is stored but never used. Needs refactoring.

        self._bkd = bkd
        self._nvars = nvars
        self._univariate_basis = univariate_basis

        # Refinement criteria
        if refinement_priority is None:
            refinement_priority = LocalHierarchicalRefinementCriteria(bkd)
        self._refinement_priority = refinement_priority

        # Index generator for basis functions
        self._index_gen = LocalIndexGenerator(bkd, nvars)

        # Priority queue for candidates
        self._candidate_queue: Optional[PriorityQueue[Array]] = None

        # Data storage
        self._samples: Optional[Array] = None
        self._values: Optional[Array] = None
        self._nqoi: Optional[int] = None

        # Mapping from basis index to sample/value index
        self._basis_to_sample_idx: Dict[Tuple[int, ...], int] = {}

        # Tracking for step pattern
        self._first_step = True
        self._basis_errors: List[float] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        if self._nqoi is None:
            raise ValueError("Values not set. Call step_values() first.")
        return self._nqoi

    def _basis_index_to_sample(self, basis_index: Array) -> Array:
        """Convert basis index to sample location.

        For piecewise linear basis on [0, 1]:
        - Index 0 -> x = 0.5 (midpoint)
        - Index 1 -> x = 0.0 (left)
        - Index 2 -> x = 1.0 (right)
        - Index k (k >= 3) -> x = (k - 2^floor(log2(k))) / 2^floor(log2(k))

        Parameters
        ----------
        basis_index : Array
            Multi-index. Shape: (nvars,)

        Returns
        -------
        Array
            Sample location. Shape: (nvars, 1)
        """
        import math
        sample = self._bkd.zeros((self._nvars, 1))
        for d in range(self._nvars):
            idx = int(basis_index[d])
            if idx == 0:
                x = 0.5
            elif idx == 1:
                x = 0.0
            elif idx == 2:
                x = 1.0
            else:
                # Hierarchical dyadic points
                level = int(math.floor(math.log2(idx + 1)))
                offset = idx - (2**level - 1)
                x = (2 * offset + 1) / (2 ** (level + 1))
            sample[d, 0] = x
        return sample

    def _get_basis_sample(self, basis_index: Array) -> Array:
        """Get sample location for a basis index."""
        return self._basis_index_to_sample(basis_index)

    def step_samples(self) -> Optional[Array]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Array]
            New samples of shape (nvars, nnew) or None if converged.
        """
        if self._first_step:
            return self._first_step_samples()
        return self._next_step_samples()

    def _first_step_samples(self) -> Array:
        """Get samples for the first step."""
        # Initialize with root basis
        self._index_gen.initialize()

        # Add candidates from root
        root = self._bkd.zeros((self._nvars,), dtype=self._bkd.int64_dtype())
        candidates = self._index_gen.add_candidates(root)

        # Collect all samples (root + candidates)
        all_indices = [root] + candidates
        samples_list: List[Array] = []

        for i, idx in enumerate(all_indices):
            sample = self._basis_index_to_sample(idx)
            samples_list.append(sample)
            self._basis_to_sample_idx[self._index_gen._hash_index(idx)] = i

        self._samples = self._bkd.hstack(samples_list)
        self._first_step = False
        return self._samples

    def _next_step_samples(self) -> Optional[Array]:
        """Get samples for subsequent steps."""
        if self._candidate_queue is None or self._candidate_queue.empty():
            return None

        # Get best candidate
        priority, error, best_idx = self._candidate_queue.get()
        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return None

        # Find the corresponding basis index
        best_basis_index = None
        for key, idx in self._index_gen._cand_basis_indices_dict.items():
            if idx == best_idx:
                best_basis_index = self._bkd.asarray(
                    list(key), dtype=self._bkd.int64_dtype()
                )
                break

        if best_basis_index is None:
            return None

        # Select this candidate and add its children
        self._index_gen.select_candidate(best_basis_index)
        new_candidates = self._index_gen.add_candidates(best_basis_index)

        if len(new_candidates) == 0:
            # No new samples, try next candidate
            return self._next_step_samples()

        # Collect new samples
        assert self._samples is not None
        samples_list: List[Array] = []
        current_count = int(self._samples.shape[1])

        for i, cand_idx in enumerate(new_candidates):
            sample = self._basis_index_to_sample(cand_idx)
            samples_list.append(sample)
            self._basis_to_sample_idx[self._index_gen._hash_index(cand_idx)] = (
                current_count + i
            )

        new_samples = self._bkd.hstack(samples_list)
        self._samples = self._bkd.hstack([self._samples, new_samples])
        return new_samples

    def step_values(self, values: Array) -> None:
        """Provide function values for samples from step_samples.

        Parameters
        ----------
        values : Array
            Values of shape (nnew, nqoi)
        """
        if self._values is None:
            self._values = values
            self._nqoi = values.shape[1]
        else:
            self._values = self._bkd.vstack((self._values, values))

        # Prioritize candidates
        self._prioritize_candidates()

    def _prioritize_candidates(self) -> None:
        """Compute priorities for all candidate basis functions."""
        self._candidate_queue = PriorityQueue(max_priority=True)

        cand_indices = self._index_gen.get_candidate_indices()
        if cand_indices is None:
            return

        if self._values is None:
            return

        for j in range(cand_indices.shape[1]):
            basis_index = cand_indices[:, j]
            key = self._index_gen._hash_index(basis_index)

            if key not in self._basis_to_sample_idx:
                continue

            sample_idx = self._basis_to_sample_idx[key]
            if sample_idx >= self._values.shape[0]:
                continue

            basis_value = self._values[sample_idx, :]
            priority, error = self._refinement_priority(
                basis_index, basis_value, self
            )

            idx = self._index_gen._cand_basis_indices_dict[key]
            self._candidate_queue.put(priority, error, idx)

    def _evaluate_with_selected_indices(self, samples: Array) -> Array:
        """Evaluate interpolant using only selected basis functions.

        This is a simplified evaluation that sums contributions from
        selected basis functions.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values. Shape: (npoints, nqoi)
        """
        if self._values is None or self._nqoi is None:
            return self._bkd.zeros((samples.shape[1], 1))

        npoints = samples.shape[1]
        result = self._bkd.zeros((npoints, self._nqoi))

        # Sum over selected basis functions
        for key, sample_idx in self._basis_to_sample_idx.items():
            if key not in self._index_gen._sel_basis_indices_dict:
                continue

            basis_index = self._bkd.asarray(
                list(key), dtype=self._bkd.int64_dtype()
            )

            # Evaluate basis function at all points
            basis_vals = self._evaluate_basis(basis_index, samples)

            # Get hierarchical coefficient (value at node)
            if sample_idx < self._values.shape[0]:
                coef = self._values[sample_idx, :]  # Shape: (nqoi,)
                # Add contribution: basis_vals (npoints,) * coef (nqoi,)
                result = result + basis_vals[:, None] * coef[None, :]

        return result

    def _evaluate_basis(self, basis_index: Array, samples: Array) -> Array:
        """Evaluate a single tensor-product basis function.

        Parameters
        ----------
        basis_index : Array
            Basis index. Shape: (nvars,)
        samples : Array
            Evaluation points. Shape: (nvars, npoints)

        Returns
        -------
        Array
            Basis values. Shape: (npoints,)
        """
        npoints = samples.shape[1]
        result = self._bkd.ones((npoints,))

        for d in range(self._nvars):
            # Get 1D basis function value
            idx = int(basis_index[d])
            x = samples[d, :]
            vals_1d = self._evaluate_1d_basis(idx, x)
            result = result * vals_1d

        return result

    def _evaluate_1d_basis(self, idx: int, x: Array) -> Array:
        """Evaluate 1D hierarchical basis function.

        Piecewise linear hat function centered at the node.

        Parameters
        ----------
        idx : int
            1D basis index.
        x : Array
            Evaluation points. Shape: (npoints,)

        Returns
        -------
        Array
            Basis values. Shape: (npoints,)
        """
        import math

        # Get center and half-width of support
        if idx == 0:
            center = 0.5
            half_width = 0.5
        elif idx == 1:
            center = 0.0
            half_width = 0.5
        elif idx == 2:
            center = 1.0
            half_width = 0.5
        else:
            level = int(math.floor(math.log2(idx + 1)))
            offset = idx - (2**level - 1)
            center = (2 * offset + 1) / (2 ** (level + 1))
            half_width = 1.0 / (2 ** (level + 1))

        # Hat function: max(0, 1 - |x - center| / half_width)
        dist = self._bkd.abs(x - center) / half_width
        one_minus_dist = 1.0 - dist
        zeros = self._bkd.zeros_like(one_minus_dist)

        # Use where to compute max(0, 1 - dist)
        # Cast condition to Array since Array comparison always returns Array
        vals = self._bkd.where(
            cast(Array, one_minus_dist > 0), one_minus_dist, zeros
        )

        # Handle boundary bases (half hats at x=0 and x=1)
        if idx == 1:
            # Left boundary: only active for x <= half_width
            vals = self._bkd.where(
                cast(Array, x <= half_width), 1.0 - x / half_width, zeros
            )
        elif idx == 2:
            # Right boundary: only active for x >= 1 - half_width
            vals = self._bkd.where(
                cast(Array, x >= 1.0 - half_width),
                1.0 - (1.0 - x) / half_width,
                zeros
            )

        return vals

    def __call__(self, samples: Array) -> Array:
        """Evaluate the sparse grid interpolant.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values. Shape: (npoints, nqoi)
        """
        return self._evaluate_with_selected_indices(samples)

    def error_estimate(self) -> float:
        """Return current error estimate."""
        if self._candidate_queue is None:
            return float("inf")
        # Sum of errors from remaining candidates
        return sum(
            err for _, err, _ in self._candidate_queue._queue
        ) if hasattr(self._candidate_queue, '_queue') else 0.0

    def nselected(self) -> int:
        """Return number of selected basis functions."""
        return self._index_gen.nselected()

    def ncandidates(self) -> int:
        """Return number of candidate basis functions."""
        return self._index_gen.ncandidates()

    def _integrate_1d_basis(self, idx: int) -> float:
        """Compute integral of 1D hierarchical basis function over [0, 1].

        For piecewise linear hat functions:
        - Interior hat: area = base * height / 2 = 2*half_width * 1 / 2 = half_width
        - Boundary half-hat: area = half_width * 1 / 2 = half_width / 2

        Parameters
        ----------
        idx : int
            1D basis index.

        Returns
        -------
        float
            Integral of the basis function over [0, 1].
        """
        import math

        if idx == 0:
            # Root: center=0.5, half_width=0.5, full hat
            return 0.5
        elif idx == 1:
            # Left boundary: half-hat at x=0
            return 0.25
        elif idx == 2:
            # Right boundary: half-hat at x=1
            return 0.25
        else:
            # Interior hierarchical: level determines half_width
            level = int(math.floor(math.log2(idx + 1)))
            half_width: float = 1.0 / (2 ** (level + 1))
            return half_width

    def _integrate_basis(self, basis_index: Array) -> float:
        """Compute integral of tensor-product basis function over [0, 1]^nvars.

        Parameters
        ----------
        basis_index : Array
            Basis index. Shape: (nvars,)

        Returns
        -------
        float
            Integral of the basis function.
        """
        integral = 1.0
        for d in range(self._nvars):
            idx = int(basis_index[d])
            integral *= self._integrate_1d_basis(idx)
        return integral

    def mean(self) -> Array:
        """Compute the mean (expectation) of the interpolant.

        Computes E[f] = integral f(x) dx over [0, 1]^nvars using the
        hierarchical representation:
            E[f] = sum_i c_i * integral(phi_i)

        where c_i are the hierarchical coefficients (function values at nodes)
        and phi_i are the basis functions.

        Returns
        -------
        Array
            Mean values of shape (nqoi,)

        Notes
        -----
        This computes the expectation with respect to the uniform probability
        measure on [0, 1]^nvars. For more accurate integration, use more
        refinement steps.
        """
        if self._values is None or self._nqoi is None:
            raise ValueError("Values not set. Call step_values() first.")

        result = self._bkd.zeros((self._nqoi,))

        # Sum over selected basis functions only
        for key, sample_idx in self._basis_to_sample_idx.items():
            if key not in self._index_gen._sel_basis_indices_dict:
                continue

            basis_index = self._bkd.asarray(
                list(key), dtype=self._bkd.int64_dtype()
            )

            # Compute integral of this basis function
            basis_integral = self._integrate_basis(basis_index)

            # Get coefficient (value at node)
            if sample_idx < self._values.shape[0]:
                coef = self._values[sample_idx, :]  # Shape: (nqoi,)
                result = result + basis_integral * coef

        return result

    def __repr__(self) -> str:
        return (
            f"LocallyAdaptiveCombinationSparseGrid(nvars={self._nvars}, "
            f"nselected={self.nselected()}, ncandidates={self.ncandidates()})"
        )
