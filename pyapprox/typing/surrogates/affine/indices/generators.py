"""Multi-index generators for polynomial expansions.

This module provides index generators for constructing multi-index sets
used in polynomial chaos and tensor-product approximations.

Classes
-------
IndexGenerator
    Abstract base class for index generators.
IterativeIndexGenerator
    Base class for iterative/adaptive index generation.
HyperbolicIndexGenerator
    Generator for hyperbolic cross index sets.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Dict

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.indices.utils import (
    hash_index,
    indices_pnorm,
)
from pyapprox.typing.surrogates.affine.indices.admissibility import (
    AdmissibilityCriteria,
    MaxLevelCriteria,
    Max1DLevelsCriteria,
    CompositeCriteria,
)


class IndexGenerator(ABC, Generic[Array]):
    """Abstract base class for multi-index generators.

    Parameters
    ----------
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]):
        self._bkd = bkd
        self._nvars = nvars
        self._indices = self._bkd.zeros(
            (nvars, 0), dtype=self._bkd.int64_dtype()
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nindices(self) -> int:
        """Return the number of indices."""
        return self._indices.shape[1]

    def _hash_index(self, array: Array) -> int:
        """Compute hash for an index."""
        return hash_index(array, self._bkd)

    @abstractmethod
    def _get_indices(self) -> Array:
        """Internal method to compute indices."""
        raise NotImplementedError

    def get_indices(self) -> Array:
        """Return the multi-indices.

        Returns
        -------
        Array
            Multi-indices. Shape: (nvars, nindices)
        """
        indices = self._get_indices()
        return indices

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nvars={self.nvars()}, nindices={self.nindices()})"


class IterativeIndexGenerator(IndexGenerator[Array], Generic[Array]):
    """Base class for iterative/adaptive index generators.

    Maintains separate selected and candidate index sets for adaptive
    refinement of polynomial approximations.

    Parameters
    ----------
    nvars : int
        Number of variables.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]):
        super().__init__(nvars, bkd)
        self._verbosity = 0
        self._sel_indices_dict: Dict[int, int] = {}
        self._cand_indices_dict: Dict[int, int] = {}
        self._admis_criteria: Optional[AdmissibilityCriteria[Array]] = None

    def set_verbosity(self, verbosity: int) -> None:
        """Set verbosity level for debugging."""
        self._verbosity = verbosity

    def set_admissibility_criteria(
        self, criteria: AdmissibilityCriteria[Array]
    ) -> None:
        """Set the admissibility criteria for index generation."""
        self._admis_criteria = criteria

    def _get_forward_neighbor(self, index: Array, dim_id: int) -> Array:
        """Get forward neighbor in specified dimension."""
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] += 1
        return neighbor

    def _get_backward_neighbor(self, index: Array, dim_id: int) -> Array:
        """Get backward neighbor in specified dimension."""
        neighbor = self._bkd.copy(index)
        neighbor[dim_id] -= 1
        return neighbor

    def _index_on_margin(self, index: Array) -> bool:
        """Check if index is on the margin (no forward neighbors selected)."""
        for dim_id in range(self.nvars()):
            neighbor = self._get_forward_neighbor(index, dim_id)
            if self._hash_index(neighbor) in self._sel_indices_dict:
                return False
        return True

    def _indices_are_downward_closed(self, indices: Array) -> bool:
        """Check if indices form a downward closed set."""
        for index in indices.T:
            for dim_id in range(self.nvars()):
                if index[dim_id] > 0:
                    neighbor = self._get_backward_neighbor(index, dim_id)
                    if self._hash_index(neighbor) not in self._sel_indices_dict:
                        return False
        return True

    def _is_admissible(self, index: Array) -> bool:
        """Check if an index is admissible."""
        fail_msg = f"Index {self._bkd.to_numpy(index)} is not admissible: "

        # Check if already in selected or candidate sets
        key = self._hash_index(index)
        if key in self._sel_indices_dict:
            if self._verbosity > 1:
                print(fail_msg + "already in selected set")
            return False
        if key in self._cand_indices_dict:
            if self._verbosity > 1:
                print(fail_msg + "already in candidate set")
            return False

        # Check downward closure
        for dim_id in range(self.nvars()):
            if index[dim_id] > 0:
                neighbor = self._get_backward_neighbor(index, dim_id)
                if self._hash_index(neighbor) not in self._sel_indices_dict:
                    if self._verbosity > 1:
                        print(fail_msg + "not downward closed")
                    return False

        # Check admissibility criteria
        if self._admis_criteria is not None:
            is_admissible = self._admis_criteria(index)
            if not is_admissible and self._verbosity > 1:
                print(fail_msg + self._admis_criteria.failure_message())
            return is_admissible

        return True

    def _get_new_candidate_indices(self, index: Array) -> Array:
        """Get new candidate indices from forward neighbors of given index."""
        if self._admis_criteria is None:
            raise RuntimeError("Must set admissibility criteria")

        new_candidates = []
        for dim_id in range(self.nvars()):
            neighbor = self._get_forward_neighbor(index, dim_id)
            if self._is_admissible(neighbor):
                new_candidates.append(neighbor)
                if self._verbosity > 1:
                    print(f"Adding candidate: {self._bkd.to_numpy(neighbor)}")

        if len(new_candidates) > 0:
            return self._bkd.stack(new_candidates, axis=1)
        return self._bkd.zeros(
            (self.nvars(), 0), dtype=self._bkd.int64_dtype()
        )

    def _find_candidate_indices(self) -> Array:
        """Find all candidate indices from current selected set."""
        candidates = []
        idx = self.nselected_indices()

        for index in self._indices.T:
            if not self._index_on_margin(index):
                continue
            new_candidates = self._get_new_candidate_indices(index)
            for cand in new_candidates.T:
                key = self._hash_index(cand)
                if key not in self._cand_indices_dict:
                    self._cand_indices_dict[key] = idx
                    candidates.append(cand)
                    idx += 1

        if len(candidates) > 0:
            return self._bkd.stack(candidates, axis=1)
        return self._bkd.zeros(
            (self.nvars(), 0), dtype=self._bkd.int64_dtype()
        )

    def set_selected_indices(self, selected_indices: Array) -> None:
        """Set the initial selected indices.

        Parameters
        ----------
        selected_indices : Array
            Initial selected indices. Shape: (nvars, nindices)
        """
        self._sel_indices_dict = {}
        self._cand_indices_dict = {}

        if selected_indices.ndim != 2 or selected_indices.shape[0] != self.nvars():
            raise ValueError("selected_indices must have shape (nvars, nindices)")

        self._indices = self._bkd.copy(selected_indices)
        for idx, index in enumerate(self._indices.T):
            self._sel_indices_dict[self._hash_index(index)] = idx

        if not self._indices_are_downward_closed(self._indices):
            raise ValueError("Selected indices must be downward closed")

        # Find candidate indices
        cand_indices = self._find_candidate_indices()
        self._indices = self._bkd.hstack((self._indices, cand_indices))

    def nselected_indices(self) -> int:
        """Return the number of selected indices."""
        return len(self._sel_indices_dict)

    def ncandidate_indices(self) -> int:
        """Return the number of candidate indices."""
        return len(self._cand_indices_dict)

    def _get_selected_idx(self) -> Array:
        """Get array indices of selected indices."""
        return self._bkd.asarray(
            list(self._sel_indices_dict.values()), dtype=self._bkd.int64_dtype()
        )

    def get_selected_indices(self) -> Array:
        """Return the selected indices.

        Returns
        -------
        Array
            Selected indices. Shape: (nvars, nselected)
        """
        idx = self._get_selected_idx()
        return self._indices[:, idx]

    def _get_candidate_idx(self) -> Array:
        """Get array indices of candidate indices."""
        return self._bkd.asarray(
            list(self._cand_indices_dict.values()), dtype=self._bkd.int64_dtype()
        )

    def get_candidate_indices(self) -> Optional[Array]:
        """Return the candidate indices.

        Returns
        -------
        Array or None
            Candidate indices. Shape: (nvars, ncandidate)
        """
        if self.ncandidate_indices() > 0:
            return self._indices[:, self._get_candidate_idx()]
        return None

    def refine_index(self, index: Array) -> Array:
        """Move an index from candidates to selected.

        Parameters
        ----------
        index : Array
            Index to refine. Shape: (nvars,)

        Returns
        -------
        Array
            New candidate indices. Shape: (nvars, nnew)
        """
        if self._verbosity > 0:
            print(f"Refining index {self._bkd.to_numpy(index)}")

        key = self._hash_index(index)
        self._sel_indices_dict[key] = self._cand_indices_dict[key]
        del self._cand_indices_dict[key]

        # Find new candidates from this index
        new_candidates = self._get_new_candidate_indices(index)
        idx = self._indices.shape[1]
        for cand in new_candidates.T:
            self._cand_indices_dict[self._hash_index(cand)] = idx
            idx += 1

        if new_candidates.shape[1] > 0:
            self._indices = self._bkd.hstack((self._indices, new_candidates))

        return new_candidates

    def step(self) -> None:
        """Move all candidates to selected and find new candidates."""
        # Move all candidates to selected
        for key, item in self._cand_indices_dict.items():
            self._sel_indices_dict[key] = item
        self._cand_indices_dict = {}

        # Find new candidates
        new_candidates = self._find_candidate_indices()
        self._indices = self._bkd.hstack((self._indices, new_candidates))

    def _get_indices(self) -> Array:
        return self._indices

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nvars={self.nvars()}, "
            f"nsel={self.nselected_indices()}, ncand={self.ncandidate_indices()})"
        )


class HyperbolicIndexGenerator(IterativeIndexGenerator[Array], Generic[Array]):
    """Generator for hyperbolic cross index sets.

    Generates multi-indices satisfying ||index||_p <= max_level where
    ||.||_p is the p-norm.

    Parameters
    ----------
    nvars : int
        Number of variables.
    max_level : int
        Maximum hyperbolic level.
    pnorm : float
        p-norm exponent (1.0 = total degree).
    bkd : Backend[Array]
        Computational backend.
    max_1d_levels : Array, optional
        Maximum level per dimension.
    """

    def __init__(
        self,
        nvars: int,
        max_level: int,
        pnorm: float,
        bkd: Backend[Array],
        max_1d_levels: Optional[Array] = None,
    ):
        super().__init__(nvars, bkd)

        # Build composite criteria
        level_criteria = MaxLevelCriteria(max_level, pnorm, bkd)
        if max_1d_levels is not None:
            criteria = CompositeCriteria(
                level_criteria,
                Max1DLevelsCriteria(max_1d_levels, bkd),
            )
        else:
            criteria = level_criteria

        self.set_admissibility_criteria(criteria)
        self._level_criteria = level_criteria
        self._initialize()

    def _initialize(self) -> None:
        """Initialize with zero index and compute all indices."""
        if self.nindices() != 0:
            raise ValueError("Can only initialize if nindices == 0")

        # Start with zero index
        zero_index = self._bkd.zeros(
            (self.nvars(), 1), dtype=self._bkd.int64_dtype()
        )
        self.set_selected_indices(zero_index)

        # Compute all indices up to max_level
        self._compute_indices()

    def _next_index_to_refine(self) -> int:
        """Find the candidate index with smallest norm to refine next."""
        candidates = self.get_candidate_indices()
        norms = indices_pnorm(
            candidates, self._level_criteria._pnorm, self._bkd
        )
        return int(self._bkd.to_numpy(self._bkd.argmin(norms)))

    def _compute_indices(self) -> None:
        """Compute all indices by iteratively refining smallest norm."""
        while self.ncandidate_indices() > 0:
            idx = self._next_index_to_refine()
            cand_idx = self._get_candidate_idx()
            index = self._indices[:, cand_idx[idx]]
            self.refine_index(index)

    def step(self) -> None:
        """Increment max_level by 1 and compute new indices."""
        self._level_criteria.max_level += 1
        super().step()
        self._compute_indices()

    def _get_indices(self) -> Array:
        return self._indices
