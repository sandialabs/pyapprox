"""Single cached constraint object aggregating essential BCs.

``DirichletConstraintSet`` concatenates the constrained DOFs of a list
of ``EssentialBCProtocol`` objects once at construction, validates them
(in range, no repeats within one BC) and resolves DOFs shared by
several BCs last-wins, then provides sparse-aware application methods
with no per-BC python loops and no per-call mass rebuilds.

All dense/vector operations stay in the computational backend
(autograd-safe ``index_update``); numpy is used only at the scipy
sparse-matrix interface, which lives outside the backend system.
"""

from typing import Any, Generic, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags, issparse, spmatrix

from pyapprox.pde.boundary.classification import BCDofClassification
from pyapprox.pde.boundary.protocols import EssentialBCProtocol
from pyapprox.pde.sparse_utils import apply_dirichlet_rows
from pyapprox.util.backends.protocols import Array, Backend


class DirichletConstraintSet(Generic[Array]):
    """Aggregated essential (Dirichlet) constraints for one physics.

    Satisfies ``ConstraintSetProtocol``. Constrained DOF LOCATIONS are
    fixed at construction (values may vary with time); the constrained
    mass matrix and the DOF classification are cached. DOFs shared by
    several BCs (corners of adjacent boundaries) take the last BC's
    value, matching legacy sequential BC application.

    **Empty-set no-op invariant**: constructed from ``bcs=[]`` (periodic
    meshes, pure-Neumann problems) every method returns its input
    unchanged.

    Parameters
    ----------
    bcs : sequence of EssentialBCProtocol
        Essential BCs to aggregate. May be empty.
    nstates : int
        Total number of DOFs; used to validate DOF indices.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        bcs: Sequence[EssentialBCProtocol[Array]],
        nstates: int,
        bkd: Backend[Array],
    ) -> None:
        for bc in bcs:
            if not isinstance(bc, EssentialBCProtocol):
                raise TypeError(
                    "bcs must satisfy EssentialBCProtocol, got "
                    f"{type(bc).__name__}"
                )
        if nstates <= 0:
            raise ValueError(f"nstates must be positive, got {nstates}")
        self._bcs = list(bcs)
        self._nstates = nstates
        self._bkd = bkd

        # DOFs shared by several BCs (e.g. corners of adjacent 2D
        # boundaries) resolve last-wins, matching the legacy sequential
        # application order in which later BCs overwrote earlier ones.
        # The dict loop (not fancy-assignment scatter) keeps last-wins
        # deterministic on torch, and runs once at construction.
        position_of: dict[int, int] = {}
        position = 0
        for ibc, bc in enumerate(self._bcs):
            bc_dofs = [int(d) for d in bc.constrained_dofs()]
            if len(set(bc_dofs)) != len(bc_dofs):
                raise ValueError(
                    f"essential BC {ibc} ({bc!r}) lists a DOF more "
                    "than once"
                )
            for dof in bc_dofs:
                if dof < 0 or dof >= nstates:
                    raise ValueError(
                        "constrained DOF indices must lie in "
                        f"[0, {nstates}), got {dof} from BC {ibc}"
                    )
                position_of[dof] = position
                position += 1
        unique_dofs = sorted(position_of)
        self._dofs = bkd.asarray(unique_dofs, dtype=bkd.int64_dtype())
        # Gather indices into the per-BC value concatenation, one per
        # unique DOF (its winning occurrence).
        self._value_sel = bkd.asarray(
            [position_of[dof] for dof in unique_dofs],
            dtype=bkd.int64_dtype(),
        )
        # Numpy copy of the indices, built lazily and used only at the
        # scipy sparse-matrix interface (outside the backend system).
        self._dofs_np: Optional[NDArray[np.int64]] = None
        self._cached_classification: Optional[BCDofClassification] = None
        # Single-slot cache for apply_to_mass: holding a reference to
        # the input matrix keeps the identity comparison valid.
        self._cached_mass_input: Union[spmatrix, Array, None] = None
        self._cached_mass_result: Union[spmatrix, Array, None] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def dofs(self) -> Array:
        """Return all constrained DOF indices. Shape: (ndofs,)"""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of constrained DOFs."""
        return len(self._dofs)

    def nstates(self) -> int:
        """Return the total number of DOFs."""
        return self._nstates

    def _dofs_numpy(self) -> NDArray[np.int64]:
        """Return numpy indices for scipy sparse interop only."""
        if self._dofs_np is None:
            self._dofs_np = self._bkd.to_numpy(self._dofs).astype(np.int64)
        return self._dofs_np

    def values(self, time: float) -> Array:
        """Return all prescribed values at ``time``. Shape: (ndofs,)

        Ordered to match ``dofs()``; at DOFs shared by several BCs the
        last BC's value wins.
        """
        if not self._bcs:
            return self._bkd.zeros((0,))
        all_values = self._bkd.concatenate(
            [bc.constrained_values(time) for bc in self._bcs]
        )
        return all_values[self._value_sel]

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Replace constrained rows with the constraint violation.

        Sets ``residual[d] = state[d] - g(time)`` for constrained DOFs.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        if not self.ndofs():
            return residual
        return self._bkd.index_update(
            residual, self._dofs, state[self._dofs] - self.values(time)
        )

    def apply_to_jacobian(
        self, jacobian: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Replace constrained rows with identity rows ``e_d``.

        Parameters
        ----------
        jacobian : sparse matrix or Array
            Jacobian matrix. Shape: (nstates, nstates)

        Returns
        -------
        sparse matrix or Array
            Modified Jacobian (same type as input).
        """
        if not self.ndofs():
            return jacobian
        if issparse(jacobian):
            return apply_dirichlet_rows(jacobian, self._dofs_numpy())
        jac = self._bkd.index_update(jacobian, self._dofs, 0.0)
        return self._bkd.index_update(jac, (self._dofs, self._dofs), 1.0)

    def apply_to_mass(
        self, mass: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Return the mass matrix with identity rows at constrained DOFs.

        The result is cached: repeated calls with the same matrix object
        (the common case — a physics' constant mass matrix) return the
        cached result without rebuilding.

        Parameters
        ----------
        mass : sparse matrix or Array
            Mass matrix. Shape: (nstates, nstates)

        Returns
        -------
        sparse matrix or Array
            Modified mass matrix (same type as input).
        """
        if not self.ndofs():
            return mass
        if mass is not self._cached_mass_input:
            self._cached_mass_input = mass
            self._cached_mass_result = self.apply_to_jacobian(mass)
        cached = self._cached_mass_result
        if cached is None:
            raise RuntimeError("apply_to_mass cache is unexpectedly empty")
        return cached

    def _sparse_row_mask(self) -> NDArray[np.floating[Any]]:
        """Return a 0/1 row mask for scipy sparse interop only."""
        mask = np.ones(self._nstates, dtype=np.float64)
        mask[self._dofs_numpy()] = 0.0
        return mask

    def zero_rows(
        self, matrix: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Zero constrained rows (e.g. dR/dp, same-step HVP outputs).

        Parameters
        ----------
        matrix : sparse matrix or Array
            Matrix with ``nstates`` rows. Shape: (nstates, ncols)

        Returns
        -------
        sparse matrix or Array
            Matrix with constrained rows zeroed (sparse inputs are
            returned in CSR format).
        """
        if not self.ndofs():
            return matrix
        if issparse(matrix):
            return (diags(self._sparse_row_mask()) @ matrix).tocsr()
        return self._bkd.index_update(matrix, self._dofs, 0.0)

    def zero_cols(
        self, matrix: Union[spmatrix, Array]
    ) -> Union[spmatrix, Array]:
        """Zero constrained columns (transposed off-diagonal blocks).

        Parameters
        ----------
        matrix : sparse matrix or Array
            Matrix with ``nstates`` columns. Shape: (nrows, nstates)

        Returns
        -------
        sparse matrix or Array
            Matrix with constrained columns zeroed (sparse inputs are
            returned in CSR format).
        """
        if not self.ndofs():
            return matrix
        if issparse(matrix):
            return (matrix @ diags(self._sparse_row_mask())).tocsr()
        return self._bkd.index_update(
            matrix, (slice(None), self._dofs), 0.0
        )

    def zero_entries(self, vec: Array) -> Array:
        """Zero constrained entries of a vector (adjoint RHS).

        Parameters
        ----------
        vec : Array
            Vector. Shape: (nstates,)

        Returns
        -------
        Array
            Vector with constrained entries zeroed.
        """
        if not self.ndofs():
            return vec
        return self._bkd.index_update(vec, self._dofs, 0.0)

    def inject(self, state: Array, time: float) -> Array:
        """Return ``state`` with prescribed values injected.

        Sets ``state[d] = g(time)`` for constrained DOFs (used for
        initial guesses and initial conditions).

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            State with prescribed values at constrained DOFs.
        """
        if not self.ndofs():
            return state
        return self._bkd.index_update(state, self._dofs, self.values(time))

    def classification(self) -> BCDofClassification:
        """Return the DOF classification.

        For Galerkin essential constraints every constrained DOF is
        row-replaced, so ``essential == row_replaced``. The result is
        cached: DOF locations are fixed at construction, so the
        per-element conversion to python ints runs at most once.
        """
        if self._cached_classification is None:
            dof_list = [int(d) for d in self._dofs]
            self._cached_classification = BCDofClassification(
                essential=dof_list, row_replaced=list(dof_list)
            )
        return self._cached_classification

    def __repr__(self) -> str:
        return (
            f"DirichletConstraintSet(ndofs={self.ndofs()}, "
            f"nstates={self._nstates}, nbcs={len(self._bcs)})"
        )
