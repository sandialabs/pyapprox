# from abc import abstractmethod
from typing import List, Tuple

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.interface.model import Model
from pyapprox.surrogates.affine.kle import PrincipalComponentAnalysis
from pyapprox.surrogates.affine.linearsystemsolvers import LinearSystemSolver
from pyapprox.surrogates.affine.basis import Basis, MultiIndexBasis
from pyapprox.surrogates.affine.basisexp import BasisExpansion


class DynamicOperatorInference(Model):
    def __init__(self, nstates: int, nvars: int, backend: BackendMixin):
        """
        Initialize the DynamicOperatorInference class.
        """
        super().__init__(backend)
        self._nvars = nvars
        self._nstates = nstates

    def stack_raw_snapshots(
        self, raw_snapshots: List[Array], samples: Array
    ) -> Array:
        """
        Parameters
        ----------
        raw_snapshots : List[Array]
            List of trajectories of len [ntrajectories].
            Each trajectory is an Array [nstates, ntsteps].

        samples: Array [nvars, ntrajectories]
            Realizations of the random parameters that produced each snapshot.

        Result
        ------
        snapshots: Array [nstates, ntsteps*ntrajectories]
            Stack trajectories.

        snapshot_samples: [nvars, ntsteps*ntrajectories]
            Samples corresponding to each time snapshot

        ntsteps: int
            The number of tsteps in each trajetory
        """
        ntrajectories = len(raw_snapshots)
        if samples.ndim != 2 or samples.shape[1] != ntrajectories:
            raise ValueError(
                f"samples must be 2D array with {ntrajectories} columns"
            )

        for trajectory in raw_snapshots:
            if trajectory.shape != raw_snapshots[0].shape:
                raise ValueError(
                    "trajectory had shape {0} must have shape {1}".format(
                        trajectory.shape, raw_snapshots[0].shape
                    )
                )
        ntsteps = raw_snapshots[0].shape[1]
        snapshot_samples = self._bkd.repeat(samples, (ntsteps,), axis=1)
        return self._bkd.hstack(raw_snapshots), snapshot_samples, ntsteps

    def set_state_compressor(
        self, compressor: PrincipalComponentAnalysis
    ) -> None:
        if not isinstance(compressor, PrincipalComponentAnalysis):
            raise TypeError(
                "compressor must be an instance of PrincipalComponentAnalysis"
                "In the future we will allow other compressors, e.g. "
                "autoencoders"
            )
        self._compressor = compressor

    def nvars(self) -> int:
        """The number of parameters"""
        return self._nvars

    def nqoi(self) -> int:
        """The number of full-dimensional states"""
        return self._nstates

    def nreduced_states(self) -> int:
        """The number of reduced-dimensional states"""
        if not hasattr(self, "_compressor"):
            raise AttributeError("must call set_state_compressor")
        return self._compressor.nvars()

    def _values(self, samples: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        String representation of the DynamicOperatorInference object.
        """
        return "{0}(nstates={1},nvars={2})".format(
            self.__class__.__name__, self.nqoi(), self.nvars()
        )

    def _evolve(self, initial_cond: Array):
        raise NotImplementedError()

    def _reduce_snapshots(
        self, snapshots: Array, nreduced_states: int
    ) -> Array:
        self._pca = PrincipalComponentAnalysis(
            snapshots, nreduced_states, backend=self._bkd
        )
        return self._pca.reduce_state(snapshots)

    def set_time_derivative_operator_bases(
        self, state_basis: MultiIndexBasis, param_basis: MultiIndexBasis
    ):
        if not isinstance(state_basis, MultiIndexBasis):
            raise TypeError(
                "state_basis must be an instance of MultiIndexBasis"
            )
        self._state_basis = state_basis
        if not isinstance(param_basis, MultiIndexBasis):
            raise TypeError(
                "param_basis must be an instance of MultiIndexBasis"
            )
        self._param_basis = param_basis
        self._time_deriv_basis = MultiIndexBasis(
            state_basis._bases_1d + param_basis._bases_1d
        )

    def set_linear_system_solver(self, lin_solver: LinearSystemSolver):
        if not isinstance(lin_solver, LinearSystemSolver):
            raise TypeError(
                "lin_solver must be an instance of LinearSystemSolver"
            )
        if not self._bkd.bkd_equal(self._bkd, lin_solver._bkd):
            raise TypeError(
                "solver backend {0} is inconsistent with backend {1}".format(
                    lin_solver._bkd, self._bkd
                )
            )
        self._lin_solver = lin_solver

    def _setup_reduced_linear_system(
        self, snapshot_samples: Array, ntsteps: int
    ) -> Tuple[Array, Array]:

        full_snapshots = self._compressor.snapshots()
        reduced_snapshots = self._compressor.reduce_state(full_snapshots)

        rhs_vec = []
        state_samples = []
        param_samples = []
        lb = 0
        ub = 0
        ntsteps = snapshot_samples
        # trajectories are stored sequentially.
        print(ntsteps, "a")
        ntrajectories = reduced_snapshots.shape[1] // ntsteps
        for ii in range(ntrajectories):
            ub += lb + ntsteps
            reduced_trajectory = reduced_snapshots[:, lb:ub]
            rhs_vec.append(reduced_trajectory[:, 1:])
            state_samples.sappend(reduced_trajectory[:, :-1])
            param_samples.sappend(reduced_trajectory[:, :-1])
            lb = ub
        rhs_vec = self._bkd.hstack(rhs_vec)
        state_samples = self._bkd.hstack(state_samples)
        param_samples = self._bkd.hstack(param_samples)

        combined_samples = self._bkd.vstack((state_samples, param_samples))
        basis_mat = self._time_deriv_basis(combined_samples)

        return basis_mat, rhs_vec

    def fit(self, snapshot_samples: Array, ntsteps: int) -> None:
        """
        Fit the operator using snapshots and time steps.

        Parameters
        ----------
         snapshot_samples: [nvars, ntsteps*ntrajectories]
            Samples corresponding to each time snapshot
        """
        if not hasattr(self, "_lin_solver"):
            raise AttributeError("must call set_linear_system_solver")
        if self._compressor.snapshots().shape[1] != snapshot_samples.shape[1]:
            raise ValueError(
                "snapshot_samples shape does not match number of "
                "compressor snapshots"
            )

        if snapshot_samples.shape[1] % ntsteps != 0:
            raise ValueError("snapshot_samples shape does not match ntsteps")
        print(ntsteps, "a")
        self._fit(snapshot_samples, ntsteps)

    def _fit(self, snapshot_samples: Array, ntsteps: int) -> None:
        print(ntsteps, "b")
        lhs_matrix, rhs_vec = self._setup_reduced_linear_system(
            snapshot_samples, ntsteps
        )
        coef = self._lin_solver.solve(lhs_matrix, rhs_vec)[0]
        # consider if I can use fit of basis expansion
        # for now just use basis expansion for evaluation by computing coef
        # outside the expansion and setting them
        self._time_deriv_operator = BasisExpansion(
            self._opbasis, nqoi=self.nreduced_states()
        )
        self._time_deriv_operator.set_coefficient_basis(coef)

    def time_derivative_operator(self) -> BasisExpansion:
        if not hasattr(self, "_time_deriv_operator"):
            raise AttributeError("must call fit")
        return self._time_deriv_operator

    def time_derivative_operator_basis(self) -> MultiIndexBasis:
        return self._time_deriv_basis
