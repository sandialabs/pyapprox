# from abc import abstractmethod
from typing import List, Tuple, Type

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.interface.model import SingleSampleModel
from pyapprox.surrogates.affine.kle import PrincipalComponentAnalysis
from pyapprox.surrogates.affine.linearsystemsolvers import LinearSystemSolver
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.basisexp import BasisExpansion
from pyapprox.pde.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    TimeIntegratorNewtonResidual,
)
from pyapprox.util.newton import NewtonSolver, NewtonResidual

from abc import abstractmethod
from functools import partial


class SteadyStateSpaceNewtonModel(SteadyStateSpaceModel):
    def __init__(
        self,
        residual: ParameterizedNewtonResidual,
        newton_solver: NewtonSolver = None,
    ):
        """
        Model that maps an initial iterate + parameters to all states
        """
        self._check_residual(residual)
        super().__init__(residual._bkd)
        self._residual = residual
        if newton_solver is None:
            # use default newton solver
            newton_solver = NewtonSolver()
        self.set_newton_solver(newton_solver)

    def set_newton_solver(self, newton_solver: NewtonSolver) -> None:
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self._newton_solver = newton_solver
        self._newton_solver.set_residual(self._residual)

    def _check_residual(self, residual: ParameterizedNewtonResidual) -> None:
        if not isinstance(residual, ParameterizedNewtonResidual):
            raise ValueError(
                "residual must be an instance of "
                "ParameterizedNewtonResidual"
            )

    def set_initial_newton_iterate(self, iterate: Array) -> None:
        if iterate.ndim != 1:
            raise ValueError("iterate must be 1D array")
        self._init_iterate = iterate

    def split_sample(self, sample: Array) -> Array:
        if sample.ndim != 1:
            raise ValueError(
                f"sample must have shape {(self.nstates()+self.nparams())}"
            )
        init_iterate = sample[: self.nstates()]
        params = sample[self.nstates() :]
        return init_iterate, params

    def _set_parameters(self, params: Array) -> None:
        self._newton_solver.residual().set_parameters(params)

    def _evaluate(self, sample: Array) -> Array:
        init_iterate, params = self.split_sample(sample[:, 0])
        self._set_parameters(params)
        states = self._newton_solver.solve(init_iterate)
        return states

    def nvars(self) -> int:
        return self.nstates() + self.nparams()

    def nqoi(self) -> int:
        return self.nstates()


class TransientTimeIntegrationNewtonModel(TransientStateSpaceModel):
    def __init__(
        self,
        time_integrator: ImplicitTimeIntegrator,
    ):
        """
        Model that maps an initial condition + parameters to all states for all
        timesteps
        """
        if not isinstance(time_integrator, ImplicitTimeIntegrator):
            raise ValueError(
                "time_residual must be an instance of "
                "TimeIntegratorNewtonResidual"
            )
        super().__init__(time_integrator._bkd)
        self._time_itegrator = time_integrator

    def time_integrator(self) -> ImplicitTimeIntegrator:
        return self._time_itegrator

    def _set_parameters(self, params: Array) -> None:
        self._time_itegrator.time_residual().state_residual().set_parameters(
            params
        )

    def _evaluate(self, sample: Array) -> Array:
        init_iterate, params = self.split_sample(sample[:, 0])
        self._newton_solver.residual().set_parameters(params)
        states = self._newton_solver.solve(init_iterate)
        return states

    def nvars(self) -> int:
        return self.nstates() + self.nparams()

    def nqoi(self) -> int:
        return self.nstates() * self.ntsteps()


class DynamicOperatorInferenceResidual(TransientNewtonResidual):
    def __init__(self, bexp: BasisExpansion):
        if not isinstance(bexp, BasisExpansion):
            raise ValueError("bexp must be an instance of BasisExpansion")
        super().__init__(bexp._bkd)
        self._bexp = bexp
        print(self._bexp.get_coefficients())

    def set_time(self, time: float) -> None:
        # time is not used because operator inference
        # must evolve using same timesteps as training trajectories
        self._time = time

    def set_params(self, params: Array) -> None:
        if params.ndim != 1:
            raise ValueError("params must be a 1D array")
        self._param = params

    def _expand_state(self, state: Array) -> Array:
        if not hasattr(self, "_param"):
            raise AttributeError("must call set_param")
        return self._bkd.hstack((state, self._param))

    def __call__(self, state: Array) -> Array:
        if state.ndim != 1:
            raise ValueError("state.ndim must equal 1")
        print(state, self._time)
        expanded_state = self._expand_state(state)[:, None]
        values = self._bexp(expanded_state)[0]
        if values.ndim != 1:
            raise ValueError(
                f"self._value must return 1D array with shape {state.shape}"
            )
        return values


class DynamicOperatorInference(SingleSampleModel):
    def __init__(self, nstates: int, nparams: int, backend: BackendMixin):
        """
        Initialize the DynamicOperatorInference class.
        """
        super().__init__(backend)
        self._nparams = nparams
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
        """The number of parameters + number of states"""
        return self._nparams + self._nstates

    def nqoi(self) -> int:
        """The number of full-dimensional states"""
        return self.nreduced_states() * self._ntsteps

    def nfull_states(self) -> int:
        return self._nstates

    def nreduced_states(self) -> int:
        """The number of reduced-dimensional states"""
        if not hasattr(self, "_compressor"):
            raise AttributeError("must call set_state_compressor")
        return self._compressor.nvars()

    def __repr__(self) -> str:
        """
        String representation of the DynamicOperatorInference object.
        """
        return "{0}(nfull_states={1},nreduced_states={2}, nparams={3})".format(
            self.__class__.__name__,
            self.nfull_states(),
            self.nreduced_states(),
            self.nparams(),
        )

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
        # trajectories are stored sequentially.
        ntrajectories = reduced_snapshots.shape[1] // ntsteps
        for ii in range(ntrajectories):
            ub += lb + ntsteps
            reduced_trajectory = reduced_snapshots[:, lb:ub]
            rhs_vec.append(reduced_trajectory[:, 1:])
            state_samples.append(reduced_trajectory[:, :-1])
            trajectory_samples = snapshot_samples[:, lb:ub]
            param_samples.append(trajectory_samples[:, :-1])
            lb = ub
        rhs_vec = self._bkd.hstack(rhs_vec).T
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
        self._fit(snapshot_samples, ntsteps)
        self._ntsteps = ntsteps

    def _fit(self, snapshot_samples: Array, ntsteps: int) -> None:
        lhs_matrix, rhs_vec = self._setup_reduced_linear_system(
            snapshot_samples, ntsteps
        )
        print(self._bkd.cond(lhs_matrix))
        coef = self._lin_solver.solve(lhs_matrix, rhs_vec)
        # consider if I can use fit of basis expansion
        # for now just use basis expansion for evaluation by computing coef
        # outside the expansion and setting them
        self._time_deriv_operator = BasisExpansion(
            self.time_derivative_operator_basis(), nqoi=self.nreduced_states()
        )
        self._time_deriv_operator.set_coefficients(coef)

    def time_derivative_operator(self) -> BasisExpansion:
        if not hasattr(self, "_time_deriv_operator"):
            raise AttributeError("must call fit")
        return self._time_deriv_operator

    def time_derivative_operator_basis(self) -> MultiIndexBasis:
        return self._time_deriv_basis

    def setup_time_integration(
        self,
        time_residual_cls: Type[TimeIntegratorNewtonResidual],
        deltat: float,
        newton_solver: NewtonSolver = None,
    ):
        self._state_residual = DynamicOperatorInferenceResidual(
            self.time_derivative_operator()
        )
        time_residual = time_residual_cls(self._state_residual)
        self._time_int = ImplicitTimeIntegrator(
            time_residual,
            0.0,
            deltat * (self._ntsteps - 1),
            deltat,
            newton_solver=newton_solver,
            verbosity=0,
        )

    def _evaluate(self, sample: Array) -> Array:
        init_condition = sample[: self._nstates, 0]
        params = sample[self._nstates :, 0]
        self._state_residual.set_params(params)
        states = self._time_int.solve(init_condition)[0]
        return states.flatten()[None, :]

    def stack_initial_conditions_and_parameters(
        self, init_conds: Array, params: Array
    ) -> Array:
        if init_conds.ndim != 2 or init_conds.shape[0] != self._nstates:
            raise ValueError(
                f"init_cond must be 2D and have {self._nstates} rows"
                f"but has shape {init_conds.shape}"
            )
        if params.shape[0] != self._nparams:
            raise ValueError(
                "params must be have shape {0} but has shape {1}".format(
                    (self._nparams, init_conds.shape[1]), params.shape
                )
            )
        return self._bkd.vstack((init_conds, params))

    def __call__(self, samples: Array) -> Array:
        """
        Return the flattened trajectories at a set of initial conditions
        and random parameter realizations.


        This class wrapper is useful because it invokes all the
        parallelization, tests, etc implemented in Model base class.

        Parameters
        ----------
        samples : Array (nstates+nparams, nsamples)
            The initial conditions and random parameter realizations.

        Returns
        -------
        values: (nsamples, nstates * ntsteps)
            The flattened trajectories
            [t(0,p0), t(1, p1), ..., t(0, p1), t(1, p1) ...].
        """
        return super().__call__(samples)

    def predict(self, samples: Array) -> Array:
        """
        Return the trajectories at a set of initial conditions and random
        parameter realizations.

        Parameters
        ----------
        samples : Array (nstates+nparams, nsamples)
            The initial conditions and random parameter realizations.

        Returns
        -------
        trajectories: Array [nsamples, nstates, ntsteps]
            The trajectories at each random parameter realization
        """
        flattened_trajectories = self(samples)
        return self._bkd.stack(
            [
                self._bkd.reshape(trajectory, (self.nreduced_states(), -1))
                for trajectory in flattened_trajectories
            ]
        )
