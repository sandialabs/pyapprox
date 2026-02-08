"""Steady-state forward model for parameterized collocation PDEs.

Provides CollocationStateEquationAdapter (wraps CollocationModel + parameterized
physics as ParameterizedStateEquationWithJacobianProtocol) and SteadyForwardModel
(satisfies FunctionProtocol with adjoint-based Jacobian computation).
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.typing.optimization.implicitfunction.operator.operator_with_jacobian import (
    AdjointOperatorWithJacobian,
)
from pyapprox.typing.optimization.implicitfunction.operator.sensitivities import (
    VectorAdjointOperatorWithJacobian,
)
from pyapprox.typing.optimization.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesAdjointFunctional,
)
from pyapprox.typing.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)


class CollocationStateEquationAdapter(Generic[Array]):
    """Adapts CollocationModel for use with AdjointOperatorWithJacobian.

    Wraps CollocationModel + parameterized physics as
    ParameterizedStateEquationWithJacobianProtocol.

    Handles shape conversion between collocation convention (1D arrays)
    and protocol convention (2D column vectors):
        - State: collocation (nstates,) <-> protocol (nstates, 1)
        - Param: collocation (nparams,) <-> protocol (nparams, 1)
        - Residual return: (nstates, 1)
        - Jacobians: (nstates, nstates) and (nstates, nparams) -- same in both

    Parameters
    ----------
    model : CollocationModel
        Collocation model (must wrap parameterized physics with set_param,
        param_jacobian, nparams methods).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        model: CollocationModel[Array],
        bkd: Backend[Array],
    ):
        self._model = model
        self._bkd = bkd
        self._physics = model.physics()
        self._adapter = model.adapter()
        self._bc_indices = self._collect_bc_indices()

    def _collect_bc_indices(self):
        """Collect all boundary DOF indices from physics BCs."""
        indices = []
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
                bc_idx = bc.boundary_indices()
                for ii in range(bc_idx.shape[0]):
                    indices.append(int(bc_idx[ii]))
        return indices

    def _zero_bc_rows(self, matrix: Array) -> Array:
        """Zero rows of a matrix at boundary DOF indices."""
        matrix = self._bkd.copy(matrix)
        for idx in self._bc_indices:
            if matrix.ndim == 1:
                matrix[idx] = 0.0
            else:
                matrix[idx, :] = 0.0
        return matrix

    def _set_param(self, param: Array) -> None:
        """Set parameter on physics (converts 2D column to 1D)."""
        self._physics.set_param(param[:, 0])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return number of state variables."""
        return self._model.nstates()

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._physics.nparams()

    def solve(self, init_state: Array, param: Array) -> Array:
        """Solve the steady-state problem R(u, p) = 0 for u.

        Parameters
        ----------
        init_state : Array
            Initial guess for Newton solver. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Solution state. Shape: (nstates, 1).
        """
        self._set_param(param)
        sol_1d = self._model.solve_steady(init_state[:, 0])
        return sol_1d[:, None]

    def __call__(self, state: Array, param: Array) -> Array:
        """Compute the residual R(u, p) with boundary conditions applied.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Residual vector. Shape: (nstates, 1).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        self._adapter.set_time(0.0)
        residual = self._adapter(state_1d)
        jacobian = self._adapter.jacobian(state_1d)
        residual, _ = self._model._apply_boundary_conditions(
            residual, jacobian, state_1d, 0.0
        )
        return residual[:, None]

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute Jacobian of residual w.r.t. state, dR/du.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, nstates).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        self._adapter.set_time(0.0)
        residual = self._adapter(state_1d)
        jacobian = self._adapter.jacobian(state_1d)
        _, jacobian = self._model._apply_boundary_conditions(
            residual, jacobian, state_1d, 0.0
        )
        return jacobian

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute Jacobian of residual w.r.t. parameters, dR/dp.

        Boundary rows are zeroed since boundary conditions do not depend
        on the PDE parameters.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        pjac = self._physics.param_jacobian(state_1d, 0.0)
        return self._zero_bc_rows(pjac)


class SteadyForwardModel(Generic[Array]):
    """Steady-state parameterized PDE forward model.

    Maps PDE parameters to quantities of interest extracted from the
    steady-state solution. Satisfies FunctionProtocol with adjoint-based
    Jacobian computation.

    Parameters
    ----------
    physics : object
        Parameterized collocation physics (must have set_param, param_jacobian,
        nparams, and satisfy PhysicsProtocol).
    bkd : Backend
        Computational backend.
    init_state : Array
        Initial guess for steady-state Newton solver. Shape: (nstates,).
    functional : ParameterizedFunctionalWithJacobianProtocol, optional
        QoI functional. If None, uses SubsetOfStatesAdjointFunctional with
        all indices (nqoi = nstates, identity functional).
    """

    def __init__(
        self,
        physics,
        bkd: Backend[Array],
        init_state: Array,
        functional: Optional[
            ParameterizedFunctionalWithJacobianProtocol[Array]
        ] = None,
    ):
        self._bkd = bkd
        self._physics = physics
        self._init_state_1d = init_state
        self._init_state_2d = init_state[:, None]

        model = CollocationModel(physics, bkd)
        self._state_eq = CollocationStateEquationAdapter(model, bkd)

        nstates = self._state_eq.nstates()
        nparams = self._state_eq.nparams()

        if functional is None:
            functional = SubsetOfStatesAdjointFunctional(
                nstates, nparams, bkd.arange(nstates), bkd
            )
        self._functional = functional

        if functional.nqoi() == 1:
            self._adjoint_op = AdjointOperatorWithJacobian(
                self._state_eq, functional
            )
        else:
            self._adjoint_op = VectorAdjointOperatorWithJacobian(
                self._state_eq, functional
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._state_eq.nparams()

    def nqoi(self) -> int:
        """Return number of output quantities of interest."""
        return self._functional.nqoi()

    def state_equation(self) -> CollocationStateEquationAdapter[Array]:
        """Return the state equation adapter."""
        return self._state_eq

    def adjoint_operator(self):
        """Return the adjoint operator."""
        return self._adjoint_op

    def __call__(self, samples: Array) -> Array:
        """Evaluate forward model for multiple parameter samples.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nvars, nsamples).

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, nsamples).
        """
        bkd = self._bkd
        nsamples = samples.shape[1]
        nqoi = self.nqoi()
        result = bkd.zeros((nqoi, nsamples))
        for ii in range(nsamples):
            param = samples[:, ii:ii+1]
            qoi = self._adjoint_op(self._init_state_2d, param)
            if qoi.ndim == 2:
                result[:, ii:ii+1] = qoi
            else:
                result[:, ii] = qoi
        return result

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of QoI w.r.t. parameters using adjoint method.

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars).
        """
        return self._adjoint_op.jacobian(self._init_state_2d, sample)
