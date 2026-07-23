"""Steady-state state-equation adapter for parameterized galerkin PDEs.

Provides GalerkinStateEquationWithHVPAdapter, which wraps a galerkin physics +
parameterization as ParameterizedStateEquationWithJacobianAndHVPProtocol
for the steady adjoint operator family
(AdjointOperatorWithJacobian/AdjointOperatorWithJacobianAndHVP, whose
linear solves are sparse-aware).
"""

from typing import Generic, Optional

from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsWithStateStateHVPProtocol,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.parameterizations.derivatives import (
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinStateEquationWithHVPAdapter(Generic[Array]):
    """Adapts galerkin physics + parameterization for the steady
    adjoint operators.

    Implements ParameterizedStateEquationWithJacobianAndHVPProtocol
    with the 2D column convention (state (nstates, 1), param
    (nparams, 1)); the physics side uses 1D arrays. Jacobians are
    returned in the physics' native form (sparse at the skfem seam) —
    the steady adjoint operators solve them sparse-aware.

    All Dirichlet handling is owned here (parameterizations return RAW
    dR/dp and HVP contractions): parameter-Jacobian rows at constrained
    DOFs are zeroed, and every second-derivative contraction receives
    the adjoint with constrained entries zeroed. That is the whole
    correction (true-tensor convention): constraint rows are linear in
    (y, p) so their true second derivatives vanish, while state-shaped
    outputs are NOT masked — their entries at constrained indices are
    genuine, and the second-adjoint component they feed is decoupled
    from the HVP (identity Jacobian rows; dR/dp rows zeroed).

    Parameters
    ----------
    physics : GalerkinPhysicsWithStateStateHVPProtocol
        Galerkin physics providing BC-applied ``residual``/``jacobian``,
        ``constraint_set``, and the raw ``state_state_hvp`` contraction
        (exact zeros for linear physics).
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients. Its
        ParamDerivatives bundle must be second order (param_jacobian
        plus all three parameter-facing HVPs) — this adapter is the
        HVP tier.
    bkd : Backend
        Computational backend.
    solver : SteadyStateSolver, optional
        Forward solver. Defaults to Newton with tol=1e-12.
    """

    def __init__(
        self,
        physics: GalerkinPhysicsWithStateStateHVPProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        bkd: Backend[Array],
        solver: Optional[SteadyStateSolver[Array]] = None,
    ) -> None:
        if not isinstance(
            physics, GalerkinPhysicsWithStateStateHVPProtocol
        ):
            raise TypeError(
                "physics must satisfy "
                "GalerkinPhysicsWithStateStateHVPProtocol, got "
                f"{type(physics).__name__}"
            )
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                "parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        derivs = parameterization.param_derivatives()
        if (
            derivs.param_jacobian is None
            or derivs.param_param_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
        ):
            raise TypeError(
                "parameterization must provide a second-order "
                "ParamDerivatives bundle (param_jacobian and all three "
                "parameter-facing HVPs); this adapter is the HVP tier"
            )
        self._physics = physics
        self._parameterization = parameterization
        self._bkd = bkd
        self._constraint_set = physics.constraint_set()
        self._param_jacobian_fn: ParamJacobianFn[Array] = (
            derivs.param_jacobian
        )
        self._param_param_hvp_fn: ParamHVPFn[Array] = derivs.param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = derivs.state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = derivs.param_state_hvp
        if solver is None:
            solver = SteadyStateSolver(physics, tol=1e-12)
        self._solver = solver

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._physics.nstates()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def _set_param(self, param: Array) -> None:
        """Apply the parameter column through the parameterization."""
        self._parameterization.apply(param[:, 0])

    def _zeroed_adjoint(self, adj_state: Array) -> Array:
        """Adjoint column as 1D with constrained entries zeroed."""
        return self._constraint_set.zero_entries(adj_state[:, 0])

    def solve(self, init_state: Array, param: Array) -> Array:
        """Solve R(u, p) = 0 for u.

        Parameters
        ----------
        init_state : Array
            Initial Newton guess. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Solution state. Shape: (nstates, 1).
        """
        self._set_param(param)
        result = self._solver.solve(init_state[:, 0])
        if not result.converged:
            raise RuntimeError(
                "steady-state solve did not converge: "
                f"{result.message} (residual_norm="
                f"{result.residual_norm:.3e})"
            )
        return result.solution[:, None]

    def __call__(self, state: Array, param: Array) -> Array:
        """Compute the BC-applied residual R(u, p). Shape: (nstates, 1)."""
        self._set_param(param)
        return self._physics.residual(state[:, 0], 0.0)[:, None]

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dR/du with Dirichlet identity rows.

        Returned in the physics' native form (sparse at the skfem
        seam). Shape: (nstates, nstates).
        """
        self._set_param(param)
        return self._physics.jacobian(state[:, 0], 0.0)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dR/dp with constrained rows zeroed.

        The parameterization returns the RAW parameter Jacobian;
        essential constraints are parameter-independent, so their rows
        vanish. Shape: (nstates, nparams).
        """
        self._set_param(param)
        raw = self._param_jacobian_fn(state[:, 0], 0.0, param[:, 0])
        return self._constraint_set.zero_rows(raw)

    def state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/du^2) w with the constrained adjoint zeroed.

        Shape: (nstates, 1).
        """
        self._set_param(param)
        return self._physics.state_state_hvp(
            state[:, 0], self._zeroed_adjoint(adj_state), wvec[:, 0], 0.0
        )[:, None]

    def param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/dp^2) v with the constrained adjoint zeroed.

        Shape: (nparams, 1).
        """
        self._set_param(param)
        return self._param_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/du dp) v with the constrained adjoint zeroed.

        Shape: (nstates, 1).
        """
        self._set_param(param)
        return self._state_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/dp du) w with the constrained adjoint zeroed.

        Shape: (nparams, 1).
        """
        self._set_param(param)
        return self._param_state_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            wvec[:, 0],
        )[:, None]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={type(self._physics).__name__}, "
            f"parameterization="
            f"{type(self._parameterization).__name__}, "
            f"nstates={self.nstates()}, nparams={self.nparams()})"
        )
