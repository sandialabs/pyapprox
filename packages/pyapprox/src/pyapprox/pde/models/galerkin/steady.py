"""Steady-state state-equation adapter for parameterized galerkin PDEs.

Provides GalerkinStateEquationAdapter, which wraps a steady view
(``compose_galerkin_system(physics, bcs).steady()``) + parameterization as
ParameterizedStateEquationWithJacobianProtocol for the steady
AdjointOperator, whose linear solves are sparse-aware.
"""

from typing import Generic, Optional

from pyapprox.optimization.implicitfunction.state_equations.derivatives import (
    SteadyStateEquationDerivatives,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    SteadyParamHVPFn,
    SteadyParamJacobianFn,
    SteadyStateStateHVPFn,
    steady_constrained_derivatives,
)
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.pde.steady_view import SteadyViewProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinStateEquationAdapter(Generic[Array]):
    """Adapts galerkin physics + parameterization for the steady
    adjoint operator.

    Implements ParameterizedStateEquationWithJacobianProtocol
    with the 2D column convention (state (nstates, 1), param
    (nparams, 1)); the physics side uses 1D arrays. Jacobians are
    returned in the physics' native form (sparse at the skfem seam) —
    the steady adjoint operator solves them sparse-aware. The
    ``derivatives()`` bundle carries the four adjoint-weighted HVPs when
    the spatial operator supplies state curvature and the
    parameterization's bundle carries all three parameter-facing HVPs;
    otherwise it is empty.

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
    view : SteadyViewProtocol
        The steady problem (``compose_galerkin_system(physics, bcs).steady()``, or a
        ``steady_snapshot``): the constrained residual and Jacobian, the
        constraint set, the bound time at which every derivative is
        evaluated, and a spatial operator that supplies the state
        curvature (exact zeros for linear physics).
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients. Its
        ParamDerivatives bundle must provide param_jacobian.
    bkd : Backend
        Computational backend.
    solver : SteadyStateSolver, optional
        Forward solver. Defaults to Newton with tol=1e-12.
    """

    def __init__(
        self,
        view: SteadyViewProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        bkd: Backend[Array],
        solver: Optional[SteadyStateSolver[Array]] = None,
    ) -> None:
        if not isinstance(view, SteadyViewProtocol):
            raise TypeError(
                "view must satisfy SteadyViewProtocol (build one with "
                "compose_galerkin_system(physics, bcs).steady()), got "
                f"{type(view).__name__}"
            )
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                "parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        require_owned_targets(parameterization, view)
        derivs = steady_constrained_derivatives(
            view, parameterization.param_derivatives()
        )
        if derivs.param_jacobian is None:
            raise TypeError(
                "parameterization must provide param_jacobian in its "
                "ParamDerivatives bundle"
            )
        self._view = view
        self._parameterization = parameterization
        self._bkd = bkd
        self._param_jacobian_fn: SteadyParamJacobianFn[Array] = (
            derivs.param_jacobian
        )
        # Capability is fixed at construction: the narrowed second-order
        # callables are captured once, and None when any is absent.
        self._state_state_hvp_fn: Optional[SteadyStateStateHVPFn[Array]] = (
            None
        )
        self._param_param_hvp_fn: Optional[SteadyParamHVPFn[Array]] = None
        self._state_param_hvp_fn: Optional[SteadyParamHVPFn[Array]] = None
        self._param_state_hvp_fn: Optional[SteadyParamHVPFn[Array]] = None
        self._derivs: SteadyStateEquationDerivatives[Array] = (
            SteadyStateEquationDerivatives.none()
        )
        if (
            derivs.state_state_hvp is not None
            and derivs.param_param_hvp is not None
            and derivs.state_param_hvp is not None
            and derivs.param_state_hvp is not None
        ):
            self._state_state_hvp_fn = derivs.state_state_hvp
            self._param_param_hvp_fn = derivs.param_param_hvp
            self._state_param_hvp_fn = derivs.state_param_hvp
            self._param_state_hvp_fn = derivs.param_state_hvp
            self._derivs = SteadyStateEquationDerivatives.second_order(
                self._state_state_hvp,
                self._state_param_hvp,
                self._param_state_hvp,
                self._param_param_hvp,
            )
        if solver is None:
            solver = SteadyStateSolver(view, tol=1e-12)
        self._solver = solver

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the number of state variables."""
        return self._view.nstates()

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def _set_param(self, param: Array) -> None:
        """Apply the parameter column through the parameterization."""
        self._parameterization.apply(param[:, 0])

    def derivatives(self) -> SteadyStateEquationDerivatives[Array]:
        """Return the optional derivatives this state equation provides."""
        return self._derivs

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
        return self._view.steady_residual(state[:, 0])[:, None]

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dR/du with Dirichlet identity rows.

        Returned in the physics' native form (sparse at the skfem
        seam). Shape: (nstates, nstates).
        """
        self._set_param(param)
        return self._view.steady_jacobian(state[:, 0])

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dR/dp with constrained rows zeroed.

        The parameterization returns the RAW parameter Jacobian;
        essential constraints are parameter-independent, so their rows
        vanish. Shape: (nstates, nparams).
        """
        self._set_param(param)
        return self._param_jacobian_fn(state[:, 0], param[:, 0])

    @staticmethod
    def _unavailable() -> RuntimeError:
        return RuntimeError(
            "the HVPs are unavailable; check derivatives() before calling"
        )

    def _state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/du^2) w with the constrained adjoint zeroed.

        Shape: (nstates, 1).
        """
        state_state_hvp_fn = self._state_state_hvp_fn
        if state_state_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return state_state_hvp_fn(
            state[:, 0], adj_state[:, 0], wvec[:, 0]
        )[:, None]

    def _param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/dp^2) v with the constrained adjoint zeroed.

        Shape: (nparams, 1).
        """
        param_param_hvp_fn = self._param_param_hvp_fn
        if param_param_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return param_param_hvp_fn(
            state[:, 0], param[:, 0], adj_state[:, 0], vvec[:, 0]
        )[:, None]

    def _state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/du dp) v with the constrained adjoint zeroed.

        Shape: (nstates, 1).
        """
        state_param_hvp_fn = self._state_param_hvp_fn
        if state_param_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return state_param_hvp_fn(
            state[:, 0], param[:, 0], adj_state[:, 0], vvec[:, 0]
        )[:, None]

    def _param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/dp du) w with the constrained adjoint zeroed.

        Shape: (nparams, 1).
        """
        param_state_hvp_fn = self._param_state_hvp_fn
        if param_state_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return param_state_hvp_fn(
            state[:, 0], param[:, 0], adj_state[:, 0], wvec[:, 0]
        )[:, None]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"view={self._view!r}, "
            f"parameterization="
            f"{type(self._parameterization).__name__}, "
            f"nstates={self.nstates()}, nparams={self.nparams()})"
        )
