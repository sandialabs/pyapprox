"""Steady-state models for parameterized galerkin PDEs.

Provides GalerkinParameterizedStateEquation, which wraps a steady view
(``compose_galerkin_system(physics, bcs).steady()``) + parameterization as
SteadyStateEquationProtocol for the steady
AdjointOperator, whose linear solves are sparse-aware; and
GalerkinSteadyForwardModel, the FunctionProtocol forward model built on
it.
"""

import warnings
from typing import Generic, Optional, Union

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (  # noqa: E501
    SubsetOfStatesFunctional,
)
from pyapprox.optimization.implicitfunction.operator.adjoint_operator import (
    AdjointOperator,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    OperatorHVPFn,
)
from pyapprox.optimization.implicitfunction.operator.sensitivities import (
    ForwardSensitivityOperator,
)
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


class GalerkinParameterizedStateEquation(Generic[Array]):
    """Adapts galerkin physics + parameterization for the steady
    adjoint operator.

    Implements SteadyStateEquationProtocol
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


class GalerkinSteadyForwardModel(Generic[Array]):
    """Steady-state parameterized galerkin PDE forward model.

    Satisfies ``FunctionProtocol``: ``__call__`` maps parameter samples
    to QoI values by a steady solve per sample. ``derivatives()`` is
    decided once at construction from the bundles of the parameterization,
    the functional and the state equation:

    - none when the parameterization has no parameter Jacobian or the
      functional has no Jacobians (the model is evaluated only);
    - the adjoint Jacobian for a scalar QoI, plus the second-order adjoint
      HVP when the state equation and the functional both carry their
      second derivatives (a downgrade caused only by the physics' missing
      state curvature warns);
    - the forward-sensitivity Jacobian for a vector QoI.

    Parameters
    ----------
    view : SteadyViewProtocol
        The steady problem: ``system.steady()``, or
        ``system.steady_snapshot(time)`` for time-dependent data.
    parameterization : ParameterizationProtocol
        Maps parameter vectors to coefficients; its targets must belong to
        ``view``.
    init_state : Array
        Initial Newton guess. Shape: (nstates,).
    bkd : Backend
        Computational backend.
    functional : SteadyFunctionalProtocol, optional
        QoI functional. Defaults to every state
        (``SubsetOfStatesFunctional`` over all indices).
    solver : SteadyStateSolver, optional
        Forward solver; defaults to the state equation's.
    """

    def __init__(
        self,
        view: SteadyViewProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        init_state: Array,
        bkd: Backend[Array],
        functional: Optional[SteadyFunctionalProtocol[Array]] = None,
        solver: Optional[SteadyStateSolver[Array]] = None,
    ) -> None:
        # The state equation checks the view, the parameterization and
        # that the parameterization's targets belong to the view.
        self._state_eq = GalerkinParameterizedStateEquation(
            view, parameterization, bkd, solver=solver
        )
        self._bkd = bkd
        nstates = self._state_eq.nstates()
        if init_state.shape != (nstates,):
            raise ValueError(
                f"init_state must have shape ({nstates},), got "
                f"{init_state.shape}"
            )
        self._init_state = init_state[:, None]
        if functional is None:
            functional = SubsetOfStatesFunctional(
                nstates, self._state_eq.nparams(), bkd.arange(nstates), bkd
            )
        if not isinstance(functional, SteadyFunctionalProtocol):
            raise TypeError(
                "functional must satisfy SteadyFunctionalProtocol, got "
                f"{type(functional).__name__}"
            )
        # A transient functional has every member of the steady protocol,
        # so isinstance cannot reject it; the declared time dependence
        # does.
        if functional.time_dependence() != "steady":
            raise TypeError(
                f"{type(functional).__name__} is a "
                f"{functional.time_dependence()} functional; a steady model "
                "needs one that reads a single state"
            )
        self._functional = functional

        self._jacobian_op: Optional[
            Union[AdjointOperator[Array], ForwardSensitivityOperator[Array]]
        ] = None
        self._operator_hvp: Optional[OperatorHVPFn[Array]] = None
        self._derivs: Derivatives[Array] = Derivatives.none()
        if (
            parameterization.param_derivatives().param_jacobian is None
            or functional.derivatives().state_jacobian is None
        ):
            return
        if functional.nqoi() != 1:
            self._jacobian_op = ForwardSensitivityOperator(
                self._state_eq, functional
            )
            self._derivs = Derivatives.first_order(jacobian=self._jacobian)
            return
        adjoint_op = AdjointOperator(self._state_eq, functional)
        self._jacobian_op = adjoint_op
        operator_hvp = adjoint_op.derivatives().hvp
        if operator_hvp is not None:
            self._operator_hvp = operator_hvp
            self._derivs = Derivatives.second_order(
                jacobian=self._jacobian, hvp=self._hvp
            )
            return
        param_derivs = parameterization.param_derivatives()
        if (
            param_derivs.param_param_hvp is not None
            and param_derivs.state_param_hvp is not None
            and param_derivs.param_state_hvp is not None
            and functional.derivatives().state_state_hvp is not None
        ):
            # Everything else is second order, so the physics' missing
            # state curvature is what costs the HVP; say so rather than
            # downgrade silently.
            warnings.warn(
                "second-order parameterization and functional downgraded "
                "to the Jacobian tier: the spatial operator supplies no "
                "state curvature (state_derivatives().state_state_hvp is "
                "None)",
                UserWarning,
                stacklevel=2,
            )
        self._derivs = Derivatives.first_order(jacobian=self._jacobian)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables (parameters)."""
        return self._state_eq.nparams()

    def nqoi(self) -> int:
        """Return the number of output quantities of interest."""
        return self._functional.nqoi()

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle (with respect to parameters)."""
        return self._derivs

    def state_equation(self) -> GalerkinParameterizedStateEquation[Array]:
        """Return the state equation."""
        return self._state_eq

    def functional(self) -> SteadyFunctionalProtocol[Array]:
        """Return the QoI functional."""
        return self._functional

    def __call__(self, samples: Array) -> Array:
        """Evaluate the QoIs at each parameter sample.

        Parameters
        ----------
        samples : Array
            Shape: (nvars, nsamples).

        Returns
        -------
        Array
            Shape: (nqoi, nsamples).
        """
        validate_samples(self.nvars(), samples)
        values = []
        for ii in range(samples.shape[1]):
            param = samples[:, ii : ii + 1]
            state = self._state_eq.solve(self._init_state, param)
            values.append(self._functional(state, param))
        return self._bkd.hstack(values)

    def _jacobian(self, sample: Array) -> Array:
        """dQ/dp at one sample. Shape: (nqoi, nvars)."""
        jacobian_op = self._jacobian_op
        if jacobian_op is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        validate_sample(self.nvars(), sample)
        return jacobian_op.jacobian(self._init_state, sample)

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """(d^2Q/dp^2) v at one sample. Shape: (nvars, 1)."""
        operator_hvp = self._operator_hvp
        if operator_hvp is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        return operator_hvp(self._init_state, sample, vec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"functional={type(self._functional).__name__})"
        )
