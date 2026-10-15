"""Problem factories for the physics-agnostic projection-ROM tests.

A reduced model's guarantees depend only on the discrete system, so the
same tests run against every factory here (the fixtures live in
``packages/pyapprox/tests/pde/rom/conftest.py``). Adding a physics, or a
discretization, to the suite means adding one factory that returns a
:class:`RomProblem`. Each factory states what it expects: whether the
residual is affine in state and parameters (so the affine fit applies)
and whether it is coercive (so Galerkin quasi-optimality applies).

Requires skfem; import only after skipping when it is absent.
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.pde.boundary import ConstraintSetProtocol, DirichletConstraintSet
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.models.galerkin import GalerkinParameterizedResidual
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.rom import (
    AlgebraicLift,
    ComposedParameterizedResidualProtocol,
    LiftProtocol,
    SnapshotSet,
    collect_steady_snapshots,
    homogenize,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    InnerProductProtocol,
    MassInnerProduct,
)
from tests._helpers.adjoint_checks import NumpyArray

_Points = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class RomProblem:
    """Everything a physics-agnostic ROM test needs, and what to expect.

    ``affine``: the residual is affine in the state and the parameters,
    so the affine fit must succeed and be exact. ``coercive``: Galerkin
    quasi-optimality holds, so the ROM error is a bounded multiple of the
    best approximation error.
    """

    name: str
    bkd: NumpyBkd
    residual: ComposedParameterizedResidualProtocol[NumpyArray]
    state_equation: SteadyStateEquationProtocol[NumpyArray]
    constraint_set: ConstraintSetProtocol[NumpyArray]
    lift: LiftProtocol[NumpyArray]
    metric: InnerProductProtocol[NumpyArray]
    train: NumpyArray
    held_out: NumpyArray
    reference_params: NumpyArray
    affine: bool
    coercive: bool

    def nstates(self) -> int:
        return self.residual.nstates()

    def solve(self, samples: NumpyArray) -> SnapshotSet[NumpyArray]:
        """Full-order steady solutions at each column of ``samples``."""
        return collect_steady_snapshots(
            self.state_equation, samples, self.bkd.zeros((self.nstates(), 1))
        )

    def homogenized(self, samples: NumpyArray) -> SnapshotSet[NumpyArray]:
        """Full-order solutions with the lift removed."""
        return homogenize(self.solve(samples), self.lift, self.constraint_set)


def _square(x: _Points, u: _Points) -> _Points:
    return u**2


def _square_deriv(x: _Points, u: _Points) -> _Points:
    return 2.0 * u


def _square_deriv2(x: _Points, u: _Points) -> _Points:
    return np.full_like(u, 2.0)


def _galerkin_adr(
    bkd: NumpyBkd, name: str, reaction: bool, exp_diffusivity: bool = False
) -> RomProblem:
    """1D ADR, -(kappa u')' [+ u^2] = f, u(0) = 1, u(1) = 2, P1 on 10 cells.

    kappa = 1 + Phi p_kappa (or exp(Phi p_kappa)) and f = 1 + Phi p_f with
    Phi the modes 0.2 sin(k pi x), k = 1..3: six parameters, kappa >= 0.4
    on [-1, 1]^6. The u^2 reaction makes F nonlinear in u; the
    exponential kappa makes it nonlinear in p while staying linear and
    coercive in u. Twelve training and four held-out samples, uniform on
    [-1, 1]^6 (seed 0); reference parameter zero; mass metric.
    """
    mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis),
        bkd=bkd,
        forcing=NodalFieldForcing(basis),
        reaction=(
            CallableReaction(_square, _square_deriv, _square_deriv2)
            if reaction
            else None
        ),
    )
    system = compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 1.0, bkd),
            DirichletBC(basis, "right", 2.0, bkd),
        ],
    )
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = bkd.asarray(
        np.stack([0.2 * np.sin((k + 1) * np.pi * coords) for k in range(3)], axis=1)
    )
    ones = bkd.asarray(np.ones(coords.shape[0]))
    diffusivity_map: FieldMapProtocol[NumpyArray] = MeshKLEFieldMap(bkd, ones, modes)
    if exp_diffusivity:
        exp = _ExpTransform(bkd)
        diffusivity_map = TransformedFieldMap(
            MeshKLEFieldMap(bkd, 0.0 * ones, modes),
            exp,
            exp,
            bkd,
            transform_deriv2=exp,
        )
    param_obj = AdvectionDiffusionParameterization(
        physics,
        diffusivity_map=diffusivity_map,
        forcing_map=MeshKLEFieldMap(bkd, ones, modes),
        bkd=bkd,
    )
    rng = np.random.default_rng(0)
    nparams = param_obj.nparams()
    constraint_set = system.constraint_set()
    return RomProblem(
        name=name,
        bkd=bkd,
        residual=GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        state_equation=GalerkinParameterizedStateEquation(
            system.steady(), param_obj, bkd
        ),
        constraint_set=constraint_set,
        lift=AlgebraicLift(constraint_set, system.nstates()),
        metric=MassInnerProduct(physics.mass_matrix(), bkd),
        train=bkd.asarray(rng.uniform(-1.0, 1.0, size=(nparams, 12))),
        held_out=bkd.asarray(rng.uniform(-1.0, 1.0, size=(nparams, 4))),
        reference_params=bkd.zeros((nparams,)),
        affine=not (reaction or exp_diffusivity),
        coercive=not reaction,
    )


def galerkin_adr_linear(bkd: NumpyBkd) -> RomProblem:
    """Affine in state and parameters, coercive."""
    return _galerkin_adr(bkd, "galerkin-adr-linear", reaction=False)


def galerkin_adr_nonlinear_state(bkd: NumpyBkd) -> RomProblem:
    """Nonlinear in the state (u^2 reaction): exact projection only."""
    return _galerkin_adr(bkd, "galerkin-adr-u2-reaction", reaction=True)


def galerkin_adr_nonlinear_params(bkd: NumpyBkd) -> RomProblem:
    """Nonlinear in the parameters (exp kappa), linear and coercive in u."""
    return _galerkin_adr(
        bkd, "galerkin-adr-exp-kappa", reaction=False, exp_diffusivity=True
    )


class _FixedValue:
    """An essential condition fixing one DOF to a constant."""

    def __init__(self, dof: int, value: float, bkd: NumpyBkd) -> None:
        self._dofs = bkd.asarray([dof], dtype=bkd.int64_dtype())
        self._values = bkd.asarray([value])
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def constrained_dofs(self) -> NumpyArray:
        return self._dofs

    def constrained_values(self, time: float) -> NumpyArray:
        return self._values

    def is_time_invariant(self) -> bool:
        return True

    def constrained_values_derivative(
        self, order: int
    ) -> Callable[[float], NumpyArray]:
        zeros = self._bkd.zeros((1,))
        return lambda time: zeros


class _FiniteDifferenceADR:
    r"""Second-order finite differences for (kappa u')' [- u^2] + f = 0.

    Satisfies ``SpatialOperatorProtocol``. ``kappa`` lives at the
    ``n - 1`` cell midpoints and ``f`` at the ``n`` nodes. Interior row
    ``i`` is :math:`(\kappa_i (u_{i+1} - u_i) - \kappa_{i-1} (u_i -
    u_{i-1}))/h^2 + f_i [- u_i^2]`; the two end rows are :math:`u_0` and
    :math:`u_{n-1}`, deliberately nonzero at a solution, because a
    reduced model must never read rows the constraints replace.
    """

    def __init__(self, nnodes: int, reaction: bool, bkd: NumpyBkd) -> None:
        self._n = nnodes
        self._h = 1.0 / (nnodes - 1)
        self._reaction = reaction
        self._bkd = bkd
        self._kappa = bkd.ones((nnodes - 1,))
        self._forcing = bkd.ones((nnodes,))

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nstates(self) -> int:
        return self._n

    def set_coefficients(self, kappa: NumpyArray, forcing: NumpyArray) -> None:
        self._kappa = kappa
        self._forcing = forcing

    def spatial_residual(self, state: NumpyArray, time: float) -> NumpyArray:
        flux = self._kappa * (state[1:] - state[:-1]) / self._h
        interior = (flux[1:] - flux[:-1]) / self._h + self._forcing[1:-1]
        if self._reaction:
            interior = interior - state[1:-1] ** 2
        return self._bkd.concatenate([state[:1], interior, state[-1:]])

    def spatial_jacobian(self, state: NumpyArray, time: float) -> NumpyArray:
        n, h2 = self._n, self._h**2
        jac = np.zeros((n, n))
        jac[0, 0] = jac[-1, -1] = 1.0
        kappa = self._bkd.to_numpy(self._kappa)
        for ii in range(1, n - 1):
            jac[ii, ii - 1] = kappa[ii - 1] / h2
            jac[ii, ii] = -(kappa[ii - 1] + kappa[ii]) / h2
            jac[ii, ii + 1] = kappa[ii] / h2
            if self._reaction:
                jac[ii, ii] -= 2.0 * float(self._bkd.to_numpy(state)[ii])
        return self._bkd.asarray(jac)

    def state_derivatives(self) -> StateDerivatives[NumpyArray]:
        # Linear without the reaction; the reaction's curvature is not
        # supplied, which only rules out second-order adjoints.
        if self._reaction:
            return StateDerivatives.none()
        return StateDerivatives.linear(self._bkd)

    def is_time_invariant(self) -> bool:
        return True

    def diffusivity_jacobian(self, state: NumpyArray) -> NumpyArray:
        """dF/dkappa, shape (n, n - 1)."""
        n, h2 = self._n, self._h**2
        u = self._bkd.to_numpy(state)
        jac = np.zeros((n, n - 1))
        for ii in range(1, n - 1):
            jac[ii, ii] = (u[ii + 1] - u[ii]) / h2
            jac[ii, ii - 1] = -(u[ii] - u[ii - 1]) / h2
        return self._bkd.asarray(jac)


class _FiniteDifferenceParameterization:
    """kappa = 1 + Phi_mid p[:3], f = 1 + Phi_nodes p[3:]; affine in p."""

    def __init__(self, operator: _FiniteDifferenceADR, bkd: NumpyBkd) -> None:
        n = operator.nstates()
        nodes = np.linspace(0.0, 1.0, n)
        mids = 0.5 * (nodes[1:] + nodes[:-1])
        self._mid_modes = bkd.asarray(
            np.stack([0.2 * np.sin((k + 1) * np.pi * mids) for k in range(3)], axis=1)
        )
        self._node_modes = bkd.asarray(
            np.stack([0.2 * np.sin((k + 1) * np.pi * nodes) for k in range(3)], axis=1)
        )
        self._operator = operator
        self._bkd = bkd

    def nparams(self) -> int:
        return 6

    def targets(self) -> Tuple[object, ...]:
        return (self._operator,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("diffusivity", "forcing")

    def apply(self, params_1d: NumpyArray) -> None:
        bkd = self._bkd
        self._operator.set_coefficients(
            1.0 + bkd.dot(self._mid_modes, params_1d[:3]),
            1.0 + bkd.dot(self._node_modes, params_1d[3:]),
        )

    def is_affine_in_params(self) -> bool:
        return True

    def param_derivatives(self) -> ParamDerivatives[NumpyArray]:
        return ParamDerivatives.first_order(
            self._param_jacobian, self._initial_param_jacobian
        )

    def _param_jacobian(
        self, state: NumpyArray, time: float, params_1d: NumpyArray
    ) -> NumpyArray:
        bkd = self._bkd
        kappa_part = bkd.dot(
            self._operator.diffusivity_jacobian(state), self._mid_modes
        )
        forcing_part = self._node_modes * 1.0
        # The end rows are constraint rows and depend on no parameter.
        forcing_part = bkd.concatenate(
            [bkd.zeros((1, 3)), forcing_part[1:-1], bkd.zeros((1, 3))]
        )
        return bkd.hstack([kappa_part, forcing_part])

    def _initial_param_jacobian(self, params_1d: NumpyArray) -> NumpyArray:
        return self._bkd.zeros((self._operator.nstates(), 6))


def _finite_difference_adr(bkd: NumpyBkd, name: str, reaction: bool) -> RomProblem:
    """The ADR problem by finite differences: no Galerkin code at all.

    11 nodes on [0, 1], u(0) = 1, u(1) = 2, the same parameters and
    samples as the Galerkin factories, trapezoid-rule diagonal metric.
    Built only from discretization-neutral pieces: a
    ``SpatialOperatorProtocol``, ``DirichletConstraintSet`` over
    ``EssentialBCProtocol`` objects, a parameterization targeting the
    operator, and the generic steady system and state equation.
    """
    nnodes = 11
    operator = _FiniteDifferenceADR(nnodes, reaction, bkd)
    param_obj = _FiniteDifferenceParameterization(operator, bkd)
    constraint_set = DirichletConstraintSet(
        [_FixedValue(0, 1.0, bkd), _FixedValue(nnodes - 1, 2.0, bkd)],
        nnodes,
        bkd,
    )
    system = GalerkinSteadySystem(operator, constraint_set)
    weights = np.full(nnodes, 1.0 / (nnodes - 1))
    weights[[0, -1]] *= 0.5
    rng = np.random.default_rng(0)
    return RomProblem(
        name=name,
        bkd=bkd,
        residual=GalerkinParameterizedResidual(operator, param_obj),
        state_equation=GalerkinParameterizedStateEquation(
            system.steady(), param_obj, bkd
        ),
        constraint_set=constraint_set,
        lift=AlgebraicLift(constraint_set, nnodes),
        metric=DiagonalInnerProduct(bkd.asarray(weights), bkd),
        train=bkd.asarray(rng.uniform(-1.0, 1.0, size=(6, 12))),
        held_out=bkd.asarray(rng.uniform(-1.0, 1.0, size=(6, 4))),
        reference_params=bkd.zeros((6,)),
        affine=not reaction,
        coercive=not reaction,
    )


def finite_difference_adr_linear(bkd: NumpyBkd) -> RomProblem:
    """Affine and coercive, with no Galerkin assembly."""
    return _finite_difference_adr(bkd, "finite-difference-adr-linear", False)


def finite_difference_adr_nonlinear_state(bkd: NumpyBkd) -> RomProblem:
    """Nonlinear in the state (u^2 reaction), with no Galerkin assembly."""
    return _finite_difference_adr(bkd, "finite-difference-adr-u2-reaction", True)


RomProblemFactory = Callable[[NumpyBkd], RomProblem]
ALL_ROM_FACTORIES: List[RomProblemFactory] = [
    galerkin_adr_linear,
    galerkin_adr_nonlinear_state,
    galerkin_adr_nonlinear_params,
    finite_difference_adr_linear,
    finite_difference_adr_nonlinear_state,
]
AFFINE_ROM_FACTORIES: List[RomProblemFactory] = [
    galerkin_adr_linear,
    finite_difference_adr_linear,
]
