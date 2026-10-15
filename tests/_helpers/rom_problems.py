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
from typing import Any, Callable, List

import numpy as np
from numpy.typing import NDArray

from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.pde.boundary import ConstraintSetProtocol
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
from pyapprox.pde.models.galerkin import GalerkinParameterizedResidual
from pyapprox.pde.models.galerkin.steady import (
    GalerkinParameterizedStateEquation,
)
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


RomProblemFactory = Callable[[NumpyBkd], RomProblem]
ALL_ROM_FACTORIES: List[RomProblemFactory] = [
    galerkin_adr_linear,
    galerkin_adr_nonlinear_state,
    galerkin_adr_nonlinear_params,
]
AFFINE_ROM_FACTORIES: List[RomProblemFactory] = [galerkin_adr_linear]
