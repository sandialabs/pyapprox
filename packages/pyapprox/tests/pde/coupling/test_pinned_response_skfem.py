r"""Mesh subdomains on skfem, alone and mixed with the closed-form P1
operator, coupled through the skeleton. NumPy only: skfem is.

The problem is :math:`-u'' [+ u^2] = 1` on [0, 2], u(0) = 1, u(2) = 2,
split at x = 1 into two pieces of 8 uniform P1 cells. Each skeleton is
solved by Newton (atol 1e-11, matrix-free Schur, GMRES tol 1e-13).

- skfem on both pieces, with and without the reaction, against one
  skfem solve on [0, 2] with 16 cells: the same element integrals and
  quadrature, so the same discrete equations. The interface value and
  both pieces' states equal the single-domain solve to 1e-10, and the
  pieces' interface fluxes balance to 1e-10.
- Mixed: one piece on the closed-form P1 operator (dense backend
  arrays, dense LU), the other on skfem (SciPy sparse, sparse LU),
  meeting only through port blocks, in both orders. Linear, where both
  give the same element equations (the forcing's consistent load and
  the stiffness are exact under either), against the same single-domain
  skfem solve to 1e-10.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.boundary import DirichletConstraintSet
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    ConstantDiffusion,
    TimeIndependent,
)
from pyapprox.pde.coupling.basis import lobatto_lagrange_basis
from pyapprox.pde.coupling.closure_numbering import VertexWeightNumbering
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    PinnedReactionFlux,
    SteadyPinnedPoint,
    SteadyPinnedResponse,
)
from pyapprox.pde.coupling.ports import (
    BackendPortRows,
    MatchingNodalTransfer,
    subdomain_port_dofs,
)
from pyapprox.pde.coupling.responses import LocalResponseProtocol, TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    MatrixFreeSchur,
    SkeletonResidual,
)
from pyapprox.pde.coupling.spaces import ContinuousSkeletonSpace, UniformBases
from pyapprox.pde.coupling.topology import TensorBoxPartition
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSteadySystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.galerkin.time_integration.dense_discretization import (
    DenseSteadyDiscretization,
)
from pyapprox.pde.models.galerkin.steady_residual import (
    galerkin_steady_residual,
    solve_galerkin_steady,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendRowSet,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol, MatrixFree
from pyapprox.util.linalg.operators.krylov import GMRES
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.mock_elementwise import MockP1Operator
from tests._helpers.rom_problem import FixedValue

_Np = NDArray[Any]
_NCELLS = 8
_ENDS = (1.0, 2.0)
_STEADY = TimeWindow.steady()
# A response of either stack, as the skeleton sees it.
_Response = LocalResponseProtocol[_Np]


def _unit(x: _Np) -> _Np:
    return np.ones(x.shape[1])


# The skfem physics adds R(u) as a source (du/dt = (D u')' + R(u) + f),
# so the sink of -u'' + u^2 = 1 is R(u) = -u^2.
def _sink(x: _Np, u: _Np) -> _Np:
    return -(u**2)


def _sink_deriv(x: _Np, u: _Np) -> _Np:
    return -2.0 * u


def _newton() -> Newton[_Np, _Np]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


def _skfem_system(
    bkd: NumpyBkd,
    bounds: tuple[float, float],
    ncells: int,
    reaction: bool,
    essentials: Sequence[tuple[str, float]],
) -> Any:
    mesh = StructuredMesh1D(nx=ncells, bounds=bounds, bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=ConstantDiffusion(1.0),
        bkd=bkd,
        reaction=CallableReaction(_sink, _sink_deriv) if reaction else None,
        forcing=TimeIndependent(_unit),
    )
    conditions = [DirichletBC(basis, side, value, bkd) for side, value in essentials]
    return compose_galerkin_system(physics, conditions), basis


def _skfem_piece(bkd: NumpyBkd, left: bool, reaction: bool) -> _Response:
    """The left piece [0, 1] (u(0) = 1) or the right one [1, 2]
    (u(2) = 2) on skfem; the port is the node at x = 1."""
    bounds = (0.0, 1.0) if left else (1.0, 2.0)
    essential = ("left", _ENDS[0]) if left else ("right", _ENDS[1])
    system, basis = _skfem_system(bkd, bounds, _NCELLS, reaction, [essential])
    disc = skfem_discretization(system)
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    port = int(np.argmin(np.abs(coords - 1.0)))
    states, theta = disc.space(), disc.parameter_space(1)
    # The stack's space has an inner product; its declared type does not.
    assert isinstance(states, InnerProductSpaceProtocol)
    ports: PortPinnedData[_Np, _Np] = PortPinnedData(
        NumpyRowSet(states.descriptor(), [port]),
        BackendPortRows(states.descriptor(), [port], bkd),
        BackendPortRows(theta.descriptor(), [0], bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc)
    return SteadyPinnedResponse(
        residual,
        ports,
        PinnedReactionFlux(residual, ports),
        states,
        RecomputeAlways(ScipySparseLU()),
        _newton(),
        bkd,
    )


def _closed_form_piece(bkd: NumpyBkd, left: bool) -> _Response:
    """The same piece on the closed-form P1 operator, linear."""
    n = _NCELLS + 1
    node, value, port = (0, _ENDS[0], n - 1) if left else (n - 1, _ENDS[1], 0)
    operator: MockP1Operator[_Np] = MockP1Operator(_NCELLS, False, bkd)
    disc: DenseSteadyDiscretization[_Np] = DenseSteadyDiscretization(
        GalerkinSteadySystem(
            operator,
            DirichletConstraintSet([FixedValue(node, value, bkd)], n, bkd),
        )
    )
    states, theta = disc.space(), disc.parameter_space(1)
    ports: PortPinnedData[_Np, _Np] = PortPinnedData(
        BackendRowSet(states.descriptor(), [port], bkd),
        BackendPortRows(states.descriptor(), [port], bkd),
        BackendPortRows(theta.descriptor(), [0], bkd),
        states,
        theta,
    )
    residual = galerkin_steady_residual(disc)
    return SteadyPinnedResponse(
        residual,
        ports,
        PinnedReactionFlux(residual, ports),
        states,
        RecomputeAlways(BackendDenseLU(bkd)),
        _newton(),
        bkd,
    )


def _single_domain(bkd: NumpyBkd, reaction: bool) -> _Np:
    ends = [("left", _ENDS[0]), ("right", _ENDS[1])]
    system, _ = _skfem_system(bkd, (0.0, 2.0), 2 * _NCELLS, reaction, ends)
    _, point = solve_galerkin_steady(
        skfem_discretization(system), RecomputeAlways(ScipySparseLU()), _newton()
    )
    assert point.converged()
    return point.state()


def _solve_skeleton(bkd: NumpyBkd, pieces: Sequence[_Response]) -> _Np:
    topo = TensorBoxPartition((2,))
    bases = UniformBases(lobatto_lagrange_basis(0, 1, bkd))
    numbering = VertexWeightNumbering(topo, bases)
    skeleton = ContinuousSkeletonSpace(topo, bases, numbering)
    space = BackendVectorSpace(skeleton.descriptor(), bkd)
    subdomains = [
        CoupledSubdomain(
            piece,
            MatchingNodalTransfer(
                BackendPortRows(
                    skeleton.descriptor(), subdomain_port_dofs(topo, skeleton, s), bkd
                )
            ),
        )
        for s, piece in enumerate(pieces)
    ]
    schur: MatrixFreeSchur[_Np, _Np] = MatrixFreeSchur()
    residual = SkeletonResidual(space, subdomains, schur)
    setups: RecomputeAlways[_Np, MatrixFree] = RecomputeAlways(
        GMRES(space, bkd, tol=1e-13)
    )
    newton: Newton[_Np, _Np] = Newton(atol=1e-11, rtol=1e-11, maxiters=20)
    point = SteadySystem(residual, space, setups).solve(newton, space.zeros(1))
    assert point.converged()
    return point.solution()


def _check(bkd: NumpyBkd, pieces: Sequence[_Response], reference: _Np) -> None:
    lam = _solve_skeleton(bkd, pieces)
    bkd.assert_allclose(lam, reference[_NCELLS : _NCELLS + 1], atol=1e-10)
    at = [piece.linearize(lam, bkd.zeros((0, 1)), _STEADY) for piece in pieces]
    bkd.assert_allclose(at[0].output() + at[1].output(), bkd.zeros((1, 1)), atol=1e-10)


@pytest.mark.parametrize("reaction", [False, True])
def test_skfem_pieces_equal_the_single_domain(reaction: bool) -> None:
    bkd = NumpyBkd()
    left, right = _skfem_piece(bkd, True, reaction), _skfem_piece(bkd, False, reaction)
    reference = _single_domain(bkd, reaction)
    _check(bkd, [left, right], reference)
    lam = reference[_NCELLS : _NCELLS + 1]
    for piece, half in ((left, slice(0, _NCELLS + 1)), (right, slice(_NCELLS, None))):
        point = piece.linearize(lam, bkd.zeros((0, 1)), _STEADY)
        assert isinstance(point, SteadyPinnedPoint)
        bkd.assert_allclose(point.state(), reference[half], atol=1e-10)


@pytest.mark.parametrize("closed_form_left", [True, False])
def test_mixed_stacks_equal_the_single_domain(closed_form_left: bool) -> None:
    bkd = NumpyBkd()
    if closed_form_left:
        pieces = [_closed_form_piece(bkd, True), _skfem_piece(bkd, False, False)]
    else:
        pieces = [_skfem_piece(bkd, True, False), _closed_form_piece(bkd, False)]
    _check(bkd, pieces, _single_domain(bkd, False))
