r"""Two-dimensional linear elasticity on skfem subdomains coupled through
a two-component skeleton, against the single-domain skfem solve. NumPy
only: skfem is.

The skeleton unknown is the interface displacement, a
:class:`ComponentSkeletonSpace` with two components per node; each
subdomain pins both components of its port nodes and returns the
reaction, the tested traction per component. The meshes match the
skeleton's Lobatto nodes (:mod:`tests._helpers.coupling_skfem`).
E = 1, nu = 0.3, a smooth body force.

- Two subdomains (cut at x = 1/2), clamped on the left edge, traction
  free elsewhere; the interface ends lie on traction-free edges.
- Four subdomains (2 x 2, one cross point), clamped on the left edge:
  the horizontal cut ends on the clamp, so the skeleton pins both
  components of that node and the two left subdomains exempt it from
  their clamp. Matrix-free and dense Schur paths.

Checks: the skeleton solution and every subdomain's state equal the
single-domain solve at the same points (1e-10), and the free interface
tractions balance per component (1e-10). Without supports, adding a
rigid motion to the skeleton data changes no residual, while a stretch
does: the components pair and the tractions carry the right signs.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Sequence

import numpy as np

from pyapprox.pde.boundary import ExceptDofs
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from tests._helpers.coupling_skfem import (
    CoupledProblem,
    Np,
    SkeletonPins,
    TensorQuadMesh,
    Value,
    box_problem,
    check_coupled,
)


def _body_force(x: Np) -> Np:
    force: Np = np.stack([1.0 + x[0], 0.5 - x[1] + x[0] * x[1]])
    return force


def _elasticity(
    mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
) -> Any:
    bkd = mesh.bkd()
    basis = VectorLagrangeBasis(mesh, degree=1)
    physics = CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=1.0,
        poisson_ratio=0.3,
        bkd=bkd,
        body_force=TimeIndependent(_body_force),
    )
    conditions = []
    for side, v in sides:
        bc = DirichletBC(
            basis, side, v if isinstance(v, float) else TimeIndependent(v), bkd
        )
        on_side = exempt & {int(d) for d in bc.constrained_dofs()}
        conditions.append(ExceptDofs(bc, sorted(on_side)))
    return compose_galerkin_system(physics, conditions)


def _on_left(x: Np) -> Np:
    mask: Np = np.abs(x[0]) < 1e-12
    return mask


def _clamped(x: Np) -> Np:
    return np.zeros((2, x.shape[1]))


def test_two_subdomains_equal_the_single_domain() -> None:
    problem = box_problem((2, 1), _elasticity, [("left", 0.0)], ncomponents=2)
    assert problem.pinned() == []
    check_coupled(problem)


@pytest.mark.parametrize("dense", [False, True])
def test_four_subdomains_with_a_clamped_cut_end(dense: bool) -> None:
    pins = SkeletonPins(_on_left, _clamped)
    problem = box_problem(
        (2, 2), _elasticity, [("left", 0.0)], ncomponents=2, pins=pins
    )
    # The horizontal cut's left end, both components.
    pinned = problem.pinned()
    assert len(pinned) == 2 and pinned[1] == pinned[0] + 1
    check_coupled(problem, dense)


def _interleave(field: Np) -> Np:
    """A (2, n) nodal field as an interleaved (2n, 1) skeleton vector."""
    return np.asarray(field.T.reshape(-1, 1))


def _residual_change(problem: CoupledProblem, lam: Np, shift: Np) -> float:
    pieces = [problem.piece(s)[0] for s in range(problem.topo.nsubdomains())]
    residual = problem.residual(pieces)
    change = residual(lam + shift) - residual(lam)
    return float(np.linalg.norm(change) / np.linalg.norm(residual(lam)))


def test_rigid_motions_change_no_residual() -> None:
    problem = box_problem((2, 2), _elasticity, [], ncomponents=2)
    x = problem.skeleton_coords()
    lam = np.random.default_rng(0).normal(size=(problem.space.descriptor().size, 1))
    # Translation (0.3, -0.2) plus rotation 0.7 about the origin.
    rigid = _interleave(np.stack([0.3 - 0.7 * x[1], -0.2 + 0.7 * x[0]]))
    assert _residual_change(problem, lam, rigid) < 1e-10
    stretch = _interleave(np.stack([x[0], np.zeros_like(x[0])]))
    assert _residual_change(problem, lam, stretch) > 1e-2
