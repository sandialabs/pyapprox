r"""Different diffusivity laws on the two sides of an interface. NumPy
only: skfem is.

- skfem beside skfem: the unit square cut at x = 1/2, Q1 meshes on the
  skeleton's Lobatto grids (``tests/_helpers/coupling_skfem.py``),
  :math:`-\nabla\cdot(D \nabla u) = f` with :math:`D = 1` on the left
  and :math:`D = 2 + xy` on the right, u = 1 on the left edge and u = 2
  on the right. The interface is an element boundary of the
  single-domain mesh, so the single-domain solve with the piecewise
  coefficient is the same discrete problem: the skeleton, every state
  and the flux balance agree to 1e-10.
- Collocation beside Q2 skfem (``tests/_helpers/coupling_mixed.py``):
  advection-diffusion-reaction with v = (1, 0.5), r = -1, D = 1 on the
  collocation side and :math:`D = 2 + xy` on the skfem side, against a
  piecewise manufactured solution: :math:`e^x \sin 3x \cos 2y` on the
  left, and on the right :math:`A(y) + B(y)(x - 1/2) + (x - 1/2)^2
  \cos(x + y)` with A the left solution's trace at x = 1/2 and B = u_x /
  D_R there, so u and D ∂u/∂n are continuous. For n = 4, 8, 16 the skfem
  L2 error converges at rate within 0.1 of 3, the collocation max error
  at least 2.5, and the fluxes cancel against every free skeleton
  function (1e-9). Control: the skfem side given the left law, D = 1,
  stalls (errors above 1e-2 at every level).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Sequence

import numpy as np

from pyapprox.pde.boundary import ExceptDofs
from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    CoordinateDiffusion,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from tests._helpers.coupling_collocation import SMOOTH, Coefficients
from tests._helpers.coupling_mixed import MixedStrip
from tests._helpers.coupling_skfem import (
    Np,
    TensorQuadMesh,
    Value,
    box_problem,
    check_coupled,
)

_RIGHT_D = "2 + x*y"
# u_R = A(y) + B(y) (x - 1/2) + (x - 1/2)^2 cos(x + y): A the left trace,
# B = D_L u_x / D_R at x = 1/2 with D_L = 1 and D_R(1/2, y) = 2 + y/2.
_TRACE = "exp(1/2)*sin(3/2)*cos(2*y)"
_SLOPE = "exp(1/2)*(sin(3/2) + 3*cos(3/2))*cos(2*y)/(2 + y/2)"
_RIGHT = f"{_TRACE} + ({_SLOPE})*(x - 1/2) + (x - 1/2)**2*cos(x + y)"
_LEVELS = (4, 8, 16)


def _piecewise(x: Np) -> Np:
    return np.asarray(np.where(x[0] < 0.5, 1.0, 2.0 + x[0] * x[1]))


def _forcing(x: Np) -> Np:
    return np.asarray(1.0 + x[0] + 2.0 * x[1])


def _no_sink(x: Np, u: Np) -> Np:
    return np.zeros_like(u)


def _build(
    mesh: TensorQuadMesh, sides: Sequence[tuple[str, Value]], exempt: set[int]
) -> Any:
    bkd = mesh.bkd()
    basis = LagrangeBasis(mesh, degree=1)
    # Quadrature points lie inside elements, so each element sees its own
    # side's law.
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=CoordinateDiffusion(TimeIndependent(_piecewise)),
        bkd=bkd,
        reaction=CallableReaction(_no_sink, _no_sink),
        forcing=TimeIndependent(_forcing),
    )
    conditions = []
    for side, v in sides:
        bc = DirichletBC(
            basis, side, v if isinstance(v, float) else TimeIndependent(v), bkd
        )
        on_side = exempt & {int(d) for d in bc.constrained_dofs()}
        conditions.append(ExceptDofs(bc, sorted(on_side)))
    return compose_galerkin_system(physics, conditions)


def test_skfem_pieces_equal_the_piecewise_single_domain() -> None:
    check_coupled(box_problem((2, 1), _build, [("left", 1.0), ("right", 2.0)]))


def _rate(errors: list[float]) -> float:
    slope = np.polyfit(np.log(1.0 / np.asarray(_LEVELS)), np.log(errors), 1)[0]
    return float(slope)


_LEFT = Coefficients(1.0, (1.0, 0.5), -1.0)


def test_mixed_strip_converges_to_the_piecewise_solution() -> None:
    right = Coefficients(_RIGHT_D, (1.0, 0.5), -1.0)
    collocation, galerkin = [], []
    for n in _LEVELS:
        strip = MixedStrip(_LEFT, SMOOTH, 18, n, 14, galerkin=(right, _RIGHT))
        lam = strip.solve_skeleton()
        collocation.append(strip.errors(lam)[0])
        galerkin.append(strip.galerkin_l2_error(lam))
        left, right_flux = strip.gathered(lam)
        free = [d for d in range(lam.shape[0]) if d not in strip.pinned()]
        assert np.abs(left[free] + right_flux[free]).max() < 1e-9
    assert abs(_rate(galerkin) - 3.0) < 0.1, galerkin
    assert _rate(collocation) > 2.5, collocation


def test_the_wrong_law_on_one_side_stalls() -> None:
    """The skfem side solved with the left law while its forcing and data
    come from the right one's solution."""
    wrong = Coefficients(1.0, (1.0, 0.5), -1.0)
    for n in _LEVELS:
        strip = MixedStrip(_LEFT, SMOOTH, 18, n, 14, galerkin=(wrong, _RIGHT))
        assert max(strip.errors(strip.solve_skeleton())) > 1e-2
