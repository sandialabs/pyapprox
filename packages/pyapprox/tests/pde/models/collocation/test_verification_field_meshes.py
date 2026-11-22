"""Fields on meshes of their own in the collocation verification adapter.

A staggered box level puts the pressure on ``N - 2`` interior
Chebyshev-Gauss nodes per direction and the velocity on the level's
``N`` Gauss-Lobatto nodes: its layout sizes each field by its mesh, the
forcing of each field is taken at its own nodes, and an approximation
measures each field from its own mesh. Boundary rows, seams and stacked
forcings refuse a field off the level's mesh; a field mesh must share the
level's dimension and map.
"""

from typing import Dict, List, Mapping, Union

import pytest

from pyapprox.pde.collocation.basis import ChebyshevBasis, ChebyshevGauss1D
from pyapprox.pde.collocation.mesh import TransformedMesh
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    ChebyshevLevel,
    CollocationApproximation,
    FieldMesh,
    PeriodicSeam,
    boundary_conditions,
    forcing,
    mean_value_rows,
    stacked_forcing,
)
from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend

_SQUARE = Box([(0.0, 1.0), (0.0, 1.0)])
_FIELDS = ["ux", "uy", "p"]


def _stokes() -> ManufacturedProblem:
    """Stokes on the unit square: velocity of degree at most 2 per
    direction, pressure of degree 1 per direction, all sides Dirichlet."""
    terms: List[TermProtocol] = [
        Divergence("ux", ["-mu*ux_x + p", "-mu*ux_y"]),
        Divergence("uy", ["-mu*uy_x", "-mu*uy_y + p"]),
        Source("p", "ux_x + uy_y"),
    ]
    conditions: Dict[str, Union[ConditionProtocol, Mapping[str, ConditionProtocol]]]
    conditions = {side: Dirichlet() for side in _SQUARE.boundaries()}
    return ManufacturedProblem(
        {"ux": "x*y**2", "uy": "x**2", "p": "x*y"},
        terms,
        _SQUARE,
        conditions,
        constants={"mu": 1.0},
    )


def _staggered(bkd: Backend[Array], n: int = 5) -> ChebyshevLevel[Array]:
    (level,) = ChebyshevBox([n], bkd, interior_fields=["p"]).levels(_SQUARE)
    return level


def test_layout_sizes_each_field_by_its_mesh(bkd: Backend[Array]) -> None:
    """5 x 5 velocity nodes and 3 x 3 interior pressure nodes: offsets 0,
    25, 50 and 59 states; the pressure nodes reach no boundary and share
    the level's map."""
    level = _staggered(bkd)
    assert not level.is_uniform()
    layout = level.layout(_FIELDS)
    assert [layout.npts(k) for k in range(3)] == [25, 25, 9]
    assert [layout.offset(k) for k in range(3)] == [0, 25, 50]
    assert layout.nstates() == 59
    assert level.field_mesh("ux") is level.mesh()
    pressure = level.field_mesh("p")
    assert pressure.npts_per_dim() == (3, 3)
    assert bkd.to_bool(bkd.all(bkd.abs(pressure.reference_points()) < 1.0))
    assert pressure.transform() is level.mesh().transform()
    assert level.field_basis("p").npts() == 9
    assert "p=(3, 3)" in repr(level)


def test_uniform_level_is_unchanged(bkd: Backend[Array]) -> None:
    """Without interior fields every field is on the level's mesh."""
    (level,) = ChebyshevBox([4], bkd).levels(_SQUARE)
    assert level.is_uniform()
    assert level.field_mesh("p") is level.mesh()
    assert level.layout(_FIELDS).is_uniform()


def test_forcing_at_each_fields_nodes(bkd: Backend[Array]) -> None:
    """The pressure row's forcing, div u = y^2, at the 9 pressure nodes;
    the velocity's at the 25 level nodes."""
    problem, level = _stokes(), _staggered(bkd)
    pressure = forcing(problem, "p", level)
    assert isinstance(pressure, NodalSupplier)
    y = level.field_mesh("p").points()[1]
    bkd.assert_allclose(pressure.values(0.0), y**2)
    velocity = forcing(problem, "ux", level)
    assert isinstance(velocity, NodalSupplier)
    assert velocity.values(0.0).shape == (25,)


def test_approximation_measures_each_field_on_its_mesh(bkd: Backend[Array]) -> None:
    """The exact values at each field's nodes have no error; adding 1 to
    the pressure block makes both errors 1 on the unit square, so the
    pressure is interpolated from its own nodes."""
    problem, level = _stokes(), _staggered(bkd)
    exact = []
    for name in _FIELDS:
        mesh = level.field_mesh(name)
        x, y = mesh.points()[0], mesh.points()[1]
        exact.append({"ux": x * y**2, "uy": x**2, "p": x * y}[name])
    state = bkd.hstack(exact)
    approximation = CollocationApproximation(problem, level, state)
    assert approximation.nodal_error() < 1e-14
    assert approximation.l2_error() < 1e-13
    shifted = bkd.hstack([exact[0], exact[1], exact[2] + 1.0])
    off = CollocationApproximation(problem, level, shifted)
    assert off.nodal_error() == pytest.approx(1.0, abs=1e-13)
    assert off.l2_error() == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match=r"shape \(59,\)"):
        CollocationApproximation(problem, level, bkd.zeros((75,)))


def test_mean_value_rows(bkd: Backend[Array]) -> None:
    """The pressure's integral row on the staggered level: it replaces
    the first pressure row (state row 50), its weights integrate the
    interpolant on the 9 pressure nodes (summing to the area 1), and its
    value is the integral 1/4 of p = x y, so the exact state satisfies
    it. A name that is not a field is refused."""
    problem, level = _stokes(), _staggered(bkd)
    rows = mean_value_rows(problem, level, "p")
    assert rows.rows().indices() == (50,)
    jacobian = rows.jacobian(bkd.zeros((59,)), 0.0)
    bkd.assert_allclose(bkd.sum(jacobian[50]), bkd.asarray(1.0), atol=1e-14)
    mesh = level.field_mesh("p")
    pressure = mesh.points()[0] * mesh.points()[1]
    state = bkd.concatenate([bkd.zeros((50,)), pressure])
    bkd.assert_allclose(rows.residual(state, 0.0), bkd.zeros((59,)), atol=1e-14)
    with pytest.raises(ValueError, match="'q' is not one of the problem's fields"):
        mean_value_rows(problem, level, "q")


def test_refusals(bkd: Backend[Array]) -> None:
    """A level's own fields outside the state; boundary rows, a seam or a
    stacked forcing on a field off the level's mesh; a field mesh of
    another dimension or map; interior fields on 2-node levels."""
    problem, level = _stokes(), _staggered(bkd)
    with pytest.raises(ValueError, match=r"places \['p'\] on their own meshes"):
        level.layout(["u"])
    with pytest.raises(ValueError, match="take each field's forcing"):
        stacked_forcing(problem, level)
    (on_velocity,) = ChebyshevBox([5], bkd, interior_fields=["ux"]).levels(_SQUARE)
    with pytest.raises(ValueError, match=r"\['ux'\] take boundary conditions"):
        boundary_conditions(problem, on_velocity)
    parts = {side: (i,) for i, side in enumerate(_SQUARE.boundaries())}
    placed = FieldMesh(level.field_mesh("p"), level.field_basis("p"))
    seamed = ChebyshevLevel(
        _SQUARE,
        level.mesh(),
        level.basis(),
        parts,
        closures=(PeriodicSeam[Array](0),),
        field_meshes={"p": placed},
    )
    with pytest.raises(ValueError, match=r"seam joins the level's mesh; \['p'\]"):
        PeriodicSeam[Array](0).rows(seamed, seamed.layout(_FIELDS))
    other_map = TransformedMesh(
        (3, 3),
        bkd,
        AffineTransform([(0.0, 1.0), (0.0, 1.0)], bkd),
        bases=[ChebyshevGauss1D[Array]()] * 2,
    )
    with pytest.raises(ValueError, match="share the level mesh's map"):
        ChebyshevLevel(
            _SQUARE,
            level.mesh(),
            level.basis(),
            parts,
            field_meshes={"p": FieldMesh(other_map, ChebyshevBasis(other_map, bkd))},
        )
    line = TransformedMesh((3,), bkd, bases=[ChebyshevGauss1D[Array]()])
    with pytest.raises(ValueError, match="mesh is 1D, the level's 2D"):
        ChebyshevLevel(
            _SQUARE,
            level.mesh(),
            level.basis(),
            parts,
            field_meshes={"p": FieldMesh(line, ChebyshevBasis(line, bkd))},
        )
    with pytest.raises(ValueError, match="at least 3 nodes per direction"):
        ChebyshevBox([2, 5], bkd, interior_fields=["p"])
