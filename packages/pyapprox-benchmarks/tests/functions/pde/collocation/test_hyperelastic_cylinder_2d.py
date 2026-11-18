"""Tests for the 2D hyperelastic pressurized cylinder builder.

Verifies:
1. Small pressure: hyperelastic QoI matches linear QoI
2. Parameter Jacobian on full forward model via DerivativeChecker
3. Builder produces a valid function with correct shapes

The manufactured tests of the physics on the polar domain are in
``tests/pde/collocation/physics/test_hyperelasticity_polar.py``.
"""

import math
from typing import Callable, Optional

from pyapprox.implicitfunction.functional_terms import StateTermProtocol
from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.functionals.observations import (
    average_radial_displacement_observation,
)
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox_benchmarks.functions.pde.collocation.hyperelastic_cylinder_2d import (
    build_hyperelastic_pressurized_cylinder_2d,
)
from pyapprox_benchmarks.functions.pde.collocation.pressurized_cylinder_2d import (
    build_linear_pressurized_cylinder_2d,
)
from tests._helpers.markers import slow_test

_Builder = Callable[
    [TransformedMesh2D[Array], float, float, FieldMapProtocol[Array]],
    ParameterizedCollocationSystem[Array, ChebyshevBasis2D[Array]],
]

# ======================================================================
# Helpers
# ======================================================================


def _polar_mesh(
    bkd: Backend[Array],
    npts_r: int,
    npts_theta: int,
    r_inner: float,
    r_outer: float,
) -> TransformedMesh2D[Array]:
    """Create the polar mesh on the quarter-annulus."""
    transform = PolarTransform(
        (r_inner, r_outer),
        (0.0, math.pi / 2.0),
        bkd,
    )
    return TransformedMesh2D(npts_r, npts_theta, bkd, transform)


def _make_kle_field_map_2d(
    bkd: Backend[Array], mesh: TransformedMesh2D[Array], num_kle_terms: int = 2
) -> TransformedFieldMap[Array]:
    """Create lognormal KLE field map on 2D polar mesh nodes."""
    physical_pts = mesh.points()  # (2, npts)
    npts = physical_pts.shape[1]

    x = physical_pts[0, :]
    y = physical_pts[1, :]
    x_min, x_max = bkd.to_float(bkd.min(x)), bkd.to_float(bkd.max(x))
    y_min, y_max = bkd.to_float(bkd.min(y)), bkd.to_float(bkd.max(y))
    x_range = max(x_max - x_min, 1e-12)
    y_range = max(y_max - y_min, 1e-12)
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    mesh_coords = bkd.stack([x_norm, y_norm], axis=0)  # (2, npts)

    mean_log = bkd.zeros((npts,))
    return create_lognormal_kle_field_map(
        mesh_coords,
        mean_log,
        bkd,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
    )


def _outer_wall_term(
    mesh: TransformedMesh2D[Array],
) -> StateTermProtocol[Array, Array]:
    """The average radial displacement over the outer wall."""
    bkd = mesh.bkd()
    outer_idx = mesh.boundary_indices(BoundarySide.RIGHT)
    pts = mesh.points()
    outer_pts = pts[:, outer_idx]
    r = bkd.sqrt(outer_pts[0, :] ** 2 + outer_pts[1, :] ** 2)
    return ObservedValue(
        average_radial_displacement_observation(
            bkd,
            [bkd.to_int(outer_idx[k]) for k in range(outer_idx.shape[0])],
            outer_pts[0, :] / r,
            outer_pts[1, :] / r,
            mesh.npts(),
        )
    )


def _function(
    builder: _Builder[Array],
    mesh: TransformedMesh2D[Array],
    pressure: float,
    field_map: FieldMapProtocol[Array],
    term: Optional[StateTermProtocol[Array, Array]] = None,
) -> SteadyImplicitFunction[Array, DenseFormat[Array]]:
    """The builder's function of ``term``, by default every state entry."""
    bkd = mesh.bkd()
    configured = builder(mesh, 0.3, pressure, field_map)
    nstates = configured.system.nstates()
    if term is None:
        term = ObservedValue(StateEntries(bkd, nstates, list(range(nstates))))
    functional = StateFunctional.from_terms(bkd, term.nqoi(), state_terms=(term,))
    return collocation_steady_function(
        configured.system, ParameterState(configured.parameterization), functional
    )


# ======================================================================
# Test class
# ======================================================================


@slow_test
class TestHyperelasticCylinder2D:
    """Tests for 2D hyperelastic pressurized cylinder."""

    # ------------------------------------------------------------------
    # Small pressure: hyperelastic ≈ linear
    # ------------------------------------------------------------------

    def test_small_pressure_matches_linear(self, bkd: Backend[Array]) -> None:
        """Low pressure: hyperelastic QoI matches linear QoI."""
        npts_r, npts_theta = 10, 10
        r_inner, r_outer = 1.0, 2.0
        pressure = 1e-3
        num_kle_terms = 2

        mesh = _polar_mesh(bkd, npts_r, npts_theta, r_inner, r_outer)
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd_hyper = _function(
            build_hyperelastic_pressurized_cylinder_2d,
            mesh,
            pressure,
            field_map,
            _outer_wall_term(mesh),
        )
        fwd_linear = _function(
            build_linear_pressurized_cylinder_2d,
            mesh,
            pressure,
            field_map,
            _outer_wall_term(mesh),
        )

        sample = bkd.zeros((num_kle_terms, 1))
        qoi_hyper = fwd_hyper(sample)
        qoi_linear = fwd_linear(sample)

        bkd.assert_allclose(qoi_hyper, qoi_linear, rtol=1e-2)

    # ------------------------------------------------------------------
    # Parameter Jacobian via DerivativeChecker
    # ------------------------------------------------------------------

    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on KLE-parameterized hyperelastic cylinder."""
        npts_r, npts_theta = 8, 8
        r_inner, r_outer = 1.0, 2.0
        pressure = 0.1
        num_kle_terms = 2

        mesh = _polar_mesh(bkd, npts_r, npts_theta, r_inner, r_outer)
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = _function(
            build_hyperelastic_pressurized_cylinder_2d, mesh, pressure, field_map
        )

        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None

        checker = DerivativeChecker(fwd)
        sample = bkd.array([0.1, -0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        assert checker.error_ratio(errors) <= 1e-5

    # ------------------------------------------------------------------
    # Factory produces valid model
    # ------------------------------------------------------------------

    def test_factory_produces_valid_model(self, bkd: Backend[Array]) -> None:
        """The builder's system gives a function with correct shapes."""
        npts_r, npts_theta = 6, 6
        r_inner, r_outer = 1.0, 2.0
        num_kle_terms = 2

        mesh = _polar_mesh(bkd, npts_r, npts_theta, r_inner, r_outer)
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = _function(
            build_hyperelastic_pressurized_cylinder_2d, mesh, 1.0, field_map
        )

        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None

        npts = npts_r * npts_theta
        assert fwd.nvars() == num_kle_terms
        assert fwd.nqoi() == 2 * npts

        sample = bkd.zeros((num_kle_terms, 1))
        result = fwd(sample)
        assert result.shape == (2 * npts, 1)

        jacobian = fwd.derivatives().jacobian
        assert jacobian is not None
        jac = jacobian(sample)
        assert jac.shape == (2 * npts, num_kle_terms)


# ======================================================================
# Backend-specific test classes
# ======================================================================
