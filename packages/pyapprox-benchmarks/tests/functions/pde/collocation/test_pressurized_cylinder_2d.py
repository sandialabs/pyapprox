"""Tests for the 2D linear elastic pressurized cylinder builder.

Verifies:
1. Lame analytical solution with full cylinder BC structure (constant E
   from a one-term ``BasisExpansion`` at p = 0, 16 x 16 nodes, to 1e-8)
2. Parameter Jacobian on full forward model via DerivativeChecker
3. Builder produces a valid function with correct shapes

The manufactured and variable-Lame tests of the physics on the polar
domain are in ``tests/pde/collocation/physics/test_linear_elasticity_polar.py``.
"""

import math
from typing import Tuple

import pytest

from pyapprox.implicitfunction.observation_terms import ObservedValue, StateEntries
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.field_maps.kle_factory import (
    create_lognormal_kle_field_map,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.field_maps.transformed import TransformedFieldMap
from pyapprox.pde.models.collocation.steady_function import (
    collocation_steady_function,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox_benchmarks.functions.pde.collocation.pressurized_cylinder_2d import (
    build_linear_pressurized_cylinder_2d,
)

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


def _every_state_function(
    mesh: TransformedMesh2D[Array],
    poisson_ratio: float,
    pressure: float,
    field_map: FieldMapProtocol[Array],
) -> SteadyImplicitFunction[Array, DenseFormat[Array]]:
    """The cylinder's displacement at every node, components stacked."""
    bkd = mesh.bkd()
    configured = build_linear_pressurized_cylinder_2d(
        mesh, poisson_ratio, pressure, field_map
    )
    nstates = configured.system.nstates()
    functional = StateFunctional.from_terms(
        bkd,
        nstates,
        state_terms=(ObservedValue(StateEntries(bkd, nstates, list(range(nstates)))),),
    )
    return collocation_steady_function(
        configured.system, ParameterState(configured.parameterization), functional
    )


def _lame_analytical(
    bkd: Backend[Array],
    pts: Array,
    r_inner: float,
    r_outer: float,
    pressure: float,
    lamda: float,
    mu: float,
) -> Tuple[Array, Array]:
    """Compute analytical Lame solution in Cartesian coordinates.

    Returns (u_x, u_y) each of shape (npts,) for plane strain.

    Parameters
    ----------
    pts : Array
        Physical coordinates, shape (2, npts).
    """
    x = pts[0, :]
    y = pts[1, :]
    r = bkd.sqrt(x**2 + y**2)

    Ri, Ro = r_inner, r_outer
    A = pressure * Ri**2 / (Ro**2 - Ri**2)
    B = -pressure * Ri**2 * Ro**2 / (Ro**2 - Ri**2)

    # Plane strain Poisson ratio from Lame params
    nu = lamda / (2.0 * (lamda + mu))

    # Radial displacement (plane strain)
    u_r = (1.0 / (2.0 * mu)) * ((1.0 - 2.0 * nu) * A * r - B / r)

    # Convert to Cartesian
    u_x = u_r * x / r
    u_y = u_r * y / r
    return u_x, u_y


def _make_kle_field_map_2d(
    bkd: Backend[Array], mesh: TransformedMesh2D[Array], num_kle_terms: int = 2
) -> TransformedFieldMap[Array]:
    """Create lognormal KLE field map on 2D polar mesh nodes."""
    physical_pts = mesh.points()  # (2, npts)
    npts = physical_pts.shape[1]

    # Normalize coordinates to [0, 1]^2 for correlation kernel
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


# ======================================================================
# Test class
# ======================================================================


class TestPressurizedCylinder2D:
    """Tests for 2D linear elastic pressurized cylinder."""

    # ------------------------------------------------------------------
    # Lame analytical solution
    # ------------------------------------------------------------------

    def test_lame_solution(self, bkd: Backend[Array]) -> None:
        """Solve with constant E, compare to Lame analytical solution."""
        npts_r, npts_theta = 16, 16
        r_inner, r_outer = 1.0, 2.0
        pressure = 1.0
        lamda, mu = 1.0, 1.0

        mesh = _polar_mesh(bkd, npts_r, npts_theta, r_inner, r_outer)
        npts = mesh.npts()
        pts = mesh.points()

        # The constant E and nu of (lamda, mu): E = exactly that at p = 0
        poisson_ratio = lamda / (2.0 * (lamda + mu))
        youngs = mu * (3.0 * lamda + 2.0 * mu) / (lamda + mu)
        fwd = _every_state_function(
            mesh,
            poisson_ratio,
            pressure,
            BasisExpansion(bkd, youngs, [bkd.ones((npts,))]),
        )
        sol = fwd(bkd.zeros((1, 1)))[:, 0]

        # Analytical Lame solution
        u_x_exact, u_y_exact = _lame_analytical(
            bkd,
            pts,
            r_inner,
            r_outer,
            pressure,
            lamda,
            mu,
        )
        exact_state = bkd.concatenate([u_x_exact, u_y_exact])
        bkd.assert_allclose(sol, exact_state, atol=1e-8)

    # ------------------------------------------------------------------
    # Parameter Jacobian on full forward model
    # ------------------------------------------------------------------

    @pytest.mark.slow_on("TorchBkd")
    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        """DerivativeChecker on KLE-parameterized cylinder forward model."""
        npts_r, npts_theta = 8, 8
        r_inner, r_outer = 1.0, 2.0
        nu = 0.3
        pressure = 0.5
        num_kle_terms = 2

        mesh = _polar_mesh(bkd, npts_r, npts_theta, r_inner, r_outer)
        field_map = _make_kle_field_map_2d(bkd, mesh, num_kle_terms)

        fwd = _every_state_function(mesh, nu, pressure, field_map)

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

        fwd = _every_state_function(mesh, 0.3, 1.0, field_map)

        # Check protocol compliance
        assert isinstance(fwd, FunctionProtocol)
        assert fwd.derivatives().jacobian is not None

        # Check dimensions
        npts = npts_r * npts_theta
        assert fwd.nvars() == num_kle_terms
        assert fwd.nqoi() == 2 * npts

        # Evaluate at zero parameters
        sample = bkd.zeros((num_kle_terms, 1))
        result = fwd(sample)
        assert result.shape == (2 * npts, 1)

        # Jacobian shape
        jacobian = fwd.derivatives().jacobian
        assert jacobian is not None
        jac = jacobian(sample)
        assert jac.shape == (2 * npts, num_kle_terms)


# ======================================================================
# Backend-specific test classes
# ======================================================================
