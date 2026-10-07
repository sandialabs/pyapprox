"""Tests for Stokes and Navier-Stokes physics.

This module contains:
1. Unit tests for matrix properties (TestStokesBase)
2. Parametrized steady-state manufactured solution tests (4 legacy cases)
3. Parametrized transient manufactured solution tests (4 cases: 1D/2D x BE/CN)

The typing Stokes class solves:
    -viscosity * Lap(u) + grad(p) = f_vel     (momentum)
    -div(u) = f_pres                            (continuity)

Uses Taylor-Hood elements: P2 velocity, P1 pressure.

For transient tests, solutions linear in time are used so that both backward
Euler (1st order) and Crank-Nicolson (2nd order) reproduce the exact solution.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis, MixedBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.manufactured.stokes import (
    ManufacturedStokes,
)
from pyapprox.util.backends.numpy import NumpyBkd

# These tests are numpy-only (numpy_bkd fixture), so generics are
# instantiated with the concrete numpy array type rather than erased.
# NDArray[Any] is the instantiation NumpyBkd itself is declared with
# (Backend[NDArray[Any]]); a floating-dtype parameter fails the
# ArrayProtocol bound.
_NumpyArray = NDArray[Any]
_ManufacturedFuncs = Dict[str, Callable[..., _NumpyArray]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dense(mat: Union[spmatrix, _NumpyArray]) -> _NumpyArray:
    """Convert sparse matrix to dense numpy array if needed."""
    if isinstance(mat, spmatrix):
        return np.asarray(mat.toarray())
    return np.asarray(mat)


def _boundary_names(nvars: int) -> List[str]:
    """Return boundary names for the given number of spatial dimensions."""
    if nvars == 1:
        return ["left", "right"]
    return ["left", "right", "bottom", "top"]


def _get_exact_state(
    funcs: _ManufacturedFuncs,
    nvars: int,
    vel_basis: VectorLagrangeBasis[_NumpyArray],
    pres_basis: LagrangeBasis[_NumpyArray],
    bkd: NumpyBkd,
    time: float = 0.0,
    transient: bool = False,
) -> _NumpyArray:
    """Evaluate manufactured solution at DOF locations.

    Returns the exact state vector [vel_dofs | pres_dofs].
    """
    vel_coords = bkd.to_numpy(vel_basis.dof_coordinates())
    pres_coords = bkd.to_numpy(pres_basis.dof_coordinates())
    n_vel = vel_basis.ndofs()

    if nvars == 1:
        x_vel = vel_coords[0]
        x_pres = pres_coords[0]
    else:
        x_vel = vel_coords
        x_pres = pres_coords

    if transient:
        sol_vel = funcs["solution"](x_vel, time)
        sol_pres = funcs["solution"](x_pres, time)
    else:
        sol_vel = funcs["solution"](x_vel)
        sol_pres = funcs["solution"](x_pres)

    # Extract velocity: for 2D, DOFs are interleaved [ux0, uy0, ux1, uy1, ...]
    if nvars == 1:
        vel_exact = sol_vel[:, 0]
    else:
        vel_exact = np.zeros(n_vel)
        for i in range(n_vel):
            vel_exact[i] = sol_vel[i, i % nvars]

    pres_exact = sol_pres[:, nvars]

    return np.asarray(
        np.concatenate([vel_exact, pres_exact]).astype(np.float64)
    )


def _stokes(
    mesh: Any, bkd: NumpyBkd, **kwargs: Any
) -> StokesPhysics[_NumpyArray]:
    """Taylor-Hood Stokes on ``mesh``: P2 velocity, P1 pressure."""
    basis = MixedBasis(
        [
            ("velocity", VectorLagrangeBasis(mesh, degree=2)),
            ("pressure", LagrangeBasis(mesh, degree=1)),
        ]
    )
    return StokesPhysics(basis, bkd, **kwargs)


def _build_stokes_from_manufactured(
    funcs: _ManufacturedFuncs,
    nvars: int,
    basis: MixedBasis[_NumpyArray],
    bkd: NumpyBkd,
    navier_stokes: bool = False,
    transient: bool = False,
) -> Tuple[GalerkinSystem[_NumpyArray], List[Any]]:
    """Compose Stokes with the manufactured forcing and Dirichlet data on
    every boundary, for both fields. Returns the system and its BCs.

    Parameters
    ----------
    funcs : dict
        Manufactured solution functions.
    nvars : int
        Number of spatial dimensions.
    transient : bool
        If True, use full forcing (including du/dT) for the Galerkin load.
        If False, use spatial-only vel_forcing/pres_forcing.
    """
    bc_names = _boundary_names(nvars)

    def _make_vel_bc(
        funcs: _ManufacturedFuncs, nvars: int, transient: bool
    ) -> Callable[..., _NumpyArray]:
        def vel_bc(coords: _NumpyArray, time: float = 0.0) -> _NumpyArray:
            x_eval = coords[0] if nvars == 1 else coords
            if transient:
                vals = funcs["solution"](x_eval, time)
            else:
                vals = funcs["solution"](x_eval)
            # One row per component (a 1D velocity is scalar-valued).
            if nvars == 1:
                return np.asarray(vals[:, 0])
            return np.asarray(vals[:, :nvars].T)

        return (
            TimeDependent(vel_bc) if transient else TimeIndependent(vel_bc)
        )

    def _make_pres_bc(
        funcs: _ManufacturedFuncs, nvars: int, transient: bool
    ) -> Callable[..., _NumpyArray]:
        def pres_bc(coords: _NumpyArray, time: float = 0.0) -> _NumpyArray:
            x_eval = coords[0] if nvars == 1 else coords
            if transient:
                vals = funcs["solution"](x_eval, time)
            else:
                vals = funcs["solution"](x_eval)
            return np.asarray(vals[:, nvars])

        return (
            TimeDependent(pres_bc) if transient else TimeIndependent(pres_bc)
        )

    if transient:
        # For transient: use full forcing (includes du/dT for velocity)
        def _make_vel_forcing(
            funcs: _ManufacturedFuncs, nvars: int
        ) -> Callable[..., _NumpyArray]:
            def vel_forcing(x_eval: _NumpyArray, time: float) -> _NumpyArray:
                vals = funcs["forcing"](x_eval, time)
                return np.asarray(vals[:, :nvars])

            return TimeDependent(vel_forcing)

        def _make_pres_forcing(
            funcs: _ManufacturedFuncs, nvars: int
        ) -> Callable[..., _NumpyArray]:
            def pres_forcing(x_eval: _NumpyArray, time: float) -> _NumpyArray:
                vals = funcs["forcing"](x_eval, time)
                return np.asarray(vals[:, nvars])

            return TimeDependent(pres_forcing)
    else:
        # For steady state: use spatial-only forcing
        def _make_vel_forcing(
            funcs: _ManufacturedFuncs, nvars: int
        ) -> Callable[..., _NumpyArray]:
            def vel_forcing(x_eval: _NumpyArray) -> _NumpyArray:
                return funcs["vel_forcing"](x_eval)

            return vel_forcing

        def _make_pres_forcing(
            funcs: _ManufacturedFuncs, nvars: int
        ) -> Callable[..., _NumpyArray]:
            def pres_forcing(x_eval: _NumpyArray) -> _NumpyArray:
                return funcs["pres_forcing"](x_eval)

            return pres_forcing

    vel_bc = _make_vel_bc(funcs, nvars, transient)
    pres_bc = _make_pres_bc(funcs, nvars, transient)
    vel_forcing = _make_vel_forcing(funcs, nvars)
    pres_forcing = _make_pres_forcing(funcs, nvars)

    physics = StokesPhysics(
        basis,
        bkd,
        navier_stokes=navier_stokes,
        viscosity=1.0,
        vel_forcing=vel_forcing,
        pres_forcing=pres_forcing,
    )
    bcs: List[Any] = [
        DirichletBC(basis.field("velocity"), bn, vel_bc, bkd)
        for bn in bc_names
    ] + [
        DirichletBC(basis.field("pressure"), bn, pres_bc, bkd)
        for bn in bc_names
    ]
    return compose_galerkin_system(physics, bcs), bcs


# ---------------------------------------------------------------------------
# Unit Tests (dual-backend)
# ---------------------------------------------------------------------------


def _mesh_2d(bkd: NumpyBkd) -> StructuredMesh2D[_NumpyArray]:
    return StructuredMesh2D(
        nx=3, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd
    )


class TestStokesBase:
    """Base test class for Stokes physics."""

    def test_nstates(self, numpy_bkd: NumpyBkd) -> None:
        """nstates is vel_ndofs + pres_ndofs, the mixed basis size."""
        bkd = numpy_bkd
        for mesh in (
            StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd),
            _mesh_2d(bkd),
        ):
            physics = _stokes(mesh, bkd)
            assert (
                physics.nstates()
                == physics.vel_ndofs() + physics.pres_ndofs()
                == physics.basis().ndofs()
            )

    def test_requires_velocity_then_pressure(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        mesh = _mesh_2d(bkd)
        swapped = MixedBasis(
            [
                ("pressure", LagrangeBasis(mesh, degree=1)),
                ("velocity", VectorLagrangeBasis(mesh, degree=2)),
            ]
        )
        with pytest.raises(ValueError, match="velocity"):
            StokesPhysics(swapped, bkd)

    def test_composed_shapes(self, numpy_bkd: NumpyBkd) -> None:
        """The composed steady residual and Jacobian have the state size."""
        bkd = numpy_bkd
        physics = _stokes(StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd), bkd)
        steady = compose_galerkin_system(physics).steady()
        u0 = bkd.zeros((physics.nstates(),))
        assert steady.steady_residual(u0).shape == (physics.nstates(),)
        assert steady.steady_jacobian(u0).shape == (
            physics.nstates(),
            physics.nstates(),
        )

    def test_mass_matrix_block_structure(self, numpy_bkd: NumpyBkd) -> None:
        """Test mass matrix has [M_vel, 0; 0, 0] block structure."""
        bkd = numpy_bkd
        physics = _stokes(StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd), bkd)
        M = _to_dense(physics.mass_matrix())
        n_vel = physics.vel_ndofs()
        assert M.shape == (physics.nstates(), physics.nstates())

        # Pressure block should be zero
        bkd.assert_allclose(bkd.asarray(M[n_vel:, :]), bkd.zeros(M[n_vel:, :].shape))
        bkd.assert_allclose(bkd.asarray(M[:, n_vel:]), bkd.zeros(M[:, n_vel:].shape))

        # Velocity block should be symmetric positive definite
        M_vel = M[:n_vel, :n_vel]
        bkd.assert_allclose(bkd.asarray(M_vel), bkd.asarray(M_vel.T), atol=1e-14)
        assert np.all(np.linalg.eigvalsh(M_vel) > 0)

    def test_ndim(self, numpy_bkd: NumpyBkd) -> None:
        """Test ndim returns spatial dimension."""
        bkd = numpy_bkd
        mesh1d = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        assert _stokes(mesh1d, bkd).ndim() == 1
        assert _stokes(_mesh_2d(bkd), bkd).ndim() == 2

    def test_field_bcs_constrain_global_dofs(self, numpy_bkd: NumpyBkd) -> None:
        """A pressure BC constrains pressure DOFs (offset by the velocity
        DOFs) and a velocity BC velocity DOFs, with each value taken
        from its own component row."""
        bkd = numpy_bkd
        physics = _stokes(_mesh_2d(bkd), bkd)
        basis = physics.basis()
        n_vel = physics.vel_ndofs()

        def inflow(coords: _NumpyArray) -> _NumpyArray:
            return np.stack([1.0 + coords[1], -2.0 + 0.0 * coords[0]])

        vel_bc = DirichletBC(basis.field("velocity"), "left", inflow, bkd)
        pres_bc = DirichletBC(basis.field("pressure"), "right", 3.0, bkd)
        vel_dofs = bkd.to_numpy(vel_bc.constrained_dofs())
        pres_dofs = bkd.to_numpy(pres_bc.constrained_dofs())
        assert np.all(vel_dofs < n_vel) and np.all(pres_dofs >= n_vel)

        coords = bkd.to_numpy(basis.dof_coordinates())
        values = bkd.to_numpy(vel_bc.constrained_values(0.0))
        expected = np.where(
            vel_dofs % 2 == 0, 1.0 + coords[1, vel_dofs], -2.0
        )
        bkd.assert_allclose(bkd.asarray(values), bkd.asarray(expected))
        bkd.assert_allclose(
            pres_bc.constrained_values(0.0),
            bkd.full((pres_dofs.shape[0],), 3.0),
        )


# ---------------------------------------------------------------------------
# Parametrized Steady-State Tests
# ---------------------------------------------------------------------------

# Format: (name, bounds, nrefine, vel_strs, pres_str, bndry_types, navier_stokes)
STOKES_STEADY_CASES: List[
    Tuple[str, List[float], int, List[str], str, List[str], bool]
] = [
    # 1D linear Stokes, all Dirichlet
    (
        "1d_stokes_DD",
        [0, 1],
        1,
        ["((2*x-1))**2"],
        "x*(1-1e-16*x)",
        ["D", "D"],
        False,
    ),
    # 1D Navier-Stokes, all Dirichlet
    (
        "1d_ns_DD",
        [0, 1],
        0,
        ["((2*x-1))*(1+1e-16*x)"],
        "x*(1-1e-16*x)",
        ["D", "D"],
        True,
    ),
    # 2D linear Stokes, all Dirichlet
    (
        "2d_stokes_DDDD",
        [0, 1, 0, 1],
        0,
        ["1e-16*x+y", "(x+1)*(y+1)"],
        "x*(1+1e-16*x)+1e-16*y*(1+1e-16*y)",
        ["D", "D", "D", "D"],
        False,
    ),
    # 2D Navier-Stokes, all Dirichlet
    (
        "2d_ns_DDDD",
        [0, 1, 0, 1],
        0,
        ["x**2*y**2", "(x+1)*(y+1)"],
        "x*y",
        ["D", "D", "D", "D"],
        True,
    ),
]


class TestParametrizedSteadyStokes:
    """Parametrized steady-state Stokes manufactured solution tests.

    Replicates the 4 legacy test cases from
    pyapprox/pde/galerkin/tests/test_finite_elements.py::test_stokes.

    Uses Taylor-Hood P2/P1 elements. For polynomial solutions within the
    approximation space, the FE solution should match exactly (up to
    round-off).
    """

    @pytest.mark.parametrize(
        "name,bounds,nrefine,vel_strs,pres_str,bndry_types,navier_stokes",
        STOKES_STEADY_CASES,
    )
    def test_manufactured_stokes(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        bounds: List[float],
        nrefine: int,
        vel_strs: List[str],
        pres_str: str,
        bndry_types: List[str],
        navier_stokes: bool,
    ) -> None:
        """Test manufactured solution for Stokes/NS equations."""
        bkd = numpy_bkd
        nvars = len(bounds) // 2
        sol_strs = vel_strs + [pres_str]

        man_sol = ManufacturedStokes(
            sol_strs, nvars, navier_stokes=navier_stokes, bkd=bkd, oned=True
        )
        funcs = man_sol.functions

        # Create mesh
        mesh: Union[
            StructuredMesh1D[_NumpyArray], StructuredMesh2D[_NumpyArray]
        ]
        if nvars == 1:
            nx = 10 * (2**nrefine)
            mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
        else:
            nx = 5 * (2**nrefine)
            mesh = StructuredMesh2D(
                nx=nx,
                ny=nx,
                bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
                bkd=bkd,
            )

        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        basis = MixedBasis([("velocity", vel_basis), ("pressure", pres_basis)])

        system, _ = _build_stokes_from_manufactured(
            funcs, nvars, basis, bkd, navier_stokes=navier_stokes
        )

        # Navier-Stokes starts from the linear Stokes solution with the
        # same forcing and boundary data.
        guess = None
        if navier_stokes:
            linear_system, _ = _build_stokes_from_manufactured(
                funcs, nvars, basis, bkd, navier_stokes=False
            )
            guess = SteadyStateSolver(linear_system.steady()).solve().solution
        result = SteadyStateSolver(system.steady(), tol=1e-12).solve(guess)

        assert result.converged, (
            f"Test {name}: solver did not converge "
            f"(residual={result.residual_norm:.2e})"
        )

        # Compare to exact solution
        u_num = bkd.to_numpy(result.solution)
        u_exact = _get_exact_state(funcs, nvars, vel_basis, pres_basis, bkd)

        max_err = np.max(np.abs(u_num - u_exact))
        assert max_err < 5e-7


# ---------------------------------------------------------------------------
# Parametrized Transient Tests
# ---------------------------------------------------------------------------

# Format: (name, bounds, nrefine, vel_strs, pres_str, bndry_types, method)
STOKES_TRANSIENT_CASES: List[
    Tuple[str, List[float], int, List[str], str, List[str], str]
] = [
    # 1D Stokes, backward Euler
    (
        "1d_BE",
        [0, 1],
        1,
        ["((2*x-1))**2*(1+T)"],
        "x*(1-1e-16*x)*(1+T)",
        ["D", "D"],
        "backward_euler",
    ),
    # 1D Stokes, Crank-Nicolson
    (
        "1d_CN",
        [0, 1],
        1,
        ["((2*x-1))**2*(1+T)"],
        "x*(1-1e-16*x)*(1+T)",
        ["D", "D"],
        "crank_nicolson",
    ),
    # 2D Stokes, backward Euler
    (
        "2d_BE",
        [0, 1, 0, 1],
        0,
        ["(1e-16*x+y)*(1+T)", "((x+1)*(y+1))*(1+T)"],
        "(x*(1+1e-16*x)+1e-16*y*(1+1e-16*y))*(1+T)",
        ["D", "D", "D", "D"],
        "backward_euler",
    ),
    # 2D Stokes, Crank-Nicolson
    (
        "2d_CN",
        [0, 1, 0, 1],
        0,
        ["(1e-16*x+y)*(1+T)", "((x+1)*(y+1))*(1+T)"],
        "(x*(1+1e-16*x)+1e-16*y*(1+1e-16*y))*(1+T)",
        ["D", "D", "D", "D"],
        "crank_nicolson",
    ),
]


class TestParametrizedTransientStokes:
    """Parametrized transient Stokes manufactured solution tests.

    Uses time-linear solutions u(x,t) = u0(x)*(1+T) which are exactly
    reproduced by both backward Euler (1st order) and Crank-Nicolson
    (2nd order) with any time step size.

    For transient manufactured solutions, the Galerkin load uses the FULL
    forcing (including du/dT) because the weak form is:
        (du/dt, v) + a(u, v) = (f_full, v)
    where f_full = du/dt - Lap(u) + grad(p).
    """

    @pytest.mark.parametrize("navier_stokes", [False, True])
    @pytest.mark.parametrize(
        "name,bounds,nrefine,vel_strs,pres_str,bndry_types,method",
        STOKES_TRANSIENT_CASES,
    )
    def test_transient_stokes(
        self,
        numpy_bkd: NumpyBkd,
        name: str,
        bounds: List[float],
        nrefine: int,
        vel_strs: List[str],
        pres_str: str,
        bndry_types: List[str],
        method: str,
        navier_stokes: bool,
    ) -> None:
        """Test transient manufactured solution for (Navier-)Stokes.

        Time-linear solutions are reproduced exactly by BE and CN even
        for the nonlinear convective term (u_dot is constant in time,
        so the manufactured forcing balances F at every time level).
        The Navier-Stokes cases exercise genuine multi-iteration
        Newton through the unified DAE pipeline.
        """
        bkd = numpy_bkd
        nvars = len(bounds) // 2
        sol_strs = vel_strs + [pres_str]

        man_sol = ManufacturedStokes(
            sol_strs, nvars, navier_stokes=navier_stokes, bkd=bkd, oned=True
        )
        funcs = man_sol.functions

        # Create mesh
        mesh: Union[
            StructuredMesh1D[_NumpyArray], StructuredMesh2D[_NumpyArray]
        ]
        if nvars == 1:
            nx = 10 * (2**nrefine)
            mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
        else:
            nx = 5 * (2**nrefine)
            mesh = StructuredMesh2D(
                nx=nx,
                ny=nx,
                bounds=[(bounds[0], bounds[1]), (bounds[2], bounds[3])],
                bkd=bkd,
            )

        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        basis = MixedBasis([("velocity", vel_basis), ("pressure", pres_basis)])

        # Build physics with full (transient) forcing
        system, _ = _build_stokes_from_manufactured(
            funcs,
            nvars,
            basis,
            bkd,
            navier_stokes=navier_stokes,
            transient=True,
        )

        # Time stepping through the unified DAE pipeline
        model = GalerkinModel(system, bkd)
        y0 = bkd.asarray(
            _get_exact_state(
                funcs,
                nvars,
                vel_basis,
                pres_basis,
                bkd,
                time=0.0,
                transient=True,
            )
        )
        config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=0.5,
            deltat=0.1,
            newton_tol=1e-12,
            newton_maxiter=50,
            lumped_mass=False,
            verbosity=0,
        )
        solutions, times = model.solve_transient(y0, config)
        y = solutions[:, -1]
        t = float(times[-1])

        # Compare to exact at final time
        y_exact = _get_exact_state(
            funcs,
            nvars,
            vel_basis,
            pres_basis,
            bkd,
            time=t,
            transient=True,
        )
        max_err = np.max(np.abs(bkd.to_numpy(y) - y_exact))

        assert max_err < 5e-7
