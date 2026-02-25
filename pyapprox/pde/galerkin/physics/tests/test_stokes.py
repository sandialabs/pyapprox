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

import unittest
from typing import Generic, Any, List, Tuple, Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.time_integration.stokes_time_stepper import (
    StokesTimeStepResidual,
)
from pyapprox.pde.collocation.manufactured_solutions.stokes import (
    ManufacturedStokes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _boundary_names(nvars: int) -> List[str]:
    """Return boundary names for the given number of spatial dimensions."""
    if nvars == 1:
        return ["left", "right"]
    return ["left", "right", "bottom", "top"]


def _get_exact_state(
    funcs: Dict[str, Callable],
    nvars: int,
    vel_basis: VectorLagrangeBasis,
    pres_basis: LagrangeBasis,
    bkd: Backend,
    time: float = 0.0,
    transient: bool = False,
) -> np.ndarray:
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

    return np.concatenate([vel_exact, pres_exact]).astype(np.float64)


def _build_stokes_from_manufactured(
    funcs: Dict[str, Callable],
    nvars: int,
    vel_basis: VectorLagrangeBasis,
    pres_basis: LagrangeBasis,
    bkd: Backend,
    navier_stokes: bool = False,
    transient: bool = False,
) -> StokesPhysics:
    """Create StokesPhysics from manufactured solution functions.

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

    def _make_vel_bc(funcs, nvars, transient):
        def vel_bc(coords, time=0.0):
            x_eval = coords[0] if nvars == 1 else coords
            if transient:
                vals = funcs["solution"](x_eval, time)
            else:
                vals = funcs["solution"](x_eval)
            return vals[:, :nvars]
        return vel_bc

    def _make_pres_bc(funcs, nvars, transient):
        def pres_bc(coords, time=0.0):
            x_eval = coords[0] if nvars == 1 else coords
            if transient:
                vals = funcs["solution"](x_eval, time)
            else:
                vals = funcs["solution"](x_eval)
            return vals[:, nvars]
        return pres_bc

    if transient:
        # For transient: use full forcing (includes du/dT for velocity)
        def _make_vel_forcing(funcs, nvars):
            def vel_forcing(x_eval, time):
                vals = funcs["forcing"](x_eval, time)
                return vals[:, :nvars]
            return vel_forcing

        def _make_pres_forcing(funcs, nvars):
            def pres_forcing(x_eval, time):
                vals = funcs["forcing"](x_eval, time)
                return vals[:, nvars]
            return pres_forcing
    else:
        # For steady state: use spatial-only forcing
        def _make_vel_forcing(funcs, nvars):
            def vel_forcing(x_eval, time=0.0):
                return funcs["vel_forcing"](x_eval)
            return vel_forcing

        def _make_pres_forcing(funcs, nvars):
            def pres_forcing(x_eval, time=0.0):
                return funcs["pres_forcing"](x_eval)
            return pres_forcing

    vel_bc = _make_vel_bc(funcs, nvars, transient)
    pres_bc = _make_pres_bc(funcs, nvars, transient)
    vel_forcing = _make_vel_forcing(funcs, nvars)
    pres_forcing = _make_pres_forcing(funcs, nvars)

    return StokesPhysics(
        vel_basis=vel_basis,
        pres_basis=pres_basis,
        bkd=bkd,
        navier_stokes=navier_stokes,
        viscosity=1.0,
        vel_forcing=vel_forcing,
        pres_forcing=pres_forcing,
        vel_dirichlet_bcs=[(bn, vel_bc) for bn in bc_names],
        pres_dirichlet_bcs=[(bn, pres_bc) for bn in bc_names],
    )


# ---------------------------------------------------------------------------
# Unit Tests (dual-backend)
# ---------------------------------------------------------------------------

class TestStokesBase(Generic[Array], unittest.TestCase):
    """Base test class for Stokes physics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_nstates_1d(self) -> None:
        """Test nstates matches vel_ndofs + pres_ndofs in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        self.assertEqual(
            physics.nstates(),
            physics.vel_ndofs() + physics.pres_ndofs(),
        )

    def test_nstates_2d(self) -> None:
        """Test nstates matches vel_ndofs + pres_ndofs in 2D."""
        mesh = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=self.bkd_inst,
        )
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        self.assertEqual(
            physics.nstates(),
            physics.vel_ndofs() + physics.pres_ndofs(),
        )

    def test_residual_shape_1d(self) -> None:
        """Test residual has correct shape in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        u0 = self.bkd_inst.asarray(
            np.zeros(physics.nstates(), dtype=np.float64)
        )
        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_jacobian_shape_1d(self) -> None:
        """Test Jacobian has correct shape in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        u0 = self.bkd_inst.asarray(
            np.zeros(physics.nstates(), dtype=np.float64)
        )
        jac = physics.jacobian(u0, 0.0)
        self.assertEqual(
            jac.shape, (physics.nstates(), physics.nstates())
        )

    def test_mass_matrix_shape_1d(self) -> None:
        """Test mass matrix has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        M = physics.mass_matrix()
        self.assertEqual(M.shape, (physics.nstates(), physics.nstates()))

    def test_mass_matrix_block_structure(self) -> None:
        """Test mass matrix has [M_vel, 0; 0, 0] block structure."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)
        physics = StokesPhysics(
            vel_basis=vel_basis, pres_basis=pres_basis, bkd=self.bkd_inst
        )
        M = _to_dense(physics.mass_matrix())
        n_vel = physics.vel_ndofs()

        # Pressure block should be zero
        np.testing.assert_array_equal(M[n_vel:, :], 0.0)
        np.testing.assert_array_equal(M[:, n_vel:], 0.0)

        # Velocity block should be symmetric positive definite
        M_vel = M[:n_vel, :n_vel]
        np.testing.assert_array_almost_equal(M_vel, M_vel.T)
        eigenvalues = np.linalg.eigvalsh(M_vel)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_ndim(self) -> None:
        """Test ndim returns spatial dimension."""
        mesh1d = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        vel1d = VectorLagrangeBasis(mesh1d, degree=2)
        pres1d = LagrangeBasis(mesh1d, degree=1)
        physics1d = StokesPhysics(
            vel_basis=vel1d, pres_basis=pres1d, bkd=self.bkd_inst
        )
        self.assertEqual(physics1d.ndim(), 1)

        mesh2d = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=self.bkd_inst,
        )
        vel2d = VectorLagrangeBasis(mesh2d, degree=2)
        pres2d = LagrangeBasis(mesh2d, degree=1)
        physics2d = StokesPhysics(
            vel_basis=vel2d, pres_basis=pres2d, bkd=self.bkd_inst
        )
        self.assertEqual(physics2d.ndim(), 2)


class TestStokesNumpy(TestStokesBase[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


try:
    import torch
    from pyapprox.util.backends.torch import TorchBkd

    class TestStokesTorch(TestStokesBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            torch.set_default_dtype(torch.float64)
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

except ImportError:
    pass


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
        [0, 1], 1,
        ["((2*x-1))**2"],
        "x*(1-1e-16*x)",
        ["D", "D"],
        False,
    ),
    # 1D Navier-Stokes, all Dirichlet
    (
        "1d_ns_DD",
        [0, 1], 0,
        ["((2*x-1))*(1+1e-16*x)"],
        "x*(1-1e-16*x)",
        ["D", "D"],
        True,
    ),
    # 2D linear Stokes, all Dirichlet
    (
        "2d_stokes_DDDD",
        [0, 1, 0, 1], 0,
        ["1e-16*x+y", "(x+1)*(y+1)"],
        "x*(1+1e-16*x)+1e-16*y*(1+1e-16*y)",
        ["D", "D", "D", "D"],
        False,
    ),
    # 2D Navier-Stokes, all Dirichlet
    (
        "2d_ns_DDDD",
        [0, 1, 0, 1], 0,
        ["x**2*y**2", "(x+1)*(y+1)"],
        "x*y",
        ["D", "D", "D", "D"],
        True,
    ),
]


class TestParametrizedSteadyStokes(ParametrizedTestCase):
    """Parametrized steady-state Stokes manufactured solution tests.

    Replicates the 4 legacy test cases from
    pyapprox/pde/galerkin/tests/test_finite_elements.py::test_stokes.

    Uses Taylor-Hood P2/P1 elements. For polynomial solutions within the
    approximation space, the FE solution should match exactly (up to
    round-off).
    """

    @parametrize(
        "name,bounds,nrefine,vel_strs,pres_str,bndry_types,navier_stokes",
        STOKES_STEADY_CASES,
    )
    def test_manufactured_stokes(
        self,
        name: str,
        bounds: List[float],
        nrefine: int,
        vel_strs: List[str],
        pres_str: str,
        bndry_types: List[str],
        navier_stokes: bool,
    ) -> None:
        """Test manufactured solution for Stokes/NS equations."""
        bkd = NumpyBkd()
        nvars = len(bounds) // 2
        sol_strs = vel_strs + [pres_str]

        man_sol = ManufacturedStokes(
            sol_strs, nvars, navier_stokes=navier_stokes, bkd=bkd, oned=True
        )
        funcs = man_sol.functions

        # Create mesh
        if nvars == 1:
            nx = 10 * (2 ** nrefine)
            mesh = StructuredMesh1D(
                nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
            )
        else:
            nx = 5 * (2 ** nrefine)
            mesh = StructuredMesh2D(
                nx=nx, ny=nx,
                bounds=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
                bkd=bkd,
            )

        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)

        # Build physics from manufactured solution
        physics = _build_stokes_from_manufactured(
            funcs, nvars, vel_basis, pres_basis, bkd,
            navier_stokes=navier_stokes, transient=False,
        )

        # Solve
        solver = SteadyStateSolver(physics, tol=1e-12)
        if navier_stokes:
            init_guess = physics.init_guess()
            result = solver.solve(init_guess)
        else:
            result = solver.solve_linear()

        self.assertTrue(
            result.converged,
            f"Test {name}: solver did not converge "
            f"(residual={result.residual_norm:.2e})",
        )

        # Compare to exact solution
        u_num = bkd.to_numpy(result.solution)
        u_exact = _get_exact_state(
            funcs, nvars, vel_basis, pres_basis, bkd
        )

        max_err = np.max(np.abs(u_num - u_exact))
        self.assertLess(
            max_err, 5e-7,
            f"Test {name}: max_err={max_err:.2e} should be < 5e-7",
        )


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
        [0, 1], 1,
        ["((2*x-1))**2*(1+T)"],
        "x*(1-1e-16*x)*(1+T)",
        ["D", "D"],
        "backward_euler",
    ),
    # 1D Stokes, Crank-Nicolson
    (
        "1d_CN",
        [0, 1], 1,
        ["((2*x-1))**2*(1+T)"],
        "x*(1-1e-16*x)*(1+T)",
        ["D", "D"],
        "crank_nicolson",
    ),
    # 2D Stokes, backward Euler
    (
        "2d_BE",
        [0, 1, 0, 1], 0,
        ["(1e-16*x+y)*(1+T)", "((x+1)*(y+1))*(1+T)"],
        "(x*(1+1e-16*x)+1e-16*y*(1+1e-16*y))*(1+T)",
        ["D", "D", "D", "D"],
        "backward_euler",
    ),
    # 2D Stokes, Crank-Nicolson
    (
        "2d_CN",
        [0, 1, 0, 1], 0,
        ["(1e-16*x+y)*(1+T)", "((x+1)*(y+1))*(1+T)"],
        "(x*(1+1e-16*x)+1e-16*y*(1+1e-16*y))*(1+T)",
        ["D", "D", "D", "D"],
        "crank_nicolson",
    ),
]


class TestParametrizedTransientStokes(ParametrizedTestCase):
    """Parametrized transient Stokes manufactured solution tests.

    Uses time-linear solutions u(x,t) = u0(x)*(1+T) which are exactly
    reproduced by both backward Euler (1st order) and Crank-Nicolson
    (2nd order) with any time step size.

    For transient manufactured solutions, the Galerkin load uses the FULL
    forcing (including du/dT) because the weak form is:
        (du/dt, v) + a(u, v) = (f_full, v)
    where f_full = du/dt - Lap(u) + grad(p).
    """

    @parametrize(
        "name,bounds,nrefine,vel_strs,pres_str,bndry_types,method",
        STOKES_TRANSIENT_CASES,
    )
    def test_transient_stokes(
        self,
        name: str,
        bounds: List[float],
        nrefine: int,
        vel_strs: List[str],
        pres_str: str,
        bndry_types: List[str],
        method: str,
    ) -> None:
        """Test transient manufactured solution for Stokes equations."""
        bkd = NumpyBkd()
        nvars = len(bounds) // 2
        sol_strs = vel_strs + [pres_str]

        man_sol = ManufacturedStokes(
            sol_strs, nvars, navier_stokes=False, bkd=bkd, oned=True
        )
        funcs = man_sol.functions

        # Create mesh
        if nvars == 1:
            nx = 10 * (2 ** nrefine)
            mesh = StructuredMesh1D(
                nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
            )
        else:
            nx = 5 * (2 ** nrefine)
            mesh = StructuredMesh2D(
                nx=nx, ny=nx,
                bounds=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
                bkd=bkd,
            )

        vel_basis = VectorLagrangeBasis(mesh, degree=2)
        pres_basis = LagrangeBasis(mesh, degree=1)

        # Build physics with full (transient) forcing
        physics = _build_stokes_from_manufactured(
            funcs, nvars, vel_basis, pres_basis, bkd,
            navier_stokes=False, transient=True,
        )

        # Time stepping
        stepper = StokesTimeStepResidual(physics, method=method)
        y = bkd.asarray(
            _get_exact_state(
                funcs, nvars, vel_basis, pres_basis, bkd,
                time=0.0, transient=True,
            )
        )

        dt = 0.1
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)
            y = stepper.solve_step()
            t += dt

        # Compare to exact at final time
        y_exact = _get_exact_state(
            funcs, nvars, vel_basis, pres_basis, bkd,
            time=t, transient=True,
        )
        max_err = np.max(np.abs(bkd.to_numpy(y) - y_exact))

        self.assertLess(
            max_err, 5e-7,
            f"Test {name}: max_err={max_err:.2e} should be < 5e-7",
        )


from pyapprox.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
