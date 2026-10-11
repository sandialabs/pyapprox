"""Boundary condition tests for HyperelasticityPhysics.

Tests Neumann, Robin, and mixed BC combinations in 1D and 2D using
manufactured solutions with non-zero boundary values.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.boundary import BoundaryConditionRole
from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_hyperelasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.pde.galerkin.physics import HyperelasticityPhysics
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.util.backends.numpy import NumpyBkd

_BOUNDARY_NAMES = ["left", "right", "bottom", "top", "front", "back"]

_Problem = Tuple[
    GalerkinSystem[NDArray[Any]],
    Dict[str, Callable[..., Any]],
    VectorLagrangeBasis[NDArray[Any]],
]


def _dirichlet(
    ms: GalerkinManufacturedSolution[NDArray[Any]],
    name: str,
    alpha: float,
    law: NormalFluxLawProtocol,
) -> BoundaryConditionRole[NDArray[Any]]:
    return ms.dirichlet(name)


def _neumann(
    ms: GalerkinManufacturedSolution[NDArray[Any]],
    name: str,
    alpha: float,
    law: NormalFluxLawProtocol,
) -> BoundaryConditionRole[NDArray[Any]]:
    return ms.neumann(name, law)


def _robin(
    ms: GalerkinManufacturedSolution[NDArray[Any]],
    name: str,
    alpha: float,
    law: NormalFluxLawProtocol,
) -> BoundaryConditionRole[NDArray[Any]]:
    return ms.robin(name, alpha, law)


# Test-data codes for the boundary kinds, in boundary order.
_BC_BUILDERS: Dict[
    str,
    Callable[
        [GalerkinManufacturedSolution[NDArray[Any]], str, float, NormalFluxLawProtocol],
        BoundaryConditionRole[NDArray[Any]],
    ],
] = {"D": _dirichlet, "N": _neumann, "R": _robin}


def _manufactured_system(
    basis: VectorLagrangeBasis[NDArray[Any]],
    functions: Dict[str, Callable[..., Any]],
    stress: NeoHookeanStress,
    bc_types: List[str],
    robin_alpha: float,
    bkd: NumpyBkd,
) -> GalerkinSystem[NDArray[Any]]:
    """Hyperelasticity with the manufactured forcing and conditions."""
    ms = GalerkinManufacturedSolution(basis, functions, bkd)
    physics = HyperelasticityPhysics(
        basis=basis,
        stress_model=stress,
        bkd=bkd,
        body_force=ms.forcing(),
    )
    law = physics.flux_law()
    bcs = [
        _BC_BUILDERS[code](ms, name, robin_alpha, law)
        for code, name in zip(bc_types, _BOUNDARY_NAMES)
    ]
    return compose_galerkin_system(physics, bcs)


def _get_exact_displacement(funcs, basis, bkd):
    """Evaluate manufactured solution at DOF locations for vector basis."""
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()
    sol = funcs["solution"](dof_coords)
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]
    return exact


# =========================================================================
# 1D BC Tests
# =========================================================================


class TestHyperelasticityBCs1DBase:
    """1D boundary condition tests with non-zero boundary values."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Solution non-zero at x=1: u(0)=0, u(1)=0.05
        self._sol_strs = ["0.1*x**2*(1-x) + 0.05*x"]

    def _setup_problem(
        self,
        bkd: NumpyBkd,
        bc_types: List[str],
        nx: int = 30,
        degree: int = 2,
        robin_alpha: float = 1.0,
    ) -> _Problem:
        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        system = _manufactured_system(
            basis, functions, self._stress, bc_types, robin_alpha, bkd
        )
        return system, functions, basis

    def _check_newton_solve(self, bkd, system, functions, basis, tol=1e-4) :
        exact = _get_exact_displacement(functions, basis, bkd)
        solver = SteadyStateSolver(
            system.steady(), tol=1e-10, max_iter=20, line_search=True
        )
        init_guess = bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        assert result.converged, f"Newton did not converge: {result.residual_norm:.2e}"
        u_np = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        assert rel_error < tol

    def test_bc_dirichlet_neumann_1d(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet left, Neumann right."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "N"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_dirichlet_robin_1d(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet left, Robin right."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "R"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_robin_dirichlet_1d(self, numpy_bkd: NumpyBkd) -> None:
        """Robin left, Dirichlet right."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["R", "D"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_residual_at_exact_neumann_1d(self, numpy_bkd: NumpyBkd) -> None:
        """Residual at exact solution should be small with Neumann BC."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "N"])
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = system.steady_snapshot(0.0).steady_residual(state)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3




# =========================================================================
# 2D BC Tests
# =========================================================================


class TestHyperelasticityBCs2DBase:
    """2D boundary condition tests with non-zero boundary values."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Non-zero on right/top: u vanishes at x=0, y=0 but not at x=1, y=1
        self._sol_strs = [
            "0.1*x**2*(1-x)*y**2*(1-y) + 0.02*x*y",
            "0.05*x**2*(1-x)*y**2*(1-y) + 0.01*x*y",
        ]

    def _setup_problem(
        self,
        bkd: NumpyBkd,
        bc_types: List[str],
        nx: int = 12,
        ny: int = 12,
        degree: int = 2,
        robin_alpha: float = 1.0,
    ) -> _Problem:
        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh2D(nx=nx, ny=ny, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        system = _manufactured_system(
            basis, functions, self._stress, bc_types, robin_alpha, bkd
        )
        return system, functions, basis

    def _check_newton_solve(self, bkd, system, functions, basis, tol=1e-3) :
        exact = _get_exact_displacement(functions, basis, bkd)
        solver = SteadyStateSolver(
            system.steady(), tol=1e-10, max_iter=20, line_search=True
        )
        init_guess = bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        assert result.converged, f"Newton did not converge: {result.residual_norm:.2e}"
        u_np = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        assert rel_error < tol

    def test_bc_mixed_DN_2d(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet left/bottom, Neumann right/top."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "N"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_mixed_DR_2d(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet left/bottom, Robin right/top."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "R", "D", "R"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_mixed_DNR_2d(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet left, Neumann right, Dirichlet bottom, Robin top."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "R"])
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_residual_at_exact_mixed_2d(self, numpy_bkd: NumpyBkd) -> None:
        """Residual at exact solution should be small with mixed BCs."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "R"])
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = system.steady_snapshot(0.0).steady_residual(state)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3




# =========================================================================
# 3D BC Tests
# =========================================================================


class TestHyperelasticityBCs3DBase:
    """3D boundary condition tests with non-zero boundary values.

    All prior 3D coverage was Dirichlet-only; these exercise the
    traction (Neumann) and Robin paths, whose data t = P.n comes from the
    physics' flux law with the facet normals of the 6 faces.
    """

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)
        # quadratic solutions in the degree-2 FE space, nonzero on all faces
        self._sol_strs = [
            "0.02*x*y + 0.01*z**2",
            "0.01*x**2 - 0.02*y*z",
            "0.005*x*z + 0.015*y**2",
        ]

    def _setup_problem(
        self,
        bkd: NumpyBkd,
        bc_types: List[str],
        nx: int = 2,
        degree: int = 2,
        robin_alpha: float = 1.0,
    ) -> _Problem:
        from pyapprox.pde.galerkin.mesh import StructuredMesh3D

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh3D(
            nx=nx,
            ny=nx,
            nz=nx,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=degree)
        system = _manufactured_system(
            basis, functions, self._stress, bc_types, robin_alpha, bkd
        )
        return system, functions, basis

    def _check_newton_solve(self, bkd, system, functions, basis, tol=1e-3):
        exact = _get_exact_displacement(functions, basis, bkd)
        solver = SteadyStateSolver(
            system.steady(), tol=1e-10, max_iter=20, line_search=True
        )
        init_guess = bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        assert result.converged, f"Newton did not converge: {result.residual_norm:.2e}"
        u_np = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        assert rel_error < tol

    def test_bc_mixed_DN_3d(self, numpy_bkd: NumpyBkd) -> None:
        """Neumann traction on the right face, Dirichlet elsewhere."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(
            bkd, ["D", "N", "D", "D", "D", "D"]
        )
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_mixed_DNR_3d(self, numpy_bkd: NumpyBkd) -> None:
        """Neumann on right, Robin on back, Dirichlet elsewhere."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(
            bkd, ["D", "N", "D", "D", "D", "R"]
        )
        self._check_newton_solve(bkd, system, functions, basis)

    def test_bc_residual_at_exact_mixed_3d(self, numpy_bkd: NumpyBkd) -> None:
        """Residual at exact solution should be small with mixed BCs."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_problem(
            bkd, ["D", "N", "D", "D", "D", "R"]
        )
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = system.steady_snapshot(0.0).steady_residual(state)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3
