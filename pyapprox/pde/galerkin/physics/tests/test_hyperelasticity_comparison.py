"""Galerkin vs Collocation comparison tests for hyperelasticity.

Both methods solve the same manufactured problem on [0,1]^d with
homogeneous Dirichlet BCs and Neo-Hookean material. Each solution
is compared to the exact manufactured solution. If both recover
the exact solution to high accuracy, they implicitly agree.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

# -- Collocation imports --
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.pde.collocation.manufactured_solutions.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    AffineTransform2D,
    TransformedMesh1D,
    TransformedMesh2D,
    create_uniform_mesh_1d,
    create_uniform_mesh_2d,
)
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics as CollocationHyperelasticityPhysics,
)

# -- Shared --
from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.collocation.time_integration import CollocationModel
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinHyperelasticityAdapter,
    create_hyperelasticity_manufactured_test,
)

# -- Galerkin imports --
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.pde.galerkin.physics import (
    HyperelasticityPhysics as GalerkinHyperelasticityPhysics,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401

# =========================================================================
# Helpers
# =========================================================================


def _solve_collocation_1d(sol_strs, stress, npts, bkd):
    """Solve 1D hyperelasticity with collocation and return (u_exact, u_num)."""
    transform = AffineTransform1D((0.0, 1.0), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_bc = create_uniform_mesh_1d(npts, (0.0, 1.0), bkd)
    nodes = mesh.points()

    man_sol = ManufacturedHyperelasticityEquations(
        sol_strs=sol_strs,
        nvars=1,
        stress_model=stress,
        bkd=bkd,
        oned=True,
    )

    u_exact = man_sol.functions["solution"](nodes)[:, 0]
    forcing = man_sol.functions["forcing"](nodes)[:, 0]

    physics = CollocationHyperelasticityPhysics(
        basis,
        bkd,
        stress,
        forcing=lambda t: forcing,
    )

    bcs = []
    for side in range(mesh_bc.nboundaries()):
        boundary_idx = mesh_bc.boundary_indices(side)
        bc_idx = bkd.asarray([int(idx) for idx in boundary_idx])
        bcs.append(zero_dirichlet_bc(bkd, bc_idx))
    physics.set_boundary_conditions(bcs)

    model = CollocationModel(physics, bkd)
    u_num = model.solve_steady(bkd.zeros((npts,)), tol=1e-10, maxiter=50)

    return bkd.to_numpy(u_exact), bkd.to_numpy(u_num)


def _solve_collocation_2d(sol_strs, stress, npts_x, npts_y, bkd):
    """Solve 2D hyperelasticity with collocation and return (u_exact, u_num)."""
    transform = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
    mesh = TransformedMesh2D(npts_x, npts_y, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    mesh_bc = create_uniform_mesh_2d(
        (npts_x, npts_y),
        (0.0, 1.0, 0.0, 1.0),
        bkd,
    )
    nodes = mesh.points()
    npts = basis.npts()
    ndim = 2

    man_sol = ManufacturedHyperelasticityEquations(
        sol_strs=sol_strs,
        nvars=2,
        stress_model=stress,
        bkd=bkd,
        oned=True,
    )

    u_exact_raw = man_sol.functions["solution"](nodes)
    forcing_raw = man_sol.functions["forcing"](nodes)
    u_exact_flat = bkd.concatenate([u_exact_raw[:, 0], u_exact_raw[:, 1]])
    forcing_flat = bkd.concatenate([forcing_raw[:, 0], forcing_raw[:, 1]])

    physics = CollocationHyperelasticityPhysics(
        basis,
        bkd,
        stress,
        forcing=lambda t: forcing_flat,
    )

    bcs = []
    for side in range(mesh_bc.nboundaries()):
        boundary_idx = mesh_bc.boundary_indices(side)
        for comp in range(ndim):
            bc_idx = bkd.asarray([int(idx) + comp * npts for idx in boundary_idx])
            bcs.append(zero_dirichlet_bc(bkd, bc_idx))
    physics.set_boundary_conditions(bcs)

    model = CollocationModel(physics, bkd)
    u_num = model.solve_steady(
        bkd.zeros((ndim * npts,)),
        tol=1e-10,
        maxiter=50,
    )

    return bkd.to_numpy(u_exact_flat), bkd.to_numpy(u_num)


def _make_vector_dirichlet_value_func(sol_func, ndim):
    """Create a DirichletBC value_func for vector basis."""

    def value_func(coords, time=0.0):
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


def _get_exact_displacement_galerkin(funcs, basis, bkd):
    """Evaluate manufactured solution at Galerkin DOF locations."""
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()
    sol = funcs["solution"](dof_coords)
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]
    return exact


def _solve_galerkin_1d(sol_strs, stress, nx, degree, bkd):
    """Solve 1D hyperelasticity with Galerkin and return (u_exact, u_num)."""
    functions, nvars = create_hyperelasticity_manufactured_test(
        bounds=[0.0, 1.0],
        sol_strs=sol_strs,
        stress_model=stress,
        bkd=bkd,
    )

    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = VectorLagrangeBasis(mesh, degree=degree)

    adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
    body_force = adapter.forcing_for_galerkin()
    value_func = _make_vector_dirichlet_value_func(
        functions["solution"],
        nvars,
    )
    bc_list = [DirichletBC(basis, name, value_func, bkd) for name in ["left", "right"]]

    physics = GalerkinHyperelasticityPhysics(
        basis=basis,
        stress_model=stress,
        bkd=bkd,
        body_force=body_force,
        boundary_conditions=bc_list,
    )

    exact = _get_exact_displacement_galerkin(functions, basis, bkd)
    solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
    result = solver.solve(bkd.asarray(exact + 0.01))

    return exact, bkd.to_numpy(result.solution), result.converged


def _solve_galerkin_2d(sol_strs, stress, nx, ny, degree, bkd):
    """Solve 2D hyperelasticity with Galerkin and return (u_exact, u_num)."""
    functions, nvars = create_hyperelasticity_manufactured_test(
        bounds=[0.0, 1.0, 0.0, 1.0],
        sol_strs=sol_strs,
        stress_model=stress,
        bkd=bkd,
    )

    mesh = StructuredMesh2D(
        nx=nx,
        ny=ny,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=degree)

    adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
    body_force = adapter.forcing_for_galerkin()
    value_func = _make_vector_dirichlet_value_func(
        functions["solution"],
        nvars,
    )
    bc_list = [
        DirichletBC(basis, name, value_func, bkd)
        for name in ["left", "right", "bottom", "top"]
    ]

    physics = GalerkinHyperelasticityPhysics(
        basis=basis,
        stress_model=stress,
        bkd=bkd,
        body_force=body_force,
        boundary_conditions=bc_list,
    )

    exact = _get_exact_displacement_galerkin(functions, basis, bkd)
    solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
    result = solver.solve(bkd.asarray(exact + 0.005))

    return exact, bkd.to_numpy(result.solution), result.converged


# =========================================================================
# 1D Comparison
# =========================================================================


class TestHyperelasticityComparison1D(Generic[Array], unittest.TestCase):
    """Compare Galerkin and Collocation 1D hyperelasticity solvers."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    @slow_test
    def test_both_recover_exact_1d(self) -> None:
        """Both methods recover the exact 1D MMS solution."""
        bkd = self.bkd()
        stress = NeoHookeanStress(1.0, 1.0)
        sol_strs = ["0.1*x**2*(1-x)**2"]

        # Collocation: npts=20 Chebyshev points
        coll_exact, coll_num = _solve_collocation_1d(
            sol_strs,
            stress,
            npts=20,
            bkd=bkd,
        )
        coll_norm = np.linalg.norm(coll_exact)
        coll_rel_err = np.linalg.norm(coll_num - coll_exact) / coll_norm

        # Galerkin: nx=30 elements, degree=2
        gal_exact, gal_num, gal_conv = _solve_galerkin_1d(
            sol_strs,
            stress,
            nx=30,
            degree=2,
            bkd=bkd,
        )
        self.assertTrue(gal_conv, "Galerkin Newton did not converge")
        gal_norm = np.linalg.norm(gal_exact)
        gal_rel_err = np.linalg.norm(gal_num - gal_exact) / gal_norm

        self.assertLess(
            coll_rel_err,
            1e-6,
            f"Collocation rel error too large: {coll_rel_err:.2e}",
        )
        self.assertLess(
            gal_rel_err,
            1e-6,
            f"Galerkin rel error too large: {gal_rel_err:.2e}",
        )


class TestHyperelasticityComparison1DNumpy(
    TestHyperelasticityComparison1D[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


try:
    import torch

    from pyapprox.util.backends.torch import TorchBkd

    class TestHyperelasticityComparison1DTorch(
        TestHyperelasticityComparison1D[torch.Tensor]
    ):
        __test__ = True

        def setUp(self) -> None:
            torch.set_default_dtype(torch.float64)

        def bkd(self) -> TorchBkd:
            return TorchBkd()

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_both_recover_exact_1d(self) -> None:
            super().test_both_recover_exact_1d()

except ImportError:
    pass


# =========================================================================
# 2D Comparison
# =========================================================================


class TestHyperelasticityComparison2D(Generic[Array], unittest.TestCase):
    """Compare Galerkin and Collocation 2D hyperelasticity solvers."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    @slow_test
    def test_both_recover_exact_2d(self) -> None:
        """Both methods recover the exact 2D MMS solution."""
        bkd = self.bkd()
        stress = NeoHookeanStress(1.0, 1.0)
        sol_strs = [
            "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
            "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
        ]

        # Collocation: 10x10 Chebyshev points
        coll_exact, coll_num = _solve_collocation_2d(
            sol_strs,
            stress,
            npts_x=10,
            npts_y=10,
            bkd=bkd,
        )
        coll_norm = np.linalg.norm(coll_exact)
        coll_rel_err = np.linalg.norm(coll_num - coll_exact) / coll_norm

        # Galerkin: 12x12 elements, degree=2
        gal_exact, gal_num, gal_conv = _solve_galerkin_2d(
            sol_strs,
            stress,
            nx=12,
            ny=12,
            degree=2,
            bkd=bkd,
        )
        self.assertTrue(gal_conv, "Galerkin Newton did not converge")
        gal_norm = np.linalg.norm(gal_exact)
        gal_rel_err = np.linalg.norm(gal_num - gal_exact) / gal_norm

        self.assertLess(
            coll_rel_err,
            1e-6,
            f"Collocation rel error too large: {coll_rel_err:.2e}",
        )
        self.assertLess(
            gal_rel_err,
            1e-4,
            f"Galerkin rel error too large: {gal_rel_err:.2e}",
        )


class TestHyperelasticityComparison2DNumpy(
    TestHyperelasticityComparison2D[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


try:
    import torch

    from pyapprox.util.backends.torch import TorchBkd

    class TestHyperelasticityComparison2DTorch(
        TestHyperelasticityComparison2D[torch.Tensor]
    ):
        __test__ = True

        def setUp(self) -> None:
            torch.set_default_dtype(torch.float64)

        def bkd(self) -> TorchBkd:
            return TorchBkd()

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_both_recover_exact_2d(self) -> None:
            super().test_both_recover_exact_2d()

except ImportError:
    pass
