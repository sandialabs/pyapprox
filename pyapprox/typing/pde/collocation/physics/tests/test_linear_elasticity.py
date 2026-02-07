"""Transient tests for 2D linear elasticity physics.

Uses quadratic-in-time manufactured solutions so Crank-Nicolson (2nd order)
integrates the time derivative exactly while backward Euler (1st order) has
O(dt) temporal error.
"""

import unittest
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.typing.pde.collocation.mesh import (
    create_uniform_mesh_2d,
    TransformedMesh2D,
)
from pyapprox.typing.pde.collocation.boundary import zero_dirichlet_bc
from pyapprox.typing.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.typing.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedLinearElasticityEquations,
)


class TestLinearElasticityTransient(Generic[Array], unittest.TestCase):
    """Test transient linear elasticity with manufactured solutions.

    Solution: u = (1-x^2)(1-y^2)(1+T+T^2), v = (1-x^2)(1-y^2)x(1+T+T^2)
    Homogeneous Dirichlet on all boundaries. Quadratic in time so
    du/dt = f(x)(1+2T), which CN integrates exactly but BE does not.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def _setup_transient_elasticity(self):
        """Create elasticity physics with time-dependent manufactured solution.

        Returns physics, manufactured solution, nodes, and backend.
        """
        bkd = self.bkd()
        npts_1d = 8
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        mesh_obj = create_uniform_mesh_2d(
            (npts_1d, npts_1d), (-1.0, 1.0, -1.0, 1.0), bkd
        )
        npts = basis.npts()

        nodes_x = basis.nodes_x()
        nodes_y = basis.nodes_y()
        xx, yy = bkd.meshgrid(nodes_x, nodes_y, indexing="xy")
        nodes = bkd.stack([xx.flatten(), yy.flatten()], axis=0)

        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=[
                "(1 - x**2)*(1 - y**2)*(1 + T + T**2)",
                "(1 - x**2)*(1 - y**2)*x*(1 + T + T**2)",
            ],
            nvars=2,
            lambda_str="1.0",
            mu_str="1.0",
            bkd=bkd,
            oned=True,
        )

        def forcing_fn(t):
            forcing = man_sol.functions["forcing"](nodes, t)
            return bkd.concatenate([forcing[:, 0], forcing[:, 1]])

        physics = LinearElasticityPhysics(
            basis, bkd, lamda=1.0, mu=1.0, forcing=forcing_fn
        )

        # Homogeneous Dirichlet on all 4 sides for both u and v
        bcs = []
        for side in range(4):
            boundary_idx = mesh_obj.boundary_indices(side)
            bc_u = zero_dirichlet_bc(bkd, boundary_idx)
            bcs.append(bc_u)
            boundary_idx_v = bkd.asarray(
                [idx + npts for idx in boundary_idx]
            )
            bc_v = zero_dirichlet_bc(bkd, boundary_idx_v)
            bcs.append(bc_v)
        physics.set_boundary_conditions(bcs)

        return physics, man_sol, nodes, npts, bkd

    def _run_transient_elasticity(self, method, atol):
        physics, man_sol, nodes, npts, bkd = (
            self._setup_transient_elasticity()
        )
        model = CollocationModel(physics, bkd)

        # Initial condition at t=0
        u_exact_0 = man_sol.functions["solution"](nodes, 0.0)
        u0 = bkd.concatenate([u_exact_0[:, 0], u_exact_0[:, 1]])

        final_time = 0.1
        config = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=final_time,
            deltat=0.01,
        )

        solutions, times = model.solve_transient(u0, config)
        t_final = float(bkd.to_numpy(times[-1]))

        u_exact_final = man_sol.functions["solution"](nodes, t_final)
        u_exact_flat = bkd.concatenate(
            [u_exact_final[:, 0], u_exact_final[:, 1]]
        )

        bkd.assert_allclose(solutions[:, -1], u_exact_flat, atol=atol)

    def test_transient_backward_euler(self):
        """Test transient elasticity with backward Euler.

        BE is 1st order: O(dt) temporal error for quadratic-in-time solution.
        """
        self._run_transient_elasticity("backward_euler", atol=0.01)

    def test_transient_crank_nicolson(self):
        """Test transient elasticity with Crank-Nicolson.

        CN is 2nd order: exact for quadratic-in-time solution.
        """
        self._run_transient_elasticity("crank_nicolson", atol=1e-8)


class TestLinearElasticityTransientNumpy(TestLinearElasticityTransient):
    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
