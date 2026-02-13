"""Transient hyperelasticity manufactured solution tests.

Tests backward Euler and Crank-Nicolson time stepping for
HyperelasticityPhysics using manufactured solutions with
time-dependent displacement fields.

NumPy only — skfem assembly at each nonlinear step is numpy-based.
"""

import unittest
from typing import List, Tuple

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics import HyperelasticityPhysics
from pyapprox.typing.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.typing.pde.galerkin.manufactured.adapter import (
    create_hyperelasticity_manufactured_test,
    GalerkinHyperelasticityAdapter,
)
from pyapprox.typing.pde.galerkin.time_integration import (
    GalerkinPhysicsODEAdapter,
    ConstrainedTimeStepResidual,
)
from pyapprox.typing.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)
from pyapprox.typing.optimization.rootfinding.newton import NewtonSolver
from pyapprox.typing.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_vector_dirichlet_value_func(sol_func, ndim):
    """Create a DirichletBC value_func for vector basis from manufactured solution.

    The manufactured solution returns (npts, ncomponents).
    For vector DOFs (interleaved), DOF j corresponds to component j % ndim.
    """
    def value_func(coords, time):
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords, time)  # (nbndry_dofs, ncomponents)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


def _get_exact_displacement(funcs, basis, bkd, time):
    """Evaluate manufactured solution at DOF locations for vector basis."""
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()

    sol = funcs["solution"](dof_coords, time)  # (ndofs, ncomponents)

    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]

    return exact


# =========================================================================
# 1D Transient Tests
# =========================================================================


TRANSIENT_1D_CASES: List[Tuple[str, str]] = [
    ("BE", "backward_euler"),
    ("CN", "crank_nicolson"),
]


class TestTransientHyperelasticity1D(ParametrizedTestCase):
    """Transient 1D hyperelasticity with manufactured solutions.

    Uses time-linear solution u(x,t) = u0(x)*(1+T) which is exactly
    reproduced by both backward Euler and Crank-Nicolson.
    """

    @parametrize("name,method", TRANSIENT_1D_CASES)
    def test_transient_1d(self, name: str, method: str) -> None:
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        sol_strs = ["0.1*x**2*(1-x)**2*(1+T)"]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0], sol_strs=sol_strs,
            stress_model=stress, bkd=bkd,
        )

        mesh = StructuredMesh1D(nx=30, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=2)

        adapter = GalerkinHyperelasticityAdapter(
            basis, functions, bkd, time_dependent=True,
        )
        body_force = adapter.forcing_for_galerkin()

        sol_func = functions["solution"]
        value_func = _make_vector_dirichlet_value_func(sol_func, nvars)
        bc_list = [
            DirichletBC(basis, bname, value_func, bkd)
            for bname in ["left", "right"]
        ]

        physics = HyperelasticityPhysics(
            basis=basis, stress_model=stress, bkd=bkd,
            body_force=body_force, boundary_conditions=bc_list,
        )

        # Time stepping
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        if method == "backward_euler":
            stepper = BackwardEulerResidual(ode_adapter)
        else:
            stepper = CrankNicolsonResidual(ode_adapter)
        constrained = ConstrainedTimeStepResidual(stepper, ode_adapter)

        newton = NewtonSolver(constrained)
        newton.set_options(maxiters=20, atol=1e-10, rtol=0.0)

        y = bkd.asarray(
            _get_exact_displacement(functions, basis, bkd, time=0.0)
        )

        dt = 0.1
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            stepper.set_time(t, dt, y)
            constrained.set_bc_time(t_np1)

            # Inject Dirichlet values into initial guess
            d_dofs, d_vals = ode_adapter.dirichlet_dof_info(t_np1)
            d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
            guess = bkd.copy(y)
            if len(d_dofs_np) > 0:
                guess_np = bkd.to_numpy(guess).copy()
                guess_np[d_dofs_np] = bkd.to_numpy(d_vals)
                guess = bkd.asarray(guess_np.astype(np.float64))

            y = newton.solve(guess)
            t = t_np1

        # Compare to exact at final time
        u_exact_final = _get_exact_displacement(
            functions, basis, bkd, time=t,
        )
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 1e-5,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-5",
        )


# =========================================================================
# 2D Transient Tests
# =========================================================================


TRANSIENT_2D_CASES: List[Tuple[str, str]] = [
    ("BE", "backward_euler"),
    ("CN", "crank_nicolson"),
]


class TestTransientHyperelasticity2D(ParametrizedTestCase):
    """Transient 2D hyperelasticity with manufactured solutions.

    Uses time-linear solution u(x,y,t) = u0(x,y)*(1+T) which is exactly
    reproduced by both backward Euler and Crank-Nicolson.
    """

    @parametrize("name,method", TRANSIENT_2D_CASES)
    @slow_test
    def test_transient_2d(self, name: str, method: str) -> None:
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        sol_strs = [
            "0.1*x**2*(1-x)**2*y**2*(1-y)**2*(1+T)",
            "0.05*x**2*(1-x)**2*y**2*(1-y)**2*(1+T)",
        ]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0], sol_strs=sol_strs,
            stress_model=stress, bkd=bkd,
        )

        mesh = StructuredMesh2D(
            nx=12, ny=12,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=2)

        adapter = GalerkinHyperelasticityAdapter(
            basis, functions, bkd, time_dependent=True,
        )
        body_force = adapter.forcing_for_galerkin()

        sol_func = functions["solution"]
        value_func = _make_vector_dirichlet_value_func(sol_func, nvars)
        bc_list = [
            DirichletBC(basis, bname, value_func, bkd)
            for bname in ["left", "right", "bottom", "top"]
        ]

        physics = HyperelasticityPhysics(
            basis=basis, stress_model=stress, bkd=bkd,
            body_force=body_force, boundary_conditions=bc_list,
        )

        # Time stepping
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        if method == "backward_euler":
            stepper = BackwardEulerResidual(ode_adapter)
        else:
            stepper = CrankNicolsonResidual(ode_adapter)
        constrained = ConstrainedTimeStepResidual(stepper, ode_adapter)

        newton = NewtonSolver(constrained)
        newton.set_options(maxiters=20, atol=1e-10, rtol=0.0)

        y = bkd.asarray(
            _get_exact_displacement(functions, basis, bkd, time=0.0)
        )

        dt = 0.1
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            t_np1 = t + dt
            stepper.set_time(t, dt, y)
            constrained.set_bc_time(t_np1)

            # Inject Dirichlet values into initial guess
            d_dofs, d_vals = ode_adapter.dirichlet_dof_info(t_np1)
            d_dofs_np = bkd.to_numpy(d_dofs).astype(np.intp)
            guess = bkd.copy(y)
            if len(d_dofs_np) > 0:
                guess_np = bkd.to_numpy(guess).copy()
                guess_np[d_dofs_np] = bkd.to_numpy(d_vals)
                guess = bkd.asarray(guess_np.astype(np.float64))

            y = newton.solve(guess)
            t = t_np1

        # Compare to exact at final time
        u_exact_final = _get_exact_displacement(
            functions, basis, bkd, time=t,
        )
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        self.assertLess(
            rel_error, 2e-4,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 2e-4",
        )


if __name__ == "__main__":
    unittest.main()
