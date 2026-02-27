"""Transient linear elasticity manufactured solution tests.

Tests backward Euler and Crank-Nicolson time stepping for LinearElasticity
using manufactured solutions with time-dependent displacement fields.
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import List, Tuple

import numpy as np

from pyapprox.optimization.rootfinding.newton import NewtonSolver
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.manufactured.adapter import (
    create_elasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity as LinearElasticity,
)
from pyapprox.pde.galerkin.time_integration import (
    ConstrainedTimeStepResidual,
    GalerkinPhysicsODEAdapter,
)
from pyapprox.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)
from pyapprox.util.backends.numpy import NumpyBkd

# =========================================================================
# Helpers
# =========================================================================


def _make_vector_dirichlet_value_func(sol_func, ndim):
    """Create a DirichletBC value_func for vector basis from manufactured solution.

    The manufactured solution returns (npts, ncomponents).
    For vector DOFs (interleaved), each DOF has a coordinate and the
    value at that DOF is the appropriate component based on DOF index.

    However, DirichletBC.boundary_values() calls value_func(bndry_coords, time)
    where bndry_coords are the DOF coordinates for ALL boundary DOFs.
    For a vector basis with get_dofs(), boundary DOFs for component idx
    have the pattern: DOF i corresponds to component i % ndim.

    The value_func must return (nbndry_dofs,) with the correct component
    value for each DOF.
    """

    def value_func(coords, time):
        # coords: (ndim, nbndry_dofs) — coordinates of boundary DOFs
        # For vector basis, DOFs are interleaved, so DOF j corresponds
        # to component j % ndim
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords, time)  # (nbndry_dofs, ncomponents)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


def _get_exact_displacement(funcs, basis, bkd, time):
    """Evaluate manufactured solution at DOF locations for vector basis.

    Returns the exact DOF values as a flat interleaved array.
    """
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()

    sol = funcs["solution"](dof_coords, time)  # (ndofs, ncomponents)

    # Extract correct component for each interleaved DOF
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]

    return exact


# =========================================================================
# Parametrized transient elasticity tests
# =========================================================================


# Format: (name, method)
TRANSIENT_ELASTICITY_2D_CASES: List[Tuple[str, str]] = [
    ("DD_BE", "backward_euler"),
    ("DD_CN", "crank_nicolson"),
]


class TestTransientElasticity2D:
    """Transient 2D linear elasticity with manufactured solutions.

    Uses time-linear solutions u(x,y,t) = u0(x,y)*(1+T) which are
    exactly reproduced by both backward Euler and Crank-Nicolson.
    """

    @pytest.mark.parametrize(
        "name,method",
        TRANSIENT_ELASTICITY_2D_CASES,
    )
    def test_transient_elasticity_2d(
        self,
        numpy_bkd,
        name: str,
        method: str,
    ) -> None:
        bkd = numpy_bkd
        bounds = [0.0, 1.0, 0.0, 1.0]
        sol_strs = [
            "(1-x)*x*(1-y)*y*(1+T)",
            "(1-x)*x*(1-y)*y*(1+T)",
        ]
        lambda_str = "1.0"
        mu_str = "1.0"

        functions, nvars = create_elasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            lambda_str=lambda_str,
            mu_str=mu_str,
            bkd=bkd,
        )

        # Mesh and basis
        mesh = StructuredMesh2D(
            nx=10,
            ny=10,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=2)

        # Forcing function for the physics: body_force(x, time)
        forcing_func = functions["forcing"]

        def body_force(x, time):
            vals = forcing_func(x, time)  # (npts, 2)
            return vals.T  # (2, npts) — physics expects (ndim, npts)

        # Dirichlet BCs on all 4 boundaries
        bndry_names = ["left", "right", "bottom", "top"]
        sol_func = functions["solution"]
        value_func = _make_vector_dirichlet_value_func(sol_func, nvars)

        bc_list = []
        for bname in bndry_names:
            bc = DirichletBC(
                basis=basis,
                boundary_name=bname,
                value_func=value_func,
                bkd=bkd,
            )
            bc_list.append(bc)

        # Create physics
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=2.5,  # E computed from lambda=1, mu=1
            poisson_ratio=0.25,  # nu computed from lambda=1, mu=1
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )

        # Verify Lame parameters match
        assert abs(physics.lame_lambda() - 1.0) < 1e-10
        assert abs(physics.lame_mu() - 1.0) < 1e-10

        # Time stepping with constrained wrapper
        ode_adapter = GalerkinPhysicsODEAdapter(physics)

        if method == "backward_euler":
            stepper = BackwardEulerResidual(ode_adapter)
        else:
            stepper = CrankNicolsonResidual(ode_adapter)
        constrained = ConstrainedTimeStepResidual(stepper, ode_adapter)

        newton = NewtonSolver(constrained)
        newton.set_options(maxiters=20, atol=1e-10, rtol=0.0)

        y = bkd.asarray(_get_exact_displacement(functions, basis, bkd, time=0.0))

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
        u_exact_final = _get_exact_displacement(functions, basis, bkd, time=t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        assert rel_error < 1e-5
