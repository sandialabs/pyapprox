"""Periodic Burgers on the endpoint-identified 1D mesh.

The discriminating checks:
- the assembled mass matrix is full rank with row sums integrating to
  the domain length (no dangling identified dof);
- discrete conservation: with no forcing the constant test function
  annihilates the residual (momentum rate zero, up to ASSEMBLY scale,
  not machine epsilon) and forward Euler conserves total momentum
  1^T M u_k exactly per step;
- the physics runs the transient drivers with NO boundary conditions
  (empty Dirichlet info);
- a manufactured-solution convergence gate: the exact space-periodic
  solution is recovered at the FEM rate under the SAME periodic
  boundary treatment the OpInf benchmarks use.

Numpy appears ONLY at the skfem seam (sparse mass-matrix contractions,
rank checks); state construction and analysis are backend-generic.
"""

import math
import sys

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.collocation.manufactured_solutions.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import PeriodicStructuredMesh1D
from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.util.backends.numpy import NumpyBkd


def _make_periodic_burgers(bkd, nx, viscosity=0.1, degree=1, forcing=None):
    mesh = PeriodicStructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    physics = BurgersPhysics(
        basis=basis, viscosity=viscosity, bkd=bkd, forcing=forcing
    )
    return basis, physics


def _mass_row_sums(physics, bkd):
    """1^T M as a backend array -- the single skfem-seam contraction."""
    return bkd.asarray(
        np.asarray(physics.mass_matrix().sum(axis=0)).ravel()
    )


def _wave_state(basis, bkd, offset):
    coords = basis.dof_coordinates()
    return offset + bkd.sin(2.0 * math.pi * coords[0])


class TestPeriodicAssembly:
    @pytest.mark.parametrize("degree", [1, 2])
    def test_mass_matrix_full_rank_and_partition_of_unity(self, degree):
        bkd = NumpyBkd()
        _, physics = _make_periodic_burgers(bkd, nx=16, degree=degree)
        assert isinstance(physics, GalerkinPhysicsProtocol)
        # rank and total mass live at the skfem seam (scipy sparse)
        mass = physics.mass_matrix().toarray()
        assert np.linalg.matrix_rank(mass) == mass.shape[0]
        assert abs(float(mass.sum()) - 1.0) < 1e-13

    def test_no_dirichlet_dofs(self):
        bkd = NumpyBkd()
        _, physics = _make_periodic_burgers(bkd, nx=16)
        dofs, values = physics.dirichlet_dof_info(0.0)
        assert dofs.shape[0] == 0
        assert values.shape[0] == 0


class TestDiscreteConservation:
    def test_momentum_rate_is_zero(self):
        """1^T f(u) = 0: convection telescopes, diffusion pairs with
        the constant test function.  Tolerance from the ASSEMBLY scale
        (quadrature/summation roundoff), not machine epsilon."""
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32)
        u0 = _wave_state(basis, bkd, offset=0.5)
        residual = physics.spatial_residual(u0, 0.0)
        scale = bkd.to_float(bkd.max(bkd.abs(residual))) * float(
            residual.shape[0]
        )
        total = abs(bkd.to_float(bkd.sum(residual)))
        assert total < 100 * sys.float_info.epsilon * max(scale, 1.0)

    def test_forward_euler_conserves_total_momentum(self):
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32)
        u0 = _wave_state(basis, bkd, offset=0.5)
        config = TimeIntegrationConfig(
            method="forward_euler", init_time=0.0, final_time=0.02,
            deltat=1e-4,
        )
        states, _ = GalerkinModel(physics, bkd).solve_transient(u0, config)
        row_sums = _mass_row_sums(physics, bkd)
        momenta = bkd.sum(row_sums[:, None] * states, axis=0)
        drift = bkd.to_float(bkd.max(bkd.abs(momenta - momenta[0])))
        assert drift < 1e-12

    def test_viscous_energy_decay(self):
        """Zero-mean initial data: the M-weighted energy decays."""
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32, viscosity=0.05)
        u0 = _wave_state(basis, bkd, offset=0.0)
        config = TimeIntegrationConfig(
            method="backward_euler", init_time=0.0, final_time=0.2,
            deltat=5e-3,
        )
        states, _ = GalerkinModel(physics, bkd).solve_transient(u0, config)
        # M @ states is the skfem-seam contraction; analysis stays bkd
        weighted = bkd.asarray(
            physics.mass_matrix() @ np.asarray(bkd.to_numpy(states))
        )
        energies = bkd.sum(states * weighted, axis=0)
        assert bkd.all_bool(bkd.diff(energies) <= 1e-14)


class TestManufacturedSolutionRecovery:
    def test_periodic_exact_solution_recovered_at_fem_rate(self):
        """The MMS gate under the SAME periodic boundary treatment the
        OpInf benchmarks use: a space-periodic manufactured solution is
        recovered with second-order spatial convergence (P1), with the
        temporal error held below the spatial error by a small dt."""
        bkd = NumpyBkd()
        man_sol = ManufacturedBurgers1D(
            sol_str="(0.5 + 0.2*sin(2*pi*x))*exp(-T)",
            visc_str="0.1",
            bkd=bkd,
            oned=True,
        )
        solution = man_sol.functions["solution"]
        forcing = man_sol.functions["forcing"]

        final_time = 0.1
        errors = []
        for nx in (16, 32, 64):
            basis, physics = _make_periodic_burgers(
                bkd, nx=nx, viscosity=0.1, forcing=forcing
            )
            coords = basis.dof_coordinates()
            u0 = bkd.ravel(solution(coords, 0.0))
            config = TimeIntegrationConfig(
                method="crank_nicolson", init_time=0.0,
                final_time=final_time, deltat=1e-3,
            )
            states, times = GalerkinModel(physics, bkd).solve_transient(
                u0, config
            )
            exact = bkd.ravel(solution(coords, bkd.to_float(times[-1])))
            errors.append(
                bkd.to_float(
                    bkd.norm(states[:, -1] - exact) / bkd.norm(exact)
                )
            )
        rates = [
            math.log2(errors[k] / errors[k + 1])
            for k in range(len(errors) - 1)
        ]
        assert min(rates) > 1.7, (errors, rates)
