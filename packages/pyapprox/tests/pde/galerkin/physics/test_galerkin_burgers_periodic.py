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
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import PeriodicStructuredMesh1D
from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import exact_dof_values, forcing
from pyapprox.pde.verification.conditions import ConditionProtocol, Periodic
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]


def _make_periodic_burgers(
    bkd: NumpyBkd,
    nx: int,
    viscosity: float = 0.1,
    degree: int = 1,
    forcing: Optional[Callable[..., Any]] = None,
) -> Tuple[LagrangeBasis[_Np], BurgersPhysics[_Np]]:
    mesh = PeriodicStructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    physics = BurgersPhysics(basis=basis, viscosity=viscosity, bkd=bkd, forcing=forcing)
    return basis, physics


def _mass_row_sums(physics: BurgersPhysics[_Np], bkd: NumpyBkd) -> _Np:
    """1^T M as a backend array -- the single skfem-seam contraction."""
    return bkd.asarray(np.asarray(physics.mass_matrix().sum(axis=0)).ravel())


def _wave_state(basis: LagrangeBasis[_Np], bkd: NumpyBkd, offset: float) -> _Np:
    coords = basis.dof_coordinates()
    state: _Np = offset + bkd.sin(2.0 * math.pi * coords[0])
    return state


def _solve(
    physics: BurgersPhysics[_Np],
    make: Callable[[], ButcherTableau],
    deltat: float,
    final_time: float,
    init_state: _Np,
) -> Tuple[_Np, _Np]:
    """Every stored state, shape (nstates, ntimes), and the times, of the
    periodic system (no boundary conditions) stepped from ``init_state``
    (shape (nstates,))."""
    transient = galerkin_transient_system(
        skfem_discretization(compose_galerkin_system(physics)), ScipySparseLU()
    )
    tableau = make()
    method: RungeKutta[_Np, _Np] = RungeKutta(
        tableau,
        None if tableau.is_explicit() else Newton(atol=1e-10, rtol=0.0, maxiters=20),
    )
    store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
    solve_galerkin_transient(
        transient,
        method,
        RecomputeAlways(transient.solved.solver),
        FixedStep(deltat, (0.0, final_time)),
        init_state[:, None],
        store,
    )
    states = np.concatenate([store.state(n) for n in range(store.ntimes())], axis=1)
    return states, np.asarray(store.times())


class TestPeriodicAssembly:
    @pytest.mark.parametrize("degree", [1, 2])
    def test_mass_matrix_full_rank_and_partition_of_unity(self, degree: int) -> None:
        bkd = NumpyBkd()
        _, physics = _make_periodic_burgers(bkd, nx=16, degree=degree)
        assert isinstance(physics, GalerkinPhysicsProtocol)
        # rank and total mass live at the skfem seam (scipy sparse)
        raw = physics.mass_matrix()
        mass = np.asarray(raw.toarray() if isinstance(raw, spmatrix) else raw)
        assert np.linalg.matrix_rank(mass) == mass.shape[0]
        assert abs(float(mass.sum()) - 1.0) < 1e-13

    def test_periodic_mesh_has_no_boundary(self) -> None:
        """The endpoints are identified, so there is no boundary on which
        an essential condition could be imposed."""
        bkd = NumpyBkd()
        basis, _ = _make_periodic_burgers(bkd, nx=16)
        assert len(basis.skfem_basis().mesh.boundary_facets()) == 0
        with pytest.raises(ValueError, match="no boundaries"):
            basis.get_dofs("left")


class TestDiscreteConservation:
    def test_momentum_rate_is_zero(self) -> None:
        """1^T f(u) = 0: convection telescopes, diffusion pairs with
        the constant test function.  Tolerance from the ASSEMBLY scale
        (quadrature/summation roundoff), not machine epsilon."""
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32)
        u0 = _wave_state(basis, bkd, offset=0.5)
        operator = compose_galerkin_system(physics).spatial_operator()
        residual = operator.spatial_residual(u0, 0.0)
        scale = bkd.to_float(bkd.max(bkd.abs(residual))) * float(residual.shape[0])
        total = abs(bkd.to_float(bkd.sum(residual)))
        assert total < 100 * sys.float_info.epsilon * max(scale, 1.0)

    def test_forward_euler_conserves_total_momentum(self) -> None:
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32)
        u0 = _wave_state(basis, bkd, offset=0.5)
        states, _ = _solve(physics, forward_euler, 1e-4, 0.02, u0)
        row_sums = _mass_row_sums(physics, bkd)
        momenta = bkd.sum(row_sums[:, None] * states, axis=0)
        drift = bkd.to_float(bkd.max(bkd.abs(momenta - momenta[0])))
        assert drift < 1e-12

    def test_viscous_energy_decay(self) -> None:
        """Zero-mean initial data: the M-weighted energy decays."""
        bkd = NumpyBkd()
        basis, physics = _make_periodic_burgers(bkd, nx=32, viscosity=0.05)
        u0 = _wave_state(basis, bkd, offset=0.0)
        states, _ = _solve(physics, backward_euler, 5e-3, 0.2, u0)
        # M @ states is the skfem-seam contraction; analysis stays bkd
        weighted = bkd.asarray(physics.mass_matrix() @ np.asarray(bkd.to_numpy(states)))
        energies = bkd.sum(states * weighted, axis=0)
        assert bkd.to_bool(bkd.all(bkd.diff(energies) <= 1e-14))


class TestManufacturedSolutionRecovery:
    def test_periodic_exact_solution_recovered_at_fem_rate(self) -> None:
        """The MMS gate under the SAME periodic boundary treatment the
        OpInf benchmarks use: a space-periodic manufactured solution is
        recovered with second-order spatial convergence (P1), with the
        temporal error held below the spatial error by a small dt."""
        bkd = NumpyBkd()
        terms: List[TermProtocol] = [
            TimeDerivative("u"),
            Divergence("u", ["-(0.1)*u_x"]),
            Source("u", "u*u_x"),
        ]
        periodic: dict[str, ConditionProtocol] = {
            "left": Periodic(),
            "right": Periodic(),
        }
        problem = ManufacturedProblem(
            {"u": "(0.5 + 0.2*sin(2*pi*x))*exp(-t)"},
            terms,
            Box([(0.0, 1.0)]),
            periodic,
        )

        final_time = 0.1
        errors = []
        for nx in (16, 32, 64):
            basis, physics = _make_periodic_burgers(
                bkd, nx=nx, viscosity=0.1, forcing=forcing(problem, "u")
            )
            u0 = exact_dof_values(problem, basis, ["u"], 0.0)
            states, times = _solve(physics, crank_nicolson, 1e-3, final_time, u0)
            exact = exact_dof_values(problem, basis, ["u"], float(times[-1]))
            errors.append(
                bkd.to_float(bkd.norm(states[:, -1] - exact) / bkd.norm(exact))
            )
        rates = [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
        assert min(rates) > 1.7, (errors, rates)
