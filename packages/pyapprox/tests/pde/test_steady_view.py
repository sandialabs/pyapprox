"""SteadyView: a time-free steady problem from a time-dependent F(u, t)."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC, RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.steady_view import (
    SteadyOperatorProtocol,
    SteadyViewProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def _steady_source(x: NDArray[Any]) -> NDArray[Any]:
    return np.sin(np.pi * x[0])


def _ramped_source(x: NDArray[Any], t: float) -> NDArray[Any]:
    return np.sin(np.pi * x[0]) * (1.0 - np.exp(-t))


def _physics_and_system(
    bkd: Backend[Array], forcing: Any, robin: bool = False
) -> tuple[AdvectionDiffusionReaction[Array], GalerkinSystem[Array]]:
    basis = LagrangeBasis(StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd), 1)
    bcs: list[Any] = [DirichletBC(basis, "left", 1.0, bkd)]
    if robin:
        bcs.append(RobinBC(basis, "right", alpha=2.0, value_func=0.5, bkd=bkd))
    else:
        bcs.append(DirichletBC(basis, "right", 0.0, bkd))
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=1.0,
        bkd=bkd,
        forcing=forcing,
    )
    return physics, compose_galerkin_system(physics, bcs)


class TestSteadyView:
    def test_satisfies_protocols(self, bkd: Backend[Array]) -> None:
        _, system = _physics_and_system(bkd, TimeIndependent(_steady_source))
        view = system.steady()
        assert isinstance(view, SteadyOperatorProtocol)
        assert isinstance(view, SteadyViewProtocol)

    def test_of_refuses_time_dependent_data(
        self, bkd: Backend[Array]
    ) -> None:
        _, system = _physics_and_system(bkd, TimeDependent(_ramped_source))
        with pytest.raises(ValueError, match="snapshot"):
            system.steady()

    def test_snapshot_evaluates_at_its_time(
        self, bkd: Backend[Array]
    ) -> None:
        physics, system = _physics_and_system(
            bkd, TimeDependent(_ramped_source)
        )
        state = bkd.linspace(1.0, 0.0, physics.nstates())
        for time in (0.0, 1.0):
            view = system.steady_snapshot(time)
            assert view.time() == time
            # F at this time, with the constraint rows applied at this time.
            expected = system.constraint_set().apply_to_residual(
                system.spatial_operator().spatial_residual(state, time),
                state,
                time,
            )
            bkd.assert_allclose(view.steady_residual(state), expected)
        assert not np.allclose(
            bkd.to_numpy(system.steady_snapshot(0.0).steady_residual(state)),
            bkd.to_numpy(system.steady_snapshot(1.0).steady_residual(state)),
        )

    def test_constrained_rows_replaced_and_others_untouched(
        self, bkd: Backend[Array]
    ) -> None:
        """Dirichlet rows become u_d - g; Robin rows keep F (a term)."""
        physics, system = _physics_and_system(
            bkd, TimeIndependent(_steady_source), robin=True
        )
        view = system.steady()
        n = physics.nstates()
        state = bkd.linspace(0.3, 0.7, n)
        raw = system.spatial_operator().spatial_residual(state, 0.0)
        residual = view.steady_residual(state)
        dofs = [int(d) for d in system.constraint_set().dofs()]
        assert dofs == [0]
        bkd.assert_allclose(residual[:1], state[:1] - 1.0)
        bkd.assert_allclose(residual[1:], raw[1:])
        jac = view.steady_jacobian(state)
        jac_np = jac.toarray() if hasattr(jac, "toarray") else bkd.to_numpy(jac)
        expected_row = np.zeros(n)
        expected_row[0] = 1.0
        bkd.assert_allclose(bkd.asarray(jac_np[0]), bkd.asarray(expected_row))

    def test_view_sees_coefficient_changes(
        self, bkd: Backend[Array]
    ) -> None:
        """The view looks data up on every call, so a coefficient changed
        after it is built (as a parameterization does, via ``set_dofs``)
        changes its residual to that of the new coefficient."""
        basis = LagrangeBasis(
            StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd), 1
        )

        def build(kappa: float) -> Any:
            diffusivity = NodalFieldDiffusion(
                basis, np.full(basis.ndofs(), kappa)
            )
            physics = AdvectionDiffusionReaction(
                basis=basis,
                diffusivity=diffusivity,
                bkd=bkd,
                forcing=TimeIndependent(_steady_source),
            )
            system = compose_galerkin_system(
                physics, [DirichletBC(basis, "left", 1.0, bkd)]
            )
            return system, diffusivity

        system, diffusivity = build(1.0)
        view = system.steady()
        state = bkd.linspace(1.0, 0.0, basis.ndofs())

        diffusivity.set_dofs(np.full(basis.ndofs(), 2.0))

        fresh, _ = build(2.0)
        bkd.assert_allclose(
            view.steady_residual(state), fresh.steady().steady_residual(state)
        )

    def test_solver_on_view_matches_closed_form(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        """-u'' = sin(pi x), u(0)=1, u(1)=0: u = 1 - x + sin(pi x)/pi^2,
        checked at the nodes to discretization accuracy."""
        bkd = numpy_bkd
        physics, system = _physics_and_system(
            bkd, TimeIndependent(_steady_source)
        )
        result = SteadyStateSolver(system.steady(), tol=1e-12).solve(
            bkd.zeros((physics.nstates(),))
        )
        assert result.converged
        x = np.linspace(0.0, 1.0, physics.nstates())
        exact = 1.0 - x + np.sin(np.pi * x) / np.pi**2
        bkd.assert_allclose(result.solution, bkd.asarray(exact), atol=5e-3)

    def test_solver_rejects_a_physics(self, numpy_bkd: Backend[Array]) -> None:
        physics, _ = _physics_and_system(
            numpy_bkd, TimeIndependent(_steady_source)
        )
        with pytest.raises(TypeError, match="SteadyOperatorProtocol"):
            SteadyStateSolver(physics)

