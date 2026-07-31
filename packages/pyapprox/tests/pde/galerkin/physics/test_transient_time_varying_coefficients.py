"""Transient ADR with coefficients that vary in TIME, not just space.

The existing transient tests hold the diffusivity and velocity constant
(``diff_str="4+1e-16*x"`` with ``diffusivity=4.0``), so nothing there
exercises a coefficient that must be re-evaluated as the integrator
steps. These do: the manufactured coefficient carries a ``T``.

Both error sources are eliminated rather than bounded, so the assertion
is at round-off instead of a tolerance that would forgive a real defect:

* SPACE. The solution ``(1-x)*x*(1+T)`` is a quadratic in ``x`` and the
  basis is P2, so it lies IN the finite element space and the spatial
  discretization error is exactly zero.
* TIME. The solution is linear in ``T``, so its time derivative is
  constant and backward Euler's difference quotient reproduces it
  exactly. The coefficients and forcing are evaluated at ``t_n``, where
  the manufactured residual vanishes identically --- so a time-varying
  ``D(t)`` or ``v(t)`` does not degrade this, provided it is evaluated
  at the right time. That proviso is the whole point of the test.

A coefficient frozen at its first evaluation still assembles a
plausible operator, so an error appears only in the answer --- and it
compounds with every step, which is why the comparison runs over the
whole trajectory rather than the final state alone.
"""

from typing import List

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolutionAdapter,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.time_integration import (
    GalerkinModel,
    TimeIntegrationConfig,
)


def _build(bkd, diff_str, vel_strs, nx=8, diffusivity_override=None):
    """Transient ADR whose coefficients come from the manufactured set.

    ``diffusivity_override`` replaces ONLY the diffusivity handed to the
    physics, leaving the forcing, boundary data and exact solution as
    generated for ``diff_str``. That mismatch is what
    :meth:`test_frozen_coefficient_would_be_detected` needs.
    """
    bounds = [0.0, 1.0]
    functions, _ = create_adr_manufactured_test(
        bounds=bounds,
        sol_str="(1-x)*x*(1+T)",
        diff_str=diff_str,
        react_str="0*u",
        vel_strs=vel_strs,
        bkd=bkd,
        time_dependent=True,
    )
    mesh = StructuredMesh1D(nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)
    adapter = GalerkinManufacturedSolutionAdapter(
        basis, functions, bkd, time_dependent=True
    )
    bc_set = adapter.create_boundary_conditions(["D", "D"])

    # The coefficients themselves, not constants standing in for them.
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=(
            functions["diffusion"]
            if diffusivity_override is None
            else diffusivity_override
        ),
        velocity=adapter.velocity_for_galerkin(),
        forcing=adapter.forcing_for_galerkin(),
        boundary_conditions=bc_set.all_conditions(),
        bkd=bkd,
    )

    exact_func = adapter.solution_function()
    coords = bkd.to_numpy(basis.dof_coordinates())

    def exact_at_time(time):
        values = exact_func(coords, time)
        if hasattr(values, "shape") and values.ndim > 1:
            return (
                values[:, 0] if values.shape[1] == 1 else values.flatten()
            )
        return values

    return GalerkinModel(physics, bkd), exact_at_time


def _config(deltat=0.5, final_time=2.0):
    return TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=final_time,
        deltat=deltat,
        newton_tol=1e-12,
        newton_maxiter=30,
        lumped_mass=False,
        verbosity=0,
    )


def _trajectory_errors(bkd, model, exact_at_time, config):
    """Relative L2 error in the solution at every step, not just the last."""
    solutions, times = model.solve_transient(
        bkd.asarray(exact_at_time(0.0)), config
    )
    errors = []
    for step, time in enumerate(times):
        exact = exact_at_time(float(time))
        norm = np.linalg.norm(exact)
        if norm > 1e-12:
            numeric = bkd.to_numpy(solutions[:, step])
            errors.append(np.linalg.norm(numeric - exact) / norm)
    return errors, times


_CASES: List[tuple] = [
    # Diffusivity grows with time; velocity fixed.
    ("time_varying_diffusivity", "4+T", ["0+1e-16*x"]),
    # Velocity grows with time; diffusivity fixed.
    ("time_varying_velocity", "4+1e-16*x", ["(1+T)/10"]),
    # Both vary, which is the transient-flow case.
    ("both_time_varying", "4+T", ["(1+T)/10"]),
]

# Round-off for this problem size, with headroom for conditioning: the
# measured worst error across all cases is ~3e-14. This is deliberately
# NOT a discretization tolerance --- there is no discretization error to
# tolerate, so anything above this bound is a threading defect.
_EXACT_TOL = 1e-11


class TestTransientTimeVaryingCoefficients:
    @pytest.mark.parametrize("name,diff_str,vel_strs", _CASES)
    def test_trajectory_is_exact(
        self, numpy_bkd, name, diff_str, vel_strs
    ) -> None:
        """The solution is reproduced to round-off at every step."""
        bkd = numpy_bkd
        model, exact_at_time = _build(bkd, diff_str, vel_strs)
        errors, times = _trajectory_errors(
            bkd, model, exact_at_time, _config()
        )

        worst = max(errors)
        assert worst < _EXACT_TOL, (
            f"{name}: worst relative L2 error {worst:.3e} over "
            f"{len(times)} steps ({[f'{e:.2e}' for e in errors]})"
        )

    @pytest.mark.parametrize("name,diff_str,vel_strs", _CASES)
    def test_exactness_is_independent_of_step_size(
        self, numpy_bkd, name, diff_str, vel_strs
    ) -> None:
        """Refining the step must not improve the answer.

        An exact scheme has no time-discretization error to refine away.
        If halving the step reduced the error, the trajectory would be
        merely convergent rather than exact --- which would mean the
        tight bound above was passing by luck of the step size.
        """
        bkd = numpy_bkd
        for deltat in (0.5, 0.25, 0.125):
            model, exact_at_time = _build(bkd, diff_str, vel_strs)
            errors, _ = _trajectory_errors(
                bkd, model, exact_at_time, _config(deltat=deltat)
            )
            assert max(errors) < _EXACT_TOL, (
                f"{name}: deltat={deltat} gave {max(errors):.3e}"
            )

    def test_frozen_coefficient_would_be_detected(self, numpy_bkd) -> None:
        """The guard on the guard.

        The bound above is only meaningful if a coefficient evaluated at
        the WRONG time actually breaks it. Here the diffusivity is
        pinned to its t=0 value while the forcing, boundary data and
        exact solution still come from the time-varying problem --- i.e.
        precisely the state the code would be in if the assembly failed
        to thread time into the coefficient.

        Note this cannot be simulated by building a second manufactured
        problem with a constant ``diff_str``: that regenerates a matching
        forcing, so the frozen model solves its own consistent problem
        exactly and the two agree to round-off. The coefficient must be
        replaced UNDER a forcing that was not told about the change.
        """
        bkd = numpy_bkd
        model, exact_at_time = _build(bkd, "4+T", ["0+1e-16*x"])

        # Same manufactured problem, but the physics is handed a
        # diffusivity frozen at its t=0 value (D(0) = 4).
        frozen_model, _ = _build(
            bkd, "4+T", ["0+1e-16*x"], diffusivity_override=4.0
        )

        config = _config()
        init = bkd.asarray(exact_at_time(0.0))
        varying, times = model.solve_transient(init, config)
        frozen, _ = frozen_model.solve_transient(init, config)

        separations = []
        for step in range(len(times)):
            reference = bkd.to_numpy(varying[:, step])
            norm = np.linalg.norm(reference)
            if norm > 1e-12:
                separations.append(
                    np.linalg.norm(reference - bkd.to_numpy(frozen[:, step]))
                    / norm
                )
        assert max(separations) > 1e-3, (
            "a frozen diffusivity is indistinguishable from a varying one, "
            "so these tests could not detect a dropped time"
        )
