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

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import (
    boundary_conditions,
    exact_dof_values,
    field,
    forcing,
    vector_field,
)
from pyapprox.pde.verification.conditions import Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_UNIT = Box([(0.0, 1.0)])


def _build(
    bkd: NumpyBkd,
    diff_str: str,
    vel_strs: Sequence[str],
    nx: int = 8,
    diffusivity_override: Optional[float] = None,
) -> Tuple["GalerkinTransientSystem[_Np, CSRFormat]", Callable[[float], _Np]]:
    r"""Transient ADR :math:`\partial_t u = \nabla\cdot(D\nabla u) -
    v\cdot\nabla u + f` whose coefficients are the strings themselves.

    ``diffusivity_override`` replaces ONLY the diffusivity handed to the
    physics, leaving the forcing, boundary data and exact solution as
    generated for ``diff_str``. That mismatch is what
    :meth:`test_frozen_coefficient_would_be_detected` needs.
    """
    terms: List[TermProtocol] = [
        TimeDerivative("u"),
        Divergence("u", [f"-({diff_str})*u_x"]),
        Source("u", " + ".join(f"({v})*u_x" for v in vel_strs)),
    ]
    problem = ManufacturedProblem(
        {"u": "(1 - x)*x*(1 + t)"},
        terms,
        _UNIT,
        {side: Dirichlet() for side in _UNIT.boundaries()},
    )
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=2)

    # The coefficients themselves, not constants standing in for them.
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=(
            field(problem, diff_str)
            if diffusivity_override is None
            else diffusivity_override
        ),
        velocity=vector_field(problem, vel_strs),
        forcing=forcing(problem, "u"),
        bkd=bkd,
    )
    system = compose_galerkin_system(physics, boundary_conditions(problem, basis, "u"))
    transient = galerkin_transient_system(skfem_discretization(system), ScipySparseLU())

    def exact_at_time(time: float) -> _Np:
        return exact_dof_values(problem, basis, ["u"], time=time)

    return transient, exact_at_time


def _solve(
    bkd: NumpyBkd,
    transient: "GalerkinTransientSystem[_Np, CSRFormat]",
    init_state: _Np,
    deltat: float = 0.5,
    final_time: float = 2.0,
) -> Tuple[_Np, _Np]:
    """Backward Euler from ``init_state`` (shape (nstates,)); every state,
    shape (nstates, ntimes), and the times."""
    store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
    solve_galerkin_transient(
        transient,
        RungeKutta(backward_euler(), Newton(atol=1e-12, rtol=0.0, maxiters=30)),
        RecomputeAlways(transient.solved.solver),
        FixedStep(deltat, (0.0, final_time)),
        bkd.reshape(bkd.asarray(init_state), (-1, 1)),
        store,
    )
    states = bkd.concatenate([store.state(n) for n in range(store.ntimes())], axis=1)
    return states, bkd.asarray(store.times())


def _trajectory_errors(
    bkd: NumpyBkd,
    transient: "GalerkinTransientSystem[_Np, CSRFormat]",
    exact_at_time: Callable[[float], _Np],
    deltat: float = 0.5,
) -> Tuple[List[float], _Np]:
    """Relative L2 error in the solution at every step, not just the last."""
    solutions, times = _solve(bkd, transient, exact_at_time(0.0), deltat=deltat)
    errors = []
    for step, time in enumerate(times):
        exact = exact_at_time(float(time))
        norm = np.linalg.norm(exact)
        if norm > 1e-12:
            numeric = bkd.to_numpy(solutions[:, step])
            errors.append(float(np.linalg.norm(numeric - exact) / norm))
    return errors, times


_CASES: List[Tuple[str, str, List[str]]] = [
    # Diffusivity grows with time; velocity fixed.
    ("time_varying_diffusivity", "4 + t", ["0 + 1e-16*x"]),
    # Velocity grows with time; diffusivity fixed.
    ("time_varying_velocity", "4 + 1e-16*x", ["(1 + t)/10"]),
    # Both vary, which is the transient-flow case.
    ("both_time_varying", "4 + t", ["(1 + t)/10"]),
]

# Round-off for this problem size, with headroom for conditioning: the
# measured worst error across all cases is ~3e-14. This is deliberately
# NOT a discretization tolerance --- there is no discretization error to
# tolerate, so anything above this bound is a threading defect.
_EXACT_TOL = 1e-11


class TestTransientTimeVaryingCoefficients:
    @pytest.mark.parametrize("name,diff_str,vel_strs", _CASES)
    def test_trajectory_is_exact(
        self, numpy_bkd: NumpyBkd, name: str, diff_str: str, vel_strs: List[str]
    ) -> None:
        """The solution is reproduced to round-off at every step."""
        bkd = numpy_bkd
        transient, exact_at_time = _build(bkd, diff_str, vel_strs)
        errors, times = _trajectory_errors(bkd, transient, exact_at_time)

        worst = max(errors)
        assert worst < _EXACT_TOL, (
            f"{name}: worst relative L2 error {worst:.3e} over "
            f"{len(times)} steps ({[f'{e:.2e}' for e in errors]})"
        )

    @pytest.mark.parametrize("name,diff_str,vel_strs", _CASES)
    def test_exactness_is_independent_of_step_size(
        self, numpy_bkd: NumpyBkd, name: str, diff_str: str, vel_strs: List[str]
    ) -> None:
        """Refining the step must not improve the answer.

        An exact scheme has no time-discretization error to refine away.
        If halving the step reduced the error, the trajectory would be
        merely convergent rather than exact --- which would mean the
        tight bound above was passing by luck of the step size.
        """
        bkd = numpy_bkd
        for deltat in (0.5, 0.25, 0.125):
            transient, exact_at_time = _build(bkd, diff_str, vel_strs)
            errors, _ = _trajectory_errors(bkd, transient, exact_at_time, deltat=deltat)
            assert max(errors) < _EXACT_TOL, (
                f"{name}: deltat={deltat} gave {max(errors):.3e}"
            )

    def test_frozen_coefficient_would_be_detected(self, numpy_bkd: NumpyBkd) -> None:
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
        transient, exact_at_time = _build(bkd, "4 + t", ["0 + 1e-16*x"])

        # Same manufactured problem, but the physics is handed a
        # diffusivity frozen at its t=0 value (D(0) = 4).
        frozen_transient, _ = _build(
            bkd, "4 + t", ["0 + 1e-16*x"], diffusivity_override=4.0
        )

        init = exact_at_time(0.0)
        varying, times = _solve(bkd, transient, init)
        frozen, _ = _solve(bkd, frozen_transient, init)

        separations = []
        for step in range(len(times)):
            reference = bkd.to_numpy(varying[:, step])
            norm = np.linalg.norm(reference)
            if norm > 1e-12:
                separations.append(
                    np.linalg.norm(reference - bkd.to_numpy(frozen[:, step])) / norm
                )
        assert max(separations) > 1e-3, (
            "a frozen diffusivity is indistinguishable from a varying one, "
            "so these tests could not detect a dropped time"
        )
