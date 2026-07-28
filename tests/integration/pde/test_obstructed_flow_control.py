"""Optimizer-in-the-loop acceptance test for the plume-control problem.

Exercises the composed adjoint stack end to end on
``ObstructedFlowControlProblem``: frozen Navier-Stokes flow, transient
advection-diffusion with an affine actuator forcing map, a
time-integrated zone QoI with Tikhonov actuation cost, and the
transient adjoint gradient + second-order-adjoint HVP consumed through
the public ``ObjectiveProtocol`` surface. An optimizer iterating on
adjoint gradients finds cross-component bugs unit tests cannot.

Run with printed FD sweeps:
    PYAPPROX_RUN_SLOW=1 pytest tests/integration/pde/test_obstructed_flow_control.py -s
"""

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.optimization.minimize.scipy.lbfgsb import LBFGSBOptimizer
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.problems.optimization import (
    ObstructedFlowControlProblem,
)

from tests._helpers.markers import slow_test

# Actuators sitting on the plume's feed paths; the two others
# (outflow_lower, downstream_of_zone) are deliberate low-leverage
# placements whose amplitudes should stay comparatively small.
_FEED_PATH_LABELS = frozenset(
    {
        "below_B_gap",
        "above_B",
        "corridor_mid",
        "gap_AC",
        "corridor_upper",
        "zone_inlet_left",
    }
)


def _make_problem(
    bkd: NumpyBkd, final_time: float = 1.0, deltat: float = 0.1
) -> ObstructedFlowControlProblem:
    """Reduced-resolution configuration for CI runtimes.

    The coarser transport mesh needs a larger diffusivity than the
    production default to keep the cell Peclet below one (unstabilized
    Galerkin oscillates beyond it).
    """
    return ObstructedFlowControlProblem(
        bkd,
        nstokes_refine=1,
        ntransport_refine=1,
        diffusivity=0.02,
        final_time=final_time,
        deltat=deltat,
    )


class TestObstructedFlowControlAcceptance:
    @slow_test
    def test_gradient_and_hvp_verification(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """FD sweeps of the adjoint gradient and one HVP direction at
        p = 0 and at a nonzero p, plus the FD-noise-immune HVP
        symmetry identity."""
        bkd = numpy_bkd
        problem = _make_problem(bkd)
        model = problem.model()
        assert model.nvars() == problem.ncontrols()
        assert model.derivatives().hvp is not None

        rng = np.random.default_rng(7)
        direction = bkd.asarray(
            rng.normal(0.0, 1.0, (model.nvars(), 1))
        )
        for label, sample_np in (
            ("p=0", np.zeros((model.nvars(), 1))),
            ("p~N(0,1)", rng.normal(0.0, 1.0, (model.nvars(), 1))),
        ):
            sample = bkd.asarray(sample_np)
            checker = DerivativeChecker(model)
            errors = checker.check_derivatives(
                sample, direction=direction, relative=True
            )
            jac_ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
            hvp_min = float(bkd.to_numpy(bkd.min(errors[1])))
            print(
                f"[{label}] gradient V-ratio {jac_ratio:.2e}, "
                f"HVP V-bottom {hvp_min:.2e}"
            )
            # Provisional sanity bounds; hardened to 2x observed CI
            # drift once the optimization stage lands.
            assert jac_ratio <= 1e-3
            assert hvp_min <= 1e-3

        sample = bkd.asarray(rng.normal(0.0, 1.0, (model.nvars(), 1)))
        other = bkd.asarray(rng.normal(0.0, 1.0, (model.nvars(), 1)))
        hvp_fn = model.derivatives().hvp
        assert hvp_fn is not None
        h_dir = bkd.flatten(hvp_fn(sample, direction))
        h_other = bkd.flatten(hvp_fn(sample, other))
        bkd.assert_allclose(
            bkd.sum(h_dir * bkd.flatten(other)),
            bkd.sum(h_other * bkd.flatten(direction)),
            rtol=1e-10,
        )

    @slow_test
    def test_uncontrolled_objective_positive(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The uncontrolled plume contaminates the zone: J(0) > 0, and
        the release actually reaches the zone (contamination term, not
        just numerical dust). Horizon matched to the release-to-zone
        transit time so this measures plume arrival."""
        bkd = numpy_bkd
        problem = _make_problem(bkd, final_time=20.0, deltat=1.0)
        value = float(
            bkd.to_numpy(
                problem.model()(bkd.zeros((problem.ncontrols(), 1)))
            )[0, 0]
        )
        print(f"J(0) = {value:.3e}")
        assert value > 1e-2

    @slow_test
    def test_optimization_learns_interception(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The optimizer-in-the-loop check: both optimizers drive J
        down by orders of magnitude, agree with each other, and the
        learned strategy places its dominant SINKS on the plume's feed
        paths rather than at the low-leverage placements. Horizon set
        to the release-to-zone transit time so the uncontrolled plume
        genuinely contaminates the zone."""
        bkd = numpy_bkd
        problem = _make_problem(bkd, final_time=20.0, deltat=1.0)
        model = problem.model()
        controls0 = bkd.zeros((problem.ncontrols(), 1))
        j0 = float(bkd.to_numpy(model(controls0))[0, 0])

        lbfgs = LBFGSBOptimizer(verbosity=0, maxiter=100)
        lbfgs.bind(model, problem.bounds())
        p_lbfgs = lbfgs.minimize(controls0).optima()
        j_lbfgs = float(bkd.to_numpy(model(p_lbfgs))[0, 0])

        newton = ScipyTrustConstrOptimizer(
            verbosity=0, maxiter=30, gtol=1e-8
        )
        newton.bind(model, problem.bounds())
        p_newton = newton.minimize(controls0).optima()
        j_newton = float(bkd.to_numpy(model(p_newton))[0, 0])

        print(
            f"J(0)={j0:.3e}  J*(L-BFGS-B)={j_lbfgs:.3e}  "
            f"J*(trust-constr)={j_newton:.3e}"
        )
        amplitudes = bkd.to_numpy(p_lbfgs)[:, 0]
        labels = problem.actuator_labels()
        for label, val in zip(labels, amplitudes):
            print(f"  {label:>20s}: {val:+.3f}")

        # Provisional sanity bounds; hardened to 2x observed CI drift
        # in the follow-up that finalizes the acceptance assertions.
        assert j_lbfgs < 0.05 * j0
        assert j_newton < 0.05 * j0
        # Both optimizers minimize the same strictly convex quadratic.
        assert abs(j_lbfgs - j_newton) <= 0.2 * max(j_lbfgs, j_newton)
        # Learned strategy: dominant actuator is a SINK on a feed path,
        # and the low-leverage placements carry comparatively little
        # amplitude (guards against the optimizer parking effort where
        # the physics barely sees it).
        dominant = int(np.argmax(np.abs(amplitudes)))
        assert labels[dominant] in _FEED_PATH_LABELS
        assert amplitudes[dominant] < 0.0
        feed_mass = sum(
            abs(val)
            for label, val in zip(labels, amplitudes)
            if label in _FEED_PATH_LABELS
        )
        off_path_mass = sum(
            abs(val)
            for label, val in zip(labels, amplitudes)
            if label not in _FEED_PATH_LABELS
        )
        print(
            f"feed-path |p| mass {feed_mass:.3f}, "
            f"off-path {off_path_mass:.3f}"
        )
        assert off_path_mass < 0.5 * feed_mass
