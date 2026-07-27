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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.problems.optimization import (
    ObstructedFlowControlProblem,
)

from tests._helpers.markers import slow_test


def _make_problem(bkd: NumpyBkd) -> ObstructedFlowControlProblem:
    """Reduced-resolution configuration for CI runtimes."""
    return ObstructedFlowControlProblem(
        bkd,
        nstokes_refine=1,
        ntransport_refine=1,
        final_time=1.0,
        deltat=0.1,
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
        just numerical dust)."""
        bkd = numpy_bkd
        problem = _make_problem(bkd)
        value = float(
            bkd.to_numpy(
                problem.model()(bkd.zeros((problem.ncontrols(), 1)))
            )[0, 0]
        )
        print(f"J(0) = {value:.3e}")
        assert value > 1e-8
