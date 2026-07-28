"""Optimizer-in-the-loop acceptance test for the plume-control problem.

Exercises the composed adjoint stack end to end on
``ObstructedFlowControlProblem``: frozen Navier-Stokes flow, transient
advection-diffusion with a bilinear extraction-rate reaction map, a
time-integrated zone QoI with Tikhonov actuation cost, and the
transient adjoint gradient + second-order-adjoint HVP consumed through
the public ``ObjectiveProtocol`` surface. An optimizer iterating on
adjoint gradients finds cross-component bugs unit tests cannot.

Run with printed FD sweeps:
    PYAPPROX_RUN_SLOW=1 pytest tests/integration/pde/test_obstructed_flow_control.py -s
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

if TYPE_CHECKING:
    from pyapprox.pde.models.galerkin.transient import (
        GalerkinTransientForwardModel,
    )
    from skfem.assembly.form.form import FormExtraParams

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

from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.markers import slow_test

# Extraction devices sitting on the plume's feed paths; the two others
# (outflow_lower, downstream_of_zone) are deliberate low-leverage
# placements whose rates should stay comparatively small.
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


def _mass_ledger(
    problem: ObstructedFlowControlProblem[NumpyArray],
    model: "GalerkinTransientForwardModel[NumpyArray]",
    controls: NumpyArray,
    bkd: NumpyBkd,
    diffusivity: float,
) -> Tuple[float, float, float, float, float]:
    """Discrete mass bookkeeping of a controlled trajectory.

    Returns (mass_change, released, extracted, outflowed, residual):
    the change of int(u) over the horizon against the exchanges the
    DISCRETE Galerkin dynamics actually make (CN trapezoid in time).
    Summing the weak form against the all-ones test function (the
    partition of unity) shows what those are: the release int(s), the
    extraction int(r u), the advective loss as the VOLUME integral
    int(v.grad(u)) — NOT a boundary trace — and the Danckwerts inlet
    exchange int_left(|v.n| u) ds from the Robin operator. Diffusion
    contributes exactly zero: 1^T K_diff u = int(kappa grad(u).grad(1))
    vanishes identically, so do-nothing boundaries lose no diffusive
    mass discretely (the continuum's diffusive boundary loss is a
    MODELING limit of those BCs, not a ledger line). Every line is
    integrated with the diagnostic's own quadrature, independent of
    the solver's assembled operators, so bookkeeping/BC/form errors
    still surface — but the floor is quadrature-level, not
    boundary-gradient-level. ``diffusivity`` is unused by the ledger
    for exactly this reason; it is kept so callers state the physics
    they think they are balancing.
    """
    from skfem import FacetBasis, Functional, asm

    if problem.inlet_bc() != "danckwerts":
        raise ValueError(
            "the discrete ledger models the Robin inlet exchange; the "
            "Dirichlet inlet replaces rows and needs different "
            "bookkeeping"
        )
    del diffusivity
    sols, times = model.forward_solve(controls)
    sols_np = bkd.to_numpy(sols)
    times_np = bkd.to_numpy(times)
    skfem_basis = problem.basis().skfem_basis()
    mesh = skfem_basis.mesh
    inlet_basis = FacetBasis(
        mesh, skfem_basis.elem, facets=mesh.boundaries["left"]
    )
    velocity = problem.velocity()

    def _integrate(w: "FormExtraParams") -> np.ndarray:
        return np.asarray(w["uh"])

    def _integrate_weighted(w: "FormExtraParams") -> np.ndarray:
        return np.asarray(w["rh"] * w["uh"])

    def _advective_loss(w: "FormExtraParams") -> np.ndarray:
        vel = velocity(np.asarray(w.x))
        return np.asarray(
            vel[0] * w["uh"].grad[0] + vel[1] * w["uh"].grad[1]
        )

    def _inlet_exchange(w: "FormExtraParams") -> np.ndarray:
        # Danckwerts coefficient alpha(y) = |v.n| = v_x on the left
        # boundary (n = (-1, 0)); no gradients involved.
        vel = velocity(np.asarray(w.x))
        return np.asarray(vel[0] * w["uh"])

    integrate = Functional(_integrate)
    integrate_weighted = Functional(_integrate_weighted)
    advective_loss = Functional(_advective_loss)
    inlet_exchange = Functional(_inlet_exchange)

    release_rate = asm(
        integrate,
        skfem_basis,
        uh=skfem_basis.interpolate(problem.release_field()),
    )
    rate_field = problem.extraction_field(bkd.to_numpy(controls)[:, 0])
    ntimes = sols_np.shape[1]
    masses = np.empty(ntimes)
    extraction = np.empty(ntimes)
    outflow = np.empty(ntimes)
    for nn in range(ntimes):
        state = skfem_basis.interpolate(sols_np[:, nn])
        masses[nn] = asm(integrate, skfem_basis, uh=state)
        extraction[nn] = asm(
            integrate_weighted,
            skfem_basis,
            rh=skfem_basis.interpolate(rate_field),
            uh=state,
        )
        outflow[nn] = asm(
            advective_loss, skfem_basis, uh=state
        ) + asm(
            inlet_exchange,
            inlet_basis,
            uh=inlet_basis.interpolate(sols_np[:, nn]),
        )
    deltas = np.diff(times_np)
    released = float(release_rate * (times_np[-1] - times_np[0]))
    extracted = float(
        (deltas * 0.5 * (extraction[:-1] + extraction[1:])).sum()
    )
    outflowed = float((deltas * 0.5 * (outflow[:-1] + outflow[1:])).sum())
    mass_change = float(masses[-1] - masses[0])
    residual = mass_change - (released - extracted - outflowed)
    return mass_change, released, extracted, outflowed, residual


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
            # Calibration: worst observed locally is 2.1e-7 (gradient)
            # and 6.4e-12 (HVP); the CI matrix has drifted FD ratios up
            # to ~10x looser than local (see the calibrated FD unit
            # tolerances), and the bound doubles that. A broken adjoint
            # term produces O(1e-2)-O(1) ratios, far past these.
            assert jac_ratio <= 5e-6
            assert hvp_min <= 5e-10

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
        # Observed 1.11 at this configuration; half of it separates a
        # delivered plume from quadrature dust by orders of magnitude.
        assert value > 0.5

    @slow_test
    def test_mass_balance(self, numpy_bkd: NumpyBkd) -> None:
        """Discrete mass bookkeeping: the change of int(u) must match
        the exchanges the discrete dynamics make — release, extraction,
        advective loss, and the Danckwerts inlet exchange — with every
        line measured by the diagnostic's own quadrature. rel_residual
        is the ledger's closure defect relative to release (nothing to
        do with Newton residuals). The check is twofold: a tight
        absolute cap, and residual DECAY under simultaneous space-time
        refinement at frozen flow — genuine quadrature error converges,
        while a BC, advection-form, or bookkeeping error is O(1) and
        cannot."""
        bkd = numpy_bkd
        residuals = {}
        for refine, deltat in ((1, 1.0), (2, 0.5)):
            problem = ObstructedFlowControlProblem(
                bkd,
                nstokes_refine=1,
                ntransport_refine=refine,
                diffusivity=0.02,
                final_time=20.0,
                deltat=deltat,
            )
            model = problem.model()
            for label, controls_np in (
                ("p=0", np.zeros(problem.ncontrols())),
                ("p=1", np.ones(problem.ncontrols())),
            ):
                controls = bkd.asarray(controls_np[:, None])
                mass_change, released, extracted, outflowed, residual = (
                    _mass_ledger(problem, model, controls, bkd, 0.02)
                )
                rel_residual = abs(residual) / released
                residuals[(refine, label)] = rel_residual
                print(
                    f"[refine={refine}][{label}] dM={mass_change:.4f} "
                    f"released={released:.4f} extracted={extracted:.4f} "
                    f"outflowed={outflowed:.4f} "
                    f"residual={100 * rel_residual:.3f}% of release"
                )
                if label == "p=0":
                    assert extracted == 0.0
                else:
                    assert extracted > 0.0
                assert outflowed > 0.0
        # Absolute cap: 2x the observed 0.26% worst case at the
        # coarse configuration.
        for (refine, label), value in residuals.items():
            if refine == 1:
                assert value < 6e-3
        # Convergence: observed decay ~5.5x per refinement level;
        # requiring 2x keeps headroom while rejecting any O(1)
        # modeling error (which cannot decay).
        for label in ("p=0", "p=1"):
            assert residuals[(2, label)] < 0.5 * residuals[(1, label)]

    @slow_test
    def test_optimization_learns_interception(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The optimizer-in-the-loop check: both optimizers drive J
        down by an order of magnitude, agree with each other, and the
        learned strategy places its dominant EXTRACTION on the plume's
        feed paths rather than at the low-leverage placements. Horizon
        set to the release-to-zone transit time so the uncontrolled
        plume genuinely contaminates the zone. Extraction (removal
        proportional to concentration, rates bounded below by zero)
        must also preserve positivity of the state — checked at p = 0
        first to separate discretization undershoot from control
        artifacts."""
        bkd = numpy_bkd
        problem = _make_problem(bkd, final_time=20.0, deltat=1.0)
        model = problem.model()
        controls0 = bkd.zeros((problem.ncontrols(), 1))
        j0 = float(bkd.to_numpy(model(controls0))[0, 0])
        sols0, _ = model.forward_solve(controls0)
        min_u0 = float(bkd.to_numpy(bkd.min(sols0)))

        # Optimizers start STRICTLY INSIDE the box: scipy's
        # tr_interior_point terminates spuriously (one evaluation,
        # zero reported optimality) when started exactly on a bound,
        # and p = 0 sits on the extraction lower bound.
        interior_start = bkd.asarray(
            0.5 * np.ones((problem.ncontrols(), 1))
        )

        lbfgs: LBFGSBOptimizer[NumpyArray] = LBFGSBOptimizer(
            verbosity=0, maxiter=100
        )
        lbfgs.bind(model, problem.bounds())
        p_lbfgs = lbfgs.minimize(interior_start).optima()
        j_lbfgs = float(bkd.to_numpy(model(p_lbfgs))[0, 0])

        newton: ScipyTrustConstrOptimizer[NumpyArray] = (
            ScipyTrustConstrOptimizer(verbosity=0, maxiter=50, gtol=1e-8)
        )
        newton.bind(model, problem.bounds())
        p_newton = newton.minimize(interior_start).optima()
        j_newton = float(bkd.to_numpy(model(p_newton))[0, 0])

        print(
            f"J(0)={j0:.3e}  J*(L-BFGS-B)={j_lbfgs:.3e}  "
            f"J*(trust-constr)={j_newton:.3e}"
        )
        amplitudes = bkd.to_numpy(p_lbfgs)[:, 0]
        labels = problem.actuator_labels()
        for label, val in zip(labels, amplitudes):
            print(f"  {label:>20s}: {val:+.3f}")

        # Positivity: proportional extraction preserves the maximum
        # principle. p = 0 measures pure discretization undershoot;
        # the controlled state must not undershoot further. Guards
        # against reintroducing signed forcing (constant-rate sinks
        # manufacture negative mass once u ~ 0). Observed undershoot
        # is exactly zero at these configurations; the tolerance is
        # absolute headroom for CI drift.
        sols_ctl, _ = model.forward_solve(p_lbfgs)
        min_u_ctl = float(bkd.to_numpy(bkd.min(sols_ctl)))
        print(f"min(u): p=0 {min_u0:.3e}, controlled {min_u_ctl:.3e}")
        assert min_u0 >= -1e-10
        assert min_u_ctl >= -1e-10

        # Observed J*/J(0) = 0.040 for both optimizers; 0.1 is 2.5x
        # headroom and still demands the order-of-magnitude reduction
        # the docstring promises.
        assert j_lbfgs < 0.1 * j0
        assert j_newton < 0.1 * j0
        # Both optimizers minimize the same objective and agree to four
        # digits locally; 5% catches one of them stalling (e.g. the
        # interior-point pathology of starting on a bound).
        assert abs(j_lbfgs - j_newton) <= 0.05 * max(j_lbfgs, j_newton)
        # Learned strategy: rates are nonnegative, the dominant
        # extraction sits on a feed path, and the low-leverage
        # placements carry comparatively little rate (guards against
        # the optimizer parking effort where the physics barely sees
        # it).
        assert np.all(amplitudes >= -1e-12)
        dominant = int(np.argmax(amplitudes))
        assert labels[dominant] in _FEED_PATH_LABELS
        assert amplitudes[dominant] > 0.0
        feed_mass = sum(
            val
            for label, val in zip(labels, amplitudes)
            if label in _FEED_PATH_LABELS
        )
        off_path_mass = sum(
            val
            for label, val in zip(labels, amplitudes)
            if label not in _FEED_PATH_LABELS
        )
        print(
            f"feed-path rate mass {feed_mass:.3f}, "
            f"off-path {off_path_mass:.3f}"
        )
        # Observed ratio 0.13; 0.3 is >2x headroom while still failing
        # if the optimizer parks comparable effort at the low-leverage
        # placements (the failure this guards against).
        assert off_path_mass < 0.3 * feed_mass
