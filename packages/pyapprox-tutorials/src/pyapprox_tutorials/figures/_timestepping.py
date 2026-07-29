"""Plotting functions for the ODE time-stepping usage tutorial.

Covers: ode_timestepping_usage.qmd

The convergence study needs a problem whose exact solution is known, so
that measured errors are true errors rather than distances to a finer
approximation of the same scheme. A constant-coefficient linear system
supplies one: :math:`\\dot{y} = A y` has solution
:math:`y(t) = e^{At} y_0`.
"""

import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import expm

from ._style import COLORS


class SimpleODEResidual:
    """Minimal ODE residual for use with PyApprox steppers.

    Wraps f(y) and J(y) callables for autonomous systems without
    parameters. Satisfies ImplicitODEResidualProtocol.

    Lives here rather than with the dynamical-systems figures because
    the time-stepping tutorial is where the residual/stepper pipeline is
    introduced, and because nothing about it is specific to dynamical
    systems.
    """

    def __init__(self, f_fn, jac_fn, nstates):
        from pyapprox.ode.mass_matrix import IdentityMassMatrix
        from pyapprox.util.backends.numpy import NumpyBkd
        self._f = f_fn
        self._jac = jac_fn
        self._bkd = NumpyBkd()
        self._mass = IdentityMassMatrix(nstates, self._bkd)

    def bkd(self):
        return self._bkd

    def __call__(self, state):
        return self._f(state)

    def set_time(self, time):
        pass

    def jacobian(self, state):
        return self._jac(state)

    def mass_matrix(self):
        return self._mass

    def newton_jacobian(self, state, coefficient):
        from pyapprox.ode.linear_operator import MatrixOperator
        matrix = self._mass.as_matrix() - coefficient * self._jac(state)
        return MatrixOperator(matrix, self._bkd)


def integrate_ode(stepper_class, residual, ic, deltat, nsteps, newton_tol=1e-12):
    """Integrate an ODE with a PyApprox TimeIntegrator.

    Returns ``(states, times)`` with every state component, unlike the
    two-component convenience wrapper the phase-space figures use.
    """
    from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
    from pyapprox.util.rootfinding.newton import NewtonSolver

    stepper = stepper_class(residual)
    solver = NewtonSolver(stepper)
    solver.set_options(atol=newton_tol, rtol=newton_tol)
    integrator = TimeIntegrator(0.0, nsteps * deltat, deltat, solver)
    states, times = integrator.solve(np.array(ic, dtype=float))
    return np.asarray(states), np.asarray(times)

# Stable and non-normal, with eigenvalues spread over about a factor of
# four: the exact solution decays without the problem being stiff, so a
# convergence study is not contaminated by stability effects.
_LINEAR_MATRIX = np.array(
    [[-2.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, -1.5]]
)
_LINEAR_INITIAL = np.array([1.0, -0.5, 2.0])


def linear_reference():
    """The matrix, initial condition, and exact-solution callable.

    Returned together so a tutorial cell can state the problem and the
    figure helpers can reuse it without duplicating the constants.
    """

    def exact(time):
        return expm(_LINEAR_MATRIX * time) @ _LINEAR_INITIAL

    return _LINEAR_MATRIX, _LINEAR_INITIAL, exact


def _linear_residual():
    return SimpleODEResidual(
        lambda state: _LINEAR_MATRIX @ state,
        lambda state: _LINEAR_MATRIX,
        _LINEAR_MATRIX.shape[0],
    )


def _run(stepper_class, deltat, final_time):
    """Integrate the linear system and return every computed state.

    The Newton tolerance is tighter than the discretization error being
    measured, so a convergence study sees the scheme's truncation error
    rather than the solver's stopping criterion.
    """
    return integrate_ode(
        stepper_class,
        _linear_residual(),
        _LINEAR_INITIAL,
        deltat,
        int(round(final_time / deltat)),
        newton_tol=1e-13,
    )


def measure_temporal_order(stepper_classes, deltats, final_time=2.0):
    """Final-time errors against the matrix-exponential solution.

    Returns ``{name: (errors, rate)}``, where ``rate`` is the slope of a
    least-squares line through the log-log data --- the observed order
    of accuracy, which is what a convergence study actually measures.
    """
    _, _, exact = linear_reference()
    truth = exact(final_time)
    results = {}
    for name, stepper_class in stepper_classes.items():
        errors = []
        for deltat in deltats:
            states, _ = _run(stepper_class, deltat, final_time)
            errors.append(float(np.linalg.norm(states[:, -1] - truth)))
        errors = np.array(errors)
        rate = float(np.polyfit(np.log(deltats), np.log(errors), 1)[0])
        results[name] = (errors, rate)
    return results


def plot_temporal_convergence(deltats, results, ax):
    """ode_timestepping_usage.qmd -> fig-dt-convergence

    Error at the final time against step size, log-log, with dashed
    slope guides at the theoretical orders.
    """
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for (name, (errors, rate)), color in zip(results.items(), palette):
        ax.loglog(deltats, errors, "-o", color=color,
                  label=f"{name} (measured {rate:.2f})")
    # Anchor each guide to a curve of that order and offset it downward,
    # so the guide sits beside the data it describes instead of hiding
    # underneath it (a slope-1 guide anchored to a first-order curve is
    # invisible) or floating in empty space.
    deltats_np = np.array(deltats)
    drawn = set()
    for (errors, rate), color in zip(results.values(), palette):
        order = int(round(rate))
        if order in drawn:
            continue
        drawn.add(order)
        guide = 0.35 * errors[0] * (deltats_np / deltats_np[0]) ** order
        ax.loglog(deltats_np, guide, "--", color=color, lw=1.1, alpha=0.7,
                  label=rf"$\Delta t^{{{order}}}$ reference")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\|y_N - y(T)\|_2$")
    ax.legend(fontsize=8)
    return ax


def linear_eigenvalues():
    """Eigenvalues of the reference matrix, via the symmetric routine.

    ``A`` is symmetric, so ``eigvalsh`` applies and the spectrum is
    guaranteed real; all three values are negative, which is what makes
    the scalar test ``|1 + dt*lambda| <= 1`` reduce to the interval
    ``dt <= 2/|lambda|`` rather than a region in the complex plane.
    """
    return np.linalg.eigvalsh(_LINEAR_MATRIX)


def stability_threshold():
    """Forward Euler's step-size limit for this matrix, ``2/|lambda|_max``.

    Real negative eigenvalues make the explicit stability condition the
    interval ``dt*|lambda| < 2``, and the most negative eigenvalue —
    the fastest-decaying mode — is the binding one.
    """
    return 2.0 / float(np.max(np.abs(linear_eigenvalues())))


def measure_stability(deltats, final_time=2.0):
    """Peak state magnitude for forward and backward Euler per step size.

    A stable run stays near the size of the initial condition; an
    unstable one grows without bound. Overflow is caught and reported at
    a ceiling so a diverged run plots instead of breaking the axis.

    ``final_time`` is chosen so the unstable runs overshoot the largest
    value the TRUE solution ever takes by a factor of roughly ten, not a
    factor of a thousand: the figure should show that the explicit
    scheme leaves the physical scale, not saturate the axis. Every step
    size covers the SAME interval, so the comparison is not confounded
    by different runs seeing different dynamics.

    The unstable branch is NOT monotone in ``deltat``, and that is
    genuine rather than an artifact: growth over a fixed interval is
    ``|1 + dt*lambda|**(T/dt)``, so a larger step amplifies more per
    step but takes fewer steps, and the two effects compete.
    """
    from pyapprox.ode.explicit_steppers.forward_euler import ForwardEulerHVP
    from pyapprox.ode.implicit_steppers.backward_euler import BackwardEulerHVP

    ceiling = 1e12
    results = {}
    for name, stepper_class in (
        ("forward Euler", ForwardEulerHVP),
        ("backward Euler", BackwardEulerHVP),
    ):
        peaks = []
        for deltat in deltats:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    states, _ = _run(stepper_class, deltat, final_time)
                peak = float(np.nanmax(np.abs(states)))
            except (FloatingPointError, OverflowError, RuntimeError):
                peak = ceiling
            peaks.append(min(peak if np.isfinite(peak) else ceiling, ceiling))
        results[name] = np.array(peaks)
    return results


def plot_stability(deltats, results, ax):
    """ode_timestepping_usage.qmd -> fig-stability

    Peak state magnitude against step size. The explicit scheme diverges
    past its stability limit; the implicit one stays bounded at every
    step size shown.
    """
    palette = {"forward Euler": COLORS["secondary"],
               "backward Euler": COLORS["primary"]}
    for name, peaks in results.items():
        # Linear y: the interesting range spans well under a decade, and
        # a log axis would flatten the very departure being shown.
        ax.semilogx(deltats, peaks, "-o", color=palette[name], label=name)
    threshold = stability_threshold()
    ax.axvline(threshold, color=COLORS["reference"], ls="--", lw=1.4,
               label=rf"$2/|\lambda|_{{\max}} = {threshold:.2f}$")
    # The largest value the true solution ever takes: the scale an
    # accurate run cannot exceed. This system decays, so the peak is at
    # t = 0, but computing it from the exact solution keeps the line
    # correct for any reference problem.
    _, _, exact = linear_reference()
    true_peak = max(
        float(np.max(np.abs(exact(t)))) for t in np.linspace(0.0, 20.0, 401)
    )
    ax.axhline(true_peak, color="0.5", ls=":", lw=1.2,
               label=f"peak of exact solution ({true_peak:.1f})")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\max_n \|y_n\|_\infty$")
    ax.legend(fontsize=8)
    return ax


def _amplification_factors():
    """Amplification factor R(z) for each scheme, z = deltat * lambda.

    One step multiplies the component along an eigenvector by R(z); the
    scheme is stable for that component exactly when |R(z)| <= 1.
    """
    return {
        "forward Euler": lambda z: 1.0 + z,
        "backward Euler": lambda z: 1.0 / (1.0 - z),
        "Crank-Nicolson": lambda z: (1.0 + z / 2.0) / (1.0 - z / 2.0),
    }


def plot_stability_regions(axes, deltats_to_mark=(0.3, 0.9), extent=4.0,
                           ngrid=400):
    """ode_timestepping_usage.qmd -> fig-stability-regions

    Absolute stability regions in the complex plane, with the scaled
    eigenvalues ``deltat * lambda`` of the tutorial's matrix overlaid for
    a stable and an unstable step size. A run is stable exactly when
    EVERY scaled eigenvalue lies inside the shaded region, which is what
    turns an abstract region into a step-size limit.
    """
    real = np.linspace(-extent, extent, ngrid)
    imag = np.linspace(-extent, extent, ngrid)
    grid = real[None, :] + 1j * imag[:, None]
    eigvals = linear_eigenvalues()
    markers = ["o", "s", "^"]
    for ax, (name, factor) in zip(axes, _amplification_factors().items()):
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.abs(factor(grid))
        ax.contourf(real, imag, magnitude, levels=[0.0, 1.0],
                    colors=[COLORS["primary"]], alpha=0.22)
        ax.contour(real, imag, magnitude, levels=[1.0],
                   colors=[COLORS["primary"]], linewidths=1.6)
        for deltat, color in zip(deltats_to_mark,
                                 (COLORS["accent"], COLORS["reference"])):
            scaled = deltat * eigvals
            inside = np.abs(factor(scaled)) <= 1.0
            for point, marker, ok in zip(scaled, markers, inside):
                ax.plot(point.real, point.imag, marker, color=color,
                        markersize=8, markerfacecolor=color if ok else "none",
                        markeredgewidth=1.8, zorder=5)
        ax.axhline(0.0, color="0.7", lw=0.7)
        ax.axvline(0.0, color="0.7", lw=0.7)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(r"$\mathrm{Re}(\Delta t\,\lambda)$")
    axes[0].set_ylabel(r"$\mathrm{Im}(\Delta t\,\lambda)$")
    # The filled/hollow distinction is the whole point of the overlay, so
    # spell it out rather than leaving it to be inferred from one marker.
    handles = [
        Line2D([], [], ls="none", marker="o", color=color, markersize=8,
               label=rf"$\Delta t = {deltat}$")
        for deltat, color in zip(deltats_to_mark,
                                 (COLORS["accent"], COLORS["reference"]))
    ]
    handles += [
        Line2D([], [], ls="none", marker="o", color="0.35", markersize=8,
               label="inside: mode is damped"),
        Line2D([], [], ls="none", marker="o", color="0.35", markersize=8,
               markerfacecolor="none", markeredgewidth=1.8,
               label="outside: mode grows"),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="lower left")
    return axes


def make_sho_residual(omega=1.0):
    """Simple harmonic oscillator: H = (p^2 + omega^2 q^2)/2, quadratic."""
    matrix = np.array([[0.0, 1.0], [-omega ** 2, 0.0]])
    return SimpleODEResidual(
        f_fn=lambda state: matrix @ state,
        jac_fn=lambda state: matrix,
        nstates=2,
    )


def make_pendulum_residual():
    """Simple pendulum: H = p^2/2 - cos(q), NOT quadratic."""
    return SimpleODEResidual(
        f_fn=lambda state: np.array([state[1], -np.sin(state[0])]),
        jac_fn=lambda state: np.array(
            [[0.0, 1.0], [-np.cos(state[0]), 0.0]]
        ),
        nstates=2,
    )


def plot_integrator_energy_comparison(ax_sho, ax_pendulum):
    """hamiltonian_integration_concept.qmd -> fig-integrator-comparison.

    Left: SHO. H = 0.5*(p^2 + omega^2*q^2). BE decays; CN preserves;
    IM preserves (coincides with CN on linear problems).
    Right: Pendulum. H = 0.5*p^2 - cos(q). BE decays; CN drifts
    secularly; IM preserves.
    """
    from pyapprox.ode.implicit_steppers.backward_euler import (
        BackwardEulerStepper,
    )
    from pyapprox.ode.implicit_steppers.crank_nicolson import (
        CrankNicolsonStepper,
    )
    from pyapprox.ode.implicit_steppers.implicit_midpoint import (
        ImplicitMidpointStepper,
    )

    from ._style import apply_style

    steppers = [BackwardEulerStepper, CrankNicolsonStepper,
                ImplicitMidpointStepper]

    styles = [
        ("backward Euler", COLORS["reference"], "-", 2.0),
        ("Crank-Nicolson", COLORS["primary"], "--", 1.8),
        ("implicit midpoint", COLORS["accent"], "-.", 1.8),
    ]

    # ----- Left panel: SHO, 30 periods ---------------------------------
    omega = 1.0
    final_time = 30 * 2 * np.pi
    deltat = 0.05
    nsteps = int(np.ceil(final_time / deltat))

    initial = (1.0, 0.0)
    energy0_sho = 0.5 * (initial[1] ** 2 + omega ** 2 * initial[0] ** 2)

    sho_residual = make_sho_residual(omega)
    results_sho = {}
    for stepper_class in steppers:
        states, times = integrate_ode(
            stepper_class, sho_residual, initial, deltat, nsteps
        )
        results_sho[stepper_class] = (states[0], states[1], times)

    for stepper_class, (label, color, style, width) in zip(steppers, styles):
        coord, momentum, times = results_sho[stepper_class]
        energy = (
            0.5 * (momentum ** 2 + omega ** 2 * coord ** 2) - energy0_sho
        )
        ax_sho.plot(times, energy, color=color, lw=width, ls=style,
                    label=label)

    ax_sho.set_xlabel(r"$t$", fontsize=11)
    ax_sho.set_ylabel(r"$\mathcal{H}(t) - \mathcal{H}(0)$", fontsize=11)
    ax_sho.set_title("SHO  (linear, quadratic $\\mathcal{H}$)", fontsize=11)
    ax_sho.legend(fontsize=9, loc="lower left")
    apply_style(ax_sho)

    # ----- Right panel: pendulum ---------------------------------------
    # Coarser step so the Crank-Nicolson energy oscillation is large
    # enough to see in an inset, separate from the dominant BE decay.
    final_time_pend = 50 * 2 * np.pi
    deltat_pend = 0.3
    nsteps_pend = int(np.ceil(final_time_pend / deltat_pend))

    initial_pend = (2.8, 0.0)
    energy0_pend = 0.5 * initial_pend[1] ** 2 - np.cos(initial_pend[0])

    pendulum_residual = make_pendulum_residual()
    energies_pend = {}
    for stepper_class, (label, color, style, width) in zip(steppers, styles):
        states, times = integrate_ode(
            stepper_class, pendulum_residual, initial_pend, deltat_pend,
            nsteps_pend,
        )
        energy = 0.5 * states[1] ** 2 - np.cos(states[0]) - energy0_pend
        ax_pendulum.plot(times, energy, color=color, lw=width, ls=style,
                         label=label)
        energies_pend[stepper_class] = (times, energy)

    ax_pendulum.set_xlabel(r"$t$", fontsize=11)
    ax_pendulum.set_ylabel(r"$\mathcal{H}(t) - \mathcal{H}(0)$", fontsize=11)
    ax_pendulum.set_title(
        "Pendulum  (nonlinear $\\mathcal{H} = \\frac{1}{2} p^2 - \\cos q$)",
        fontsize=11,
    )
    ax_pendulum.legend(fontsize=9, loc="lower left")
    apply_style(ax_pendulum)

    # Inset: zoom into CN vs IM, since BE is off-scale.
    ax_inset = ax_pendulum.inset_axes([0.38, 0.35, 0.58, 0.55])
    for stepper_class, (_, color, style, width) in zip(
        steppers[1:], styles[1:],
    ):
        times, energy = energies_pend[stepper_class]
        ax_inset.plot(times, energy, color=color, lw=width, ls=style)
    ax_inset.set_ylabel(r"$\Delta\mathcal{H}$", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title("CN vs IM (zoom)", fontsize=8)


def plot_trajectories(times, states, ax, labels=None):
    """ode_timestepping_usage.qmd -> fig-lv-trajectories

    One line per state component against time.
    """
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    for index in range(states.shape[0]):
        label = None if labels is None else labels[index]
        ax.plot(times, states[index], lw=2,
                color=palette[index % len(palette)], label=label)
    ax.set_xlabel("$t$")
    ax.set_ylabel("state")
    if labels is not None:
        ax.legend(fontsize=8)
    return ax
