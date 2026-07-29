"""Plotting functions for the adjoint concept tutorials.

Covers: adjoint_concept.qmd, adjoint_hvp_concept.qmd,
adjoint_ode_concept.qmd
"""

import matplotlib.pyplot as plt
import numpy as np
from pyapprox.interface.functions.derivative_checks.plots import (
    plot_fd_error_sweep,
)
from pyapprox.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
)

from ._style import COLORS, NEON_CMAP


def _draw_block_bidiagonal(ax, nsteps, transposed, title):
    """One block-bidiagonal matrix with its substitution direction.

    Draws the all-at-once Jacobian of a time-stepping scheme: diagonal
    blocks for the step residual's dependence on its own state, and one
    off-diagonal for its dependence on the previous state. Transposing
    moves the off-diagonal from below the diagonal to above it, which is
    what turns forward substitution into backward substitution.
    """
    for row in range(nsteps):
        for col in range(nsteps):
            on_diagonal = row == col
            below = row == col + 1
            above = col == row + 1
            occupied = on_diagonal or (above if transposed else below)
            if not occupied:
                continue
            color = COLORS["primary"] if on_diagonal else COLORS["secondary"]
            ax.add_patch(
                plt.Rectangle(
                    (col, nsteps - 1 - row), 1, 1, facecolor=color,
                    alpha=0.75 if on_diagonal else 0.5,
                    edgecolor="white", lw=1.5,
                )
            )
            if on_diagonal:
                label = rf"$A_{{{row + 1}}}^\top$" if transposed \
                    else rf"$A_{{{row + 1}}}$"
            else:
                label = r"$B^\top$" if transposed else r"$B$"
            ax.text(col + 0.5, nsteps - 0.5 - row, label, ha="center",
                    va="center", fontsize=9, color="white")
    # The order the blocks are resolved in: down the diagonal for the
    # untransposed system, up it for the transposed one.
    start, stop = (0.5, nsteps - 0.5) if transposed else (nsteps - 0.5, 0.5)
    ax.annotate(
        "",
        xy=(-0.55, stop), xytext=(-0.55, start),
        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=COLORS["accent"]),
    )
    ax.text(
        -1.15, nsteps / 2,
        "solved backward\nin time" if transposed else "solved forward\nin time",
        rotation=90, ha="center", va="center", fontsize=8,
        color=COLORS["accent"],
    )
    ax.set_xlim(-1.5, nsteps)
    ax.set_ylim(-0.3, nsteps + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return ax


def plot_all_at_once_blocks(axes, nsteps=4):
    """adjoint_ode_concept.qmd -> fig-all-at-once

    The all-at-once Jacobian beside its transpose. Both are block
    bidiagonal; the transpose moves the coupling block across the
    diagonal, which reverses the direction the system is solved in.
    """
    # Plain mathtext: matplotlib does not know the document's LaTeX
    # macros, so figure labels spell the symbols out.
    _draw_block_bidiagonal(
        axes[0], nsteps, transposed=False,
        title=r"$\mathbf{C}_{\mathbf{Y}}$  (block lower bidiagonal)",
    )
    _draw_block_bidiagonal(
        axes[1], nsteps, transposed=True,
        title=r"$\mathbf{C}_{\mathbf{Y}}^\top$  (block upper bidiagonal)",
    )
    return axes


def plot_quadrature_rules(axes, nintervals=4):
    """transient_functionals_concept.qmd -> fig-quadrature-rules

    What each rule assumes about the integrand between stored states:
    a constant taken from the left endpoint, from the right endpoint,
    from the interval midpoint, or a straight line between the two ends.
    The shaded area is what the rule actually sums; the curve is the
    integrand it is approximating.
    """
    def integrand(time):
        return 1.0 + 0.6 * np.sin(1.7 * time) + 0.25 * time

    edges = np.linspace(0.0, 4.0, nintervals + 1)
    fine = np.linspace(edges[0], edges[-1], 400)
    rules = (
        ("left rectangle", "forward Euler"),
        ("right rectangle", "backward Euler"),
        ("midpoint", "implicit midpoint"),
        ("trapezoidal", "Crank-Nicolson"),
    )
    for ax, (rule, scheme) in zip(axes, rules):
        for left, right in zip(edges[:-1], edges[1:]):
            if rule == "trapezoidal":
                xs = [left, left, right, right]
                ys = [0.0, integrand(left), integrand(right), 0.0]
            else:
                sample = {
                    "left rectangle": left,
                    "right rectangle": right,
                    "midpoint": 0.5 * (left + right),
                }[rule]
                height = integrand(sample)
                xs = [left, left, right, right]
                ys = [0.0, height, height, 0.0]
            ax.fill(xs, ys, facecolor=COLORS["primary"], alpha=0.25,
                    edgecolor=COLORS["primary"], lw=1.2)
        ax.plot(fine, integrand(fine), color=COLORS["reference"], lw=2.0)
        # Mark the states each rule evaluates at.
        if rule == "midpoint":
            samples = 0.5 * (edges[:-1] + edges[1:])
        elif rule == "left rectangle":
            samples = edges[:-1]
        elif rule == "right rectangle":
            samples = edges[1:]
        else:
            samples = edges
        ax.plot(samples, integrand(samples), "o", color=COLORS["reference"],
                markersize=5, zorder=5)
        ax.set_title(f"{rule}\n({scheme})", fontsize=9)
        ax.set_xticks(edges)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_ylim(0.0, 2.6)
    axes[0].set_ylabel(r"$q(y(t))$")
    return axes


def plot_quadrature_orders(deltats, errors, ax):
    """transient_functionals_concept.qmd -> fig-quadrature-mismatch

    Error in a time-integrated quantity of interest against step size,
    log-log, one curve per scheme-and-rule pairing. Returns the observed
    order of each curve so the tutorial can assert on them.
    """
    palette = [COLORS["primary"], COLORS["reference"], COLORS["gray"]]
    styles = ["-o", "-s", "--^"]
    deltats_np = np.asarray(deltats)
    orders = {}
    for (label, curve), color, style in zip(errors.items(), palette, styles):
        curve_np = np.asarray(curve)
        order = float(np.polyfit(np.log(deltats_np), np.log(curve_np), 1)[0])
        orders[label] = order
        ax.loglog(deltats_np, curve_np, style, color=color,
                  label=f"{label} ({order:.2f})")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"error in $Q$")
    ax.legend(fontsize=7)
    return orders


def measure_gradient_cost(nparams_list, nstates=2, deltat=0.1, final_time=1.0):
    """adjoint_ode_concept.qmd -> data for fig-cost-vs-nparams

    Times one adjoint gradient against a central-difference gradient on
    a linear ODE, as the parameter count grows. Returns
    ``(adjoint_times, fd_times)`` in seconds.
    """
    import time

    from pyapprox.ode.functionals.endpoint import EndpointFunctional
    from pyapprox.ode.implicit_steppers.backward_euler import (
        BackwardEulerAdjoint,
    )
    from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.util.rootfinding.newton import NewtonSolver
    from pyapprox_benchmarks.functions.ode.linear_ode import LinearODEResidual

    bkd = NumpyBkd()
    rng = np.random.RandomState(0)
    state_matrix = bkd.asarray(np.array([[-1.0, 0.5], [0.2, -2.0]]))
    init_state = bkd.asarray(np.array([1.0, 0.5]))
    adjoint_times, fd_times = [], []
    for nparams in nparams_list:
        forcing = bkd.asarray(rng.normal(0, 1, (nstates, nparams)))
        residual = LinearODEResidual(state_matrix, forcing, bkd)
        integrator = TimeIntegrator(
            0.0, final_time, deltat,
            NewtonSolver(BackwardEulerAdjoint(residual)),
        )
        integrator.set_functional(
            EndpointFunctional(
                state_idx=0, nstates=nstates, nparams=nparams, bkd=bkd
            )
        )
        param = bkd.asarray(rng.normal(0, 1, (nparams, 1)))
        residual.set_param(bkd.flatten(param))
        states, times = integrator.solve(init_state)

        start = time.perf_counter()
        integrator.gradient(states, times, param)
        adjoint_times.append(time.perf_counter() - start)

        # Central differences: two extra forward solves per parameter.
        start = time.perf_counter()
        for index in range(nparams):
            for sign in (1.0, -1.0):
                perturbed = bkd.copy(param)
                perturbed[index, 0] += sign * 1e-6
                residual.set_param(bkd.flatten(perturbed))
                integrator.solve(init_state)
        fd_times.append(time.perf_counter() - start)
        residual.set_param(bkd.flatten(param))
    return adjoint_times, fd_times


def plot_gradient_cost_scaling(nparams_list, adjoint_times, fd_times, ax):
    """adjoint_ode_concept.qmd -> fig-cost-vs-nparams

    Wall-clock cost of one gradient against the number of parameters.
    The adjoint line is flat; finite differences grow linearly because
    each parameter needs its own perturbed solve.
    """
    ax.plot(nparams_list, np.array(fd_times) * 1e3, "-o",
            color=COLORS["secondary"], label="finite differences")
    ax.plot(nparams_list, np.array(adjoint_times) * 1e3, "-o",
            color=COLORS["primary"], label="adjoint")
    ax.set_xlabel("number of parameters")
    ax.set_ylabel("time per gradient [ms]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return ax


def _reduced_hessian(function, bkd, point):
    """Assemble the reduced Hessian by applying the hvp to unit vectors.

    Affordable only because there are two parameters; it exists so the
    figures can draw eigen-directions, not as a way to compute Hessians.
    """
    hessian = np.column_stack(
        [
            bkd.to_numpy(function.hvp(point, bkd.asarray(unit[:, None]))).ravel()
            for unit in np.eye(function.nvars())
        ]
    )
    return np.linalg.eigh(0.5 * (hessian + hessian.T))


def _draw_hessian_axes(ax, point, eigvals, eigvecs, plot_limits, fraction=0.3):
    """Level-set axes of the local quadratic model.

    An axis is drawn with length proportional to ``1/sqrt(|lambda|)``,
    the semi-axis of the ellipse ``d^T H d = const``: SOFT directions
    are long, stiff ones short, and their ratio is the square root of
    the condition number. Positive curvature is drawn in the accent
    color, negative in the reference color.

    The two axes of the plot span different ranges, so a direction is
    converted to display proportions before being scaled and converted
    back. Without that the arrows would misreport their own directions.
    """
    spans = np.array(
        [plot_limits[1] - plot_limits[0], plot_limits[3] - plot_limits[2]]
    )
    semi_axes = 1.0 / np.sqrt(np.maximum(np.abs(eigvals), 1e-12))
    scale = fraction / semi_axes.max()
    for value, vector, semi_axis in zip(eigvals, eigvecs.T, semi_axes):
        # Unit direction in display proportions, so equal-length arrows
        # look equal regardless of the axis ranges.
        display = vector / np.linalg.norm(vector / spans) / spans
        offset = scale * semi_axis * display * spans
        color = COLORS["accent"] if value > 0 else COLORS["reference"]
        ax.annotate(
            "",
            xy=point + offset,
            xytext=point - offset,
            arrowprops=dict(arrowstyle="<|-|>", lw=2.2, color=color),
        )


def plot_qoi_surface_with_gradient(
    function, point, ax, plot_limits=(0.1, 0.9, 0.2, 2.0), npts_1d=81
):
    """adjoint_concept.qmd -> fig-qoi-surface

    The implicitly defined QoI over the parameter box, rendered by
    PyApprox's own 2D plotter, with the adjoint gradient drawn at
    ``point``. The solid arrow points uphill; the dashed arrow is the
    descent direction an optimizer would follow.
    """
    bkd = function.bkd()
    plotter = Plotter2DRectangularDomain(function, plot_limits)
    contours = plotter.plot_contours(
        ax, npts_1d=npts_1d, levels=40, cmap=NEON_CMAP
    )
    point_np = bkd.to_numpy(point).ravel()
    gradient = bkd.to_numpy(function.jacobian(point)).ravel()
    span = min(plot_limits[1] - plot_limits[0], plot_limits[3] - plot_limits[2])
    # Scale for legibility against the box, not to physical units.
    scale = 0.32 * span / max(np.linalg.norm(gradient), 1e-12)
    ax.annotate(
        "",
        xy=point_np + scale * gradient,
        xytext=point_np,
        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=COLORS["secondary"]),
    )
    ax.annotate(
        "",
        xy=point_np - scale * gradient,
        xytext=point_np,
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="white", linestyle="--"),
    )
    ax.plot(*point_np, "o", color="white", markersize=8, zorder=6)
    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    return contours


def plot_curvature_panels(
    function, points, labels, axes, plot_limits=(0.1, 0.9, 0.2, 2.0), npts_1d=81
):
    """adjoint_hvp_concept.qmd -> fig-curvature

    The QoI field with the Hessian's level-set axes drawn at each point.
    Axis lengths go like ``1/sqrt(|lambda|)``, so the long axis is the
    direction the objective barely resists and the short axis the one it
    resists most; their ratio is the square root of the condition
    number.
    """
    bkd = function.bkd()
    contours = None
    for ax, point, label in zip(axes, points, labels):
        plotter = Plotter2DRectangularDomain(function, plot_limits)
        contours = plotter.plot_contours(
            ax, npts_1d=npts_1d, levels=40, cmap=NEON_CMAP
        )
        point_np = bkd.to_numpy(point).ravel()
        eigvals, eigvecs = _reduced_hessian(function, bkd, point)
        _draw_hessian_axes(ax, point_np, eigvals, eigvecs, plot_limits)
        ax.plot(*point_np, "o", color="white", markersize=8, zorder=6)
        kind = "positive definite" if eigvals.min() > 0 else "indefinite"
        ax.set_title(
            f"{label}\n$\\lambda$ = ({eigvals[0]:+.3f}, {eigvals[1]:+.3f})"
            f" --- {kind}",
            fontsize=9,
        )
        ax.set_xlabel("$a$")
    axes[0].set_ylabel("$b$")
    return contours


def _torch_newton_qoi(param, niters, init_state):
    """Newton on the coupled state equations, written so autograd can
    tape it, returning the second state component.

    This is a deliberately hand-rolled loop: the point of the timing
    comparison is what reverse-mode AD does when it differentiates the
    ITERATION rather than the converged solution, so the loop must be
    on the tape.
    """
    import torch

    a, b = param[0], param[1]
    state = init_state.clone()
    for _ in range(niters):
        first, second = state[0], state[1]
        residual = torch.stack(
            [a * first**2 + second**2 - 1.0, first**2 - b * second**2 - 1.0]
        )
        jacobian = torch.stack(
            [
                torch.stack([2 * a * first, 2 * second]),
                torch.stack([2 * first, -2 * b * second]),
            ]
        )
        state = state - torch.linalg.solve(jacobian, residual)
    return state[1]


def measure_ad_vs_adjoint(function, point, newton_iters=(5, 10, 20), repeats=3):
    """adjoint_concept.qmd -> data for the AD comparison table.

    Times one adjoint gradient against reverse-mode AD applied to a
    taped Newton loop, and counts the autograd graph nodes the tape
    holds. Returns ``(adjoint_seconds, rows)`` where each row is
    ``(iterations, ad_seconds, ratio, tape_nodes)``.
    """
    import time

    import torch

    bkd = function.bkd()
    function.jacobian(point)  # warm up caches before timing
    start = time.perf_counter()
    for _ in range(repeats):
        function.jacobian(point)
    adjoint_seconds = (time.perf_counter() - start) / repeats

    point_np = bkd.to_numpy(point).ravel()
    init_state = torch.full((2,), 0.75, dtype=torch.double)
    rows = []
    for niters in newton_iters:
        def run():
            params = torch.tensor(point_np, dtype=torch.double, requires_grad=True)
            _torch_newton_qoi(params, niters, init_state).backward()

        run()  # warm up
        start = time.perf_counter()
        for _ in range(repeats):
            run()
        ad_seconds = (time.perf_counter() - start) / repeats
        # Count nodes reachable in the autograd graph: this is what the
        # reverse sweep has to walk back through.
        params = torch.tensor(point_np, dtype=torch.double, requires_grad=True)
        value = _torch_newton_qoi(params, niters, init_state)
        seen, stack, nodes = set(), [value.grad_fn], 0
        while stack:
            node = stack.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            nodes += 1
            stack.extend(nxt for nxt, _ in node.next_functions)
        rows.append((niters, ad_seconds, ad_seconds / adjoint_seconds, nodes))
    return adjoint_seconds, rows


def plot_ad_comparison(adjoint_seconds, rows, ax):
    """adjoint_concept.qmd -> fig-ad-comparison

    Cost of one gradient against the number of Newton iterations the
    forward solve takes. The adjoint line is flat because it uses only
    the converged linearization; the AD line rises because reverse mode
    differentiates every iteration on the tape.
    """
    iterations = [row[0] for row in rows]
    ad_times = [row[1] * 1e6 for row in rows]
    tape_nodes = [row[3] for row in rows]
    ax.plot(iterations, ad_times, "-o", color=COLORS["secondary"],
            label="reverse-mode AD (taped Newton loop)")
    ax.axhline(adjoint_seconds * 1e6, color=COLORS["primary"], lw=2.2,
               label="adjoint (one transposed solve)")
    ax.set_xlabel("Newton iterations in the forward solve")
    ax.set_ylabel(r"time per gradient [$\mu$s]")
    ax.legend(fontsize=8)
    twin = ax.twinx()
    twin.plot(iterations, tape_nodes, "--s", color=COLORS["gray"], markersize=4)
    twin.set_ylabel("autograd graph nodes", color=COLORS["gray"])
    twin.tick_params(axis="y", colors=COLORS["gray"])
    return ax


def plot_gradient_cost_table(nparams_list, ax):
    """adjoint_concept.qmd -> fig-cost-table

    State-equation solves needed for one gradient, by method, rendered
    as a table. Finite differences and the tangent (forward
    sensitivity) method both scale with the number of parameters; the
    adjoint method costs two solves whatever that number is.
    """
    rows = [
        ("finite differences", lambda nz: nz + 1),
        ("tangent (forward sensitivity)", lambda nz: nz + 1),
        ("adjoint", lambda _: 2),
    ]
    cell_text = [[str(count(nz)) for nz in nparams_list] for _, count in rows]
    table = ax.table(
        cellText=cell_text,
        rowLabels=[label for label, _ in rows],
        colLabels=[rf"$n_z = {nz}$" for nz in nparams_list],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Highlight the adjoint row: the constant one.
    for col in range(len(nparams_list)):
        table[len(rows), col].set_facecolor("#dceefb")
    ax.axis("off")
    return table


def _broken_adjoint_for_figure(state_eq, functional, init_state, param):
    """A DELIBERATELY WRONG gradient, for the flat line of fig-v-curve.

    Step (2) of the adjoint recipe solves the transposed system
    ``c_u^T lambda = -f_u^T``. This drops the transpose and solves
    ``c_u lambda = -f_u^T`` instead. The result has the right shape and
    a plausible magnitude, and is wrong in every component. It exists
    only so the tutorial can show what a wrong derivative looks like
    under a step-size sweep; it is never exported from this package and
    must not be used as if it computed anything.
    """
    fwd_state = state_eq.solve(init_state, param)
    state_jac = state_eq.state_jacobian(fwd_state, param)
    qoi_state_jac = functional.state_jacobian(fwd_state, param)
    bkd = state_eq.bkd()
    adj_state = bkd.solve(state_jac, -qoi_state_jac.T)
    return functional.param_jacobian(
        fwd_state, param
    ) + adj_state.T @ state_eq.param_jacobian(fwd_state, param)


def fd_error_sweep(objective, gradient, param, direction, fd_eps, bkd):
    """adjoint_concept.qmd -> data for fig-v-curve

    Relative error between a directional derivative and its one-sided
    finite-difference quotient, at each step size in ``fd_eps``. This
    mirrors what :class:`JVPChecker` computes, for a gradient supplied
    as a plain callable (the tutorial's hand-rolled and broken
    gradients are not Derivatives bundles).
    """
    base_value = objective(param)
    directional = float(gradient(param) @ direction)
    errors = []
    for eps in bkd.to_numpy(fd_eps):
        quotient = (objective(param + eps * direction) - base_value) / eps
        errors.append(abs(quotient - directional) / abs(directional))
    return bkd.asarray(np.array(errors))


def plot_v_curve(fd_eps, correct_errors, broken_errors, bkd, ax):
    """adjoint_concept.qmd -> fig-v-curve

    The step-size sweep: the correct gradient traces a V (truncation
    error falling, a rounding-error floor, then rising), the broken
    gradient a flat line no step size can rescue.
    """
    plot_fd_error_sweep(
        fd_eps,
        [correct_errors, broken_errors],
        bkd,
        ax,
        labels=["adjoint gradient", "broken adjoint"],
        slope_guides=[1],
    )
    errors_np = bkd.to_numpy(correct_errors)
    eps_np = bkd.to_numpy(fd_eps)
    floor_index = int(errors_np.argmin())
    ax.plot(
        eps_np[floor_index],
        errors_np[floor_index],
        "*",
        color=COLORS["reference"],
        markersize=16,
        zorder=5,
        label="V bottom",
    )
    ax.set_ylabel(r"relative error in $g^\top v$")
    ax.legend(fontsize=8)
    return ax


def plot_hvp_solve_schematic(ax):
    """adjoint_hvp_concept.qmd -> fig-hvp-solves

    The four solves of one Hessian-vector product, and what each reuses
    from the ones before it. The gradient needs the first two; the HVP
    adds a tangent solve and a second adjoint solve, both of which
    reuse the SAME Jacobian (or its transpose) already formed.
    """
    steps = [
        ("1. state solve", r"$c(u, z) = 0$", r"Newton; forms $c_u$",
         COLORS["primary"]),
        ("2. adjoint solve", r"$c_u^\top \lambda = -f_u^\top$", r"reuses $c_u^\top$",
         COLORS["accent"]),
        ("3. tangent solve", r"$c_u\, w = -c_z v$", r"reuses $c_u$",
         COLORS["secondary"]),
        ("4. second adjoint", r"$c_u^\top s = \mathcal{L}_{uu} w - \mathcal{L}_{uz} v$",
         r"reuses $c_u^\top$", COLORS["purple"]),
    ]
    for index, (name, equation, note, color) in enumerate(steps):
        y = len(steps) - 1 - index
        ax.add_patch(
            plt.Rectangle(
                (0.02, y + 0.12), 0.96, 0.76, facecolor=color, alpha=0.16,
                edgecolor=color, lw=1.4,
            )
        )
        ax.text(0.06, y + 0.5, name, va="center", fontsize=10, fontweight="bold")
        ax.text(0.34, y + 0.5, equation, va="center", fontsize=11)
        ax.text(0.95, y + 0.5, note, va="center", ha="right", fontsize=8,
                color="0.35")
    # Bracket marking which solves the gradient alone already pays for.
    ax.annotate(
        "", xy=(0.005, 2.1), xytext=(0.005, 3.9),
        arrowprops=dict(arrowstyle="-", lw=2.5, color="0.45"),
    )
    ax.text(-0.02, 3.0, "gradient", rotation=90, va="center", ha="center",
            fontsize=8, color="0.45")
    ax.annotate(
        "", xy=(0.005, 0.1), xytext=(0.005, 1.9),
        arrowprops=dict(arrowstyle="-", lw=2.5, color="0.45"),
    )
    ax.text(-0.02, 1.0, "extra for HVP", rotation=90, va="center", ha="center",
            fontsize=8, color="0.45")
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(0, len(steps))
    ax.axis("off")
    return ax


def plot_hessian_matrix(hessian, ax, bkd):
    """adjoint_hvp_concept.qmd -> fig-reduced-hessian

    The assembled reduced Hessian as an annotated heatmap. Small enough
    to print every entry, which is the point: on this example the
    operator the HVP applies can be written down in full.
    """
    matrix = bkd.to_numpy(hessian)
    bound = np.abs(matrix).max()
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.4f}", ha="center",
                    va="center", fontsize=11)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    labels = [rf"$z_{i + 1}$" for i in range(matrix.shape[0])]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(r"reduced Hessian $\nabla^2 J$")
    return image
