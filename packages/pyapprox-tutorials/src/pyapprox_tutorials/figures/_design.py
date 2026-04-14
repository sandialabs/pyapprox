"""Plotting functions for design-related tutorials.

Covers: experimental_design_intro.qmd, design_under_uncertainty.qmd,
        models_decisions_uncertainty.qmd
"""

import numpy as np

# ====================================================================
# Private helpers for models_decisions_uncertainty.qmd figures
# ====================================================================

def _get_exterior_edges(connectivity):
    """Return exterior edges of a mesh as a list of [node_i, node_j]."""
    edge_count = {}
    for elem in connectivity:
        n = len(elem)
        for i in range(n):
            e = tuple(sorted([elem[i], elem[(i + 1) % n]]))
            edge_count[e] = edge_count.get(e, 0) + 1
    return [list(e) for e, c in edge_count.items() if c == 1]


def _solve_beam(basis, sub_elems, material_map, bkd, L, q0,
                nonlinear=False, max_iter=1):
    """Solve beam with given material map and return solution array."""
    from pyapprox.pde.galerkin.boundary.implementations import (
        DirichletBC,
        NeumannBC,
    )
    from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
    from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver

    bc_left = DirichletBC(
        basis, "left_edge", lambda c, t=0.0: np.zeros(c.shape[1]), bkd,
    )
    bc_top = NeumannBC(
        basis, "top_edge",
        lambda c, t=0.0, _q=q0: np.vstack(
            [np.zeros(c.shape[1]), -_q * c[0] / L]
        ),
        bkd,
    )
    if nonlinear:
        from pyapprox.pde.galerkin.physics import (
            CompositeHyperelasticityPhysics,
        )
        phys = CompositeHyperelasticityPhysics(
            basis=basis, material_map=material_map,
            element_materials=sub_elems, bkd=bkd,
            boundary_conditions=[bc_left, bc_top],
        )
    else:
        phys = CompositeLinearElasticity(
            basis=basis, material_map=material_map,
            element_materials=sub_elems, bkd=bkd,
            boundary_conditions=[bc_left, bc_top],
        )
    slvr = SteadyStateSolver(phys, tol=1e-10, max_iter=max_iter,
                             **({"line_search": True} if nonlinear else {}))
    res = slvr.solve(bkd.asarray(np.zeros(phys.nstates())))
    return bkd.to_numpy(res.solution)


def _plot_deformed(ax, coordx, coordy, conn, ext_edges, sol, material_map,
                   sub_elems, L, title, tip_dof, scale=None):
    """Plot deformed mesh colored by von Mises stress."""
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize
    from skfem.models.elasticity import lame_parameters

    from pyapprox.pde.galerkin.postprocessing import von_mises_stress_2d

    ux_loc, uy_loc = sol[0::2], sol[1::2]
    tip_val = sol[tip_dof]

    nelems = conn.shape[0]
    lam_e, mu_e = np.empty(nelems), np.empty(nelems)
    for name, (E_val, nu_val) in material_map.items():
        li, mi = lame_parameters(E_val, nu_val)
        lam_e[sub_elems[name]] = li
        mu_e[sub_elems[name]] = mi
    vm = von_mises_stress_2d(coordx, coordy, conn, ux_loc, uy_loc,
                             lam_e, mu_e)

    if scale is None:
        max_d = max(np.max(np.abs(ux_loc)), np.max(np.abs(uy_loc)))
        scale = 0.05 * L / max_d if max_d > 0 else 1.0
    dx, dy = coordx + scale * ux_loc, coordy + scale * uy_loc

    polys = [list(zip(dx[e], dy[e])) for e in conn]
    nrm = Normalize(vmin=vm.min(), vmax=vm.max())
    pc = PolyCollection(polys, array=vm, cmap="RdBu_r", norm=nrm,
                        edgecolors="k", linewidths=0.1)
    ax.add_collection(pc)

    orig = [[(coordx[e[0]], coordy[e[0]]), (coordx[e[1]], coordy[e[1]])]
            for e in ext_edges]
    ax.add_collection(LineCollection(orig, colors="k", linewidths=0.6,
                                     alpha=0.4))

    scale_str = f"{scale:.1f}" if scale < 1 else f"{scale:.0f}"
    ax.set_title(
        title + rf",  $\delta_{{\mathrm{{tip}}}} = {abs(tip_val):.2f}$"
        f"  ({scale_str}x mag.)", fontsize=9,
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    return pc


def _setup_beam_mesh(mesh_path):
    """Set up beam mesh, basis, connectivity, and exterior edges."""
    from pyapprox_benchmarks.pde.cantilever_beam import (
        _find_tip_dof,
    )
    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.util.backends.numpy import NumpyBkd

    bkd = NumpyBkd()
    L, H = 100.0, 30.0
    mesh = UnstructuredMesh2D(mesh_path, bkd, rescale_origin=(0.0, 0.0))
    basis = VectorLagrangeBasis(mesh, degree=1)
    skm = mesh.skfem_mesh()
    sub_names = mesh.subdomain_names()
    sub_elems = {n: mesh.subdomain_elements(n) for n in sub_names}
    conn = skm.t.T
    coordx, coordy = skm.p[0], skm.p[1]
    ext_edges = _get_exterior_edges(conn)
    tip_dof = _find_tip_dof(basis, L, H, bkd)

    return {
        "bkd": bkd, "mesh": mesh, "basis": basis, "skm": skm,
        "sub_elems": sub_elems, "conn": conn,
        "coordx": coordx, "coordy": coordy,
        "ext_edges": ext_edges, "tip_dof": tip_dof,
        "L": L, "H": H,
    }


# ====================================================================
# experimental_design_intro.qmd — all echo:false → Convention A
# ====================================================================

def plot_beam_setup(ax):
    """experimental_design_intro.qmd -> fig-beam-setup

    Composite cantilever beam with uncertain loading and two sensor locations.
    """
    import matplotlib.patches as mpatches

    L, H, t_s = 100.0, 30.0, 5.0
    hole_centers = [16.5, 33.5, 50.0, 66.5, 83.5]
    r_hole = 5.0

    # Material layers
    ax.add_patch(mpatches.Rectangle((0, 0), L, t_s,
                 fc="#8FBCDB", ec="none", alpha=0.8))
    ax.add_patch(mpatches.Rectangle((0, t_s), L, H - 2*t_s,
                 fc="#F4C2A1", ec="none", alpha=0.6))
    ax.add_patch(mpatches.Rectangle((0, H - t_s), L, t_s,
                 fc="#8FBCDB", ec="none", alpha=0.8))

    # Beam outline
    ax.add_patch(mpatches.Rectangle((0, 0), L, H,
                 fill=False, ec="k", lw=1.5))

    # Holes
    for cx in hole_centers:
        ax.add_patch(mpatches.Circle((cx, H / 2), r_hole,
                     fc="white", ec="k", lw=1.0, zorder=3))

    # Clamped left edge
    for yy in np.linspace(0, H, 12):
        ax.plot([-3, 0], [yy, yy - 1.5], "k-", lw=0.7)
    ax.plot([0, 0], [-1.5, H + 1.5], "k-", lw=2)

    # Load arrows (varying magnitude: constant + slope)
    x_arrows = np.linspace(8, 95, 9)
    for xa in x_arrows:
        mag = 3 + 6 * xa / L
        ax.annotate("", xy=(xa, H), xytext=(xa, H + mag + 1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#C0392B"))
    ax.text(L/2, H + 13, r"$t_y(x) = -(\theta_1 + \theta_2\, x/L)$",
            ha="center", fontsize=11, color="#C0392B")

    # Sensor locations
    sensor_tip_x = L - 1
    sensor_mid_x = L / 2
    ms = 14

    ax.plot(sensor_tip_x, 0, "v", ms=ms, color="#27AE60",
            markeredgecolor="k", markeredgewidth=1.2, zorder=5)
    ax.text(sensor_tip_x, -6, "Tip sensor", ha="center", fontsize=9,
            color="#27AE60", fontweight="bold")

    ax.plot(sensor_mid_x, 0, "v", ms=ms, color="#8E44AD",
            markeredgecolor="k", markeredgewidth=1.2, zorder=5)
    ax.text(sensor_mid_x, -6, "Mid sensor", ha="center", fontsize=9,
            color="#8E44AD", fontweight="bold")

    # Material labels
    ax.text(85, 2.5, r"Skin ($E_1$)", ha="center", va="center",
            fontsize=9, color="#1A5276")
    ax.text(85, 27.5, r"Skin ($E_1$)", ha="center", va="center",
            fontsize=9, color="#1A5276")
    ax.text(85, 8.5, r"Core ($E_2$)", ha="center", va="center",
            fontsize=9, color="#A04000")

    ax.set_xlim(-8, L + 5)
    ax.set_ylim(-10, H + 18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_two_posteriors(prior_dist, post_tip, post_mid, theta_true, ax):
    """experimental_design_intro.qmd -> fig-two-posteriors

    Prior and two single-sensor posterior ellipses.
    """
    from pyapprox.probability.gaussian import plot_gaussian_2d_contour

    # Prior
    plot_gaussian_2d_contour(prior_dist, ax, n_std=2,
                             facecolor="none", edgecolor="#2C7FB8", lw=2.5,
                             ls="--", label=r"Prior ($2\sigma$)")

    # Posterior -- tip
    plot_gaussian_2d_contour(post_tip, ax, n_std=2,
                             facecolor="#27AE60", edgecolor="#27AE60", lw=2,
                             alpha=0.15, label="Posterior --- tip sensor")
    plot_gaussian_2d_contour(post_tip, ax, n_std=2,
                             facecolor="none", edgecolor="#27AE60", lw=2)

    # Posterior -- midpoint
    plot_gaussian_2d_contour(post_mid, ax, n_std=2,
                             facecolor="#8E44AD", edgecolor="#8E44AD", lw=2,
                             alpha=0.15, label="Posterior --- mid sensor")
    plot_gaussian_2d_contour(post_mid, ax, n_std=2,
                             facecolor="none", edgecolor="#8E44AD", lw=2)

    # True parameters
    ax.plot(*theta_true, "x", ms=14, mew=3, color="#C0392B", zorder=5,
            label=(rf"True $\boldsymbol{{\theta}}$ = "
                   f"({theta_true[0]}, {theta_true[1]})"))

    ax.set_xlabel(r"$\theta_1$ (constant load)", fontsize=12)
    ax.set_ylabel(r"$\theta_2$ (slope)", fontsize=12)
    ax.set_title("Two sensors, two posteriors", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")


def plot_response_surfaces(a_tip, a_mid, y_tip, y_mid, theta_true,
                           fig, axes):
    """experimental_design_intro.qmd -> fig-response-surfaces

    Contour plots of deflection as a function of (theta_1, theta_2).
    """
    t1_grid = np.linspace(0, 12, 100)
    t2_grid = np.linspace(0, 20, 100)
    T1, T2 = np.meshgrid(t1_grid, t2_grid)

    for ax, a_vec, y_val, title, color in [
        (axes[0], a_tip, y_tip, "Tip sensor", "#27AE60"),
        (axes[1], a_mid, y_mid, "Midpoint sensor", "#8E44AD"),
    ]:
        Z = a_vec[0] * T1 + a_vec[1] * T2
        cs = ax.contourf(T1, T2, Z, levels=15, cmap="coolwarm", alpha=0.7)
        ax.contour(T1, T2, Z, levels=15, colors="gray", linewidths=0.3,
                   alpha=0.5)
        fig.colorbar(cs, ax=ax, shrink=0.8, label=r"$\delta(x_s)$")

        # Level set at observed value
        ax.contour(T1, T2, Z, levels=[y_val], colors="k", linewidths=2.5)

        # True parameters
        ax.plot(*theta_true, "x", ms=12, mew=3, color="#C0392B", zorder=5)

        ax.set_xlabel(r"$\theta_1$", fontsize=12)
        ax.set_ylabel(r"$\theta_2$", fontsize=12)
        ax.set_title(title, fontsize=11, color=color, fontweight="bold")
        ax.grid(True, alpha=0.15)


def plot_combined_experiments(prior_dist, post_tt, post_tm, post_mm,
                              theta_true, cov_det_func, fig, axes):
    """experimental_design_intro.qmd -> fig-combined-experiments

    Two-sensor posterior ellipses and bar chart comparing configurations.
    """
    from pyapprox.probability.gaussian import plot_gaussian_2d_contour

    configs = [
        ("Tip + Tip", post_tt, "#C0392B"),
        ("Tip + Mid", post_tm, "#27AE60"),
    ]

    for ax, (title, post, color) in zip(axes[:2], configs):
        # Prior
        plot_gaussian_2d_contour(prior_dist, ax, n_std=2,
                                 facecolor="none", edgecolor="#2C7FB8", lw=2,
                                 ls="--", label="Prior")
        # Posterior
        plot_gaussian_2d_contour(post, ax, n_std=2,
                                 facecolor=color, edgecolor=color, lw=2,
                                 alpha=0.2)
        plot_gaussian_2d_contour(post, ax, n_std=2,
                                 facecolor="none", edgecolor=color, lw=2,
                                 label="Posterior")
        # True
        ax.plot(*theta_true, "x", ms=12, mew=3, color="#C0392B", zorder=5)
        ax.set_xlabel(r"$\theta_1$", fontsize=11)
        ax.set_ylabel(r"$\theta_2$", fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.set_aspect("equal")

    # Bar chart
    ax_bar = axes[2]
    dets = {
        "Tip+Tip": cov_det_func(post_tt),
        "Mid+Mid": cov_det_func(post_mm),
        "Tip+Mid": cov_det_func(post_tm),
    }
    colors_bar = ["#C0392B", "#8E44AD", "#27AE60"]
    ax_bar.bar(dets.keys(), dets.values(), color=colors_bar, alpha=0.7,
               edgecolor="k", lw=0.5)
    ax_bar.set_ylabel("Posterior cov. determinant\n(smaller = more "
                       "informative)", fontsize=10)
    ax_bar.set_title("Design comparison", fontsize=10, fontweight="bold")
    ax_bar.grid(True, alpha=0.15, axis="y")

    # Annotate the winner
    min_key = min(dets, key=dets.get)
    min_idx = list(dets.keys()).index(min_key)
    min_color = colors_bar[min_idx]
    ax_bar.annotate("Best", xy=(min_idx, dets[min_key]),
                    xytext=(0, 15), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=min_color,
                    arrowprops=dict(arrowstyle="->", color=min_color))


def plot_eig_sweep(x_candidates, eig_values, L, ax):
    """experimental_design_intro.qmd -> fig-eig-sweep

    EIG as a function of single-sensor position along the beam.
    """
    x_opt = x_candidates[np.argmax(eig_values)]
    eig_opt = eig_values.max()

    ax.plot(x_candidates, eig_values, color="#2C7FB8", lw=2.5)
    ax.axvline(L, color="#27AE60", ls="--", lw=1.5, alpha=0.7, label="Tip")
    ax.axvline(L/2, color="#8E44AD", ls="--", lw=1.5, alpha=0.7,
               label="Midpoint")
    ax.plot(x_opt, eig_opt, "o", ms=12, color="#C0392B", zorder=5,
            markeredgecolor="k", markeredgewidth=1.5,
            label=f"Optimal: $x$ = {x_opt:.1f}")

    ax.set_xlabel(r"Sensor position $x$ along beam", fontsize=12)
    ax.set_ylabel("Expected Information Gain (nats)", fontsize=12)
    ax.set_title("Single-sensor design optimization", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)


def plot_eig_2sensor_heatmap(x_sweep, eig_2d, ax):
    """experimental_design_intro.qmd -> fig-eig-2sensor-heatmap

    EIG heatmap for two-sensor designs with optimal pair marked.
    """
    idx_opt = np.unravel_index(np.argmax(eig_2d), eig_2d.shape)
    x1_opt = x_sweep[idx_opt[0]]
    x2_opt = x_sweep[idx_opt[1]]

    im = ax.imshow(eig_2d, origin="lower", aspect="equal",
                   extent=[x_sweep[0], x_sweep[-1],
                           x_sweep[0], x_sweep[-1]],
                   cmap="YlOrRd")
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="EIG (nats)")

    # Diagonal line
    ax.plot([x_sweep[0], x_sweep[-1]], [x_sweep[0], x_sweep[-1]],
            "w--", lw=1.5, alpha=0.7, label="Same location (repeated)")

    # Optimal pair
    ax.plot(x2_opt, x1_opt, "*", ms=18, color="cyan", markeredgecolor="k",
            markeredgewidth=1.5, zorder=5,
            label=f"Optimal: ({x1_opt:.0f}, {x2_opt:.0f})")

    ax.set_xlabel(r"Sensor 2 position $x_2$", fontsize=12)
    ax.set_ylabel(r"Sensor 1 position $x_1$", fontsize=12)
    ax.set_title("Two-sensor EIG heatmap", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")


# ====================================================================
# design_under_uncertainty.qmd — mixed conventions
# ====================================================================

def plot_duu_beam_schematic(ax_side, ax_cross):
    """design_under_uncertainty.qmd -> fig-beam-schematic

    Cantilever beam side view and cross-section with design variables.
    """
    import matplotlib.patches as mpatches

    # -- Side view --
    beam_left, beam_bot = 1.0, 2.0
    beam_length, beam_height = 6.0, 1.0
    beam_right = beam_left + beam_length
    beam_top = beam_bot + beam_height
    beam_mid_y = beam_bot + beam_height / 2

    # Wall (hatching)
    wall_width = 0.35
    wall = mpatches.FancyBboxPatch(
        (beam_left - wall_width, beam_bot - 0.6),
        wall_width, beam_height + 1.2,
        boxstyle="square,pad=0", facecolor="#AAAAAA", edgecolor="k", lw=1.5,
    )
    ax_side.add_patch(wall)
    for yy in np.linspace(beam_bot - 0.5, beam_top + 0.5, 12):
        ax_side.plot(
            [beam_left - wall_width - 0.15, beam_left - wall_width + 0.02],
            [yy - 0.2, yy + 0.2],
            "k-", lw=0.8,
        )

    # Beam body
    beam_rect = mpatches.FancyBboxPatch(
        (beam_left, beam_bot), beam_length, beam_height,
        boxstyle="square,pad=0",
        facecolor="#D4E6F1", edgecolor="#2C3E50", lw=2,
    )
    ax_side.add_patch(beam_rect)

    # Dashed neutral axis
    ax_side.plot(
        [beam_left, beam_right], [beam_mid_y, beam_mid_y],
        "k--", lw=0.8, alpha=0.4,
    )

    # Length dimension (L) -- below beam
    dim_y = beam_bot - 0.45
    ax_side.annotate(
        "", xy=(beam_right, dim_y), xytext=(beam_left, dim_y),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#2C3E50"),
    )
    ax_side.text(
        beam_left + beam_length / 2, dim_y - 0.28,
        "$L$", ha="center", va="top", fontsize=16, fontstyle="italic",
    )

    # Depth dimension (t) -- right side
    dim_x = beam_right + 0.35
    ax_side.annotate(
        "", xy=(dim_x, beam_top), xytext=(dim_x, beam_bot),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#8E44AD"),
    )
    ax_side.text(
        dim_x + 0.2, beam_mid_y, "$t$",
        ha="left", va="center", fontsize=16, color="#8E44AD",
        fontstyle="italic",
    )

    # Horizontal load X
    arrow_x_start = beam_right + 0.15
    arrow_x_end = beam_right + 1.5
    ax_side.annotate(
        "", xy=(arrow_x_start, beam_mid_y),
        xytext=(arrow_x_end, beam_mid_y),
        arrowprops=dict(
            arrowstyle="->,head_width=0.3,head_length=0.2",
            lw=2.5, color="#E74C3C",
        ),
    )
    ax_side.text(
        arrow_x_end + 0.15, beam_mid_y, "$X$",
        ha="left", va="center", fontsize=18, color="#E74C3C",
        fontweight="bold",
    )

    # Vertical load Y
    arrow_y_start = beam_right - 0.3, beam_top + 1.2
    arrow_y_end = beam_right - 0.3, beam_top + 0.15
    ax_side.annotate(
        "", xy=arrow_y_end, xytext=arrow_y_start,
        arrowprops=dict(
            arrowstyle="->,head_width=0.3,head_length=0.2",
            lw=2.5, color="#2980B9",
        ),
    )
    ax_side.text(
        beam_right - 0.3, beam_top + 1.35, "$Y$",
        ha="center", va="bottom", fontsize=18, color="#2980B9",
        fontweight="bold",
    )

    # Material labels
    ax_side.text(
        beam_left + beam_length / 2, beam_mid_y + 0.05,
        "$E, R$", ha="center", va="center", fontsize=14,
        fontstyle="italic", color="#2C3E50", alpha=0.7,
    )

    ax_side.set_xlim(-0.2, 9.5)
    ax_side.set_ylim(0.8, 4.5)
    ax_side.set_aspect("equal")
    ax_side.axis("off")
    ax_side.set_title("Side View", fontsize=13, pad=10)

    # -- Cross-section view --
    cx, cy = 1.5, 2.5
    cw, ct = 2.0, 1.2

    cross = mpatches.FancyBboxPatch(
        (cx - cw / 2, cy - ct / 2), cw, ct,
        boxstyle="square,pad=0",
        facecolor="#D4E6F1", edgecolor="#2C3E50", lw=2,
    )
    ax_cross.add_patch(cross)

    # Width dimension (w) -- below
    wd_y = cy - ct / 2 - 0.3
    ax_cross.annotate(
        "", xy=(cx + cw / 2, wd_y), xytext=(cx - cw / 2, wd_y),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#E67E22"),
    )
    ax_cross.text(
        cx, wd_y - 0.22, "$w$",
        ha="center", va="top", fontsize=16, color="#E67E22",
        fontstyle="italic",
    )

    # Depth dimension (t) -- right
    td_x = cx + cw / 2 + 0.3
    ax_cross.annotate(
        "", xy=(td_x, cy + ct / 2), xytext=(td_x, cy - ct / 2),
        arrowprops=dict(arrowstyle="<->", lw=1.2, color="#8E44AD"),
    )
    ax_cross.text(
        td_x + 0.18, cy, "$t$",
        ha="left", va="center", fontsize=16, color="#8E44AD",
        fontstyle="italic",
    )

    ax_cross.set_xlim(-0.5, 3.5)
    ax_cross.set_ylim(1.2, 3.8)
    ax_cross.set_aspect("equal")
    ax_cross.axis("off")
    ax_cross.set_title("Cross-Section", fontsize=13, pad=10)


def plot_duu_comparison(constraint_values_det, constraint_values_rob, axes):
    """design_under_uncertainty.qmd -> fig-comparison

    Overlay histograms of constraint values at deterministic vs robust optima.
    """
    labels = ["Stress constraint", "Displacement constraint"]

    for ii, ax in enumerate(axes):
        ax.hist(constraint_values_det[ii, :], bins=60, density=True,
                alpha=0.4, color="#2C7FB8", edgecolor="k", lw=0.1,
                label="Deterministic")
        ax.hist(constraint_values_rob[ii, :], bins=60, density=True,
                alpha=0.4, color="#E67E22", edgecolor="k", lw=0.1,
                label="Robust")
        ax.axvline(0, color="red", linestyle="--", lw=2)
        ax.set_xlabel(labels[ii])
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.2)


# ====================================================================
# models_decisions_uncertainty.qmd — all echo:false → Convention A
# ====================================================================

def plot_mdu_beam_geometry(ax):
    """models_decisions_uncertainty.qmd -> fig-beam-geometry

    Composite cantilever beam schematic with material layers and dimensions.
    """
    import matplotlib.patches as patches

    L, H, t_s = 100.0, 30.0, 5.0
    hole_centers = [16.5, 33.5, 50.0, 66.5, 83.5]
    r_hole = 5.0

    # Material layers
    ax.add_patch(patches.Rectangle((0, 0), L, t_s, fc="#8FBCDB", ec="none",
                                   alpha=0.8))
    ax.add_patch(patches.Rectangle((0, t_s), L, H - 2*t_s, fc="#F4C2A1",
                                   ec="none", alpha=0.6))
    ax.add_patch(patches.Rectangle((0, H - t_s), L, t_s, fc="#8FBCDB",
                                   ec="none", alpha=0.8))

    # Beam outline
    ax.add_patch(patches.Rectangle((0, 0), L, H, fill=False, ec="k",
                                   lw=1.5))

    # Holes
    for cx in hole_centers:
        circle = patches.Circle((cx, 15.0), r_hole, fc="white", ec="k",
                                lw=1.0, zorder=3)
        ax.add_patch(circle)

    # Clamped left edge
    for yy in np.linspace(0, H, 12):
        ax.plot([-3, 0], [yy, yy - 1.5], "k-", lw=0.7)
    ax.plot([0, 0], [-1.5, H + 1.5], "k-", lw=2)

    # Traction arrows on top surface
    x_arrows = np.linspace(8, 95, 9)
    for xa in x_arrows:
        mag = 0.3 + 5.0 * xa / L
        ax.annotate("", xy=(xa, H), xytext=(xa, H + mag + 1),
                    arrowprops=dict(arrowstyle="->", color="#C0392B",
                                    lw=1.5))
    ax.text(52, H + 7, r"$t_y(x) = -q_0\, x/L$", ha="center", fontsize=11,
            color="#C0392B")

    # Dimension labels
    ax.annotate("", xy=(0, -5), xytext=(L, -5),
                arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(L / 2, -7, "$L$", ha="center", fontsize=12)

    ax.annotate("", xy=(L + 4, 0), xytext=(L + 4, H),
                arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(L + 7, H / 2, "$H$", ha="left", va="center", fontsize=12)

    ax.annotate("", xy=(L + 4, 0), xytext=(L + 4, t_s),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax.text(L + 7, t_s / 2, "$t_s$", ha="left", va="center", fontsize=10)

    # Hole radius label
    cx_mid = hole_centers[2]
    ax.plot([cx_mid, cx_mid + r_hole], [15, 15], "k-", lw=1.0, zorder=4)
    ax.text(cx_mid + r_hole / 2, 16.2, "$r$", ha="center", fontsize=10,
            zorder=4)

    # Material labels
    ax.text(85, 2.5, r"$E_1, \nu_1$", ha="center", va="center",
            fontsize=10, color="#1A5276")
    ax.text(85, 27.5, r"$E_1, \nu_1$", ha="center", va="center",
            fontsize=10, color="#1A5276")
    ax.text(85, 7.5, r"$E_2, \nu_2$", ha="center", va="center",
            fontsize=10, color="#A04000")

    ax.set_xlim(-8, L + 14)
    ax.set_ylim(-10, H + 10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_reference_solution(fig, ax):
    """models_decisions_uncertainty.qmd -> fig-reference-solution

    Von Mises stress on the deformed mesh with undeformed outline.
    """
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize
    from skfem.models.elasticity import lame_parameters

    from pyapprox_benchmarks.pde.cantilever_beam import MESH_PATHS
    from pyapprox.pde.galerkin.boundary.implementations import (
        DirichletBC,
        NeumannBC,
    )
    from pyapprox.pde.galerkin.physics import CompositeLinearElasticity
    from pyapprox.pde.galerkin.postprocessing import von_mises_stress_2d
    from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver

    info = _setup_beam_mesh(MESH_PATHS[2])
    bkd = info["bkd"]
    basis, sub_elems = info["basis"], info["sub_elems"]
    _skm, conn = info["skm"], info["conn"]
    coordx, coordy = info["coordx"], info["coordy"]
    ext_edges, tip_dof = info["ext_edges"], info["tip_dof"]
    L, _H = info["L"], info["H"]
    q0 = 10.0

    material_map = {
        "bottom_layer": (2e4, 0.3),
        "inner_core": (5e3, 0.3),
        "top_layer": (2e4, 0.3),
    }

    bc_left = DirichletBC(
        basis, "left_edge", lambda c, t=0.0: np.zeros(c.shape[1]), bkd,
    )
    bc_top = NeumannBC(
        basis, "top_edge",
        lambda c, t=0.0: np.vstack([np.zeros(c.shape[1]),
                                     -q0 * c[0] / L]),
        bkd,
    )

    physics = CompositeLinearElasticity(
        basis=basis, material_map=material_map, element_materials=sub_elems,
        bkd=bkd, boundary_conditions=[bc_left, bc_top],
    )
    solver = SteadyStateSolver(physics, tol=1e-10, max_iter=1)
    result = solver.solve(bkd.asarray(np.zeros(physics.nstates())))
    sol = bkd.to_numpy(result.solution)

    ux, uy = sol[0::2], sol[1::2]
    tip_2d = sol[tip_dof]

    # Von Mises stress
    nelems = conn.shape[0]
    lam_elem = np.empty(nelems)
    mu_elem = np.empty(nelems)
    for name, (E_val, nu_val) in material_map.items():
        lam_i, mu_i = lame_parameters(E_val, nu_val)
        lam_elem[sub_elems[name]] = lam_i
        mu_elem[sub_elems[name]] = mu_i

    vm_stress = von_mises_stress_2d(
        coordx, coordy, conn, ux, uy, lam_elem, mu_elem,
    )

    # Displacement scale
    max_disp = max(np.max(np.abs(ux)), np.max(np.abs(uy)))
    scale = 0.05 * L / max_disp if max_disp > 0 else 1.0

    disp_x = coordx + scale * ux
    disp_y = coordy + scale * uy

    # Build displaced polygons
    polys_disp = [list(zip(disp_x[e], disp_y[e])) for e in conn]
    norm = Normalize(vmin=vm_stress.min(), vmax=vm_stress.max())
    pc = PolyCollection(polys_disp, array=vm_stress, cmap="RdBu_r",
                        norm=norm, edgecolors="k", linewidths=0.1)
    ax.add_collection(pc)

    # Undeformed outline
    orig_lines = [[(coordx[e[0]], coordy[e[0]]),
                   (coordx[e[1]], coordy[e[1]])]
                  for e in ext_edges]
    lc = LineCollection(orig_lines, colors="k", linewidths=0.6, alpha=0.4)
    ax.add_collection(lc)

    cbar = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Von Mises stress")

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(
        rf"$\delta_{{\mathrm{{tip}}}} = {abs(tip_2d):.4f}$"
        f"  (deformation magnified {scale:.0f}x)",
        fontsize=11,
    )
    ax.axis("off")


def plot_uncertainty_sources(kind, fig, axes):
    """models_decisions_uncertainty.qmd -> fig-parameter-uncertainty,
    fig-model-form-uncertainty, fig-numerical-uncertainty

    Deformed-mesh comparison for different uncertainty sources.

    Parameters
    ----------
    kind : str
        One of 'parameter', 'model_form', 'numerical'.
    fig : matplotlib Figure
    axes : pair of Axes
    """
    from pyapprox_benchmarks.pde.cantilever_beam import (
        MESH_PATHS,
        _find_tip_dof,
    )
    from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
    from pyapprox.pde.galerkin.mesh import UnstructuredMesh2D
    from pyapprox.util.backends.numpy import NumpyBkd

    bkd = NumpyBkd()
    L, H, q0 = 100.0, 30.0, 10.0

    material_map = {
        "bottom_layer": (2e4, 0.3),
        "inner_core": (5e3, 0.3),
        "top_layer": (2e4, 0.3),
    }

    # Set up fine mesh (used by all kinds)
    info = _setup_beam_mesh(MESH_PATHS[2])
    basis = info["basis"]
    sub_elems = info["sub_elems"]
    skm = info["skm"]
    conn = info["conn"]
    coordx, coordy = info["coordx"], info["coordy"]
    ext_edges = info["ext_edges"]
    tip_dof = info["tip_dof"]

    ax1, ax2 = axes

    if kind == "parameter":
        mat_stiff = {
            "bottom_layer": (2.5e4, 0.3),
            "inner_core": (6e3, 0.3),
            "top_layer": (2.5e4, 0.3),
        }
        mat_soft = {
            "bottom_layer": (1.2e4, 0.3),
            "inner_core": (3e3, 0.3),
            "top_layer": (1.2e4, 0.3),
        }
        sol_stiff = _solve_beam(basis, sub_elems, mat_stiff, bkd, L, q0)
        sol_soft = _solve_beam(basis, sub_elems, mat_soft, bkd, L, q0)

        max_d_soft = max(np.max(np.abs(sol_soft[0::2])),
                         np.max(np.abs(sol_soft[1::2])))
        shared_scale = 0.05 * L / max_d_soft

        _plot_deformed(ax1, coordx, coordy, conn, ext_edges, sol_stiff,
                       mat_stiff, sub_elems, L, "Stiff materials",
                       tip_dof, scale=shared_scale)
        pc2 = _plot_deformed(ax2, coordx, coordy, conn, ext_edges,
                             sol_soft, mat_soft, sub_elems, L,
                             "Soft materials", tip_dof,
                             scale=shared_scale)
        fig.colorbar(pc2, ax=[ax1, ax2], shrink=0.8, pad=0.02,
                     label="Von Mises stress")

    elif kind == "model_form":
        q0_heavy = 120.0
        sol_le = _solve_beam(basis, sub_elems, material_map, bkd, L,
                             q0_heavy)
        sol_nh = _solve_beam(basis, sub_elems, material_map, bkd, L,
                             q0_heavy, nonlinear=True, max_iter=50)

        max_d_le = max(np.max(np.abs(sol_le[0::2])),
                       np.max(np.abs(sol_le[1::2])))
        shared_scale = 0.05 * L / max_d_le

        _plot_deformed(ax1, coordx, coordy, conn, ext_edges, sol_le,
                       material_map, sub_elems, L, "Linear elasticity",
                       tip_dof, scale=shared_scale)
        pc2 = _plot_deformed(ax2, coordx, coordy, conn, ext_edges,
                             sol_nh, material_map, sub_elems, L,
                             "Neo-Hookean", tip_dof, scale=shared_scale)
        fig.colorbar(pc2, ax=[ax1, ax2], shrink=0.8, pad=0.02,
                     label="Von Mises stress")

    elif kind == "numerical":
        # Coarse mesh
        mesh_c = UnstructuredMesh2D(MESH_PATHS[4], bkd,
                                    rescale_origin=(0.0, 0.0))
        basis_c = VectorLagrangeBasis(mesh_c, degree=1)
        skm_c = mesh_c.skfem_mesh()
        sub_names_c = mesh_c.subdomain_names()
        sub_elems_c = {n: mesh_c.subdomain_elements(n)
                       for n in sub_names_c}
        conn_c = skm_c.t.T
        coordx_c, coordy_c = skm_c.p[0], skm_c.p[1]
        ext_edges_c = _get_exterior_edges(conn_c)
        tip_dof_c = _find_tip_dof(basis_c, L, H, bkd)

        sol_coarse = _solve_beam(basis_c, sub_elems_c, material_map,
                                 bkd, L, q0)
        sol_fine = _solve_beam(basis, sub_elems, material_map, bkd, L, q0)

        max_d_fine = max(np.max(np.abs(sol_fine[0::2])),
                         np.max(np.abs(sol_fine[1::2])))
        shared_scale = 0.05 * L / max_d_fine

        _plot_deformed(
            ax1, coordx_c, coordy_c, conn_c, ext_edges_c, sol_coarse,
            material_map, sub_elems_c, L,
            f"Coarse mesh ($h=4$, {skm_c.p.shape[1]} nodes)",
            tip_dof_c, scale=shared_scale,
        )
        pc2 = _plot_deformed(
            ax2, coordx, coordy, conn, ext_edges, sol_fine,
            material_map, sub_elems, L,
            f"Fine mesh ($h=2$, {skm.p.shape[1]} nodes)",
            tip_dof, scale=shared_scale,
        )
        fig.colorbar(pc2, ax=[ax1, ax2], shrink=0.8, pad=0.02,
                     label="Von Mises stress")
    else:
        raise ValueError(
            f"kind must be 'parameter', 'model_form', or 'numerical', "
            f"got {kind!r}"
        )
