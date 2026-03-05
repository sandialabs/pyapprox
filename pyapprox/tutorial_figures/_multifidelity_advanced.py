"""Plotting functions for advanced multi-fidelity tutorials.

Covers: mlblue_concept.qmd, mlblue_analysis.qmd, pacv_concept.qmd,
        pacv_analysis.qmd, multioutput_acv_concept.qmd,
        multioutput_acv_analysis.qmd, ensemble_selection_concept.qmd,
        pilot_studies_concept.qmd, mc_budget_estimation.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# mlblue_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_mlblue_subsets(ax):
    """mlblue_concept.qmd -> fig-subsets

    MLBLUE subset allocation structure with colour-coded model blocks.
    """
    import matplotlib.patches as mpatches

    np.random.seed(42)

    subsets = [
        {"models": [0, 2, 3], "label": "$S_1$"},
        {"models": [0, 1],    "label": "$S_2$"},
        {"models": [1, 4],    "label": "$S_3$"},
        {"models": [4],       "label": "$S_4$"},
    ]

    model_colors = ["#2C7FB8", "#95a5a6", "#27AE60", "#8E44AD", "#C0392B"]
    model_labels = ["$f_0$ (HF)", "$f_1$", "$f_2$", "$f_3$", "$f_4$"]

    block_w, block_h = 0.75, 0.7
    gap_x = 1.3
    y_counter = 0

    for col, sub in enumerate(subsets):
        x = col * gap_x
        for row, midx in enumerate(sub["models"]):
            y = -(y_counter + row) * (block_h + 0.15)
            rect = mpatches.FancyBboxPatch(
                (x - block_w / 2, y - block_h / 2), block_w, block_h,
                boxstyle="round,pad=0.05", facecolor=model_colors[midx],
                edgecolor="k", lw=1.2, alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(x, y, f"$Y_{{{y_counter + row + 1}}}$",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white")
        top_y = -(y_counter) * (block_h + 0.15) + block_h / 2 + 0.3
        ax.text(x, top_y, sub["label"], ha="center", va="bottom",
                fontsize=11, fontweight="bold")
        ax.text(x, top_y + 0.35, f"$m_{{{col+1}}}$ pts", ha="center",
                va="bottom", fontsize=8, color="#555")
        y_counter += len(sub["models"])

    handles = [mpatches.Patch(facecolor=c, edgecolor="k", label=l)
               for c, l in zip(model_colors, model_labels)]
    ax.legend(handles=handles, loc="upper right", fontsize=9, ncol=1,
              framealpha=0.95, edgecolor="#ccc")

    ax.set_xlim(-0.8, (len(subsets) - 1) * gap_x + 2.5)
    ax.set_ylim(-(y_counter) * (block_h + 0.15) - 0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MLBLUE subset allocation structure", fontsize=12, pad=10)


def plot_mlblue_system(axes):
    """mlblue_concept.qmd -> fig-system

    MLBLUE system construction: subsets, Y vector, H matrix, residual.
    """
    import matplotlib.patches as mpatches

    subset_specs = [
        {"name": "$S_5$", "models": [2],    "m": 1, "color": "#95a5a6",
         "members": "$\\{f_2\\}$"},
        {"name": "$S_2$", "models": [1, 2], "m": 1, "color": "#C0392B",
         "members": "$\\{f_1, f_2\\}$"},
        {"name": "$S_3$", "models": [0, 1], "m": 2, "color": "#27AE60",
         "members": "$\\{f_0, f_1\\}$"},
    ]

    nmodels = 3
    mean_labels = ["$Q_0$", "$Q_1$", "$Q_2$"]

    H_rows = []
    row_colors = []
    for spec in subset_specs:
        for rep in range(spec["m"]):
            for midx in spec["models"]:
                row = [0] * nmodels
                row[midx] = 1
                H_rows.append(row)
                row_colors.append(spec["color"])

    H = np.array(H_rows)
    nrows = len(H_rows)

    # Panel 1: subset column labels
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Subsets", fontsize=10, fontweight="bold")
    y_pos = 0
    for spec in subset_specs:
        block_h = spec["m"] * len(spec["models"])
        rect = mpatches.FancyBboxPatch(
            (0.1, -(y_pos + block_h) + 0.1), 0.8, block_h - 0.2,
            boxstyle="round,pad=0.05", facecolor=spec["color"],
            edgecolor="k", lw=1, alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(0.5, -(y_pos + block_h / 2),
                f"{spec['name']}={spec['members']}\n$m={spec['m']}$",
                ha="center", va="center", fontsize=8, fontweight="bold")
        y_pos += block_h
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-nrows - 0.3, 0.8)

    # Panel 2: Y vector
    ax = axes[1]
    ax.axis("off")
    ax.set_title("$\\mathbf{Y}$", fontsize=11, fontweight="bold")
    for i in range(nrows):
        rect = mpatches.FancyBboxPatch(
            (0.05, -(i + 1) + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.03", facecolor=row_colors[i],
            edgecolor="k", lw=0.8, alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(0.5, -(i + 0.5), f"$Y_{{{i+1}}}$", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-nrows - 0.3, 0.8)

    # Panel 3: H matrix
    ax = axes[2]
    ax.set_title("$\\mathbf{H}$", fontsize=11, fontweight="bold")
    ax.matshow(H, cmap="Blues", alpha=0.7, extent=[0, nmodels, nrows, 0])
    for i in range(nrows):
        for j in range(nmodels):
            ax.text(j + 0.5, i + 0.5, str(H[i, j]),
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if H[i, j] == 1 else "normal",
                    color="white" if H[i, j] == 1 else "#888")
    for j in range(nmodels):
        ax.text(j + 0.5, -0.3, mean_labels[j], ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, nmodels)
    ax.set_ylim(nrows, -0.6)

    # Panel 4: residual
    ax = axes[3]
    ax.axis("off")
    ax.set_title("$\\mathbf{Y} - \\mathbf{H}\\mathbf{q}$", fontsize=11,
                 fontweight="bold")
    for i in range(nrows):
        rect = mpatches.FancyBboxPatch(
            (0.05, -(i + 1) + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.03", facecolor=row_colors[i],
            edgecolor="k", lw=0.8, alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(0.5, -(i + 0.5), f"$\\epsilon_{{{i+1}}}$", ha="center",
                va="center", fontsize=9)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-nrows - 0.3, 0.8)


def plot_mlblue_cov(ax, fig):
    """mlblue_concept.qmd -> fig-cov

    Block-diagonal error covariance C_ee for the three-subset example.
    """
    import matplotlib.patches as mpatches

    Sigma = np.array([
        [1.00, 0.70, 0.50],
        [0.70, 1.00, 0.60],
        [0.50, 0.60, 1.00],
    ])

    subset_specs = [
        {"name": "$G_5$", "models": [2],    "m": 1},
        {"name": "$G_2$", "models": [1, 2], "m": 1},
        {"name": "$G_3$", "models": [0, 1], "m": 2},
    ]

    blocks = []
    block_info = []
    pos = 0
    for spec in subset_specs:
        idx = spec["models"]
        Ck = Sigma[np.ix_(idx, idx)]
        for rep in range(spec["m"]):
            blocks.append(Ck)
            block_info.append((pos, len(idx), spec["name"]))
            pos += len(idx)

    total = pos
    C_ee = np.zeros((total, total))
    for blk, (start, size, _) in zip(blocks, block_info):
        C_ee[start:start + size, start:start + size] = blk

    im = ax.matshow(C_ee, cmap="RdBu_r", vmin=-1, vmax=1)

    for i in range(total):
        for j in range(total):
            v = C_ee[i, j]
            if abs(v) > 1e-10:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10)

    block_colors = ["#95a5a6", "#C0392B", "#27AE60", "#27AE60"]
    block_labels_used = set()
    for (start, size, name), col in zip(block_info, block_colors):
        rect = mpatches.Rectangle(
            (start - 0.5, start - 0.5), size, size,
            linewidth=2.5, edgecolor=col, facecolor="none",
        )
        ax.add_patch(rect)
        if name not in block_labels_used:
            ax.text(total - 0.1, start + size / 2 - 0.5, name,
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    color=col)
            block_labels_used.add(name)

    ax.set_xticks(range(total))
    ax.set_yticks(range(total))
    ax.set_title("$\\mathbf{C}_{\\epsilon\\epsilon}$", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.12, label="covariance")


def plot_mlblue_ceiling(ax):
    """mlblue_concept.qmd -> fig-ceiling

    Variance relative to MC as LF samples grow: MLMC, MFMC, MLBLUE.
    """
    import copy
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.polynomial_ensemble import (
        polynomial_ensemble_5model,
    )
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest.acv import MLMCEstimator, MFMCEstimator
    from pyapprox.statest.groupacv import MLBLUEEstimator
    from pyapprox.statest.groupacv.allocation import GroupACVAllocationResult

    bkd = NumpyBkd()
    benchmark = polynomial_ensemble_5model(bkd)
    nmodels = 5
    cov = benchmark.ensemble_covariance()
    costs = benchmark.costs()
    nqoi = benchmark.models()[0].nqoi()
    cov_np = bkd.to_numpy(cov)
    sigma2 = cov_np[0, 0]

    cv_lims = []
    for k in range(1, nmodels):
        sub = cov_np[:k+1, :k+1]
        c0 = sub[0, 1:]
        Sll = sub[1:, 1:]
        cv_lims.append(1 - c0 @ np.linalg.solve(Sll, c0) / sigma2)

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)

    nhf_samples = 1
    mc_var = sigma2 / nhf_samples
    npartition_ratio_base = np.array([2, 2, 2, 2])
    factors = np.arange(22)

    def compute_acv_ratios(est_template):
        est_costs, est_ratios = [], []
        for factor in factors:
            npartition_samples = bkd.asarray(
                nhf_samples * np.hstack(
                    (1, npartition_ratio_base[:est_template._npartitions - 1]
                     * 2**factor)),
            )
            est_copy = copy.deepcopy(est_template)
            nsamples_per_model = est_copy._compute_nsamples_per_model(
                npartition_samples)
            est_copy._set_optimized_params_base(
                npartition_samples, nsamples_per_model,
                est_copy._estimator_cost(npartition_samples))
            est_var = bkd.to_float(est_copy._optimized_covariance[0, 0])
            est_costs.append(
                bkd.to_float(est_copy._estimator_cost(npartition_samples)))
            est_ratios.append(est_var / mc_var)
        return est_costs, est_ratios

    mlmc_costs, mlmc_ratio = compute_acv_ratios(MLMCEstimator(stat, costs))
    mfmc_costs, mfmc_ratio = compute_acv_ratios(MFMCEstimator(stat, costs))

    mlblue_subsets = [
        bkd.asarray(list(range(nmodels)), dtype=int),
        bkd.asarray(list(range(1, nmodels)), dtype=int),
        bkd.asarray(list(range(2, nmodels)), dtype=int),
        bkd.asarray(list(range(3, nmodels)), dtype=int),
        bkd.asarray([nmodels - 1], dtype=int),
    ]
    mlblue_template = MLBLUEEstimator(
        stat, costs, model_subsets=mlblue_subsets)

    mlblue_costs, mlblue_ratio = [], []
    for factor in factors:
        npartition_samples = bkd.asarray(
            nhf_samples * np.hstack(
                (1, npartition_ratio_base * 2**factor)),
        )
        est_copy = copy.deepcopy(mlblue_template)
        nsamples_per_model = est_copy._compute_nsamples_per_model(
            npartition_samples)
        actual_cost = bkd.to_float(est_copy._estimator_cost(npartition_samples))
        result = GroupACVAllocationResult(
            npartition_samples=npartition_samples,
            nsamples_per_model=nsamples_per_model,
            actual_cost=actual_cost,
            objective_value=bkd.asarray([0.0]),
            success=True,
        )
        est_copy.set_allocation(result)
        est_var = bkd.to_float(est_copy.optimized_covariance()[0, 0])
        mlblue_costs.append(actual_cost)
        mlblue_ratio.append(est_var / mc_var)

    ax.loglog(mlmc_costs, mlmc_ratio, "-", color="#2C7FB8", lw=2,
              label="MLMC")
    ax.loglog(mfmc_costs, mfmc_ratio, "--", color="#E67E22", lw=2,
              label="MFMC")
    ax.loglog(mlblue_costs, mlblue_ratio, "-", color="#27AE60", lw=2.5,
              label="MLBLUE")

    cv_colors = ["#c0392b", "#d35400", "#f39c12", "#2c3e50"]
    cv_ls = ["--", "-.", ":", (0, (5, 1))]
    for i, (lim, col, ls) in enumerate(zip(cv_lims, cv_colors, cv_ls)):
        ax.axhline(lim, color=col, ls=ls, lw=1.5,
                   label=f"CV-{i+1} limit ({i+1} LF)")

    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.4,
               label="MC (no reduction)")
    ax.set_xlabel("Total cost", fontsize=11)
    ax.set_ylabel("Variance / MC variance", fontsize=11)
    ax.set_title("MLBLUE converges to the multi-model CV ceiling",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# mlblue_analysis.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_mlblue_verify(estimates, true_mean, var_theo, subsets_idx, m_counts,
                       ax):
    """mlblue_analysis.qmd -> fig-verify

    Empirical distribution of MLBLUE estimates vs theoretical prediction.
    """
    ax.hist(estimates, bins=60, density=True, color="#2C7FB8", alpha=0.78,
            edgecolor="k", lw=0.2,
            label=f"Empirical ({len(estimates)} trials)")
    ax.axvline(true_mean, color="#C0392B", ls="--", lw=2, label="True mean")
    ax.axvspan(true_mean - var_theo**0.5, true_mean + var_theo**0.5,
               alpha=0.14, color="#C0392B",
               label=f"$\\pm$std theory = {var_theo**0.5:.4f}")
    ax.set_xlabel(r"MLBLUE estimate $Q^B_0$", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Variance verification  |  subsets {subsets_idx}  |  m={m_counts}",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_mlblue_m_sweep(m3_vals, vars_sweep, cv1_floor, mc_var_N1, ax):
    """mlblue_analysis.qmd -> fig-m-sweep

    MLBLUE variance vs subset sample count m3.
    """
    ax.semilogx(m3_vals, vars_sweep, color="#2C7FB8", lw=2.5,
                label=r"MLBLUE variance ($m_1=1, m_2=1, m_3$ varies)")
    ax.axhline(cv1_floor, color="#C0392B", ls="--", lw=1.5, alpha=0.85,
               label=f"CV-1 floor = {cv1_floor:.5f}")
    ax.axhline(mc_var_N1, color="k", ls=":", lw=1, alpha=0.4,
               label=f"MC variance (N=1) = {mc_var_N1:.4f}")
    ax.set_xlabel(r"Sample count $m_3$ for $S_3 = \{f_0, f_1\}$",
                  fontsize=11)
    ax.set_ylabel(
        r"$\mathbb{V}[Q^B_0] = \boldsymbol{\beta}^\top"
        r"\boldsymbol{\Psi}^{-1}\boldsymbol{\beta}$",
        fontsize=11)
    ax.set_title("MLBLUE variance vs subset sample count", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# pacv_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pacv_dags(fig, axes):
    """pacv_concept.qmd -> fig-dags

    All valid recursion DAGs for a four-model ensemble.
    """
    import networkx as nx

    def all_recursion_indices(M):
        if M == 0:
            yield []
            return
        for gamma_prev in all_recursion_indices(M - 1):
            for g in range(M):
                yield gamma_prev + [g]

    def draw_dag(ax, gamma, title, node_colors=None):
        M = len(gamma)
        G = nx.DiGraph()
        nodes = list(range(M + 1))
        G.add_nodes_from(nodes)
        for j, g in enumerate(gamma):
            G.add_edge(j + 1, g)

        pos = {0: (0.5, 1.0)}
        for j in range(1, M + 1):
            pos[j] = (j / (M + 1), 0.15)

        nc = node_colors or (["#e74c3c"] + ["#2C7FB8"] * M)
        labels = {0: "$f_0$"} | {j: f"$f_{j}$" for j in range(1, M + 1)}

        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=nc,
                               node_size=700, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10,
                                font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#555", width=2,
                               arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.1", ax=ax)
        ax.set_title(f"$\\boldsymbol{{\\gamma}}={list(gamma)}$\n{title}",
                     fontsize=9, pad=4)
        ax.axis("off")

    def label(g):
        if g == [0, 1, 2]:
            return "MLMC/MFMC"
        if g == [0, 0, 0]:
            return "ACVMF/ACVIS"
        return "Mixed"

    gammas = list(all_recursion_indices(3))
    n = len(gammas)

    axes_flat = np.array(axes).ravel()
    for i, g in enumerate(gammas):
        draw_dag(axes_flat[i], g, label(g))
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "All valid recursion DAGs --- four-model ensemble ($M=3$)",
        fontsize=12, y=1.01)


def plot_pacv_family_matrix(ax):
    """pacv_concept.qmd -> fig-family-matrix

    The three PACV families (rows) vs two canonical recursion indices (cols).
    """
    import matplotlib.patches as mpatches

    ax.axis("off")

    rows = ["GMF", "GRD", "GIS"]
    cols = ["Chain: " + r"$\boldsymbol{\gamma}=[0,1,\ldots,M{-}1]$",
            "Star: " + r"$\boldsymbol{\gamma}=[0,0,\ldots,0]$"]
    data = [["MFMC", "ACVMF"],
            ["MLMC", "(novel)"],
            ["(novel)", "ACVIS"]]
    colors = [["#E67E22", "#8E44AD"],
              ["#2C7FB8", "#bdc3c7"],
              ["#bdc3c7", "#27AE60"]]

    col_w, row_h = 0.38, 0.20
    x0, y0 = 0.18, 0.05

    for c, col in enumerate(cols):
        ax.text(x0 + col_w * (c + 0.5), y0 + row_h * len(rows) + 0.06,
                col, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.text(x0 - 0.04, y0 + row_h * (len(rows) + 0.3),
            "Family", ha="center", va="bottom", fontsize=10, style="italic")
    ax.text(x0 + col_w, y0 + row_h * len(rows) + 0.14,
            "Recursion index", ha="center", va="bottom", fontsize=10,
            style="italic")

    for r, row in enumerate(rows):
        yc = y0 + row_h * (len(rows) - r - 0.5)
        ax.text(x0 - 0.04, yc, row, ha="center", va="center",
                fontsize=11, fontweight="bold")
        for c, (cell, clr) in enumerate(zip(data[r], colors[r])):
            xc = x0 + col_w * c
            rect = mpatches.FancyBboxPatch(
                (xc + 0.01, yc - row_h * 0.45),
                col_w - 0.02, row_h - 0.02,
                boxstyle="round,pad=0.02", fc=clr, ec="white", lw=2,
                alpha=0.85)
            ax.add_patch(rect)
            ax.text(xc + col_w * 0.5, yc,
                    cell, ha="center", va="center", fontsize=12,
                    fontweight="bold",
                    color="white" if cell != "(novel)" else "#333")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title(
        "PACV family grid: three base families x two recursion indices",
        fontsize=11, pad=12)


def plot_pacv_enumeration(ax):
    """pacv_concept.qmd -> fig-enumeration

    Predicted variance for all valid GMF recursion indices at budget P=100.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.polynomial_ensemble import (
        polynomial_ensemble_5model,
    )
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest.mc_estimator import MCEstimator
    from pyapprox.statest.acv.search import ACVSearch
    from pyapprox.statest.acv.allocation import (
        ACVAllocator, default_allocator_factory,
    )
    from pyapprox.statest.acv.strategies import TreeDepthRecursionStrategy
    from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer

    bkd = NumpyBkd()
    np.random.seed(1)
    benchmark = polynomial_ensemble_5model(bkd)
    nmodels = 4
    costs = benchmark.costs()[:nmodels]
    cov = benchmark.ensemble_covariance()[:nmodels, :nmodels]
    nqoi = benchmark.models()[0].nqoi()

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)

    mc_est = MCEstimator(stat, costs[:1])
    mc_est.allocate_samples(100.0)
    mc_var = bkd.to_float(mc_est.optimized_covariance()[0, 0])

    _slsqp = ScipySLSQPOptimizer(maxiter=200)

    def _fast_allocator(est):
        alloc = default_allocator_factory(est)
        if isinstance(alloc, ACVAllocator):
            return ACVAllocator(est, optimizer=_slsqp)
        return alloc

    search = ACVSearch(
        stat, costs,
        recursion_strategy=TreeDepthRecursionStrategy(
            max_depth=nmodels - 1),
        allocator_factory=_fast_allocator,
    )
    result = search.search(100.0, allow_failures=True)

    variances, labels_ri = [], []
    for est, alloc in result.all_allocations:
        if alloc.success:
            try:
                est.set_allocation(alloc)
                var = bkd.to_float(est.optimized_covariance()[0, 0])
            except np.linalg.LinAlgError:
                var = mc_var
        else:
            var = mc_var
        variances.append(var / mc_var)
        ri = bkd.to_numpy(est._recursion_index).astype(int)
        labels_ri.append(str(list(ri.tolist())))

    colors_bar = ["#8E44AD" if v == min(variances) else "#2C7FB8"
                  for v in variances]

    x = np.arange(len(variances))
    ax.bar(x, variances, color=colors_bar, edgecolor="k", lw=0.5,
           alpha=0.85)
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.4, label="MC baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\gamma={l}$" for l in labels_ri],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Predicted variance / MC variance", fontsize=11)
    ax.set_title(
        f"GMF estimators --- all {len(variances)} recursion indices  "
        f"(P=100, M={nmodels-1} LF models)",
        fontsize=11)
    ax.legend(fontsize=10)
    best_lbl = labels_ri[np.argmin(variances)]
    ax.annotate(f"Best: $\\gamma={best_lbl}$",
                xy=(np.argmin(variances), min(variances)),
                xytext=(np.argmin(variances) + 0.5,
                        min(variances) + 0.08),
                fontsize=9, color="#8E44AD",
                arrowprops=dict(arrowstyle="->", color="#8E44AD"))
    ax.grid(True, alpha=0.2, axis="y")


def plot_pacv_ceiling(ax):
    """pacv_concept.qmd -> fig-pacv-ceiling

    Variance vs cost ceiling plot: MLMC, MFMC, ACVMF, best GMF.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.polynomial_ensemble import (
        polynomial_ensemble_5model,
    )
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest.acv import MLMCEstimator, MFMCEstimator, GMFEstimator
    from pyapprox.statest.acv.strategies import TreeDepthRecursionStrategy

    bkd = NumpyBkd()
    benchmark = polynomial_ensemble_5model(bkd)
    models = benchmark.models()
    costs = benchmark.costs()
    nqoi = models[0].nqoi()
    cov = bkd.to_numpy(benchmark.ensemble_covariance())
    sigma2_hf = cov[0, 0]
    n_models = len(models)
    M = n_models - 1

    cv_limits = []
    for k in range(1, n_models):
        sub = cov[:k + 1, :k + 1]
        c0l = sub[0, 1:]
        Sl = sub[1:, 1:]
        r2 = c0l @ np.linalg.solve(Sl, c0l) / sigma2_hf
        cv_limits.append(1 - r2)

    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(bkd.asarray(cov))

    nhf_samples = 1
    mc_var = sigma2_hf / nhf_samples
    partition_ratio_base = np.array([2, 2, 2, 2])
    factors = np.arange(22)

    def compute_variance_vs_cost(estimator):
        nparts = estimator._npartitions
        est_costs, est_ratios = [], []
        for factor in factors:
            ratios = bkd.asarray(
                partition_ratio_base[:nparts - 1] * 2**factor, dtype=float)
            model_ratios = estimator._partition_ratios_to_model_ratios(ratios)
            target_cost = bkd.to_float(
                nhf_samples * (costs[0] + bkd.dot(model_ratios, costs[1:])))
            est_cov = estimator.covariance_from_ratios(target_cost, ratios)
            est_var = bkd.to_float(est_cov[0, 0])
            est_costs.append(target_cost)
            est_ratios.append(est_var / mc_var)
        return est_costs, est_ratios

    mlmc_costs, mlmc_ratio = compute_variance_vs_cost(
        MLMCEstimator(stat, costs))
    mfmc_costs, mfmc_ratio = compute_variance_vs_cost(
        MFMCEstimator(stat, costs))
    acvmf_costs, acvmf_ratio = compute_variance_vs_cost(
        GMFEstimator(stat, costs,
                     recursion_index=bkd.zeros(M, dtype=int)))

    all_ri = TreeDepthRecursionStrategy(max_depth=M).indices(n_models, bkd)
    best_costs, best_ratio = [], []
    for factor in factors:
        best_var = np.inf
        for ri in all_ri:
            est = GMFEstimator(stat, costs, recursion_index=ri)
            nparts = est._npartitions
            ratios = bkd.asarray(
                partition_ratio_base[:nparts - 1] * 2**factor, dtype=float)
            model_ratios = est._partition_ratios_to_model_ratios(ratios)
            target_cost = bkd.to_float(
                nhf_samples * (
                    costs[0] + bkd.dot(model_ratios, costs[1:])))
            est_cov = est.covariance_from_ratios(target_cost, ratios)
            est_var = bkd.to_float(est_cov[0, 0])
            if est_var < best_var:
                best_var = est_var
        best_costs.append(target_cost)
        best_ratio.append(best_var / mc_var)

    cv_colors = ["#e74c3c", "#8e44ad", "#16a085", "#2c3e50"]
    cv_ls = ["--", "-.", ":", (0, (5, 1))]

    ax.loglog(mlmc_costs, mlmc_ratio, "-", color="#2C7FB8", lw=2,
              label="MLMC")
    ax.loglog(mfmc_costs, mfmc_ratio, "--", color="#E67E22", lw=2,
              label="MFMC")
    ax.loglog(acvmf_costs, acvmf_ratio, "-.", color="#27AE60", lw=2,
              label="ACVMF")
    ax.loglog(best_costs, best_ratio, "-", color="#c0392b", lw=2.5,
              label="Best GMF")

    for k, (lim, col, ls) in enumerate(zip(cv_limits, cv_colors, cv_ls)):
        ax.axhline(lim, color=col, ls=ls, lw=1.5,
                   label=f"CV-{k+1} limit ({k+1} LF)")

    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.4,
               label="MC (no reduction)")
    ax.set_xlabel("Total cost", fontsize=11)
    ax.set_ylabel("Variance / MC variance", fontsize=11)
    ax.set_title("PACV best-GMF converges to the multi-model CV ceiling",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# pacv_analysis.qmd — echo:true, only the cross-family plot → Convention B
# ---------------------------------------------------------------------------

def plot_pacv_cross_family(ri_list, results, fam_names, fam_cols, mc_var,
                           nmodels, ax):
    """pacv_analysis.qmd -> fig-cross-family

    Grouped bar chart of predicted variance across PACV families.
    """
    x = np.arange(len(ri_list))
    width = 0.28
    offsets = [-1, 0, 1]
    for offset, fam in zip(offsets, fam_names):
        vals = results[fam]
        ax.bar(x + offset * width, vals, width * 0.92, color=fam_cols[fam],
               alpha=0.85, edgecolor="k", lw=0.4, label=fam_names[fam])

    all_vals = [(v, fam, i)
                for fam in fam_names for i, v in enumerate(results[fam])]
    best_v, best_fam, best_i = min(all_vals)
    best_offset = offsets[list(fam_names).index(best_fam)]
    ax.scatter(best_i + best_offset * width, best_v,
               s=140, marker="*", color="gold", edgecolors="k", lw=0.8,
               zorder=5,
               label=(f"Best: {fam_names[best_fam]} "
                      f"\u03b3={list(ri_list[best_i])}"))

    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.4, label="MC")
    ax.set_xticks(x)
    ax.set_xticklabels([str(list(ri)) for ri in ri_list],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Recursion index $\\boldsymbol{\\gamma}$", fontsize=11)
    ax.set_ylabel("Predicted variance / MC variance", fontsize=11)
    ax.set_title(
        f"All PACV estimators --- three families x {len(ri_list)} indices  "
        f"(P=500, M={nmodels-1})",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")


# ---------------------------------------------------------------------------
# multioutput_acv_concept.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_moacv_vs_soacv(ax):
    """multioutput_acv_concept.qmd -> fig-moacv-soacv

    Estimator variance for QoI 0 vs target cost: MOACV vs SOACV vs MC.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks.instances.multifidelity.multioutput_ensemble import (
        multioutput_ensemble_3x3,
    )
    from pyapprox.statest.statistics import MultiOutputMean
    from pyapprox.statest.acv.search import ACVSearch

    bkd = NumpyBkd()
    np.random.seed(1)
    benchmark = multioutput_ensemble_3x3(bkd)
    costs = benchmark.costs()
    nqoi = 2
    nmodels = benchmark.nmodels()
    full_nqoi = benchmark.models()[0].nqoi()

    cov_full = benchmark.ensemble_covariance()
    idx = []
    for a in range(nmodels):
        for q in range(nqoi):
            idx.append(a * full_nqoi + q)
    cov_true = cov_full[np.ix_(idx, idx)]

    stat_mo = MultiOutputMean(nqoi, bkd)
    stat_mo.set_pilot_quantities(cov_true)
    search_mo = ACVSearch(stat_mo, costs)

    qoi_idx = [0]
    (cov_so,) = stat_mo.get_pilot_quantities_subset(
        nmodels, nqoi, list(range(nmodels)), qoi_idx)
    stat_so = MultiOutputMean(len(qoi_idx), bkd)
    stat_so.set_pilot_quantities(cov_so)
    search_so = ACVSearch(stat_so, costs)

    cov_true_np = bkd.to_numpy(cov_true)
    mc_var_q0 = cov_true_np[0, 0]

    target_costs_all = np.logspace(1, 3.5, 50)

    target_costs, mo_vars, so_vars, mc_vars = [], [], [], []
    for P in target_costs_all:
        try:
            result_mo = search_mo.search(float(P))
            result_so = search_so.search(float(P))
        except RuntimeError:
            continue
        cov_mo_P = bkd.to_numpy(result_mo.estimator.optimized_covariance())
        target_costs.append(P)
        mo_vars.append(cov_mo_P[0, 0])
        so_vars.append(bkd.to_float(
            result_so.estimator.optimized_covariance()[0, 0]))
        mc_vars.append(mc_var_q0 / float(P))

    ax.loglog(target_costs, mo_vars, "-", color="#8E44AD", lw=2.5,
              label="MOACV (joint: both QoIs)")
    ax.loglog(target_costs, so_vars, "--", color="#2C7FB8", lw=2.5,
              label="SOACV (QoI 0 only)")
    ax.loglog(target_costs, mc_vars, ":", color="k", lw=1.5, alpha=0.6,
              label="MC")

    mid_i = len(target_costs) // 2
    ax.annotate("", xy=(target_costs[mid_i], mo_vars[mid_i]),
                xytext=(target_costs[mid_i], so_vars[mid_i]),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=1.5))
    ax.text(target_costs[mid_i] * 1.15,
            (mo_vars[mid_i] * so_vars[mid_i]) ** 0.5,
            "MOACV\ngain", fontsize=9, color="#e74c3c", va="center")

    ax.set_xlabel("Target cost $P$", fontsize=12)
    ax.set_ylabel("Estimator variance (QoI 0)", fontsize=12)
    ax.set_title("MOACV vs SOACV: joint estimation improves each QoI",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")


# ---------------------------------------------------------------------------
# multioutput_acv_analysis.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_mo_theory_vs_empirical(nqoi, cov_mo_np, cov_so_np, mo_vars_emp,
                                so_vars_emp, n_trials, ax):
    """multioutput_acv_analysis.qmd -> fig-mo-theory-vs-empirical

    Theoretical (bars) vs empirical (dots) variance per QoI.
    """
    x = np.arange(nqoi)
    width = 0.2
    ax.bar(x - width * 1.5,
           [cov_mo_np[qi, qi] for qi in range(nqoi)],
           width, color="#8E44AD", alpha=0.8, edgecolor="k", lw=0.5,
           label="MOACV theory")
    ax.scatter(x - width * 1.5, mo_vars_emp,
               s=70, color="#8E44AD", edgecolors="k", lw=1, zorder=4)
    ax.bar(x - width * 0.5,
           [cov_so_np[qi, qi] for qi in range(nqoi)],
           width, color="#2C7FB8", alpha=0.8, edgecolor="k", lw=0.5,
           label="SOACV theory")
    ax.scatter(x - width * 0.5, so_vars_emp,
               s=70, color="#2C7FB8", edgecolors="k", lw=1, zorder=4,
               label="Empirical (dots)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"QoI {qi}" for qi in range(nqoi)], fontsize=11)
    ax.set_ylabel("Estimator variance", fontsize=11)
    ax.set_title(f"MOACV vs SOACV  |  P=100  |  {n_trials} trials",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")


# ---------------------------------------------------------------------------
# ensemble_selection_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_bad_model(ax):
    """ensemble_selection_concept.qmd -> fig-bad-model

    Three-model vs best two-model ACVMF variance across rho_01 values.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks import tunable_ensemble_3model
    from pyapprox.statest import MultiOutputMean, MCEstimator, GMFEstimator

    bkd = NumpyBkd()
    np.random.seed(42)
    P = 100.0

    theta_vals = np.linspace(0.5, 1.5, 40)
    two_model_vars = []
    three_model_vars = []
    rho01_vals = []
    mc_vars = []

    for theta1 in theta_vals:
        bm = tunable_ensemble_3model(bkd, theta1=theta1)
        cov = bm.ensemble_covariance()
        costs = bm.costs()
        cov_np = bkd.to_numpy(cov)

        D = np.diag(1.0 / np.sqrt(np.diag(cov_np)))
        corr = D @ cov_np @ D
        rho01_vals.append(corr[0, 1])

        stat_mc = MultiOutputMean(1, bkd)
        stat_mc.set_pilot_quantities(cov)
        mc_est = MCEstimator(stat_mc, costs[:1])
        mc_est.allocate_samples(P)
        mc_var = bkd.to_float(mc_est.optimized_covariance()[0, 0])
        mc_vars.append(mc_var)

        best2 = mc_var
        for idx2 in ([0, 1], [0, 2]):
            sub_cov = cov_np[np.ix_(idx2, idx2)]
            sub_costs = bkd.asarray([bkd.to_numpy(costs)[i] for i in idx2])
            stat2 = MultiOutputMean(1, bkd)
            stat2.set_pilot_quantities(bkd.array(sub_cov))
            est2 = GMFEstimator(stat2, sub_costs,
                                recursion_index=bkd.asarray([0]))
            est2.allocate_samples(P)
            v = bkd.to_float(est2.optimized_covariance()[0, 0])
            best2 = min(best2, v)
        two_model_vars.append(best2)

        stat3 = MultiOutputMean(1, bkd)
        stat3.set_pilot_quantities(cov)
        est3 = GMFEstimator(stat3, costs,
                            recursion_index=bkd.asarray([0, 0]))
        est3.allocate_samples(P)
        v3 = bkd.to_float(est3.optimized_covariance()[0, 0])
        three_model_vars.append(v3)

    rho01_arr = np.array(rho01_vals)
    ratio_2m = np.array(two_model_vars) / np.array(mc_vars)
    ratio_3m = np.array(three_model_vars) / np.array(mc_vars)

    ax.plot(rho01_arr, ratio_3m, color="#2C7FB8", lw=2.5,
            label="Three-model ACVMF  $\\{f_0,f_1,f_2\\}$")
    ax.plot(rho01_arr, ratio_2m, color="#E67E22", ls="--", lw=2,
            label="Best two-model ACVMF")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, alpha=0.5,
               label="MC baseline")
    ax.fill_between(rho01_arr, ratio_3m, ratio_2m,
                    where=ratio_3m > ratio_2m,
                    color="#e74c3c", alpha=0.18,
                    label="Region where $f_1$ *hurts*")
    ax.set_xlabel(
        r"$\rho_{01}$ --- correlation of $f_1$ with HF model", fontsize=12)
    ax.set_ylabel("Predicted variance / MC variance", fontsize=12)
    ax.set_title(
        "ACVMF: two-model vs three-model  (tunable benchmark, $P=100$)",
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_correlation_heatmaps(axes, fig):
    """ensemble_selection_concept.qmd -> fig-heatmap

    Correlation matrices for two configurations of the tunable benchmark.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks import tunable_ensemble_3model
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    bkd = NumpyBkd()
    nmodels = 3

    im = None
    for ax, theta1, title in zip(axes, [1.4, 0.6],
                                 [r"$\theta_1 = 1.4$",
                                  r"$\theta_1 = 0.6$"]):
        bm = tunable_ensemble_3model(bkd, theta1=theta1)
        cov_np = bkd.to_numpy(bm.ensemble_covariance())
        D_inv = np.diag(1.0 / np.sqrt(np.diag(cov_np)))
        corr = D_inv @ cov_np @ D_inv

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        labels = [f"$f_{i}$" for i in range(nmodels)]
        ax.set_xticks(range(nmodels))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticks(range(nmodels))
        ax.set_yticklabels(labels, fontsize=11)
        for i in range(nmodels):
            for j in range(nmodels):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=10,
                        color="black" if abs(corr[i, j]) < 0.85
                        else "white")
        ax.set_title(title, fontsize=11)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="8%", pad=0.15)
    fig.colorbar(im, cax=cax, label="Correlation")


def plot_ensemble_nmodels(ax):
    """ensemble_selection_concept.qmd -> fig-nmodels

    Best 1-LF vs 2-LF ACVMF variance ratio across rho_01 values.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks import tunable_ensemble_3model
    from pyapprox.statest import MultiOutputMean, MCEstimator, GMFEstimator

    bkd = NumpyBkd()
    P = 100.0

    theta_vals = np.linspace(0.5, 1.5, 30)
    best_1lf = []
    two_lf = []
    rho01_vals = []

    for theta1 in theta_vals:
        bm = tunable_ensemble_3model(bkd, theta1=theta1)
        cov = bm.ensemble_covariance()
        costs = bm.costs()
        cov_np = bkd.to_numpy(cov)

        D = np.diag(1.0 / np.sqrt(np.diag(cov_np)))
        corr = D @ cov_np @ D
        rho01_vals.append(corr[0, 1])

        stat_mc = MultiOutputMean(1, bkd)
        stat_mc.set_pilot_quantities(cov)
        mc_est = MCEstimator(stat_mc, costs[:1])
        mc_est.allocate_samples(P)
        mc_var = bkd.to_float(mc_est.optimized_covariance()[0, 0])

        best1 = mc_var
        for idx2 in ([0, 1], [0, 2]):
            sub_cov = cov_np[np.ix_(idx2, idx2)]
            sub_costs = bkd.asarray([bkd.to_numpy(costs)[i] for i in idx2])
            stat2 = MultiOutputMean(1, bkd)
            stat2.set_pilot_quantities(bkd.array(sub_cov))
            est2 = GMFEstimator(stat2, sub_costs,
                                recursion_index=bkd.asarray([0]))
            est2.allocate_samples(P)
            v = bkd.to_float(est2.optimized_covariance()[0, 0])
            best1 = min(best1, v)
        best_1lf.append(best1 / mc_var)

        stat3 = MultiOutputMean(1, bkd)
        stat3.set_pilot_quantities(cov)
        est3 = GMFEstimator(stat3, costs,
                            recursion_index=bkd.asarray([0, 0]))
        est3.allocate_samples(P)
        v3 = bkd.to_float(est3.optimized_covariance()[0, 0])
        two_lf.append(v3 / mc_var)

    rho01_arr = np.array(rho01_vals)

    ax.plot(rho01_arr, two_lf, color="#2C7FB8", lw=2.5, marker="o", ms=4,
            label="2 LF models  $\\{f_0,f_1,f_2\\}$")
    ax.plot(rho01_arr, best_1lf, color="#E67E22", lw=2.5, marker="s", ms=4,
            label="Best 1 LF model")
    ax.axhline(1.0, color="k", ls=":", lw=1.2, alpha=0.5,
               label="MC baseline")
    ax.set_xlabel(
        r"$\rho_{01}$ --- correlation of $f_1$ with HF model", fontsize=12)
    ax.set_ylabel("Best variance / MC variance", fontsize=11)
    ax.set_title(
        "ACVMF: optimal ensemble size depends on correlation  ($P=100$)",
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# pilot_studies_concept.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_pilot_tradeoff(axes):
    """pilot_studies_concept.qmd -> fig-tradeoff

    MSE vs pilot size with/without pilot cost deduction.
    """
    from pyapprox.util.backends.numpy import NumpyBkd
    from pyapprox.benchmarks import polynomial_ensemble_3model
    from pyapprox.statest import MultiOutputMean, MFMCEstimator, MCEstimator

    bkd = NumpyBkd()
    np.random.seed(1)
    benchmark = polynomial_ensemble_3model(bkd)
    nmodels = 3
    nqoi = 1
    costs = bkd.array([1.0, 0.1, 0.05])
    prior = benchmark.prior()
    models = benchmark.models()
    true_mean = bkd.to_float(benchmark.ensemble_means()[0, 0])
    P_total = 100.0

    cov_oracle = benchmark.ensemble_covariance()
    oracle_stat = MultiOutputMean(nqoi, bkd)
    oracle_stat.set_pilot_quantities(cov_oracle)
    oracle_est = MFMCEstimator(oracle_stat, costs)
    oracle_est.allocate_samples(P_total)
    oracle_mse = bkd.to_float(oracle_est.optimized_covariance()[0, 0])

    mc_est = MCEstimator(oracle_stat, costs[:1])
    mc_est.allocate_samples(P_total)
    mc_mse = bkd.to_float(mc_est.optimized_covariance()[0, 0])

    def single_trial(seed, npilot, budget):
        np.random.seed(seed)
        s_p = prior.rvs(npilot)
        vals_p = [m(s_p) for m in models]
        stat = MultiOutputMean(nqoi, bkd)
        (cov_hat,) = stat.compute_pilot_quantities(vals_p)
        stat.set_pilot_quantities(cov_hat)
        est = MFMCEstimator(stat, costs)
        try:
            est.allocate_samples(budget)
            np.random.seed(seed + 1000)
            s_main = est.generate_samples_per_model(prior.rvs)
            vals_main = [m(s) for m, s in zip(models, s_main)]
            return bkd.to_float(est(vals_main))
        except Exception:
            return float("nan")

    npilot_list = [5, 10, 20, 40, 80, 160]
    ntrials = 400
    costs_np = bkd.to_numpy(costs)

    mse_free = []
    mse_paid = []

    for npilot in npilot_list:
        pilot_cost = costs_np.sum().item() * npilot
        budget_free = P_total
        budget_paid = max(P_total - pilot_cost, 1.0)

        vals_free = [single_trial(s, npilot, budget_free)
                     for s in range(ntrials)]
        vals_paid = [single_trial(s, npilot, budget_paid)
                     for s in range(ntrials)]

        vals_free = np.array([v for v in vals_free if np.isfinite(v)])
        vals_paid = np.array([v for v in vals_paid if np.isfinite(v)])

        mse_free.append(np.mean((vals_free - true_mean)**2).item())
        mse_paid.append(np.mean((vals_paid - true_mean)**2).item())

    for ax, mse_list, title in zip(
        axes,
        [mse_free, mse_paid],
        ["Pilot cost ignored\n(unrealistic)",
         "Pilot cost deducted\n(realistic)"],
    ):
        ax.plot(npilot_list, [m / mc_mse for m in mse_list], "-o",
                color="#2C7FB8", lw=2.2, ms=7, label="Pilot MSE")
        ax.axhline(oracle_mse / mc_mse, color="k", ls="--", lw=1.8,
                   label=f"Oracle ACV  ({oracle_mse/mc_mse:.3f})")
        ax.axhline(1.0, color="#c0392b", ls=":", lw=1.5,
                   label="MC baseline")
        ax.set_xlabel("Pilot samples $N_p$", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("MSE / MC MSE", fontsize=11)


# ---------------------------------------------------------------------------
# mc_budget_estimation.qmd — echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_budget_verification(all_means, nqoi, predicted_est_cov_np, axes):
    """mc_budget_estimation.qmd -> fig-budget-verification

    Histograms of mean estimates with predicted vs empirical std overlaid.
    """
    from scipy.stats import norm

    qoi_labels = [rf"$\hat{{\mu}}_{{{k+1}}}$" for k in range(nqoi)]
    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    M_alloc = all_means.shape[0]

    for col, ax in enumerate(axes):
        ax.hist(all_means[:, col], bins=30, density=True,
                color=colors[col], alpha=0.7, edgecolor="k", lw=0.3)

        pred_std = np.sqrt(predicted_est_cov_np[col, col])
        emp_mean = np.mean(all_means[:, col])
        emp_std = np.std(all_means[:, col])

        x_grid = np.linspace(all_means[:, col].min(),
                             all_means[:, col].max(), 200)
        ax.plot(x_grid, norm.pdf(x_grid, emp_mean, pred_std), "r--", lw=2,
                label=f"Predicted \u03c3 = {pred_std:.4f}")
        ax.axvline(emp_mean, color="#27AE60", ls="-", lw=1.5,
                   label=f"Empirical \u03c3 = {emp_std:.4f}")

        ax.set_xlabel(qoi_labels[col], fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Mean estimator, $M = {M_alloc}$", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)


def plot_budget_vs_statistic(stat_configs, cost_per_eval, bkd, ax):
    """mc_budget_estimation.qmd -> fig-budget-vs-statistic

    Predicted estimation error vs budget for mean, variance, mean+variance.
    """
    from pyapprox.statest.mc_estimator import MCEstimator

    budget_values = np.logspace(1, 3.5, 30)
    colors_stat = ["#2C7FB8", "#E67E22", "#C0392B"]

    for (name, stat_obj), color in zip(stat_configs.items(), colors_stat):
        traces = []
        for budget in budget_values:
            est_temp = MCEstimator(stat_obj, cost_per_eval)
            est_temp.allocate_samples(float(budget))
            est_cov = est_temp.optimized_covariance()
            traces.append(bkd.to_float(bkd.trace(est_cov)))
        ax.loglog(budget_values, traces, "-o", ms=4, color=color, label=name)

    ax.set_xlabel("Computational budget (model evaluations)", fontsize=12)
    ax.set_ylabel("tr(Cov[$\\hat{\\mathbf{Q}}_M$])", fontsize=12)
    ax.set_title("Estimation error vs. budget by statistic", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, which="both")


def plot_pilot_sensitivity(all_pred_stds, pilot_sizes, true_pred_std,
                           n_pilot_reps, target_for_test, axes, fig):
    """mc_budget_estimation.qmd -> fig-pilot-sensitivity

    Sensitivity of budget prediction to pilot size.
    """
    all_vals = np.concatenate(list(all_pred_stds.values()))
    x_pad = 0.05 * (all_vals.max() - all_vals.min())
    xlim = (all_vals.min() - x_pad, all_vals.max() + x_pad)

    for ax, M_p in zip(axes, pilot_sizes):
        ax.hist(all_pred_stds[M_p], bins=25, density=True, color="#2C7FB8",
                alpha=0.7, edgecolor="k", lw=0.3)
        ax.axvline(true_pred_std, color="#C0392B", ls="--", lw=2)
        ax.set_xlim(xlim)
        ax.set_xlabel("Predicted \u03c3", fontsize=10)
        ax.set_title(f"$M_{{\\text{{pilot}}}} = {M_p}$", fontsize=11)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Density", fontsize=11)
    fig.suptitle(
        "Predicted std of $\\hat{\\mu}_1$ at budget = "
        f"{int(target_for_test)}, across {n_pilot_reps} independent pilots",
        fontsize=11, y=1.02,
    )


def plot_est_cov_final(pred_cov_plot, stat_labels, nstats, nqoi, target_cost,
                       ax1, ax2, fig):
    """mc_budget_estimation.qmd -> fig-est-cov-final

    Estimator covariance heatmap and diagonal bar chart.
    """
    vmin = pred_cov_plot.min()
    vmax = pred_cov_plot.max()

    im1 = ax1.imshow(pred_cov_plot, cmap="RdBu_r", aspect="equal",
                     vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(nstats))
    ax1.set_xticklabels(stat_labels, fontsize=7, rotation=45)
    ax1.set_yticks(range(nstats))
    ax1.set_yticklabels(stat_labels, fontsize=7)
    ax1.set_title(f"Estimator covariance (target_cost = {target_cost:.0f})",
                  fontsize=11)
    fig.colorbar(im1, ax=ax1, shrink=0.7)

    diag_vals = np.diag(pred_cov_plot)
    colors_bar = ["#2C7FB8"] * nqoi + ["#E67E22"] * (nstats - nqoi)
    ax2.bar(range(nstats), diag_vals, color=colors_bar, edgecolor="k",
            lw=0.3)
    ax2.set_xticks(range(nstats))
    ax2.set_xticklabels(stat_labels, fontsize=8, rotation=45)
    ax2.set_ylabel("Estimator variance", fontsize=12)
    ax2.set_title("Individual estimator variances (diagonal)", fontsize=11)
    ax2.grid(True, alpha=0.2, axis="y")
