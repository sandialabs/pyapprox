"""Plotting functions for sensitivity analysis tutorials.

Covers: sensitivity_analysis_concept.qmd, sobol_sensitivity_analysis.qmd,
        bin_based_sensitivity.qmd
"""

import numpy as np

# ---------------------------------------------------------------------------
# sensitivity_analysis_concept.qmd — all echo:false → Convention A
# ---------------------------------------------------------------------------

def plot_variance_decomposition(main_effects, qoi_idx, input_labels,
                                qoi_labels, ax):
    """sensitivity_analysis_concept.qmd → fig-variance-decomposition

    Bar chart of variance fractions for a single QoI.
    """

    main_vals = main_effects[:, qoi_idx]

    # Interaction contribution (total - sum of main effects)
    interaction_fraction = max(0, 1.0 - np.sum(main_vals))

    bar_labels = list(input_labels)
    bar_vals = list(main_vals)
    colors = ["#2C7FB8", "#E67E22", "#27AE60"]
    if interaction_fraction > 0.005:
        bar_labels.append("Interactions")
        bar_vals.append(interaction_fraction)
        colors.append("#CCCCCC")

    bars = ax.bar(bar_labels, bar_vals, color=colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Fraction of total variance", fontsize=11)
    ax.set_title(f"Variance decomposition: {qoi_labels[qoi_idx]}",
                 fontsize=13)
    ax.set_ylim(0, max(bar_vals) * 1.15)
    ax.grid(True, alpha=0.2, axis="y")


def plot_sobol_bar_chart(main_effects, total_effects, plot_qois,
                         input_labels, qoi_labels, nvars, axes):
    """sensitivity_analysis_concept.qmd → fig-sobol-bar-chart

    Side-by-side first-order vs total-order Sobol indices for selected QoIs.
    """

    x = np.arange(nvars)
    width = 0.35

    for panel, (k, ax) in enumerate(zip(plot_qois, axes)):
        ax.bar(x - width / 2, main_effects[:, k], width,
               label=r"$S_i$ (first-order)", color="#2C7FB8", alpha=0.8)
        ax.bar(x + width / 2, total_effects[:, k], width,
               label=r"$S_i^T$ (total-order)", color="#E67E22", alpha=0.8)
        ax.set_xlabel("KLE input", fontsize=11)
        ax.set_title(qoi_labels[k], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(input_labels)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        if panel == 0:
            ax.set_ylabel("Sobol index", fontsize=11)
            ax.legend(fontsize=9)


def plot_scatter_dominant(scatter_samples_np, scatter_values, main_effects,
                          qoi_idx, input_labels, qoi_labels, ax_dom, ax_least):
    """sensitivity_analysis_concept.qmd → fig-scatter-dominant

    Scatter plots of QoI vs dominant and least influential inputs.
    """

    dominant_var = np.argmax(main_effects[:, qoi_idx])
    least_var = np.argmin(main_effects[:, qoi_idx])

    ax_dom.scatter(scatter_samples_np[dominant_var, :],
                   scatter_values[qoi_idx, :],
                   s=8, alpha=0.5, color="#2C7FB8")
    ax_dom.set_xlabel(input_labels[dominant_var], fontsize=12)
    ax_dom.set_ylabel(qoi_labels[qoi_idx], fontsize=12)
    ax_dom.set_title(
        f"Dominant input: {input_labels[dominant_var]}"
        f" ($S = {main_effects[dominant_var, qoi_idx]:.3f}$)",
        fontsize=11,
    )
    ax_dom.grid(True, alpha=0.2)

    ax_least.scatter(scatter_samples_np[least_var, :],
                     scatter_values[qoi_idx, :],
                     s=8, alpha=0.5, color="#AAAAAA")
    ax_least.set_xlabel(input_labels[least_var], fontsize=12)
    ax_least.set_ylabel(qoi_labels[qoi_idx], fontsize=12)
    ax_least.set_title(
        f"Least influential: {input_labels[least_var]}"
        f" ($S = {main_effects[least_var, qoi_idx]:.3f}$)",
        fontsize=11,
    )
    ax_least.grid(True, alpha=0.2)


# ---------------------------------------------------------------------------
# sobol_sensitivity_analysis.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_sobol_indices(first_order, total_order, labels, d, axes):
    """sobol_sensitivity_analysis.qmd → fig-sobol-indices

    First-order bar chart and first-order vs total-order comparison.
    """

    x = np.arange(d)
    width = 0.35

    # First-order indices
    axes[0].bar(x, first_order, width, label='First-order S_i', alpha=0.8)
    axes[0].set_xlabel('Parameter')
    axes[0].set_ylabel('Sobol Index')
    axes[0].set_title('First-Order Sensitivity Indices')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_ylim(-0.1, max(first_order) * 1.2)

    # Comparison: First-order vs Total-order
    axes[1].bar(x - width / 2, first_order, width,
                label='First-order S_i', alpha=0.8)
    axes[1].bar(x + width / 2, total_order, width,
                label='Total-order S^T_i', alpha=0.8)
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Sobol Index')
    axes[1].set_title('First-Order vs Total-Order Indices')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].legend()
    axes[1].axhline(0, color='black', linewidth=0.5)


def plot_group_pie(group_sensitivity, ax):
    """sobol_sensitivity_analysis.qmd → fig-group-pie

    Pie chart of total-order sensitivity aggregated by parameter group.
    """
    values = list(group_sensitivity.values())
    labels_pie = list(group_sensitivity.keys())
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    ax.pie(values, labels=labels_pie, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Variance Contribution by Parameter Group')


# ---------------------------------------------------------------------------
# bin_based_sensitivity.qmd — echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_bin_1d(samples_np, values_np, main_effects, nbins, axes):
    """bin_based_sensitivity.qmd → fig-bin-1d

    Conditional means per bin for each input variable (1D binning).
    """
    global_mean = np.mean(values_np)

    for var_idx in range(3):
        ax = axes[var_idx]

        # Compute bin statistics
        probs = np.linspace(0, 1, nbins + 1)
        bin_edges = np.quantile(samples_np[var_idx], probs)
        bin_indices = np.clip(
            np.digitize(samples_np[var_idx], bin_edges[:-1]) - 1,
            0, nbins - 1,
        )

        bin_means, bin_centers = [], []
        for k in range(nbins):
            mask = bin_indices == k
            if np.sum(mask) > 0:
                bin_means.append(np.mean(values_np[mask]))
                bin_centers.append(np.mean(samples_np[var_idx, mask]))

        # Plot histogram of samples
        ax.hist(samples_np[var_idx], bins=30, alpha=0.3, density=True,
                color='tab:blue', label='Sample density')

        # Plot bin means as bar chart overlay
        ax2 = ax.twinx()
        ax2.bar(bin_centers, bin_means, width=0.4, alpha=0.7, color='orange',
                label=r'$E[Y|\mathrm{bin}]$')
        ax2.axhline(global_mean, color='red', linestyle='--', label='$E[Y]$')

        ax.set_xlabel(f'$X_{var_idx + 1}$')
        ax.set_ylabel('Density')
        ax2.set_ylabel('Conditional Mean', color='orange')
        ax.set_title(f'$S_{var_idx + 1}$ = {main_effects[var_idx]:.3f}')

        # Combine legends from both axes
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if var_idx == 2:
            ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)


def plot_bin_2d(xi, xj, values_np, edges_i, edges_j, cell_means,
                nbins_2d, axes, fig=None):
    """bin_based_sensitivity.qmd → fig-bin-2d

    2D binning visualization: scatter with grid and conditional-mean heatmap.
    """
    import matplotlib.pyplot as plt

    # Left plot: scatter with bin grid
    ax1 = axes[0]
    scatter = ax1.scatter(xi, xj, c=values_np, alpha=0.3, s=5,
                          cmap='coolwarm')
    if fig is not None:
        plt.colorbar(scatter, ax=ax1, label='Y')

    for edge in edges_i:
        ax1.axvline(edge, color='black', linewidth=0.5, alpha=0.7)
    for edge in edges_j:
        ax1.axhline(edge, color='black', linewidth=0.5, alpha=0.7)

    ax1.set_xlabel('$X_1$')
    ax1.set_ylabel('$X_3$')
    ax1.set_title(
        f'Samples colored by Y with bin grid ({nbins_2d}x{nbins_2d})')

    # Right plot: cell means heatmap
    ax2 = axes[1]
    im = ax2.imshow(cell_means.T, origin='lower', cmap='coolwarm',
                    aspect='auto',
                    extent=[edges_i[0], edges_i[-1],
                            edges_j[0], edges_j[-1]])
    if fig is not None:
        plt.colorbar(im, ax=ax2, label='E[Y | cell]')

    for edge in edges_i[1:-1]:
        ax2.axvline(edge, color='white', linewidth=0.5)
    for edge in edges_j[1:-1]:
        ax2.axhline(edge, color='white', linewidth=0.5)

    ax2.set_xlabel('$X_1$')
    ax2.set_ylabel('$X_3$')
    ax2.set_title('Conditional means E[Y | X_1, X_3] per cell')


def plot_bin_vs_mc(budgets, bin_errors, mc_errors, gt_main,
                   bin_main_detail, mc_main_detail, N_detail,
                   N_base_detail, axes):
    """bin_based_sensitivity.qmd → fig-bin-vs-mc

    Convergence comparison and per-variable accuracy at equal budget.
    """

    # Left: convergence plot
    ax = axes[0]
    bin_med = [np.median(bin_errors[N]) for N in budgets]
    bin_q25 = [np.percentile(bin_errors[N], 25) for N in budgets]
    bin_q75 = [np.percentile(bin_errors[N], 75) for N in budgets]

    mc_med = [np.median(mc_errors[N]) for N in budgets]
    mc_q25 = [np.percentile(mc_errors[N], 25) for N in budgets]
    mc_q75 = [np.percentile(mc_errors[N], 75) for N in budgets]

    ax.plot(budgets, bin_med, 'o-', color='tab:blue',
            label='Bin-based ($S_i$)')
    ax.fill_between(budgets, bin_q25, bin_q75, color='tab:blue', alpha=0.2)
    ax.plot(budgets, mc_med, 's-', color='tab:orange',
            label='Saltelli/Jansen ($S_i$)')
    ax.fill_between(budgets, mc_q25, mc_q75, color='tab:orange', alpha=0.2)

    ax.set_xlabel('Total function evaluations')
    ax.set_ylabel('Max absolute error in $S_i$')
    ax.set_title('First-Order Index Error vs. Budget')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-variable comparison at N_detail
    ax = axes[1]
    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, gt_main, width, label='Ground truth',
           alpha=0.8, color='tab:green')
    ax.bar(x, bin_main_detail, width,
           label=f'Bin-based (N={N_detail})', alpha=0.8, color='tab:blue')
    ax.bar(x + width, mc_main_detail, width,
           label=f'Saltelli (N_base={N_base_detail})',
           alpha=0.8, color='tab:orange')
    ax.set_xlabel('Variable')
    ax.set_ylabel('$S_i$')
    ax.set_title(f'First-Order Indices at Budget = {N_detail}')
    ax.set_xticks(x)
    ax.set_xticklabels(['$X_1$', '$X_2$', '$X_3$'])
    ax.legend(fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)


def plot_bootstrap(median, q25, q75, gt_main, ax):
    """bin_based_sensitivity.qmd → fig-bootstrap

    Bootstrap uncertainty of bin-based main effect estimates.
    """
    x = np.arange(3)
    yerr = np.array([median - q25, q75 - median])

    ax.bar(x, median, yerr=yerr, capsize=5, alpha=0.8,
           label='Bootstrap median')
    ax.scatter(x, gt_main, marker='*', s=200, color='red', zorder=5,
               label='Ground truth')

    ax.set_xlabel('Variable')
    ax.set_ylabel('Sobol Index')
    ax.set_title('Bin-Based Estimates with Bootstrap Uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels(['$X_1$', '$X_2$', '$X_3$'])
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
