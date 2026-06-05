"""Plotting functions for multifidelity_aerospace_beam.qmd."""

import numpy as np

_ESTIMATOR_STYLES = {
    "MC": dict(color="#95a5a6", marker="o", ls="--"),
    "MLMC subsets": dict(color="#E67E22", marker="s", ls="-"),
    "SAOB subsets": dict(color="#2C7FB8", marker="^", ls="-"),
    "Full subsets": dict(color="#27AE60", marker="D", ls="-"),
    "Full + known mean": dict(color="#8E44AD", marker="*", ls="-", ms=10),
}

_MODEL_COLORS = ["#E74C3C", "#2C7FB8", "#27AE60", "#8E44AD"]


def _short_model_labels(model_names):
    """Convert full model names to compact labels for tick marks."""
    labels = []
    for i, n in enumerate(model_names):
        if ":" in n:
            parts = n.split(":", 1)
            labels.append(parts[0].strip())
        else:
            labels.append(f"f{i}")
    return labels


def _legend_model_labels(model_names):
    """Convert full model names to legend-ready labels with description."""
    return list(model_names)


def plot_variance_vs_cost(ax, results_dict, target_costs):
    """Log-log predicted estimator variance vs total cost.

    Parameters
    ----------
    ax : matplotlib Axes
    results_dict : dict[str, list[float]]
        Keys are estimator names, values are lists of predicted
        variances at each target cost.
    target_costs : array-like
        Budget levels.
    """
    for name, variances in results_dict.items():
        style = _ESTIMATOR_STYLES.get(name, {})
        ax.loglog(target_costs, variances, label=name, **style)
    ax.set_xlabel("Total cost (HF-equivalent evaluations)")
    ax.set_ylabel("Predicted estimator variance (tip deflection)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)


def plot_allocation_bars(ax, nsamples_per_model, model_names, title=""):
    """Stacked bar chart of sample counts per model.

    Parameters
    ----------
    ax : matplotlib Axes
    nsamples_per_model : array-like, shape (nmodels,)
        Number of samples allocated to each model.
    model_names : list of str
    title : str
    """
    colors = ["#2C7FB8", "#27AE60", "#E67E22", "#8E44AD"]
    x = np.arange(len(model_names))
    bars = ax.bar(x, nsamples_per_model, color=colors[: len(model_names)])
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of samples")
    if title:
        ax.set_title(title, fontsize=10)
    for bar, n in zip(bars, nsamples_per_model):
        if n > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(n)}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def plot_pilot_correlation_heatmap(ax, cov_np, nmodels, nqoi, model_names,
                                   qoi_names):
    """Heatmap of the pilot correlation matrix over (model, QoI) blocks.

    Parameters
    ----------
    ax : matplotlib Axes
    cov_np : ndarray, shape (nmodels*nqoi, nmodels*nqoi)
    nmodels, nqoi : int
    model_names : list of str
    qoi_names : list of str
    """
    n = nmodels * nqoi
    std = np.sqrt(np.diag(cov_np))
    corr = cov_np / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = []
    for mi in range(nmodels):
        for qi in range(nqoi):
            tick_labels.append(f"{model_names[mi]}\n{qoi_names[qi]}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=5.5)

    max_offdiag = np.max(np.abs(corr[np.triu_indices(n, k=1)]))
    if max_offdiag > 0.99:
        fmt = ".4f"
        fs = 4.5
    elif max_offdiag > 0.9:
        fmt = ".3f"
        fs = 5
    else:
        fmt = ".2f"
        fs = 5

    for i in range(n):
        for j in range(n):
            color = "w" if abs(corr[i, j]) > 0.7 else "k"
            ax.text(j, i, f"{corr[i, j]:{fmt}}", ha="center", va="center",
                    fontsize=fs, color=color)

    for k in range(1, nmodels):
        pos = k * nqoi - 0.5
        ax.axhline(pos, color="k", lw=0.5)
        ax.axvline(pos, color="k", lw=0.5)


def plot_pilot_means_and_variances(axes, pilot_values_np, model_names,
                                   qoi_names):
    """Bar charts of per-model pilot means (top row) and variances (bottom row).

    Parameters
    ----------
    axes : array of shape (2, nqoi) matplotlib Axes
        Row 0: means, Row 1: variances.
    pilot_values_np : list of ndarray, each shape (nqoi, npilot)
    model_names : list of str
    qoi_names : list of str
    """
    nmodels = len(pilot_values_np)
    x = np.arange(nmodels)
    legend_names = _legend_model_labels(model_names)

    for qi, qname in enumerate(qoi_names):
        means = [pilot_values_np[m][qi].mean() for m in range(nmodels)]
        variances = [pilot_values_np[m][qi].var() for m in range(nmodels)]

        ax_mean = axes[0, qi]
        bars = ax_mean.bar(x, means, color=_MODEL_COLORS[:nmodels], alpha=0.8)
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels(legend_names, fontsize=7, rotation=20,
                                ha="right")
        ax_mean.set_ylabel("Mean", fontsize=8)
        ax_mean.set_title(qname, fontsize=10)
        for bar, m in zip(bars, means):
            ax_mean.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{m:.2f}", ha="center", va="bottom", fontsize=6)

        ax_var = axes[1, qi]
        bars = ax_var.bar(x, variances, color=_MODEL_COLORS[:nmodels],
                          alpha=0.8)
        ax_var.set_xticks(x)
        ax_var.set_xticklabels(legend_names, fontsize=7, rotation=20,
                               ha="right")
        ax_var.set_ylabel("Variance", fontsize=8)
        for bar, v in zip(bars, variances):
            ax_var.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.2e}", ha="center", va="bottom", fontsize=5.5)


def plot_bias_vs_hf(ax, pilot_values_np, model_names, qoi_names):
    """Grouped bar chart of relative bias (model mean − f0 mean) / |f0 mean|.

    Parameters
    ----------
    ax : matplotlib Axes
    pilot_values_np : list of ndarray, each shape (nqoi, npilot)
    model_names : list of str
    qoi_names : list of str
    """
    nmodels = len(pilot_values_np)
    nqoi = len(qoi_names)
    legend_names = _legend_model_labels(model_names)
    hf_means = np.array([pilot_values_np[0][qi].mean() for qi in range(nqoi)])

    x = np.arange(1, nmodels)
    width = 0.35
    for qi, qname in enumerate(qoi_names):
        rel_biases = [
            (pilot_values_np[m][qi].mean() - hf_means[qi]) / abs(hf_means[qi])
            for m in range(1, nmodels)
        ]
        offset = (qi - (nqoi - 1) / 2) * width
        bars = ax.bar(x + offset, rel_biases, width, label=qname,
                      color=_MODEL_COLORS[qi], alpha=0.8)
        for bar, rb in zip(bars, rel_biases):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{rb:.1%}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(legend_names[1:], fontsize=7, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Relative bias  (model − f0) / |f0|")
    ax.legend(fontsize=8)


def plot_1d_sweep(axes, sweep_t, sweep_values_per_model, model_names,
                  qoi_names):
    """1D parameter-sweep response curves for all models, one panel per QoI.

    Parameters
    ----------
    axes : array of matplotlib Axes, one per QoI
    sweep_t : ndarray, shape (npts,)
        Canonical 1D coordinate along the sweep direction.
    sweep_values_per_model : list of ndarray, each shape (nqoi, npts)
    model_names : list of str
    qoi_names : list of str
    """
    nmodels = len(sweep_values_per_model)
    legend_names = _legend_model_labels(model_names)
    for qi, qname in enumerate(qoi_names):
        ax = axes[qi]
        for mi in range(nmodels):
            ax.plot(sweep_t, sweep_values_per_model[mi][qi],
                    color=_MODEL_COLORS[mi], lw=1.5, label=legend_names[mi])
        ax.set_xlabel("Distance along sweep direction", fontsize=8)
        ax.set_ylabel(qname, fontsize=8)
        ax.set_title(qname, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)


def plot_variance_vs_cost_grid(axes, all_results, target_costs, qoi_names,
                              stat_names):
    """2×2 grid of variance-vs-cost curves.

    Parameters
    ----------
    axes : ndarray of shape (nstats, nqoi) matplotlib Axes
        Row 0: mean, Row 1: variance (or whatever stat_names says).
    all_results : dict[(stat_name, qoi_idx), dict[str, list[float]]]
        Outer key is (stat_name, qoi_idx), inner dict maps estimator
        name to list of predicted variances at each budget.
    target_costs : array-like
    qoi_names : list of str
    stat_names : list of str
    """
    _STYLES = {
        "MC": dict(color="#95a5a6", marker="o", ls="--", ms=6),
        "GACV": dict(color="#2C7FB8", marker="^", ls="-", ms=6),
        "GACV + known": dict(color="#8E44AD", marker="*", ls="-", ms=8),
        "Joint 2-QoI": dict(color="#27AE60", marker="D", ls="-.", ms=5),
    }
    for si, sname in enumerate(stat_names):
        for qi, qname in enumerate(qoi_names):
            ax = axes[si, qi]
            key = (sname, qi)
            if key not in all_results:
                continue
            results = all_results[key]
            for ename, variances in results.items():
                style = _STYLES.get(ename, {})
                ax.loglog(target_costs, variances, label=ename, **style)
            ax.set_xlabel("Total cost (HF-equivalent)")
            ylabel = f"Estimator var ({sname})"
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{sname} of {qname}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)


def plot_joint_vs_separate_reduction(ax, joint_ratios, sep_ratios, qoi_names,
                                     budget=None):
    """Grouped bar chart of variance reduction (estimator var / MC var).

    Parameters
    ----------
    ax : matplotlib Axes
    joint_ratios : list of float
        Estimator variance / MC variance for joint estimation, per QoI.
    sep_ratios : list of float
        Estimator variance / MC variance for separate estimation, per QoI.
    qoi_names : list of str
    budget : float, optional
        Budget for the title.
    """
    nqoi = len(qoi_names)
    x = np.arange(nqoi)
    width = 0.3

    bars_sep = ax.bar(x - width / 2, sep_ratios, width, label="Separate",
                      color="#2C7FB8", edgecolor="k", lw=0.5)
    bars_joint = ax.bar(x + width / 2, joint_ratios, width, label="Joint",
                        color="#27AE60", edgecolor="k", lw=0.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(qoi_names, fontsize=9)
    ax.set_ylabel(
        r"$\mathbb{V}[\hat{\mu}]\;/\;\mathbb{V}_{\mathrm{MC}}$",
        fontsize=10,
    )
    title = "Joint vs separate estimation"
    if budget is not None:
        title += f" ($P={int(budget)}$)"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y", which="both")
    ax.set_axisbelow(True)

    for bars in [bars_sep, bars_joint]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)
