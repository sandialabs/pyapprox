"""Plotting functions for KLE tutorials.

Covers: kle_introduction.qmd, kle_mesh_and_data_driven.qmd, kle_spde.qmd
"""

import numpy as np


# ---------------------------------------------------------------------------
# kle_introduction.qmd — all echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_cov_matrices(mesh_coords, kernel_factory, ell_values, bkd, axes):
    """kle_introduction.qmd -> fig-cov-matrices

    Covariance matrix heatmaps for several correlation lengths.
    """
    for ax, ell in zip(axes, ell_values):
        kernel = kernel_factory(ell)
        C = bkd.to_numpy(kernel(mesh_coords, mesh_coords))
        ax.imshow(C, extent=[0, 2, 2, 0], cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title(f"$\\ell = {ell}$", fontsize=13)
        ax.set_xlabel("$x'$")
    axes[0].set_ylabel("$x$")


def plot_eigenvalues_eigenfunctions(x, lam, phi, nterms, ax_eig, ax_phi):
    """kle_introduction.qmd -> fig-eigenvalues-eigenfunctions

    Eigenvalue bar chart and first four eigenfunctions.
    """
    import matplotlib.pyplot as plt

    # Eigenvalue decay
    ax_eig.bar(range(1, nterms + 1), lam, color="steelblue", alpha=0.8)
    ax_eig.set_xlabel("Mode $k$", fontsize=12)
    ax_eig.set_ylabel("$\\lambda_k$", fontsize=12)
    ax_eig.set_title("Eigenvalue decay \u2014 energy per mode", fontsize=12)
    ax_eig.set_xticks(range(1, nterms + 1))
    ax_eig.set_ylim(bottom=0)

    # First 4 eigenfunctions
    colors = plt.cm.tab10(np.linspace(0, 0.5, 4))
    for k in range(4):
        ax_phi.plot(
            x, phi[:, k], color=colors[k],
            label=f"$\\phi_{k+1}$  ($\\lambda={lam[k]:.3f}$)",
        )
    ax_phi.axhline(0, color="k", lw=0.5)
    ax_phi.set_xlabel("$x$", fontsize=12)
    ax_phi.set_title("First four eigenfunctions $\\phi_k(x)$", fontsize=12)
    ax_phi.legend(fontsize=10)


def plot_sample_paths(x, fields_list, K_list, axes):
    """kle_introduction.qmd -> fig-sample-paths

    Sample paths from the KLE with increasing number of terms.
    """
    for ax, fields, K in zip(axes, fields_list, K_list):
        nsamples = fields.shape[1]
        for s in range(nsamples):
            ax.plot(x, fields[:, s], alpha=0.7, lw=1.2)
        ax.set_title(f"$K = {K}$ terms", fontsize=12)
        ax.set_xlabel("$x$")
        ax.axhline(0, color="k", lw=0.5, ls="--")
    axes[0].set_ylabel("$f(x)$")


def plot_kle_truncation_error(
    x, lam_full, nterms_full, mesh_coords, quad_weights,
    ell_values, bkd, kernel_factory, kle_factory,
    ax_cumvar, ax_decay,
):
    """kle_introduction.qmd -> fig-truncation-error

    Cumulative variance explained and eigenvalue decay by correlation length.
    """
    # Cumulative variance explained
    cumvar = np.cumsum(lam_full) / lam_full.sum()
    ax_cumvar.plot(
        range(1, nterms_full + 1), cumvar, "steelblue", lw=2,
    )
    for thresh in [0.90, 0.95, 0.99]:
        k_thresh = int(np.searchsorted(cumvar, thresh)) + 1
        ax_cumvar.axhline(thresh, color="gray", ls="--", lw=0.8)
        ax_cumvar.text(
            nterms_full * 0.55, thresh + 0.005,
            f"{int(thresh * 100)}% at $K={k_thresh}$", fontsize=9,
        )
    ax_cumvar.set_xlabel("Number of terms $K$", fontsize=12)
    ax_cumvar.set_ylabel("Cumulative variance explained", fontsize=12)
    ax_cumvar.set_title(
        "Truncation error vs $K$  ($\\ell=0.5$)", fontsize=12,
    )
    ax_cumvar.set_xlim(1, nterms_full)

    # Eigenvalue decay for three correlation lengths
    colors2 = ["steelblue", "darkorange", "green"]
    for color, ell_i in zip(colors2, ell_values):
        kernel_i = kernel_factory(ell_i)
        kle_i = kle_factory(kernel_i, nterms_full)
        lam_i = bkd.to_numpy(kle_i.eigenvalues())
        ax_decay.semilogy(
            range(1, nterms_full + 1), lam_i / lam_i[0],
            color=color, lw=1.8, label=f"$\\ell={ell_i}$",
        )
    ax_decay.set_xlabel("Mode $k$", fontsize=12)
    ax_decay.set_ylabel(
        "$\\lambda_k / \\lambda_1$ (log scale)", fontsize=12,
    )
    ax_decay.set_title(
        "Eigenvalue decay by correlation length", fontsize=12,
    )
    ax_decay.legend(fontsize=11)
    ax_decay.set_xlim(1, 40)


def plot_pointwise_variance(
    x, true_var, K_vals, approx_vars, ax_var, ax_err,
):
    """kle_introduction.qmd -> fig-pointwise-variance

    Pointwise variance recovered by truncated KLE and relative error.
    """
    import matplotlib.pyplot as plt

    colors3 = plt.cm.viridis(np.linspace(0, 0.85, len(K_vals)))
    for color, K, approx_var in zip(colors3, K_vals, approx_vars):
        ax_var.plot(x, approx_var, color=color, lw=1.8, label=f"$K={K}$")
    ax_var.plot(x, true_var, "k--", lw=2, label="exact")
    ax_var.set_xlabel("$x$", fontsize=12)
    ax_var.set_ylabel("Pointwise variance", fontsize=12)
    ax_var.set_title("Variance recovered by truncated KLE", fontsize=12)
    ax_var.legend(fontsize=10)

    for color, K, approx_var in zip(colors3, K_vals, approx_vars):
        rel_err = np.abs(true_var - approx_var) / true_var
        ax_err.semilogy(
            x, rel_err + 1e-16, color=color, lw=1.8, label=f"$K={K}$",
        )
    ax_err.set_xlabel("$x$", fontsize=12)
    ax_err.set_ylabel("Relative variance error (log)", fontsize=12)
    ax_err.set_title("Truncation error in pointwise variance", fontsize=12)
    ax_err.legend(fontsize=10)


def plot_analytical_validation(
    lam, lam_exact, nterms, npts, ax_bar, ax_err,
):
    """kle_introduction.qmd -> fig-analytical-validation

    Numerical vs analytical eigenvalues with relative error.
    """
    k_idx = np.arange(1, nterms + 1)
    ax_bar.bar(
        k_idx - 0.15, lam, width=0.3, color="steelblue", alpha=0.8,
        label="MeshKLE (numerical)",
    )
    ax_bar.bar(
        k_idx + 0.15, lam_exact, width=0.3, color="darkorange", alpha=0.8,
        label="Analytical",
    )
    ax_bar.set_xlabel("Mode $k$", fontsize=12)
    ax_bar.set_ylabel("$\\lambda_k$", fontsize=12)
    ax_bar.set_title("Eigenvalue comparison", fontsize=12)
    ax_bar.set_xticks(k_idx)
    ax_bar.legend(fontsize=10)
    ax_bar.set_ylim(bottom=0)

    rel_err_eig = np.abs(lam - lam_exact) / lam_exact
    ax_err.bar(k_idx, rel_err_eig, color="steelblue", alpha=0.8)
    ax_err.set_xlabel("Mode $k$", fontsize=12)
    ax_err.set_ylabel("Relative eigenvalue error", fontsize=12)
    ax_err.set_title(
        f"MeshKLE accuracy  ({npts} mesh points)", fontsize=12,
    )
    ax_err.set_xticks(k_idx)
    ax_err.set_yscale("log")


# ---------------------------------------------------------------------------
# kle_mesh_and_data_driven.qmd — all echo:true → Convention B
# ---------------------------------------------------------------------------

def plot_mesh_kle_overview(
    x, C, phi_mesh, nterms, w, ax_cov, ax_phi, ax_gram, fig=None,
):
    """kle_mesh_and_data_driven.qmd -> fig-mesh-kle-overview

    Covariance matrix, MeshKLE eigenfunctions, and Gram matrix check.
    """
    import matplotlib.pyplot as plt

    im = ax_cov.imshow(
        C, extent=[0, 2, 2, 0], cmap="RdBu_r", vmin=0, vmax=1,
    )
    if fig is not None:
        fig.colorbar(im, ax=ax_cov)
    ax_cov.set_title("Covariance matrix $C(x,x')$", fontsize=12)
    ax_cov.set_xlabel("$x'$")
    ax_cov.set_ylabel("$x$")

    colors = plt.cm.tab10(np.linspace(0, 0.55, nterms))
    for k in range(nterms):
        ax_phi.plot(
            x, phi_mesh[:, k], color=colors[k], label=f"$\\phi_{k+1}$",
        )
    ax_phi.axhline(0, color="k", lw=0.5)
    ax_phi.set_title("MeshKLE eigenfunctions", fontsize=12)
    ax_phi.set_xlabel("$x$")
    ax_phi.legend(fontsize=9)

    G = phi_mesh.T @ (w[:, None] * phi_mesh)
    im2 = ax_gram.imshow(G, cmap="RdBu_r", vmin=-1.1, vmax=1.1)
    if fig is not None:
        fig.colorbar(im2, ax=ax_gram)
    ax_gram.set_title("$\\Phi^T W \\Phi$ (should be $I$)", fontsize=12)


def plot_data_driven_eigenfunctions(
    x, phi_mesh, phi_dd_list, N_samples_list, nterms, axes,
):
    """kle_mesh_and_data_driven.qmd -> fig-data-driven-eigenfunctions

    DataDrivenKLE eigenfunctions vs MeshKLE reference for several sample counts.
    """
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 0.55, nterms))
    for ax, phi_dd, Ns in zip(axes, phi_dd_list, N_samples_list):
        for k in range(nterms):
            sign = np.sign(np.dot(phi_dd[:, k], phi_mesh[:, k]))
            ax.plot(
                x, sign * phi_dd[:, k], color=colors[k], lw=1.5, alpha=0.9,
            )
        for k in range(nterms):
            ax.plot(
                x, phi_mesh[:, k], color=colors[k], lw=2, ls="--", alpha=0.4,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"$N_s = {Ns}$ samples", fontsize=12)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("Eigenfunction $\\phi_k(x)$")


def plot_eigenvalue_convergence(
    x, Ns_range, errors, nterms, phi_mesh,
    phi_dd_w, phi_dd_no_w, ax_conv, ax_weights,
):
    """kle_mesh_and_data_driven.qmd -> fig-eigenvalue-convergence

    Eigenvalue error vs sample count and effect of quadrature weights.
    """
    # Relative eigenvalue error vs sample count
    for k in range(nterms):
        ax_conv.loglog(
            Ns_range, errors[:, k], lw=1.8, label=f"$\\lambda_{k+1}$",
        )
    ns_ref = np.array([100, 10000])
    ax_conv.loglog(
        ns_ref, 0.3 / np.sqrt(ns_ref), "k--", lw=1,
        label="$1/\\sqrt{N}$",
    )
    ax_conv.set_xlabel("Number of samples $N_s$", fontsize=12)
    ax_conv.set_ylabel("Relative eigenvalue error", fontsize=12)
    ax_conv.set_title("DataDrivenKLE eigenvalue convergence", fontsize=12)
    ax_conv.legend(fontsize=9)

    # Effect of quadrature weights on first two eigenfunctions
    for k in range(2):
        sign_w = np.sign(np.dot(phi_dd_w[:, k], phi_mesh[:, k]))
        sign_nw = np.sign(np.dot(phi_dd_no_w[:, k], phi_mesh[:, k]))
        ax_weights.plot(
            x, sign_w * phi_dd_w[:, k], lw=2,
            label=f"$\\phi_{k+1}$ with weights",
        )
        ax_weights.plot(
            x, sign_nw * phi_dd_no_w[:, k], lw=2, ls="--",
            label=f"$\\phi_{k+1}$ no weights",
        )
    ax_weights.plot(x, phi_mesh[:, 0], "k:", lw=1.5, label="reference")
    ax_weights.set_xlabel("$x$", fontsize=12)
    ax_weights.set_title(
        "Effect of quadrature weights  ($N_s=500$)", fontsize=12,
    )
    ax_weights.legend(fontsize=9)


def plot_mesh_convergence(
    npts_list, eig_errors_rel, nterms, table_data, ax_conv, ax_table,
):
    """kle_mesh_and_data_driven.qmd -> fig-mesh-convergence

    MeshKLE eigenvalue convergence with mesh refinement plus comparison table.
    """
    for k in range(nterms):
        ax_conv.loglog(
            npts_list, eig_errors_rel[:, k], lw=1.8,
            label=f"$\\lambda_{k+1}$",
        )
    # O(h^2) reference
    nref = np.array([npts_list[0], npts_list[-1]])
    h_ref = 2.0 / np.array(nref, dtype=float)
    ax_conv.loglog(
        nref,
        0.5 * (h_ref / h_ref[0]) ** 2 * eig_errors_rel[0, 0],
        "k--", lw=1, label="$O(h^2)$",
    )
    ax_conv.set_xlabel("Number of mesh points $N$", fontsize=12)
    ax_conv.set_ylabel("Relative eigenvalue error", fontsize=12)
    ax_conv.set_title(
        "MeshKLE convergence with mesh refinement", fontsize=12,
    )
    ax_conv.legend(fontsize=9)

    # Comparison table
    ax_table.set_axis_off()
    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#D9E1F2")
    ax_table.set_title("Method comparison", fontsize=12, pad=20)


def plot_field_reconstruction(
    x, f_true, K_list, reconstructions, ax_list,
):
    """kle_mesh_and_data_driven.qmd -> fig-field-reconstruction

    Field reconstruction from KLE projection with increasing terms.
    """
    for ax, K, f_recon in zip(ax_list, K_list, reconstructions):
        ax.plot(x, f_true, "k-", lw=2, label="true field")
        ax.plot(x, f_recon, "r--", lw=1.8, label=f"$K={K}$ terms")
        ax.fill_between(
            x, f_true, f_recon, alpha=0.2, color="red", label="error",
        )
        rmse = np.sqrt(np.mean((f_true - f_recon) ** 2))
        ax.set_title(f"$K={K}$ terms  (RMSE={rmse:.3f})", fontsize=11)
        ax.set_xlabel("$x$")
        ax.legend(fontsize=9)
    ax_list[0].set_ylabel("$f(x)$")


# ---------------------------------------------------------------------------
# kle_spde.qmd — echo:true → Convention B (5 functions)
#                echo:false → Convention A (1 function)
# ---------------------------------------------------------------------------

def plot_matern_kernels_paths(
    x_plot, kernel_curves, sample_paths_list, path_labels, axes,
):
    """kle_spde.qmd -> fig-matern-kernels-paths

    Matern kernel functions and sample paths for two smoothness values.
    """
    ax = axes[0]
    for r, k_vals, name, color in kernel_curves:
        ax.plot(x_plot, k_vals, color=color, lw=2, label=name)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$k(x_0, x)$", fontsize=12)
    ax.set_title(f"Matern kernels  ($\\ell={0.4}$)", fontsize=12)
    ax.legend(fontsize=9)

    for ax_idx, (fields, label) in enumerate(
        zip(sample_paths_list, path_labels)
    ):
        ax = axes[ax_idx + 1]
        nsamples = fields.shape[1]
        for s in range(nsamples):
            ax.plot(x_plot, fields[:, s], lw=1.2, alpha=0.8)
        ax.set_title(f"{label} sample paths", fontsize=12)
        ax.set_xlabel("$x$")


def plot_spde_overview(
    x_nodes, phi_spde, nterms, fields_spde, var_spde,
    k_idx, lam_dense, lam_spde, ell_c,
    ax_phi, ax_paths, ax_eig,
):
    """kle_spde.qmd -> fig-spde-overview

    SPDE eigenfunctions, sample paths with variance envelope, eigenvalue comparison.
    """
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10(np.linspace(0, 0.55, nterms))
    for k in range(nterms):
        sign = np.sign(phi_spde[1, k])
        ax_phi.plot(
            x_nodes, sign * phi_spde[:, k], color=colors[k],
            label=f"$\\phi_{k+1}$",
        )
    ax_phi.axhline(0, color="k", lw=0.5)
    ax_phi.set_title("SPDE eigenfunctions", fontsize=12)
    ax_phi.set_xlabel("$x$")
    ax_phi.legend(fontsize=9)

    nsamples_show = fields_spde.shape[1]
    for s in range(nsamples_show):
        ax_paths.plot(x_nodes, fields_spde[:, s], lw=1.5, alpha=0.9)
    ax_paths.fill_between(
        x_nodes,
        -2 * np.sqrt(var_spde), 2 * np.sqrt(var_spde),
        alpha=0.15, color="gray", label="$\\pm 2\\sigma$",
    )
    ax_paths.set_title(
        f"SPDE KLE sample paths  ($\\ell_c={ell_c}$, $K={nterms}$)",
        fontsize=11,
    )
    ax_paths.set_xlabel("$x$")
    ax_paths.legend(fontsize=10)

    ax_eig.semilogy(
        k_idx, lam_dense, "bs-", lw=2, ms=7, label="MeshKLE (dense)",
    )
    ax_eig.semilogy(
        k_idx, lam_spde, "r^--", lw=2, ms=7, label="SPDE KLE",
    )
    ax_eig.set_xlabel("Mode $k$", fontsize=12)
    ax_eig.set_ylabel("$\\lambda_k$ (log scale)", fontsize=12)
    ax_eig.set_title(
        f"Eigenvalue comparison  ($\\ell_c={ell_c}$)", fontsize=12,
    )
    ax_eig.legend(fontsize=11)
    ax_eig.set_xticks(k_idx)


def plot_boundary_artefacts(
    domain_data, ell_c_fixed, axes,
):
    """kle_spde.qmd -> fig-boundary-artefacts

    Pointwise variance of the SPDE KLE for several domain sizes.
    """
    for ax, (x_d, var_d, domain_len) in zip(axes, domain_data):
        ax.plot(x_d, var_d, "b-", lw=2, label="SPDE variance")
        ax.axhline(1.0, color="r", ls="--", lw=1.5, label="target $\\sigma^2=1$")
        ratio = domain_len / ell_c_fixed
        ax.set_title(
            f"Domain $[0,{domain_len}]$\n"
            f"Domain$/\\ell_c = {ratio:.0f}$",
            fontsize=11,
        )
        ax.set_xlabel("$x$")
        ax.set_ylim(0, 1.5)
    axes[0].set_ylabel("Pointwise variance")
    axes[0].legend(fontsize=9)


def plot_memory_scaling(
    N_vals, mem_dense, mem_sparse, cost_dense, cost_sparse,
    x_s, fields_by_ell, ells, ax_scaling, ax_paths,
):
    """kle_spde.qmd -> fig-memory-scaling

    Memory/cost scaling comparison and SPDE sample paths for several ell_c.
    """
    ax_scaling.loglog(
        N_vals, mem_dense, "bs-", lw=2, ms=7,
        label="Dense $C$  ($O(N^2)$)",
    )
    ax_scaling.loglog(
        N_vals, mem_sparse, "r^--", lw=2, ms=7,
        label="Sparse SPDE $A$  ($O(N)$)",
    )
    ax_scaling.loglog(
        N_vals, cost_dense / cost_dense[0] * mem_dense[0],
        "b:", lw=1.5, alpha=0.5, label="Dense solve  ($O(N^2 K)$)",
    )
    ax_scaling.loglog(
        N_vals, cost_sparse / cost_sparse[0] * mem_sparse[0],
        "r:", lw=1.5, alpha=0.5, label="Sparse solve  ($O(NK)$)",
    )
    ax_scaling.set_xlabel("Mesh size $N$", fontsize=12)
    ax_scaling.set_ylabel("Memory (MB) / relative cost", fontsize=12)
    ax_scaling.set_title("Memory and solve cost scaling", fontsize=12)
    ax_scaling.legend(fontsize=9)

    colors_ell = ["steelblue", "darkorange", "green"]
    for color, ell_target, fields_t in zip(colors_ell, ells, fields_by_ell):
        nsamples = fields_t.shape[1]
        for s in range(nsamples):
            lbl = f"$\\ell_c={ell_target}$" if s == 0 else None
            ax_paths.plot(
                x_s, fields_t[:, s], color=color, lw=1.5, alpha=0.8,
                label=lbl,
            )
    ax_paths.set_xlabel("$x$", fontsize=12)
    ax_paths.set_title(
        "SPDE sample paths for three correlation lengths", fontsize=12,
    )
    ax_paths.legend(fontsize=10)
    ax_paths.axhline(0, color="k", lw=0.4)


def plot_spde_parameters(
    domain_sizes, max_errors, x_s, cov_by_ell, ells, mid_idx,
    ax_conv, ax_cov,
):
    """kle_spde.qmd -> fig-spde-parameters

    SPDE vs dense KLE convergence with domain size, and empirical covariance.
    """
    colors_ell = ["steelblue", "darkorange", "green"]

    ax_conv.semilogy(domain_sizes, max_errors, "bs-", lw=2, ms=7)
    ax_conv.set_xlabel("Domain length $L$", fontsize=12)
    ax_conv.set_ylabel("Max relative eigenvalue error", fontsize=12)
    ax_conv.set_title(
        "SPDE vs dense KLE convergence\n(first 5 modes)", fontsize=12,
    )
    ax_conv.set_xticks(domain_sizes)

    for color, ell_target, cov_empirical in zip(
        colors_ell, ells, cov_by_ell,
    ):
        ax_cov.plot(
            x_s, cov_empirical, color=color, lw=2,
            label=f"$\\ell_c={ell_target}$",
        )
    ax_cov.axvline(x_s[mid_idx], color="k", ls=":", lw=1)
    ax_cov.set_xlabel("$x$", fontsize=12)
    ax_cov.set_ylabel(
        "Empirical $C(x_{\\mathrm{mid}}, x)$", fontsize=12,
    )
    ax_cov.set_title(
        "Empirical covariance from 2000 MC samples", fontsize=12,
    )
    ax_cov.legend(fontsize=10)


def plot_method_comparison(ax):
    """kle_spde.qmd -> fig-method-comparison

    Summary comparison table of MeshKLE, DataDrivenKLE, SPDEMaternKLE.
    """
    ax.set_axis_off()

    rows = [
        ["", "MeshKLE", "DataDrivenKLE", "SPDEMaternKLE"],
        ["Memory", "$O(N^2)$", "$O(N \\cdot N_s)$", "$O(N)$ sparse"],
        [
            "Eigensolve cost", "$O(N^2 K)$ dense",
            "$O(N N_s K)$ SVD", "$O(NK)$ iterative",
        ],
        ["Known kernel needed", "Yes", "No", "No (Matern implicit)"],
        [
            "Smoothness control", "Any kernel",
            "Empirical", "$\\nu = \\alpha - d/2$",
        ],
        [
            "Boundary artefacts", "Minimal",
            "None", "Yes (shrink with domain)",
        ],
        [
            "Scales to large $N$?", "No ($N \\lesssim 5000$)",
            "Moderate", "Yes",
        ],
        [
            "When to use", "Small mesh,\nknown kernel",
            "Data available,\nno kernel", "Large mesh,\nMatern family",
        ],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#D9E1F2")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F2F2")
