"""
Example: Kernel Matrix Visualization

This example demonstrates how to visualize kernel matrices K(X, X) for 1D kernels
using heatmaps and 3D surface plots.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pyapprox.typing.surrogates.kernels import MaternKernel, PolynomialScaling
from pyapprox.typing.surrogates.kernels.multioutput import (
    MultiLevelKernel,
    DAGMultiOutputKernel,
)
from pyapprox.typing.surrogates.kernels.plot_kernel_matrix import (
    plot_kernel_matrix_heatmap,
    plot_kernel_matrix_surface,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd


def example_1_matern_kernel_matrix():
    """Example 1: Visualize Matern kernel matrix."""
    print("=" * 70)
    print("Example 1: Matern Kernel Matrix Visualization")
    print("=" * 70)

    bkd = NumpyBkd()

    # Create 1D Matern kernel with nu=2.5
    kernel = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)

    print(f"Kernel: {kernel}")
    print(f"Input dimensions: {kernel.nvars()}")

    # Create figure with heatmap and 3D surface
    fig = plt.figure(figsize=(14, 6))

    # Heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = plot_kernel_matrix_heatmap(
        kernel,
        x_limits=(-3.0, 3.0),
        ax=ax1,
        npts=100,
        cmap="viridis"
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("x'")
    ax1.set_title("Matern 2.5 Kernel Matrix (Heatmap)")
    plt.colorbar(im, ax=ax1, label="K(x, x')")

    # 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = plot_kernel_matrix_surface(
        kernel,
        x_limits=(-3.0, 3.0),
        ax=ax2,
        npts=50,
        cmap="viridis",
        alpha=0.9
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("x'")
    ax2.set_zlabel("K(x, x')")
    ax2.set_title("Matern 2.5 Kernel Matrix (3D Surface)")
    plt.colorbar(surf, ax=ax2, shrink=0.5, label="K(x, x')")

    plt.tight_layout()
    plt.savefig("kernel_matrix_matern.png", dpi=150)
    print("Plot saved to kernel_matrix_matern.png")
    print()

    return fig


def example_2_compare_length_scales():
    """Example 2: Compare kernel matrices with different length scales."""
    print("=" * 70)
    print("Example 2: Effect of Length Scale on Kernel Matrix")
    print("=" * 70)

    bkd = NumpyBkd()

    # Create kernels with different length scales
    length_scales = [0.5, 1.0, 2.0]
    kernels = [
        MaternKernel(2.5, [ls], (0.1, 10.0), 1, bkd)
        for ls in length_scales
    ]

    # Create figure with multiple heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, kernel, ls in zip(axes, kernels, length_scales):
        im = plot_kernel_matrix_heatmap(
            kernel,
            x_limits=(-3.0, 3.0),
            ax=ax,
            npts=100,
            cmap="viridis"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("x'")
        ax.set_title(f"Length Scale = {ls}")
        plt.colorbar(im, ax=ax, label="K(x, x')")

    fig.suptitle("Matern 2.5 Kernel Matrix: Effect of Length Scale", fontsize=14)
    plt.tight_layout()
    plt.savefig("kernel_matrix_length_scales.png", dpi=150)
    print("Plot saved to kernel_matrix_length_scales.png")
    print()

    return fig


def example_3_compare_smoothness():
    """Example 3: Compare kernel matrices with different smoothness (nu)."""
    print("=" * 70)
    print("Example 3: Effect of Smoothness (nu) on Kernel Matrix")
    print("=" * 70)

    bkd = NumpyBkd()

    # Create kernels with different nu values
    nu_values = [np.inf, 1.5, 2.5]
    kernels = [
        MaternKernel(nu, [1.0], (0.1, 10.0), 1, bkd)
        for nu in nu_values
    ]

    # Create figure with 3D surfaces
    fig = plt.figure(figsize=(18, 5))

    for i, (kernel, nu) in enumerate(zip(kernels, nu_values), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        surf = plot_kernel_matrix_surface(
            kernel,
            x_limits=(-3.0, 3.0),
            ax=ax,
            npts=50,
            cmap="viridis",
            alpha=0.9
        )
        ax.set_xlabel("x")
        ax.set_ylabel("x'")
        ax.set_zlabel("K(x, x')")
        ax.set_title(f"nu = {nu}")
        plt.colorbar(surf, ax=ax, shrink=0.5, label="K(x, x')")

    fig.suptitle("Matern Kernel Matrix: Effect of Smoothness Parameter", fontsize=14)
    plt.tight_layout()
    plt.savefig("kernel_matrix_smoothness.png", dpi=150)
    print("Plot saved to kernel_matrix_smoothness.png")
    print()

    return fig


def example_4_multilevel_kernel_matrices():
    """Example 4: Visualize multi-level kernel matrices."""
    print("=" * 70)
    print("Example 4: Multi-Level Kernel Matrices")
    print("=" * 70)

    bkd = NumpyBkd()

    # Create 2-level kernel
    k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    k1 = MaternKernel(2.5, [0.5], (0.1, 10.0), 1, bkd)
    scaling = PolynomialScaling([0.9], (0.5, 1.5), bkd, nvars=1)

    ml_kernel = MultiLevelKernel([k0, k1], [scaling])

    print(f"Multi-level kernel with {ml_kernel.noutputs()} levels")

    # Sample points for each level (higher resolution)
    x_vals = np.linspace(-1.0, 1.0, 51)
    x = bkd.array([x_vals])
    X_list = [x, x]  # Same points for both levels

    # Compute full multi-level kernel matrix
    K_full = ml_kernel(X_list)  # Shape: (102, 102) - 51 points per level
    K_full_np = bkd.to_numpy(K_full)

    # Create figure with full matrix and inverse
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot full kernel matrix
    n0 = 51  # Number of points at level 0
    n1 = 51  # Number of points at level 1

    im1 = axes[0].imshow(K_full_np, cmap="viridis", aspect="auto")
    axes[0].set_title("Multi-Level Kernel Matrix K", fontsize=12)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Sample index")

    # Add block boundaries
    axes[0].axhline(y=n0-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axvline(x=n0-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add block labels
    axes[0].text(n0/2, -0.8, 'Level 0', ha='center', fontsize=10, color='red')
    axes[0].text(n0 + n1/2, -0.8, 'Level 1', ha='center', fontsize=10, color='red')
    axes[0].text(-1.2, n0/2, 'Level 0', va='center', rotation=90, fontsize=10, color='red')
    axes[0].text(-1.2, n0 + n1/2, 'Level 1', va='center', rotation=90, fontsize=10, color='red')

    plt.colorbar(im1, ax=axes[0], label="Covariance")

    # Compute and plot inverse
    K_inv = np.linalg.inv(K_full_np + 1e-6 * np.eye(K_full_np.shape[0]))  # Add jitter for stability

    im2 = axes[1].imshow(K_inv, cmap="viridis", aspect="auto")
    axes[1].set_title("Inverse Multi-Level Kernel Matrix K⁻¹", fontsize=12)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Sample index")

    # Add block boundaries
    axes[1].axhline(y=n0-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(x=n0-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add block labels
    axes[1].text(n0/2, -0.8, 'Level 0', ha='center', fontsize=10, color='red')
    axes[1].text(n0 + n1/2, -0.8, 'Level 1', ha='center', fontsize=10, color='red')
    axes[1].text(-1.2, n0/2, 'Level 0', va='center', rotation=90, fontsize=10, color='red')
    axes[1].text(-1.2, n0 + n1/2, 'Level 1', va='center', rotation=90, fontsize=10, color='red')

    plt.colorbar(im2, ax=axes[1], label="Precision")

    fig.suptitle(
        "Multi-Level Kernel: Covariance and Precision Matrices\n"
        "Red lines show block boundaries between levels",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig("kernel_matrix_multilevel.png", dpi=150)
    print("Plot saved to kernel_matrix_multilevel.png")
    print(f"Kernel matrix condition number: {np.linalg.cond(K_full_np):.2e}")
    print()

    return fig


def example_5_dag_kernel_nonhierarchical():
    """Example 5: Visualize DAG kernel with non-hierarchical structure."""
    print("=" * 70)
    print("Example 5: DAG Kernel (Non-Hierarchical) Matrix")
    print("=" * 70)

    bkd = NumpyBkd()

    # Create non-hierarchical DAG (diamond structure)
    # Output 2 depends on both outputs 0 and 1
    dag = nx.DiGraph()
    dag.add_edges_from([
        (0, 2),  # 0 -> 2
        (1, 2),  # 1 -> 2
    ])

    print(f"DAG structure: {dag.number_of_nodes()} outputs")
    print(f"Edges: {list(dag.edges())}")
    print("This is a non-hierarchical (diamond) DAG:")
    print("  Output 0 and Output 1 are independent")
    print("  Output 2 depends on both Output 0 and Output 1")

    # Create base kernels for each output
    k0 = MaternKernel(2.5, [1.0], (0.1, 10.0), 1, bkd)
    k1 = MaternKernel(2.5, [0.8], (0.1, 10.0), 1, bkd)
    k2 = MaternKernel(2.5, [0.6], (0.1, 10.0), 1, bkd)

    base_kernels = [k0, k1, k2]

    # Create scalings for edges
    scaling_02 = PolynomialScaling([0.9], (0.5, 1.5), bkd, nvars=1)  # 0 -> 2
    scaling_12 = PolynomialScaling([0.85], (0.5, 1.5), bkd, nvars=1)  # 1 -> 2

    scalings = {(0, 2): scaling_02, (1, 2): scaling_12}

    # Create DAG kernel
    dag_kernel = DAGMultiOutputKernel(dag, base_kernels, scalings)

    # Sample points for each output
    x = bkd.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])  # 5 points
    X_list = [x, x, x]  # Same points for all outputs

    # Compute full DAG kernel matrix
    K_full = dag_kernel(X_list)  # Shape: (15, 15)
    K_full_np = bkd.to_numpy(K_full)

    # Create figure with full matrix and inverse
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    n = 5  # Number of points per output

    # Plot full kernel matrix
    im1 = axes[0].imshow(K_full_np, cmap="viridis", aspect="auto")
    axes[0].set_title("DAG Kernel Matrix K (Diamond Structure)", fontsize=12)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Sample index")

    # Add block boundaries
    axes[0].axhline(y=n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axhline(y=2*n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axvline(x=n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axvline(x=2*n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add block labels
    axes[0].text(n/2, -1.0, 'Out 0', ha='center', fontsize=10, color='red')
    axes[0].text(n + n/2, -1.0, 'Out 1', ha='center', fontsize=10, color='red')
    axes[0].text(2*n + n/2, -1.0, 'Out 2', ha='center', fontsize=10, color='red')
    axes[0].text(-1.5, n/2, 'Out 0', va='center', rotation=90, fontsize=10, color='red')
    axes[0].text(-1.5, n + n/2, 'Out 1', va='center', rotation=90, fontsize=10, color='red')
    axes[0].text(-1.5, 2*n + n/2, 'Out 2', va='center', rotation=90, fontsize=10, color='red')

    plt.colorbar(im1, ax=axes[0], label="Covariance")

    # Compute and plot inverse
    K_inv = np.linalg.inv(K_full_np + 1e-6 * np.eye(K_full_np.shape[0]))  # Add jitter for stability

    im2 = axes[1].imshow(K_inv, cmap="viridis", aspect="auto")
    axes[1].set_title("Inverse DAG Kernel Matrix K⁻¹", fontsize=12)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Sample index")

    # Add block boundaries
    axes[1].axhline(y=n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axhline(y=2*n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(x=n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(x=2*n-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add block labels
    axes[1].text(n/2, -1.0, 'Out 0', ha='center', fontsize=10, color='red')
    axes[1].text(n + n/2, -1.0, 'Out 1', ha='center', fontsize=10, color='red')
    axes[1].text(2*n + n/2, -1.0, 'Out 2', ha='center', fontsize=10, color='red')
    axes[1].text(-1.5, n/2, 'Out 0', va='center', rotation=90, fontsize=10, color='red')
    axes[1].text(-1.5, n + n/2, 'Out 1', va='center', rotation=90, fontsize=10, color='red')
    axes[1].text(-1.5, 2*n + n/2, 'Out 2', va='center', rotation=90, fontsize=10, color='red')

    plt.colorbar(im2, ax=axes[1], label="Precision")

    fig.suptitle(
        "DAG Kernel (Diamond Structure): Covariance and Precision Matrices\n"
        "Output 2 depends on both Output 0 and Output 1\n"
        "Red lines show block boundaries between outputs",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig("kernel_matrix_dag_nonhierarchical.png", dpi=150)
    print("Plot saved to kernel_matrix_dag_nonhierarchical.png")
    print(f"Kernel matrix condition number: {np.linalg.cond(K_full_np):.2e}")
    print()

    return fig


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Kernel Matrix Visualization Examples")
    print("=" * 70)

    # Example 1: Basic Matern kernel matrix
    example_1_matern_kernel_matrix()

    # Example 2: Compare length scales
    example_2_compare_length_scales()

    # Example 3: Compare smoothness
    example_3_compare_smoothness()

    # Example 4: Multi-level kernel matrices
    example_4_multilevel_kernel_matrices()

    # Example 5: DAG kernel (non-hierarchical)
    example_5_dag_kernel_nonhierarchical()

    print("=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Kernel matrices K(X, X) visualize covariance structure")
    print("2. Heatmaps show correlation patterns between input points")
    print("3. 3D surfaces reveal the smoothness and decay properties")
    print("4. Length scale controls the width of the correlation")
    print("5. Smoothness (nu) controls the differentiability of the kernel")
    print("6. Multi-output kernels show block structure with output correlations")
    print("7. Inverse kernel matrices (precision) show conditional independence")
    print("8. Multi-level kernels: hierarchical correlation structure")
    print("9. DAG kernels: non-hierarchical dependencies (e.g., diamond structure)")
    print("10. Red lines highlight block boundaries between outputs/levels")
    print("=" * 70)


if __name__ == "__main__":
    main()
    plt.show()
