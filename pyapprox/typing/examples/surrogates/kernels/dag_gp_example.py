"""
Example: DAG-based Autoregressive Gaussian Process

This example demonstrates how to use the DAGMultiOutputKernel class to model
multi-fidelity or multi-output data with arbitrary autoregressive structure
specified by a directed acyclic graph (DAG).

We show three examples:
1. Sequential structure: 0 -> 1 -> 2 (standard multi-level)
2. Tree structure: 0 -> 1, 0 -> 2 (single root, multiple branches)
3. Diamond structure: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (multiple paths)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pyapprox.typing.surrogates.kernels import MaternKernel, PolynomialScaling
from pyapprox.typing.surrogates.kernels.multioutput import DAGMultiOutputKernel
from pyapprox.typing.surrogates.gaussianprocess.multioutput import MultiOutputGP
from pyapprox.typing.util.backends.numpy import NumpyBkd


def example_1_sequential():
    """Example 1: Sequential 3-level structure: 0 -> 1 -> 2."""
    print("="*70)
    print("Example 1: Sequential Structure (0 -> 1 -> 2)")
    print("="*70)

    bkd = NumpyBkd()
    nvars = 1

    # Define DAG structure using NetworkX
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2])
    dag.add_edges_from([(0, 1), (1, 2)])
    print(f"\nDAG edges: {list(dag.edges())}")
    print(f"Roots: {[n for n in dag.nodes() if dag.in_degree(n) == 0]}")
    print(f"Leaves: {[n for n in dag.nodes() if dag.out_degree(n) == 0]}")

    # Create discrepancy kernels for each level
    kernels = [
        MaternKernel(2.5, [1.0], (0.1, 10.0), nvars, bkd),  # Level 0
        MaternKernel(2.5, [0.8], (0.1, 10.0), nvars, bkd),  # Level 1
        MaternKernel(2.5, [0.5], (0.1, 10.0), nvars, bkd),  # Level 2
    ]

    # Create spatially varying scalings for edges (linear: c0 + c1*x)
    edge_scalings = {
        (0, 1): PolynomialScaling([0.9, 0.1], (0.5, 1.5), bkd),   # 0 -> 1
        (1, 2): PolynomialScaling([0.85, 0.05], (0.5, 1.5), bkd), # 1 -> 2
    }

    # Create multi-output kernel
    mo_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)
    print(f"\nTotal hyperparameters: {mo_kernel.hyp_list().nactive_params()}")

    # Generate training data
    np.random.seed(42)
    n_samples = [20, 10, 5]  # More samples at lower levels

    # True functions (for demonstration)
    def f0(x): return np.sin(2 * np.pi * x)
    def f1(x): return f0(x) + 0.2 * np.cos(8 * np.pi * x)
    def f2(x): return f1(x) + 0.1 * np.sin(16 * np.pi * x)

    X_train_list = []
    y_train_list = []
    for level, n in enumerate(n_samples):
        X = bkd.array(np.linspace(-1, 1, n).reshape(1, -1))
        if level == 0:
            y = f0(X[0, :])
        elif level == 1:
            y = f1(X[0, :])
        else:
            y = f2(X[0, :])
        X_train_list.append(X)
        y_train_list.append(bkd.array((y + 0.01 * np.random.randn(n))[:, None]))

    # Stack training outputs
    y_train_stacked = bkd.vstack(y_train_list)

    # Fit GP
    gp = MultiOutputGP(mo_kernel, noise_variance=1e-6)
    gp.fit(X_train_list, y_train_stacked)
    print(f"\nGP fitted successfully!")

    # Predict on fine grid
    X_test = bkd.array(np.linspace(-1, 1, 100).reshape(1, -1))
    X_test_list = [X_test] * 3

    y_pred, y_std = gp.predict_with_uncertainty(X_test_list)

    # Split predictions by level
    n_test = X_test.shape[1]
    y_pred_levels = [
        y_pred[:n_test, 0],
        y_pred[n_test:2*n_test, 0],
        y_pred[2*n_test:, 0],
    ]
    y_std_levels = [
        y_std[:n_test, 0],
        y_std[n_test:2*n_test, 0],
        y_std[2*n_test:, 0],
    ]

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    x_plot = X_test[0, :]

    for level, ax in enumerate(axes):
        # True function
        if level == 0:
            y_true = f0(x_plot)
            title = "Level 0 (Root)"
        elif level == 1:
            y_true = f1(x_plot)
            title = "Level 1 (depends on 0)"
        else:
            y_true = f2(x_plot)
            title = "Level 2 (depends on 1)"

        ax.plot(x_plot, y_true, 'k--', label='True', alpha=0.5)
        ax.plot(x_plot, y_pred_levels[level], 'b-', label='GP prediction', linewidth=2)
        ax.fill_between(
            x_plot,
            y_pred_levels[level] - 2 * y_std_levels[level],
            y_pred_levels[level] + 2 * y_std_levels[level],
            alpha=0.2,
            label='95% confidence'
        )
        ax.scatter(
            X_train_list[level][0, :],
            y_train_list[level][:, 0],
            c='red', marker='o', s=50, label='Training data', zorder=5
        )
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dag_gp_sequential.png', dpi=150)
    print(f"Plot saved to dag_gp_sequential.png")
    return fig


def example_2_tree():
    """Example 2: Tree structure: 0 -> 1, 0 -> 2."""
    print("\n" + "="*70)
    print("Example 2: Tree Structure (0 -> 1, 0 -> 2)")
    print("="*70)

    bkd = NumpyBkd()
    nvars = 1

    # Define DAG structure: 0 is root, 1 and 2 are independent children
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2])
    dag.add_edges_from([(0, 1), (0, 2)])
    print(f"\nDAG edges: {list(dag.edges())}")
    print(f"Roots: {[n for n in dag.nodes() if dag.in_degree(n) == 0]}")
    print(f"Leaves: {[n for n in dag.nodes() if dag.out_degree(n) == 0]}")

    # Create discrepancy kernels
    kernels = [
        MaternKernel(2.5, [1.0], (0.1, 10.0), nvars, bkd),  # Root
        MaternKernel(2.5, [0.5], (0.1, 10.0), nvars, bkd),  # Branch 1
        MaternKernel(2.5, [0.5], (0.1, 10.0), nvars, bkd),  # Branch 2
    ]

    # Create constant scalings (degree 0 polynomial)
    edge_scalings = {
        (0, 1): PolynomialScaling([0.9], (0.5, 1.5), bkd, nvars=nvars),  # 0 -> 1
        (0, 2): PolynomialScaling([0.85], (0.5, 1.5), bkd, nvars=nvars), # 0 -> 2
    }

    # Create and fit model
    mo_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

    print(f"\nTree structure allows independent branches from common root")
    print(f"Both branches share information through the root level")

    return None


def example_3_diamond():
    """Example 3: Diamond structure: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""
    print("\n" + "="*70)
    print("Example 3: Diamond Structure (0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3)")
    print("="*70)

    bkd = NumpyBkd()
    nvars = 1

    # Define DAG using NetworkX
    dag = nx.DiGraph()
    dag.add_nodes_from([0, 1, 2, 3])
    dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    print(f"\nDAG edges: {list(dag.edges())}")
    print(f"Roots: {[n for n in dag.nodes() if dag.in_degree(n) == 0]}")
    print(f"Leaves: {[n for n in dag.nodes() if dag.out_degree(n) == 0]}")

    # Create discrepancy kernels
    kernels = [
        MaternKernel(2.5, [1.0], (0.1, 10.0), nvars, bkd),   # Node 0 (root)
        MaternKernel(2.5, [0.7], (0.1, 10.0), nvars, bkd),   # Node 1
        MaternKernel(2.5, [0.7], (0.1, 10.0), nvars, bkd),   # Node 2
        MaternKernel(2.5, [0.4], (0.1, 10.0), nvars, bkd),   # Node 3
    ]

    # Create constant scalings for all edges
    edge_scalings = {
        (0, 1): PolynomialScaling([0.9], (0.5, 1.5), bkd, nvars=nvars),   # 0 -> 1
        (0, 2): PolynomialScaling([0.85], (0.5, 1.5), bkd, nvars=nvars),  # 0 -> 2
        (1, 3): PolynomialScaling([0.8], (0.5, 1.5), bkd, nvars=nvars),   # 1 -> 3
        (2, 3): PolynomialScaling([0.75], (0.5, 1.5), bkd, nvars=nvars),  # 2 -> 3
    }

    mo_kernel = DAGMultiOutputKernel(dag, kernels, edge_scalings)

    print(f"\nDiamond structure has multiple paths from node 0 to node 3:")
    print(f"  Path 1: 0 -> 1 -> 3")
    print(f"  Path 2: 0 -> 2 -> 3")
    print(f"\nNode 3 benefits from information through both paths!")

    return None


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DAG-based Autoregressive Gaussian Process Examples")
    print("="*70)

    # Example 1: Full demonstration with plots
    example_1_sequential()

    # Example 2 and 3: Structure demonstrations
    example_2_tree()
    example_3_diamond()

    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("1. DAG structure allows flexible autoregressive dependencies")
    print("2. Sequential (0->1->2): Standard multi-level structure")
    print("3. Tree (0->{1,2}): Independent branches from common root")
    print("4. Diamond (0->{1,2}->3): Multiple paths, information fusion")
    print("5. Scaling functions control correlation strength between levels")
    print("6. Each node has its own discrepancy kernel")
    print("\nAll structures work seamlessly with MultiOutputGP!")
    print("="*70)


if __name__ == "__main__":
    main()
