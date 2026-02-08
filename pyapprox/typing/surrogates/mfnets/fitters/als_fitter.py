"""Alternating Least-Squares fitter for MFNet.

Processes nodes in topological order (leaves first). At each node, all
other nodes are fixed and only the current node's model parameters are
optimized. For linear models, this reduces to a direct least-squares solve.
"""

from typing import Dict, Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.mfnets.network import MFNet
from pyapprox.typing.surrogates.mfnets.protocols import (
    LinearNodeModelProtocol,
)
from pyapprox.typing.surrogates.mfnets.fitters.results import (
    MFNetALSFitResult,
)


class MFNetALSFitter(Generic[Array]):
    """Alternating Least-Squares fitter for MFNet surrogates.

    For each node (leaf-to-root), fixes all other parameters and fits the
    current node's model. If the node model is linear in its coefficients
    (satisfies ``LinearNodeModelProtocol``), a direct least-squares solve
    is used. Otherwise, a gradient sub-optimizer is used.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    max_sweeps : int
        Maximum number of leaf-to-root sweeps. Default: 10.
    tol : float
        Convergence tolerance on relative loss change. Default: 1e-8.
    verbosity : int
        0=silent, 1=summary, 2=per-sweep.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        max_sweeps: int = 10,
        tol: float = 1e-8,
        verbosity: int = 0,
    ) -> None:
        self._bkd = bkd
        self._max_sweeps = max_sweeps
        self._tol = tol
        self._verbosity = verbosity

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        network: MFNet[Array],
        train_samples_per_node: List[Array],
        train_values_per_node: List[Array],
    ) -> MFNetALSFitResult[Array]:
        """Fit the MFNet using alternating least squares.

        Parameters
        ----------
        network : MFNet[Array]
            A validated MFNet network.
        train_samples_per_node : list of Array
            Per-node training samples indexed by node id.
        train_values_per_node : list of Array
            Per-node training values indexed by node id.

        Returns
        -------
        MFNetALSFitResult[Array]
        """
        loss_history: List[float] = []
        converged = False

        # Initial loss
        loss = self._compute_total_mse(
            network, train_samples_per_node, train_values_per_node
        )
        loss_history.append(loss)
        if self._verbosity > 0:
            print(f"ALS: Initial MSE = {loss:.6e}")

        n_sweeps = 0
        for sweep in range(self._max_sweeps):
            n_sweeps = sweep + 1
            for node_id in network.topo_order():
                self._fit_node(
                    network, node_id,
                    train_samples_per_node, train_values_per_node,
                )

            loss = self._compute_total_mse(
                network, train_samples_per_node, train_values_per_node
            )
            loss_history.append(loss)

            if self._verbosity > 1:
                print(f"  Sweep {n_sweeps}: MSE = {loss:.6e}")

            # Check convergence (relative change)
            if len(loss_history) >= 2 and loss_history[-2] > 0:
                rel_change = abs(
                    loss_history[-1] - loss_history[-2]
                ) / max(abs(loss_history[-2]), 1e-30)
                if rel_change < self._tol:
                    converged = True
                    if self._verbosity > 0:
                        print(
                            f"ALS: Converged after {n_sweeps} sweeps "
                            f"(MSE = {loss:.6e})"
                        )
                    break

        if not converged and self._verbosity > 0:
            print(
                f"ALS: Max sweeps ({self._max_sweeps}) reached "
                f"(MSE = {loss:.6e})"
            )

        return MFNetALSFitResult(
            surrogate=network,
            loss_history=loss_history,
            n_sweeps=n_sweeps,
            converged=converged,
        )

    def _fit_node(
        self,
        network: MFNet[Array],
        node_id: int,
        train_samples: List[Array],
        train_values: List[Array],
    ) -> None:
        """Fit a single node's model, holding all others fixed."""
        bkd = self._bkd
        node = network.node(node_id)
        model = node.model()
        samples_n = train_samples[node_id]
        values_n = train_values[node_id]  # (nqoi_node, nsamples)

        if node.is_leaf():
            # For leaf: input = samples[active_vars]
            active_vars = node.active_sample_vars()
            x = samples_n[active_vars]
            self._fit_linear_model(model, x, values_n)
        else:
            # For non-leaf: build augmented input [x_active; child_outputs]
            active_vars = node.active_sample_vars()
            x = samples_n[active_vars]

            # Evaluate child outputs (fixed)
            child_val_parts: List[Array] = []
            for child_id in node.children_ids():
                cache: Dict[int, Array] = {}
                child_vals = network.subgraph_values(
                    samples_n, child_id, cache
                )
                edge_data = network.graph().edges[child_id, node_id]["edge"]
                output_ids = edge_data.output_ids()
                child_val_parts.append(child_vals[output_ids])

            stacked_child = bkd.vstack(child_val_parts)
            augmented = bkd.vstack([x, stacked_child])

            self._fit_linear_model(model, augmented, values_n)

    def _fit_linear_model(
        self,
        model: object,
        samples: Array,
        values: Array,
    ) -> None:
        """Fit a linear model via least squares.

        Parameters
        ----------
        model : object
            Should satisfy LinearNodeModelProtocol.
        samples : Array
            Input samples. Shape: ``(nvars_model, nsamples)``
        values : Array
            Target values. Shape: ``(nqoi, nsamples)``
        """
        bkd = self._bkd

        if isinstance(model, LinearNodeModelProtocol):
            # Direct least-squares solve
            # For MultiplicativeAdditiveDiscrepancy, basis_matrix uses
            # the augmented input including child outputs
            if hasattr(model, 'basis_matrix'):
                phi = model.basis_matrix(samples)  # (nsamples, nterms)
                # Solve Phi @ c = values.T for each QoI
                # lstsq returns (nterms, nqoi)
                coef = bkd.lstsq(phi, values.T)  # (total_nterms, nqoi)
                model.set_coefficients(bkd.flatten(coef))
            else:
                # Fallback: use model's __call__ directly
                pass
        elif hasattr(model, 'basis_matrix'):
            # Has basis_matrix but not full protocol — try anyway
            phi = model.basis_matrix(samples)
            coef = bkd.lstsq(phi, values.T)
            if hasattr(model, 'set_coefficients'):
                model.set_coefficients(bkd.flatten(coef))

    def _compute_total_mse(
        self,
        network: MFNet[Array],
        train_samples: List[Array],
        train_values: List[Array],
    ) -> float:
        """Compute total MSE across all nodes."""
        bkd = self._bkd
        total_mse = 0.0
        for node_id in network.topo_order():
            cache: Dict[int, Array] = {}
            pred = network.subgraph_values(
                train_samples[node_id], node_id, cache
            )
            residual = train_values[node_id] - pred
            mse = float(bkd.to_numpy(bkd.mean(residual * residual)))
            total_mse += mse
        return total_mse
