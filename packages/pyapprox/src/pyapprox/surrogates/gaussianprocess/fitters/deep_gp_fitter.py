"""Fitters for DeepGaussianProcess.

Provides two fitters:
- DGPMaximumLikelihoodFitter: joint optimization of all active params
- MFDGPSequentialFitter: layer-by-layer pre-fitting in topological order

Use DGPFitterChain([MFDGPSequentialFitter, DGPMaximumLikelihoodFitter]) for
the Cutajar 2019 two-stage MF-DGP initialization pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Generic, Hashable, List, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.surrogates.gaussianprocess.deep.layer import DGPLayer

import networkx as nx

from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.surrogates.gaussianprocess.deep.deep_gp import (
    DeepGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.deep.single_layer_loss import (
    SingleLayerELBOLoss,
    TorchSingleLayerELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
    DGPELBOLoss,
    TorchDGPELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.fitters.results import (
    GPOptimizedFitResult,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.hyperparameter import HyperParameterList


def _init_variational_mean(
    layer: "DGPLayer[Array]",
    h: Array,
    y: Array,
    bkd: Backend[Array],
) -> None:
    """Set whitened variational mean so the SVGP posterior approximates y.

    Solves for m_tilde such that the posterior mean at training inputs h
    is a least-squares fit to y. For the whitened parameterization:

        mean(h) = A^T @ m_tilde + mean_fn(h)

    where A = L_uu^{-1} @ K(Z, h), shape (M, N). We solve the
    minimum-norm least-squares problem A^T @ m_tilde = residual
    via lstsq, which handles both underdetermined (N < M) and
    overdetermined (N >= M) cases stably without forming normal
    equations.
    """
    L_uu = layer.compute_L_uu()
    Z = layer.inducing_points().get_samples()
    K_uf = layer.kernel()(Z, h)
    A = bkd.solve_triangular(L_uu, K_uf, lower=True)

    mean_fn_h = layer.mean_function()(h)
    residual = y[0, :] - mean_fn_h[0, :]

    # A^T @ m_tilde = residual, solve via lstsq on A^T
    # A^T is (N, M), residual is (N,) -> m_tilde is (M,)
    m_tilde = bkd.lstsq(A.T, bkd.reshape(residual, (-1, 1)))[:, 0]

    layer.variational_dist()._mean_param.set_values(m_tilde)


def _detach_hyp_list(hyp_list: HyperParameterList[Array], bkd: Backend[Array]) -> None:
    """Detach all hyperparameter values from the computation graph.

    After optimization, torch tensors may be non-leaf (have grad_fn)
    due to in-place index assignment in set_active_values. This breaks
    deepcopy. Re-set all values via bkd.array() which produces fresh
    leaf tensors.
    """
    for hyp in hyp_list.hyperparameters():
        hyp.set_values(bkd.array(hyp.get_values()))


class DGPMaximumLikelihoodFitter(Generic[Array]):
    """Deep GP fitter with maximum likelihood hyperparameter optimization.

    Optimizes active hyperparameters by minimizing the negative ELBO,
    then marks the DGP as fitted. Uses AdamOptimizer by default since
    DGP optimization requires gradient-based methods.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for hyperparameter tuning. If None, uses
        AdamOptimizer(lr=1e-2, maxiter=5000).
    n_propagation : int
        Number of propagation samples for the ELBO.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        n_propagation: int = 10,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer
        self._n_propagation = n_propagation

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        dgp: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> GPOptimizedFitResult[Array, DeepGaussianProcess[Array]]:
        """Fit deep GP and optimize active hyperparameters.

        Parameters
        ----------
        dgp : DeepGaussianProcess[Array]
            The deep GP model.
        data : Dict[Hashable, Tuple[Array, Array]]
            Training data for each observed node: {node_id: (X, y)}.
            X has shape (nvars, N), y has shape (1, N).

        Returns
        -------
        GPOptimizedFitResult
            Result with fitted DGP and optimization metadata.
        """
        clone = dgp._clone_unfitted()

        initial_hyps = self._bkd.array(
            clone.hyp_list().get_active_values()
        )

        if clone.hyp_list().nactive_params() == 0:
            neg_elbo = DGPELBOLoss(clone, data, self._n_propagation)(
                clone.hyp_list().get_active_values()
            )
            clone.set_fitted()
            return GPOptimizedFitResult(
                surrogate=clone,
                neg_log_marginal_likelihood=neg_elbo,
                initial_hyperparameters=initial_hyps,
                optimized_hyperparameters=initial_hyps,
                optimization_result=None,
            )

        if isinstance(self._bkd, TorchBkd):
            loss = TorchDGPELBOLoss(clone, data, self._n_propagation)
        else:
            loss = DGPELBOLoss(clone, data, self._n_propagation)

        bounds = clone.hyp_list().get_active_bounds()

        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            from pyapprox.optimization.minimize.adam.adam_optimizer import (
                AdamOptimizer,
            )

            optimizer = AdamOptimizer(lr=1e-2, maxiter=5000, verbosity=0)

        optimizer.bind(loss, bounds)

        init_guess = clone.hyp_list().get_active_values()
        if len(init_guess.shape) == 1:
            init_guess = self._bkd.reshape(
                init_guess, (len(init_guess), 1)
            )

        opt_result = optimizer.minimize(init_guess)

        optimal_params = opt_result.optima()
        if len(optimal_params.shape) == 2:
            optimal_params = optimal_params[:, 0]
        clone.hyp_list().set_active_values(optimal_params)

        neg_elbo = loss(optimal_params)
        optimized_hyps = clone.hyp_list().get_active_values()

        _detach_hyp_list(clone.hyp_list(), self._bkd)
        clone.set_fitted()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=neg_elbo,
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=opt_result,
        )


def _evaluate_parent_mean(
    dag: nx.DiGraph,
    layers: Dict[Hashable, "DGPLayer[Array]"],
    node: Hashable,
    X: Array,
    bkd: Backend[Array],
) -> Array:
    """Posterior mean of a fitted node at input X.

    Recursively evaluates ancestors in topological order, building
    augmented inputs for non-root nodes. Topological ordering guarantees
    all ancestors are fitted before this node is evaluated.

    Parameters
    ----------
    dag : nx.DiGraph
        DGP connectivity graph.
    layers : Dict[Hashable, DGPLayer[Array]]
        All DGP layers.
    node : Hashable
        Node to evaluate.
    X : Array
        Original input, shape (d_x, N).
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    Array
        Posterior mean at X, shape (1, N).
    """
    layer = layers[node]
    parents = list(dag.predecessors(node))

    if not parents:
        mean, _ = layer.predict_marginal(X)
        return mean

    parent_means: List[Array] = []
    for p in parents:
        parent_means.append(
            _evaluate_parent_mean(dag, layers, p, X, bkd)
        )

    h = layer.input_builder().build(X, parent_means, bkd)
    mean, _ = layer.predict_marginal(h)
    return mean


class MFDGPSequentialFitter(Generic[Array]):
    """Sequential per-layer SVGP fit for a multi-fidelity DGP.

    Each layer in topological order is trained on its own data, with
    the parent's posterior mean (not variance) used to construct the
    augmented input. The resulting fit is a valid MF-DGP, suitable as
    a final model or as a warm-start for joint refinement via
    DGPFitterChain([MFDGPSequentialFitter, DGPMaximumLikelihoodFitter]).

    This is not NARGP (Perdikaris 2017), which uses standalone exact
    GPs at each level rather than SVGP layers of a DGP.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    optimizer : Optional[BindableOptimizerProtocol[Array]]
        Optimizer for per-layer hyperparameter tuning. A fresh copy is
        created for each layer. If None, uses AdamOptimizer(lr=1e-2,
        maxiter=5000).
    init_variational_mean : bool
        If True, initialize each layer's whitened variational mean
        from a least-squares fit to the training targets before
        optimization. Gives the optimizer a warm start instead of
        starting from the prior (zero mean).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        optimizer: Optional[BindableOptimizerProtocol[Array]] = None,
        init_variational_mean: bool = False,
    ) -> None:
        self._bkd = bkd
        self._optimizer = optimizer
        self._init_variational_mean = init_variational_mean

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def fit(
        self,
        dgp: DeepGaussianProcess[Array],
        data: Dict[Hashable, Tuple[Array, Array]],
    ) -> GPOptimizedFitResult[Array, DeepGaussianProcess[Array]]:
        """Fit DGP by optimizing each layer independently in topological order.

        For each node with training data, constructs the layer's input
        using fitted parent posterior means, then optimizes that layer's
        single-layer ELBO. Layers without data are skipped (left at prior).

        Parameters
        ----------
        dgp : DeepGaussianProcess[Array]
            The deep GP model.
        data : Dict[Hashable, Tuple[Array, Array]]
            Training data: {node_id: (X, y)}, X shape (d_x, N),
            y shape (1, N).

        Returns
        -------
        GPOptimizedFitResult
            Result with fitted DGP. neg_log_marginal_likelihood is the
            sum of per-node single-layer ELBOs (what was optimized).
        """
        bkd = self._bkd
        clone = dgp._clone_unfitted()

        initial_hyps = bkd.array(clone.hyp_list().get_active_values())

        dag = clone.dag()
        layers = clone.layers()

        saved_active: Dict[Hashable, Array] = {}
        for node_id, layer in layers.items():
            saved_active[node_id] = layer.hyp_list().get_active_indices()
            layer.hyp_list().set_all_inactive()

        total_neg_elbo = bkd.zeros((1,))[0]

        for node in nx.topological_sort(dag):
            if node not in data:
                continue

            layer = layers[node]
            X_node, y_node = data[node]
            parents = list(dag.predecessors(node))

            if not parents:
                h = X_node
            else:
                parent_means: List[Array] = []
                for p in parents:
                    parent_means.append(
                        _evaluate_parent_mean(dag, layers, p, X_node, bkd)
                    )
                h = layer.input_builder().build(X_node, parent_means, bkd)

            if self._init_variational_mean:
                _init_variational_mean(layer, h, y_node, bkd)

            layer.hyp_list().set_active_indices(saved_active[node])

            if layer.hyp_list().nactive_params() == 0:
                continue

            if isinstance(bkd, TorchBkd):
                loss = TorchSingleLayerELBOLoss(layer, h, y_node)
            else:
                loss = SingleLayerELBOLoss(layer, h, y_node)

            if self._optimizer is not None:
                optimizer = self._optimizer.copy()
            else:
                from pyapprox.optimization.minimize.adam.adam_optimizer import (
                    AdamOptimizer,
                )
                optimizer = AdamOptimizer(
                    lr=1e-2, maxiter=5000, verbosity=0,
                )

            bounds = layer.hyp_list().get_active_bounds()
            optimizer.bind(loss, bounds)

            init_guess = layer.hyp_list().get_active_values()
            if len(init_guess.shape) == 1:
                init_guess = bkd.reshape(
                    init_guess, (len(init_guess), 1),
                )

            opt_result = optimizer.minimize(init_guess)

            optimal_params = opt_result.optima()
            if len(optimal_params.shape) == 2:
                optimal_params = optimal_params[:, 0]
            layer.hyp_list().set_active_values(optimal_params)

            node_neg_elbo = loss(optimal_params)
            total_neg_elbo = total_neg_elbo + node_neg_elbo[0, 0]

            layer.hyp_list().set_all_inactive()

        for node_id, layer in layers.items():
            layer.hyp_list().set_active_indices(saved_active[node_id])

        _detach_hyp_list(clone.hyp_list(), bkd)
        optimized_hyps = clone.hyp_list().get_active_values()
        clone.set_fitted()

        return GPOptimizedFitResult(
            surrogate=clone,
            neg_log_marginal_likelihood=bkd.reshape(total_neg_elbo, (1, 1)),
            initial_hyperparameters=initial_hyps,
            optimized_hyperparameters=optimized_hyps,
            optimization_result=None,
        )
