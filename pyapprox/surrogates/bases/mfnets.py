import math
from typing import List, Tuple

import numpy as np
import networkx as nx

from pyapprox.surrogates.regressor import OptimizedRegressor
from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.interface.model import Model
from pyapprox.util.linalg import diag_of_mat_mat_product
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    LogHyperParameterTransform,
)
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.minimize import MultiStartOptimizer


class MultiplicativeAndAdditiveDiscrepancyModel(OptimizedRegressor):
    # scaling_models and delta_model can be any type of model
    def __init__(
        self, scaling_models: List[Model], delta_model: Model, nscaled_qoi: int
    ):
        super().__init__(delta_model._bkd)
        if len(scaling_models) != delta_model.nqoi():
            raise ValueError(
                "Must provide a scaling for each qoi. "
                f"{len(scaling_models)=}, {delta_model.nqoi()=}"
            )
        for scaling_model in scaling_models:
            if scaling_model.nqoi() % nscaled_qoi:
                raise ValueError("scaling and delta are inconsistent")
        self._delta = delta_model
        self._scalings = scaling_models
        self._nscaled_qoi = nscaled_qoi
        self._hyp_list = scaling_model.hyp_list() + delta_model.hyp_list()

    def nqoi(self) -> int:
        return self._delta.nqoi()

    def nvars(self) -> int:
        return self._delta.nvars() + self._nscaled_qoi

    def _values(self, samples: Array) -> Array:
        # ith output is
        # scaling[ii][0]*unscaled_qoi[0] + scaling[ii][1]*unscaled_qoi[1] + ... + delta
        delta_samples = samples[: self._delta.nvars()]
        unscaled_qoi = samples[self._delta.nvars() :].T
        values = self._delta(delta_samples)
        for ii, scaling in enumerate(self._scalings):
            scale = scaling(delta_samples)
            values[:, ii] += self._bkd.sum(scale * unscaled_qoi, axis=1)
        return values

    # def param_jacobian_implemented(self) -> bool:
    #     return True

    # def _param_jacobian(self, active_opt_params: Array) -> Array:
    #     return self._bkd.jacobian(self.)
    #     delta_samples = samples[: self._delta.nvars()]
    #     unscaled_qoi = samples[self._delta.nvars() :].T
    #     delta_jac = self._delta.jacobian(delta_samples)
    #     scale_jacs = []
    #     for ii, scaling in enumerate(self._scalings):
    #         scale = scaling(delta_samples)
    #         scale_jac = scaling.jacobian(delta_samples)
    #         # jac with respect to scale parameters
    #         tmp1 = self._bkd.sum(unscaled_qoi * scale_jac, axis=1)
    #         scale_jacs.append(tmp1 + tmp2)
    #     jac = self._bkd.hstack((delta_jac, scale_jacs))
    #     return jac


# class MultiplicativeAndAdditiveDiscrepancyModel(Model):
#     def _setup_polynomial(self):
#         poly = MonomialExpansion(
#             MultiIndexBasis(
#                 [Monomial1D(backend=self._bkd) for ii in range(self.nvars())]
#             ),
#             None,
#             self.nqoi(),
#             (np.inf, np.inf),
#             fixed=True,
#         )
#         # assume random variables (z) are first entries in samples (x) and
#         # unscaled_qoi (q) are the next set of entries, i.e. x=(z,q)
#         # M(x) =  delta @ ones + q.T @ scalings
#         delta_indices = self._bkd.zeros((self.nvars(), self._delta.nvars()))
#         qoi_indices = self._bkd.zeros((self.nvars(), self._nscaled_qoi))
#         qoi_indices[self._delta.nvars() :, :] = self._bkd.eye(
#             self._nscaled_qoi
#         )
#         indices = self._bkd.hstack((delta_indices, qoi_indices))
#         poly.set_indices(indices)
#         coefs = self._bkd.zeros((indices.shape[1], self.nqoi()))
#         poly.set_coefficients(coefs)
#         return poly

#     def _values(self, samples: Array) -> Array:
#         nsamples = samples.shape[1]
#         delta_samples = samples[: self._delta.nvars()]
#         unscaled_qoi = samples[self._delta.nvars() :].T
#         coefs = self._bkd.zeros((indices.shape[1], self.nqoi()))
#         coefs[: self._delta.nvars(), :] = 1.0
#         for ii, scaling in enumerate(self.scalings):
#             coefs[self._delta.nvars() :, ii] = scaling(delta_samples)
#         self._poly.set_coefficients(coefs)
#         return poly(samples)


class MFNetModel(OptimizedRegressor):
    def __init__(
        self,
        nvars: int,
        graph: nx.DiGraph,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._nvars = nvars
        self._graph = graph

    def nvars(self) -> int:
        return self._nvars

    def _subgraph_values(self, samples: Array, node_id: int) -> Array:
        node = self._graph.nodes[node_id]["node"]
        if node._is_leaf():
            if node._values is None:
                return node.model()(samples)
            return node._values

        req_child_vals = []
        for child_id in node._children_ids:
            child_vals = self._subgraph_values(samples, child_id)
            req_child_vals.append(
                child_vals[
                    :, self._graph.edges[child_id, node_id]["edge"]._output_ids
                ]
            )
        req_child_vals = self._bkd.hstack(req_child_vals)

        inputs = self._bkd.vstack(
            (samples[node._active_sample_vars], req_child_vals.T)
        )
        values = node.model()(inputs)
        return values

    def _neg_log_like_from_model_values(
        self, node_id: int, model_values: Array
    ) -> float:
        node = self._graph.nodes[node_id]["node"]
        nsamples = model_values.shape[0]
        sigma = node._noise_std.get_values()[0]  # assume scalar
        const = 0.5 * nsamples * math.log(
            2.0 * math.pi
        ) + nsamples * self._bkd.log(sigma)
        # or last line can be nsamples * log(sigma**2) /2
        residual = node._ctrain_values - model_values
        residual /= sigma  # not sigma **2 because residual is applied twice
        return (
            0.5
            * diag_of_mat_mat_product(residual.T, residual, self._bkd)[:, None]
        ).sum() + const

    def _node_neg_log_like(self, node_id: int) -> float:
        node = self._graph.nodes[node_id]["node"]
        return self._neg_log_like_from_model_values(
            node_id, self._subgraph_values(node._ctrain_samples, node_id)
        )

    def _reset_memory(self):
        for node_id in self._graph.nodes:
            node = self._graph.nodes[node_id]["node"]
            node._reset_memory()

    def _neg_log_like(self, active_opt_params: Array) -> float:
        self._hyp_list.set_active_opt_params(active_opt_params)
        self._reset_memory()
        loglike_vals_per_node = [
            self._node_neg_log_like(node_id) for node_id in self._graph.nodes
        ]
        return sum(loglike_vals_per_node)

    def _set_training_data(self, train_samples: list, train_values: list):
        self._ctrain_samples = [
            s for s in self._in_trans.map_to_canonical(train_samples)
        ]
        self._ctrain_values = [
            self._out_trans.map_to_canonical(v.T).T for v in train_values
        ]
        for node_id in self._graph.nodes:
            self._graph.nodes[node_id]["node"]._set_training_data(
                self._ctrain_samples[node_id], self._ctrain_values[node_id]
            )

    def add_node(self, node: "MFNetNode"):
        if not isinstance(node, MFNetNode):
            raise ValueError("node must be an instance of MFNetNode")
        self._graph.add_node(node._node_id, node=node)

    def add_edge(self, edge: "MFNetEdge"):
        if not isinstance(edge, MFNetEdge):
            raise ValueError("node must be an instance of MFNetNode")
        self._graph.add_edge(
            edge._child_node._node_id, edge._parent_node._node_id, edge=edge
        )

    def nodes(self) -> List["MFNetNode"]:
        return [graph_node["node"] for graph_node in self._graph.nodes]

    def nqoi(self) -> int:
        return sum(node.model().nqoi() for node in self._root_nodes)

    def _values(self, samples: Array) -> Array:
        if not hasattr(self, "_nodes"):
            raise RuntimeError("must call validate()")
        self._reset_memory()
        vals = [
            self._subgraph_values(samples, node._node_id)
            for node in self._root_nodes
        ]
        return self._bkd.hstack(vals)

    def validate(self):
        for node_id in self._graph.nodes:
            self._graph.nodes[node_id]["node"]._validate(self)

        self._nodes = [
            self._graph.nodes[node_id]["node"] for node_id in self._graph.nodes
        ]  # all nodes
        self._root_nodes = [node for node in self._nodes if node._is_root()]
        self._leaf_nodes = [node for node in self._nodes if node._is_leaf()]
        self._hyp_list = sum(
            [
                node.model().hyp_list() + HyperParameterList([node._noise_std])
                for node in self._nodes
            ]
        )

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        super().set_optimizer(optimizer)
        self.set_loss(MFNetNegLogLikelihoodLoss())

    def _fit(self, iterate: Array):
        if self._optimizer is None:
            self.set_optimizer(self.default_optimizer(verbosity=10, gtol=1e-8))
        return super()._fit(iterate)


class MFNetNegLogLikelihoodLoss(LossFunction):
    def _loss_values(self, active_opt_params):
        vals = self._bkd.atleast2d(
            self._model._neg_log_like(active_opt_params[:, 0])
        )
        return vals

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _check_model(self, model: MFNetModel):
        if not isinstance(MFNetModel):
            raise ValueError(
                "model must be an instance of ExactGaussianProcess"
            )
        super()._check_model(model)

    def _jacobian(self, active_opt_params: Array) -> Array:
        return super()._jacobian(active_opt_params)


class MFNetNode:
    def __init__(
        self,
        node_id: int,
        model: Model,
        noise_std: float,
        noise_std_bounds: Tuple[float, float] = (1e-8, np.inf),
        fixed_noise_std: bool = True,
        active_sample_vars: Array = None,
    ):
        """
        Terminology
        -----------
        For graph 1 -> 2 -> 3
        root of graph is 3 (most upstream node/highest-fidelity)
        leaf of graph is 3 (most downstream node/lowest-fidelity)
        predecessor (ancestor/child) of 3 is 2
        sucessor (decendent/parent) of 2 is 3
        """
        self._bkd = model._bkd
        if node_id < 0:
            raise ValueError("node ids must be non-negative")
        self._node_id = node_id
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        self._model = model
        self._noise_std = HyperParameter(
            "noise_std",
            1,
            noise_std,
            noise_std_bounds,
            LogHyperParameterTransform(backend=self._bkd),
            fixed=fixed_noise_std,
            backend=self._bkd,
        )
        # define subset of global sample variables that enter into the model
        self._active_sample_vars = active_sample_vars
        self._reset_memory()

    def _set_training_data(self, ctrain_samples: Array, ctrain_values: Array):
        if ctrain_samples.shape[1] != ctrain_values.shape[0]:
            raise ValueError(
                (
                    "Number of cols of samples {0} does not match "
                    + "number of rows of values {1}"
                ).format(ctrain_samples.shape[1], ctrain_values.shape[0])
            )
        self._ctrain_samples = ctrain_samples
        self._ctrain_values = ctrain_values

    def _is_leaf(self) -> bool:
        return self._children_ids.shape[0] == 0

    def _is_root(self) -> bool:
        return self._parent_ids.shape[0] == 0

    def model(self) -> Model:
        return self._model

    def _set_model_values(self, values: Array):
        # used to avoid recomputation
        self._values = values

    def _set_param_jacobian(self, jac: Array):
        self._jac = jac

    def _reset_memory(self):
        self._values = None
        self._jac = None

    def _check_is_mfnet_model(self, mfnet: MFNetModel):
        if not isinstance(mfnet, MFNetModel):
            raise ValueError(
                "mfnet must be an instance of MFNetModel but was type"
                f"{type(mfnet)}"
            )
        self._children_ids = self._bkd.asarray(
            list(mfnet._graph.predecessors(self._node_id)), dtype=int
        )
        self._parent_ids = self._bkd.asarray(
            list(mfnet._graph.successors(self._node_id)),
            dtype=int,
        )
        print(self._node_id, self._children_ids, self._parent_ids)
        if self._active_sample_vars is None:
            self._active_sample_vars = self._bkd.arange(mfnet.nvars())
        if len(self._active_sample_vars) != mfnet.nvars():
            raise ValueError("len(active_sample_vars) != mfnet.nvars()")
        self._active_sample_vars = self._bkd.asarray(
            self._active_sample_vars, dtype=int
        )

    def _validate(self, mfnet: MFNetModel):
        self._check_is_mfnet_model(mfnet)
        if self._is_root() or self._is_leaf():
            raise ValueError("MFNetNode must have children and parents")


class RootMFNetNode(MFNetNode):
    def _validate(self, mfnet: MFNetModel):
        self._check_is_mfnet_model(mfnet)
        if not self._is_root():
            raise ValueError("RootMFNetNode cannot have parents")


class LeafMFNetNode(MFNetNode):
    def _validate(self, mfnet: MFNetModel):
        self._check_is_mfnet_model(mfnet)
        if not self._is_leaf:
            raise ValueError("LeafMFNode cannot have children")


class MFNetEdge:
    def __init__(
        self,
        child_node: MFNetNode,
        parent_node: MFNetNode,
        child_output_ids: Array = None,
    ):
        self._check_node(child_node)
        self._check_node(parent_node)
        self._bkd = parent_node._bkd
        self._child_node = child_node
        self._parent_node = parent_node
        if child_output_ids is None:
            child_output_ids = self._bkd.arange(
                self._child_node.model().nqoi()
            )
        if self._bkd.max(child_output_ids) >= child_node.model().nqoi():
            raise ValueError(
                "child_output_ids are inconsistent with "
                "child_node.model().nqoi()"
            )
        self._output_ids = child_output_ids

    def _check_node(self, node: MFNetNode):
        if not isinstance(node, MFNetNode):
            raise ValueError("node must be an instance of MFNetNode")
