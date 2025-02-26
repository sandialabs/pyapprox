from abc import ABC, abstractmethod

import numpy as np

from pyapprox.interface.model import (
    Model,
    MultiIndexModelEnsemble,
    Array,
    LinAlgMixin,
    NumpyLinAlgMixin,
)


class ConvergenceErrorEstimator(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def _estimate(self, model: Model) -> float:
        raise NotImplementedError

    def __call__(self, model: Model):
        return self._estimate(model)


class ConvergenceStudy:
    def __init__(
        self,
        model_ensemble: MultiIndexModelEnsemble,
        error_est: ConvergenceErrorEstimator,
        multi_index_bounds: Array,
    ):
        self._bkd = model_ensemble._bkd
        self._model_ensemble = model_ensemble
        self._error_est = error_est
        print(multi_index_bounds.shape)
        if multi_index_bounds.shape != (model_ensemble.nrefinement_vars(), 2):
            raise ValueError("multi_index_bounds has the wrong shape ")
        if self._bkd.any(multi_index_bounds[:, 0] >= multi_index_bounds[:, 1]):
            raise ValueError("multi_index_bounds must be increasing")
        self._multi_index_bounds = multi_index_bounds
        self._indices_1d = [
            self._bkd.arange(bounds[0], bounds[1] + 1)
            for bounds in multi_index_bounds
        ]
        self._multi_indices = self._bkd.cartesian_product(self._indices_1d)

    def run(self):
        self._errors = dict()
        for model_id in self._multi_indices.T:
            key = self._model_ensemble._hash_model_id(model_id)
            model = self._model_ensemble.get_model(model_id)
            self._errors[key] = self._error_est(model)

    def _plot_1d(self, axs):
        label = r"$(\cdot)$"
        errors = [
                self._errors[self._hash_model_id(self._bkd.array([jj]))]
                for jj in range(self._indices_1d[0])
            ]
        return axs.loglog(costs, errors, "o-", label=label)

    def _plot_2d(self, axs):
        if len(axs) ! =2:
            raise ValueError("Must provide two axes")
        for ii in range(self._indices_1d[1]):
            label = r"$(\cdot,%d)$" % (ii)
            errors = [
                self._errors[self._hash_model_id(self._bkd.array([jj, ii]))]
                for jj in range(self._indices_1d[0])
            ]
            axs[0].loglog(costs, errors, "o-", label=label)
        # for ii in range(self._indices_1d[0]):
        #     label = r"$(%d,\cdot)$" % (ii)
        #     axs[1].loglog(costs[ii, :], errors[ii, :], "o-", label=label)

    def plot(self, axs):
        if self._model_ensemble.nrefinement_vars() == 1:
            return self._plot_1d(axs)
        if self._model_ensemble.nrefinement_vars() == 2:
            return self._plot_2d(axs)


def plot_convergence_data(data, cost_type="ndof"):

    errors, costs = data["errors"], data["costs"]
    config_idx = data["canonical_config_samples_1d"]

    if cost_type == "ndof":
        costs = data["ndofs"] / data["ndofs"].max()
    validation_levels = costs.shape
    nconfig_vars = len(validation_levels)
    fig, axs = plt.subplots(
        1, nconfig_vars, figsize=(nconfig_vars * 8, 6), sharey=False
    )
    if nconfig_vars == 1:
        label = r"$(\cdot)$"
        axs.loglog(costs, errors, "o-", label=label)
    if nconfig_vars == 2:
        for ii in range(validation_levels[1]):
            label = r"$(\cdot,%d)$" % (config_idx[1][ii])
            axs[0].loglog(costs[:, ii], errors[:, ii], "o-", label=label)
        for ii in range(validation_levels[0]):
            label = r"$(%d,\cdot)$" % (config_idx[0][ii])
            axs[1].loglog(costs[ii, :], errors[ii, :], "o-", label=label)
    if nconfig_vars == 3:
        for ii in range(validation_levels[1]):
            jj = costs.shape[2] - 1
            label = r"$(\cdot,%d,%d)$" % (config_idx[1][ii], config_idx[2][jj])
            axs[0].loglog(
                costs[:, ii, jj], errors[:, ii, jj], "o-", label=label
            )
        for ii in range(validation_levels[0]):
            jj = costs.shape[2] - 1
            label = r"$(%d,\cdot,%d)$" % (config_idx[0][ii], config_idx[2][jj])
            axs[1].loglog(
                costs[ii, :, jj], errors[ii, :, jj], "o-", label=label
            )
            jj = costs.shape[1] - 1
            label = r"$(%d,%d,\cdot)$" % (config_idx[0][ii], config_idx[1][jj])
            axs[2].loglog(
                costs[ii, jj, :], errors[ii, jj, :], "o-", label=label
            )

    for ii in range(nconfig_vars):
        axs[ii].legend()
        axs[ii].set_xlabel(mathrm_label("Work ") + r"$W_{\alpha}$")
        axs[0].set_ylabel(
            r"$\left| \mathbb{E}[f]-\mathbb{E}[f_{\alpha}]\right| / \left| \mathbb{E}[f]\right|$"
        )
    return fig, axs
