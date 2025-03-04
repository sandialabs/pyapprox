from abc import ABC, abstractmethod

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
        error, cost = self._estimate(model)
        return error, cost


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
        if multi_index_bounds.shape != (model_ensemble.nrefinement_vars(), 2):
            raise ValueError("multi_index_bounds has the wrong shape ")
        if self._bkd.any(multi_index_bounds[:, 0] >= multi_index_bounds[:, 1]):
            raise ValueError("multi_index_bounds must be increasing")
        if self._bkd.any(
            multi_index_bounds[:, 1] > self._model_ensemble._index_bounds
        ):
            raise ValueError(
                "multi_index_bounds must exceed bounds of model enesmble"
            )
        self._multi_index_bounds = multi_index_bounds
        self._indices_1d = [
            self._bkd.arange(bounds[0], bounds[1] + 1)
            for bounds in multi_index_bounds
        ]
        self._multi_indices = self._bkd.cartesian_product(self._indices_1d)

    def run(self):
        self._errors = dict()
        self._costs = dict()
        for model_id in self._multi_indices.T:
            key = self._model_ensemble._hash_model_id(model_id)
            model = self._model_ensemble.get_model(model_id)
            self._errors[key], self._costs[key] = self._error_est(model)

    def _plot_1d(self, axs):
        label = r"$(\cdot)$"
        errors, costs = [], []
        for ii in self._indices_1d[0]:
            key = self._hash_model_id(self._bkd.array([ii], dtype=int))
            errors.append(self._errors[key])
            costs.append(self._costs[key])
        axs.loglog(costs, errors, "o-", label=label)
        axs.legend()

    def _hash_model_id(self, model_id: int):
        return self._model_ensemble._hash_model_id(model_id)

    def _plot_2d(self, axs):
        if len(axs) != 2:
            raise ValueError("Must provide two axes")
        for jj in self._indices_1d[1]:
            label = r"$(\cdot,%d)$" % (jj)
            errors, costs = [], []
            for ii in self._indices_1d[0]:
                key = self._hash_model_id(self._bkd.array([ii, jj], dtype=int))
                errors.append(self._errors[key])
                costs.append(self._costs[key])
            axs[0].loglog(costs, errors, "o-", label=label)
        axs[0].legend()
        for ii in self._indices_1d[0]:
            label = r"$(%d,\cdot)$" % (ii)
            errors, costs = [], []
            for jj in self._indices_1d[1]:
                key = self._hash_model_id(self._bkd.array([ii, jj], dtype=int))
                errors.append(self._errors[key])
                costs.append(self._costs[key])
            axs[1].loglog(costs, errors, "o-", label=label)
        axs[1].legend()

    def plot(self, axs):
        if self._model_ensemble.nrefinement_vars() == 1:
            return self._plot_1d(axs)
        if self._model_ensemble.nrefinement_vars() == 2:
            return self._plot_2d(axs)
