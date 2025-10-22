from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.interface.model import Model


class FiniteDifference(ABC):
    def __init__(
        self, model: Model, fd_eps: float = 2 * np.sqrt(np.finfo(float).eps)
    ):
        self._bkd = model._bkd
        self._model = model
        self.set_step_size(fd_eps)

    def set_step_size(self, fd_eps: float):
        self._fd_eps = fd_eps

    @abstractmethod
    def jacobian(self, sample: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def apply_jacobian(self, sample: Array, vecs: Array) -> Array:
        raise NotImplementedError

    def _check_jacobian_implemented(self):
        if not (
            self._model.jacobian_implemented()
            or self._model.apply_jacobian_implemented()
        ):
            raise ValueError("model.jacobian must be implemented")

    def apply_hessian(self, sample: Array, vecs: Array) -> Array:
        self._check_jacobian_implemented()
        return self._apply_hessian(sample, vecs)

    def hessian(self, sample: Array) -> Array:
        self._check_jacobian_implemented()
        return self._hessian(sample)

    @abstractmethod
    def _hessian(self, sample: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _apply_hessian(self, sample: Array, vecs: Array) -> Array:
        raise NotImplementedError

    def nvars(self) -> int:
        return self._model.nvars()


class ForwardFiniteDifference(FiniteDifference):
    def _perturbed_samples(self, sample: Array) -> Array:
        perturbed_samples = self._bkd.tile(sample, (self.nvars(),))
        for ii in range(self.nvars()):
            perturbed_samples[ii, ii] += self._fd_eps
        return perturbed_samples

    def _directionally_perturbed_samples(self, sample: Array, vecs) -> Array:
        return sample + self._fd_eps * vecs

    def jacobian(self, sample: Array) -> Array:
        value = self._model(sample)
        perturbed_values = self._model(self._perturbed_samples(sample))
        return (perturbed_values - value).T / self._fd_eps

    def apply_jacobian(self, sample: Array, vecs: Array) -> Array:
        value = self._model(sample)
        perturbed_samples = self._directionally_perturbed_samples(sample, vecs)
        perturbed_values = self._model(perturbed_samples)
        return (perturbed_values - value).T / self._fd_eps

    def _hessian(self, sample: Array) -> Array:
        value = self._model.jacobian(sample)
        perturbed_samples = self._perturbed_samples(sample)
        perturbed_values = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples.T],
            axis=-1,
        )
        return (perturbed_values - value[..., None]) / self._fd_eps

    def _apply_hessian(self, sample: Array, vecs: Array) -> Array:
        value = self._model.jacobian(sample)
        perturbed_samples = self._directionally_perturbed_samples(sample, vecs)
        perturbed_values = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples.T],
            axis=-1,
        )
        return (perturbed_values - value[..., None]) / self._fd_eps


class BackwardFiniteDifference(FiniteDifference):
    def _perturbed_samples(self, sample: Array) -> Array:
        perturbed_samples = self._bkd.tile(sample, (self.nvars(),))
        for ii in range(self.nvars()):
            perturbed_samples[ii, ii] -= self._fd_eps
        return perturbed_samples

    def _directionally_perturbed_samples(self, sample: Array, vecs) -> Array:
        return sample - self._fd_eps * vecs

    def jacobian(self, sample: Array) -> Array:
        value = self._model(sample)
        perturbed_values = self._model(self._perturbed_samples(sample))
        return (-perturbed_values + value).T / self._fd_eps

    def apply_jacobian(self, sample: Array, vecs: Array) -> Array:
        value = self._model(sample)
        perturbed_samples = self._directionally_perturbed_samples(sample, vecs)
        perturbed_values = self._model(perturbed_samples)
        return (value - perturbed_values).T / self._fd_eps

    def _hessian(self, sample: Array) -> Array:
        value = self._model.jacobian(sample)
        perturbed_samples = self._perturbed_samples(sample)
        perturbed_values = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples.T],
            axis=-1,
        )
        return (-perturbed_values + value[..., None]) / self._fd_eps

    def _apply_hessian(self, sample: Array, vecs: Array) -> Array:
        value = self._model.jacobian(sample)
        perturbed_samples = self._directionally_perturbed_samples(sample, vecs)
        perturbed_values = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples.T],
            axis=-1,
        )
        return -(perturbed_values - value[..., None]) / self._fd_eps


class CenteredFiniteDifference(FiniteDifference):
    def _perturbed_samples(self, sample: Array) -> Array:
        perturbed_samples1 = self._bkd.tile(sample, (self.nvars(),))
        perturbed_samples2 = self._bkd.tile(sample, (self.nvars(),))
        for ii in range(self.nvars()):
            perturbed_samples1[ii, ii] -= self._fd_eps
            perturbed_samples2[ii, ii] += self._fd_eps
        return perturbed_samples1, perturbed_samples2

    def _directionally_perturbed_samples(self, sample: Array, vecs) -> Array:
        perturbed_samples1 = sample - self._fd_eps * vecs
        perturbed_samples2 = sample + self._fd_eps * vecs
        return perturbed_samples1, perturbed_samples2

    def jacobian(self, sample: Array) -> Array:
        perturbed_samples1, perturbed_samples2 = self._perturbed_samples(
            sample
        )
        perturbed_values1 = self._model(perturbed_samples1)
        perturbed_values2 = self._model(perturbed_samples2)
        return (perturbed_values2 - perturbed_values1).T / (2 * self._fd_eps)

    def apply_jacobian(self, sample: Array, vecs: Array) -> Array:
        perturbed_samples1, perturbed_samples2 = (
            self._directionally_perturbed_samples(sample, vecs)
        )
        perturbed_values1 = self._model(perturbed_samples1)
        perturbed_values2 = self._model(perturbed_samples2)
        return (perturbed_values2 - perturbed_values1).T / (2 * self._fd_eps)

    def _hessian(self, sample: Array) -> Array:
        perturbed_samples1, perturbed_samples2 = self._perturbed_samples(
            sample
        )
        perturbed_values1 = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples1.T],
            axis=-1,
        )
        perturbed_values2 = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples2.T],
            axis=-1,
        )
        return (perturbed_values2 - perturbed_values1) / (2 * self._fd_eps)

    def _apply_hessian(self, sample: Array, vecs: Array) -> Array:
        perturbed_samples1, perturbed_samples2 = (
            self._directionally_perturbed_samples(sample, vecs)
        )
        perturbed_values1 = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples1.T],
            axis=-1,
        )
        perturbed_values2 = self._bkd.stack(
            [self._model.jacobian(ps[:, None]) for ps in perturbed_samples2.T],
            axis=-1,
        )
        return (perturbed_values2 - perturbed_values1) / (2 * self._fd_eps)
