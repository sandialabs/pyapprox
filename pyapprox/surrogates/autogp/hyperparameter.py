import numpy as np
from abc import ABC, abstractmethod
from pyapprox.surrogates.autogp._torch_wrappers import (
    log, exp, atleast1d, repeat, arange, isnan, vstack, hstack, copy)


class HyperParameterTransform(ABC):
    @abstractmethod
    def to_opt_space(self, params):
        raise NotImplementedError

    @abstractmethod
    def from_opt_space(self, params):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class IdentityHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return params

    def from_opt_space(self, params):
        return params


class LogHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return log(params)

    def from_opt_space(self, params):
        return exp(params)


class HyperParameter():
    def __init__(self, name: str, nvars: int, values: np.ndarray,
                 bounds: np.ndarray, transform: HyperParameterTransform):
        self.name = name
        self._nvars = nvars
        self._values = atleast1d(values)
        if self._values.shape[0] == 1:
            self._values = repeat(self._values, self.nvars())
        if self._values.ndim == 2:
            raise ValueError("values is not a 1D array")
        if self._values.shape[0] != self.nvars():
            raise ValueError("values shape {0} inconsistent with nvars".format(
                self._values.shape))
        self.bounds = atleast1d(bounds)
        if self.bounds.shape[0] == 2:
            self.bounds = repeat(self.bounds, self.nvars())
        if self.bounds.shape[0] != 2*self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                self.bounds.shape, 2*self.nvars())
            raise ValueError(msg)
        self.bounds = self.bounds.reshape((self.bounds.shape[0]//2, 2))
        self.transform = transform
        if np.where(
                (self._values < self.bounds[:, 0]) |
                (self._values > self.bounds[:, 1]))[0].shape[0] > 0:
            raise ValueError("values outside bounds")
        self._active_indices = np.atleast_1d(
            arange(self.nvars())[~isnan(self.bounds[:, 0])])

    def nvars(self):
        return self._nvars

    def nactive_vars(self):
        return self._active_indices.shape[0]

    def set_active_opt_params(self, active_params):
        # The copy ensures that the error
        # "a leaf Variable that requires grad is being used in an in-place operation.
        # is not thrown
        self._values = copy(self._values)
        self._values[self._active_indices] = self.transform.from_opt_space(
            active_params)

    def get_active_opt_params(self):
        return self.transform.to_opt_space(self._values[self._active_indices])

    def get_active_opt_bounds(self):
        return self.transform.to_opt_space(
            self.bounds[self._active_indices, :])

    def get_values(self):
        return self._values

    def set_values(self, values):
        self._values = values

    def _short_repr(self):
        if self.nvars() > 5:
            return "{0}:nvars={1}".format(self.name, self.nvars())

        return "{0}={1}".format(
            self.name,
            "["+", ".join(map("{0:.2g}".format, self._values))+"]")

    def __repr__(self):
        if self.nvars() > 5:
            return "{0}(name={1}, nvars={2}, transform={3}, nactive={4})".format(
                self.__class__.__name__, self.name, self.nvars(),
                self.transform, self.nactive_vars())
        return "{0}(name={1}, values={2}, transform={3}, active={4})".format(
            self.__class__.__name__, self.name,
            "["+", ".join(map("{0:.2g}".format, self.get_values()))+"]",
            self.transform,
            "["+", ".join(map("{0}".format, self._active_indices))+"]")

    def detach(self):
        self.set_values(self.get_values().detach())


class HyperParameterList():
    def __init__(self, hyper_params: list):
        self.hyper_params = hyper_params

    def set_active_opt_params(self, active_params):
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_active_opt_params(
                active_params[cnt:cnt+hyp.nactive_vars()])
            cnt += hyp.nactive_vars()

    def get_active_opt_params(self):
        return hstack(
            [hyp.get_active_opt_params() for hyp in self.hyper_params])

    def get_active_opt_bounds(self):
        return vstack(
            [hyp.get_active_opt_bounds() for hyp in self.hyper_params])

    def get_values(self):
        return hstack([hyp.get_values() for hyp in self.hyper_params])

    def __add__(self, hyp_list):
        return HyperParameterList(self.hyper_params+hyp_list.hyper_params)

    def __radd__(self, hyp_list):
        if hyp_list == 0:
            # for when sum is called over list of HyperParameterLists
            return self
        return HyperParameterList(hyp_list.hyper_params+self.hyper_params)

    def _short_repr(self):
        # simpler representation used when printing kernels
        return (
            ", ".join(
                map("{0}".format,
                    [hyp._short_repr() for hyp in self.hyper_params])))

    def __repr__(self):
        return ("{0}(".format(self.__class__.__name__) +
                ",\n\t\t   ".join(map("{0}".format, self.hyper_params))+")")
