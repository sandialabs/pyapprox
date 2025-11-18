from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.template import BackendMixin, Array


class HyperParameterTransform(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend

    @abstractmethod
    def to_opt_space(self, params):
        raise NotImplementedError

    @abstractmethod
    def from_opt_space(self, params):
        raise NotImplementedError

    def __repr__(self):
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__name__
        )


class IdentityHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return params

    def from_opt_space(self, params):
        return params


class LogHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return self._bkd.log(params)

    def from_opt_space(self, params):
        return self._bkd.exp(params)


class HyperParameter:
    def __init__(
        self,
        name: str,
        nvars: int,
        values: Array,
        bounds: Union[Tuple[float, float], Array],
        transform: HyperParameterTransform = None,
        fixed: bool = False,
        backend: BackendMixin = NumpyMixin,
    ):
        """A possibly vector-valued hyper-parameter to be used with
        optimization."""
        if backend is None and transform is None:
            backend = NumpyMixin
        elif backend is None:
            backend = transform._bkd
        self._bkd = backend

        if transform is None:
            transform = IdentityHyperParameterTransform(self._bkd)
        else:
            if type(transform._bkd) is not type(backend):
                raise ValueError("transform._bkd must be the same as backend")
        self._transform = transform
        self._transform._bkd = self._bkd

        self._name = name
        self._nvars = nvars

        self._values = self._bkd.atleast1d(self._bkd.asarray(values))
        if self._values.shape[0] == 1:
            self._values = self._bkd.tile(self._values, (self.nvars(),))
        if self._values.ndim == 2:
            raise ValueError("values is not a 1D array")
        if self._values.shape[0] != self.nvars():
            raise ValueError(
                "values shape {0} inconsistent with nvars {1}".format(
                    self._values.shape, self.nvars()
                )
            )
        self.set_bounds(bounds)
        if fixed:
            self.set_all_inactive()
        else:
            self.set_all_active()

    def set_active_indices(self, indices):
        if indices.shape[0] == 0:
            self._active_indices = indices
            return

        if max(indices) >= self.nvars():
            raise ValueError("indices exceed nvars")
        if min(indices) < 0:
            raise ValueError("Ensure indices >= 0")
        self._active_indices = indices

    def get_active_indices(self):
        return self._active_indices

    def set_all_inactive(self):
        self.set_active_indices(self._bkd.zeros((0,), dtype=int))

    def set_all_active(self):
        frozen_indices = self._bkd.isnan(self._bounds[:, 0])
        self.set_active_indices(
            self._bkd.arange(self.nvars(), dtype=int)[~frozen_indices]
        )

    def set_bounds(self, bounds: Union[Tuple[float, float], Array]):
        self._bounds = self._bkd.atleast1d(self._bkd.asarray(bounds))
        if self._bounds.shape[0] == 2:
            self._bounds = self._bkd.tile(self._bounds, (self.nvars(),))
        if self._bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                self._bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)
        self._bounds = self._bkd.reshape(
            self._bounds, (self._bounds.shape[0] // 2, 2)
        )

    def nvars(self):
        """Return the number of hyperparameters."""
        return self._nvars

    def nactive_vars(self):
        """Return the number of active (to be optinized) hyperparameters."""
        return self._active_indices.shape[0]

    def set_active_opt_params(self, active_params):
        """
        Set the values of the active parameters in the optimization space.
        """
        if active_params.ndim != 1:
            raise ValueError("active_params must be a 1D array")
        # Copy does not detache self._values from graph,
        # but we only want gradient with respect to active_opt_params
        self._values = self._bkd.copy(self._bkd.detach(self._values))
        self._values = self._bkd.up(
            self._values,
            self._active_indices,
            self._transform.from_opt_space(active_params),
            axis=0,
        )

    def get_active_opt_params(self):
        """Get the values of the active parameters in the optimization space."""
        return self._transform.to_opt_space(self._values[self._active_indices])

    def get_active_opt_bounds(self):
        """Get the bounds of the active parameters in the optimization space."""
        return self._transform.to_opt_space(
            self._bounds[self._active_indices, :]
        )

    def get_bounds(self):
        """Get the bounds of the parameters in the user space."""
        return self._bounds.flatten()

    def get_values(self):
        """Get the values of the parameters in the user space."""
        return self._values

    def set_values(self, values):
        """Set the values of the parameters in the user space."""
        if values.ndim != 1:
            raise ValueError("values must be 1D")
        self._values = values

    def _short_repr(self):
        if self.nvars() > 5:
            return "{0}:nvars={1}".format(self._name, self.nvars())

        return "{0}={1}".format(
            self._name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
        )

    def __repr__(self):
        if self.nvars() > 5:
            return (
                "{0}(name={1}, nvars={2}, transform={3}, nactive={4})".format(
                    self.__class__.__name__,
                    self._name,
                    self.nvars(),
                    self._transform,
                    self.nactive_vars(),
                )
            )
        return "{0}(name={1}, values={2}, transform={3}, active={4})".format(
            self.__class__.__name__,
            self._name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
            self._transform,
            "[" + ", ".join(map("{0}".format, self._active_indices)) + "]",
        )

    def detach(self):
        """Detach the hyperparameter values from the computational graph if
        in use."""
        self.set_values(self._bkd.detach(self.get_values()))


class HyperParameterList:
    def __init__(self, hyper_params: list):
        """A list of hyper-parameters to be used with optimization."""
        self._hyper_params = hyper_params
        self._bkd = self._hyper_params[0]._bkd

    def set_active_opt_params(self, active_params):
        """Set the values of the active parameters in the optimization space."""
        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_active_opt_params(
                active_params[cnt : cnt + hyp.nactive_vars()]
            )
            cnt += hyp.nactive_vars()

    def nvars(self):
        """Return the total number of hyperparameters (active and inactive)."""
        cnt = 0
        for hyp in self._hyper_params:
            cnt += hyp.nvars()
        return cnt

    def nactive_vars(self):
        """Return the number of active (to be optinized) hyperparameters."""
        cnt = 0
        for hyp in self._hyper_params:
            cnt += hyp.nactive_vars()
        return cnt

    def get_active_opt_params(self):
        """Get the values of the active parameters in the optimization space."""
        return self._bkd.hstack(
            [hyp.get_active_opt_params() for hyp in self._hyper_params]
        )

    def get_active_opt_bounds(self):
        """Get the values of the active parameters in the optimization space."""
        return self._bkd.vstack(
            [hyp.get_active_opt_bounds() for hyp in self._hyper_params]
        )

    def get_bounds(self):
        """Get the flattned bounds of the parameters in the user space."""
        # bounds are flat because flat array is passed to set_bounds
        return self._bkd.hstack(
            [hyp.get_bounds() for hyp in self._hyper_params]
        )

    def get_values(self):
        """Get the values of the parameters in the user space."""
        return self._bkd.hstack(
            [hyp.get_values() for hyp in self._hyper_params]
        )

    def get_active_indices(self):
        cnt = 0
        active_indices = []
        for hyp in self._hyper_params:
            active_indices.append(hyp.get_active_indices() + cnt)
            cnt += hyp.nvars()
        return self._bkd.hstack(active_indices)

    def set_active_indices(self, active_indices):
        cnt = 0
        for hyp in self._hyper_params:
            hyp_indices = self._bkd.where(
                (active_indices >= cnt) & (active_indices < cnt + hyp.nvars())
            )[0]
            hyp.set_active_indices(active_indices[hyp_indices] - cnt)
            cnt += hyp.nvars()

    def set_all_inactive(self):
        for hyp in self._hyper_params:
            hyp.set_all_inactive()

    def set_all_active(self):
        for hyp in self._hyper_params:
            hyp.set_all_active()

    def set_values(self, values):
        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_values(values[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()

    def __add__(self, hyp_list):
        # self.__class__ must be because of the use of mixin with derived
        # classes
        return self.__class__(self._hyper_params + hyp_list._hyper_params)

    def __radd__(self, hyp_list):
        if hyp_list == 0:
            # for when sum is called over list of HyperParameterLists
            return self
        return self.__class__(hyp_list._hyper_params + self._hyper_params)

    def _short_repr(self):
        # simpler representation used when printing kernels
        return ", ".join(
            map(
                "{0}".format, [hyp._short_repr() for hyp in self._hyper_params]
            )
        )

    def __repr__(self):
        return (
            "{0}(".format(self.__class__.__name__)
            + ",\n\t\t   ".join(map("{0}".format, self._hyper_params))
            + ")"
        )

    def set_bounds(self, bounds: Union[Tuple[float, float], Array]):
        # used to turn params inactive to active or vice-versa
        bounds = self._bkd.atleast1d(bounds)
        if bounds.shape[0] == 2:
            bounds = self._bkd.tile(bounds, (self.nvars(),))
        if bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)
        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_bounds(bounds[cnt : cnt + 2 * hyp.nvars()])
            cnt += 2 * hyp.nvars()


class CombinedHyperParameter(HyperParameter):
    # Some times it is more intuitive for the user to pass to separate
    # hyperparameters but the code requires them to be treated
    # as a single hyperparameter, e.g. when set_active_opt_params
    # that requires both user hyperparameters must trigger an action
    # like updating of an internal variable not common to all hyperparameter
    # classes
    def __init__(self, hyper_params: list):
        self._hyper_params = hyper_params
        self._bkd = self._hyper_params[0]._bkd
        self._bounds = self._bkd.vstack(
            [hyp._bounds for hyp in self._hyper_params]
        )

    def nvars(self):
        return sum([hyp.nvars() for hyp in self._hyper_params])

    def nactive_vars(self):
        return sum([hyp.nactive_vars() for hyp in self._hyper_params])

    def set_active_opt_params(self, active_params):
        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_active_opt_params(
                active_params[cnt : cnt + hyp.nactive_vars()]
            )
            cnt += hyp.nactive_vars()

    def set_bounds(self, bounds: Union[Tuple[float, float], Array]):
        # used to turn params inactive to active or vice-versa
        bounds = self._bkd.atleast1d(bounds)
        if bounds.shape[0] == 2:
            bounds = self._bkd.tile(bounds, (self.nvars(),))
        if bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                self._bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)

        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_bounds(bounds[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()

    def get_active_opt_params(self):
        return self._bkd.hstack(
            [hyp.get_active_opt_params() for hyp in self._hyper_params]
        )

    def get_active_opt_bounds(self):
        return self._bkd.vstack(
            [hyp.get_active_opt_bounds() for hyp in self._hyper_params]
        )

    def get_bounds(self):
        return self._bkd.vstack(
            [hyp.get_bounds() for hyp in self._hyper_params]
        )

    def get_values(self):
        return self._bkd.hstack(
            [hyp.get_values() for hyp in self._hyper_params]
        )

    def set_values(self, values):
        cnt = 0
        for hyp in self._hyper_params:
            hyp.set_values(values[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()

    def get_active_indices(self):
        return self._bkd.hstack(
            [hyp.get_active_indices() for hyp in self._hyper_params]
        )


class CholeskyHyperParameter(HyperParameter):
    def __init__(
        self,
        name: str,
        nrows: int,
        values: Array,
        bounds: Union[Tuple[float, float], Array] = (-np.inf, np.inf),
        fixed: bool = False,
        backend: BackendMixin = NumpyMixin,
    ):
        self._nrows = nrows
        # do not use function flattened_lower_diagonal_matrix_entries
        # to avoid recreating mask each time values are set and get
        self._mask = backend.tril(
            backend.ones((self._nrows, self._nrows), dtype=bool)
        )
        nnonzero_chol = backend.where(self._mask)[0].shape[0]
        super().__init__(
            name,
            nnonzero_chol,
            values,
            bounds,
            IdentityHyperParameterTransform(backend=backend),
            fixed,
            backend,
        )

    def get_cholesky_factor(self) -> Array:
        # entire chol matrix must be created here. If a member
        # variable is updated autograd will not work with torch
        chol = self._bkd.zeros((self._nrows, self._nrows))
        chol[self._mask] = self.get_values()
        return chol

    def set_cholesky_factor(self, chol: Array):
        self.set_values(chol[self._mask])


def flattened_lower_diagonal_matrix_entries(
    matrix, bkd: BackendMixin = NumpyMixin
):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix[bkd.tril(bkd.ones(matrix.shape, dtype=bool))]
