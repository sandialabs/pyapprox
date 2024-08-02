from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)


class HyperParameterTransform(ABC):
    def __init__(self, backend: LinAlgMixin = None):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend

    @abstractmethod
    def to_opt_space(self, params):
        raise NotImplementedError

    @abstractmethod
    def from_opt_space(self, params):
        raise NotImplementedError

    def __repr__(self):
        return "{0}(bkd={1})".format(self.__class__.__name__, self._bkd)


class IdentityHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return params

    def from_opt_space(self, params):
        return params


class LogHyperParameterTransform(HyperParameterTransform):
    def to_opt_space(self, params):
        return self._bkd._la_log(params)

    def from_opt_space(self, params):
        return self._bkd._la_exp(params)


class HyperParameter:
    def __init__(
        self,
        name: str,
        nvars: int,
        values,
        bounds,
        transform: HyperParameterTransform = None,
        fixed: bool = False,
        backend: LinAlgMixin = None,
    ):
        """A possibly vector-valued hyper-parameter to be used with
        optimization."""
        if backend is None and transform is None:
            backend = NumpyLinAlgMixin()
        elif backend is None:
            backend = transform._bkd
        self._bkd = backend

        if transform is None:
            transform = IdentityHyperParameterTransform(self._bkd)
        else:
            if type(transform._bkd) is not type(backend):
                raise ValueError("transform._bkd must be the same as backend")
        self.transform = transform

        self.name = name
        self._nvars = nvars

        self._values = self._bkd._la_atleast1d(values)
        if self._values.shape[0] == 1:
            self._values = self._bkd._la_repeat(self._values, self.nvars())
        if self._values.ndim == 2:
            raise ValueError("values is not a 1D array")
        if self._values.shape[0] != self.nvars():
            raise ValueError(
                "values shape {0} inconsistent with nvars {1}".format(
                    self._values.shape, self.nvars()
                )
            )
        if fixed:
            self.set_all_inactive()
        else:
            self.set_all_active()
        self.set_bounds(bounds)

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
        self.set_active_indices(self._bkd._la_zeros((0, ), dtype=int))

    def set_all_active(self):
        self.set_active_indices(self._bkd._la_arange(self.nvars(), dtype=int))

    def set_bounds(self, bounds):
        self.bounds = self._bkd._la_atleast1d(bounds)
        if self.bounds.shape[0] == 2:
            self.bounds = self._bkd._la_repeat(self.bounds, self.nvars())
        if self.bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                self.bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)
        self.bounds = self._bkd._la_reshape(
            self.bounds, (self.bounds.shape[0] // 2, 2)
        )

    def nvars(self):
        """Return the number of hyperparameters."""
        return self._nvars

    def nactive_vars(self):
        """Return the number of active (to be optinized) hyperparameters."""
        return self._active_indices.shape[0]

    def set_active_opt_params(self, active_params):
        """Set the values of the active parameters in the optimization space."""
        # The copy ensures that the error
        # "a leaf Variable that requires grad is being used in an in-place
        # operation is not thrown
        self._values = self._bkd._la_copy(self._values)
        # self._values[self._active_indices] = self.transform.from_opt_space(
        #    active_params)
        self._values = self._bkd._la_up(
            self._values,
            self._active_indices,
            self.transform.from_opt_space(active_params),
            axis=0,
        )

    def get_active_opt_params(self):
        """Get the values of the active parameters in the optimization space."""
        return self.transform.to_opt_space(self._values[self._active_indices])

    def get_active_opt_bounds(self):
        """Get the bounds of the active parameters in the optimization space."""
        return self.transform.to_opt_space(
            self.bounds[self._active_indices, :]
        )

    def get_bounds(self):
        """Get the bounds of the parameters in the user space."""
        return self.bounds.flatten()

    def get_values(self):
        """Get the values of the parameters in the user space."""
        return self._values

    def set_values(self, values):
        """Set the values of the parameters in the user space."""
        self._values = values

    def _short_repr(self):
        if self.nvars() > 5:
            return "{0}:nvars={1}".format(self.name, self.nvars())

        return "{0}={1}".format(
            self.name,
            "[" + ", ".join(map("{0:.2g}".format, self._values)) + "]",
        )

    def __repr__(self):
        if self.nvars() > 5:
            return (
                "{0}(name={1}, nvars={2}, transform={3}, nactive={4})".format(
                    self.__class__.__name__,
                    self.name,
                    self.nvars(),
                    self.transform,
                    self.nactive_vars(),
                )
            )
        return "{0}(name={1}, values={2}, transform={3}, active={4})".format(
            self.__class__.__name__,
            self.name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
            self.transform,
            "[" + ", ".join(map("{0}".format, self._active_indices)) + "]",
        )

    def detach(self):
        """Detach the hyperparameter values from the computational graph if
        in use."""
        self.set_values(self._bkd._la_detach(self.get_values()))


class HyperParameterList:
    def __init__(self, hyper_params: list):
        """A list of hyper-parameters to be used with optimization."""
        self.hyper_params = hyper_params
        self._bkd = self.hyper_params[0]._bkd

    def set_active_opt_params(self, active_params):
        """Set the values of the active parameters in the optimization space."""
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_active_opt_params(
                active_params[cnt : cnt + hyp.nactive_vars()]
            )
            cnt += hyp.nactive_vars()

    def nvars(self):
        """Return the total number of hyperparameters (active and inactive)."""
        cnt = 0
        for hyp in self.hyper_params:
            cnt += hyp.nvars()
        return cnt

    def nactive_vars(self):
        """Return the number of active (to be optinized) hyperparameters."""
        cnt = 0
        for hyp in self.hyper_params:
            cnt += hyp.nactive_vars()
        return cnt

    def get_active_opt_params(self):
        """Get the values of the active parameters in the optimization space."""
        return self._bkd._la_hstack(
            [hyp.get_active_opt_params() for hyp in self.hyper_params]
        )

    def get_active_opt_bounds(self):
        """Get the values of the active parameters in the optimization space."""
        return self._bkd._la_vstack(
            [hyp.get_active_opt_bounds() for hyp in self.hyper_params]
        )

    def get_bounds(self):
        """Get the flattned bounds of the parameters in the user space."""
        # bounds are flat because flat array is passed to set_bounds
        return self._bkd._la_hstack(
            [hyp.get_bounds() for hyp in self.hyper_params]
        )

    def get_values(self):
        """Get the values of the parameters in the user space."""
        return self._bkd._la_hstack(
            [hyp.get_values() for hyp in self.hyper_params]
        )

    def get_active_indices(self):
        cnt = 0
        active_indices = []
        for hyp in self.hyper_params:
            active_indices.append(hyp.get_active_indices()+cnt)
            cnt += hyp.nvars()
        return self._bkd._la_hstack(active_indices)

    def set_active_indices(self, active_indices):
        cnt = 0
        for hyp in self.hyper_params:
            hyp_indices = self._bkd._la_where(
                (active_indices>=cnt)&(active_indices<cnt+hyp.nvars()))[0]
            hyp.set_active_indices(active_indices[hyp_indices]-cnt)
            cnt += hyp.nvars()

    def set_all_inactive(self):
        for hyp in self.hyper_params:
            hyp.set_all_inactive()

    def set_all_active(self):
        for hyp in self.hyper_params:
            hyp.set_all_active()

    def set_values(self, values):
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_values(values[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()

    def __add__(self, hyp_list):
        # self.__class__ must be because of the use of mixin with derived
        # classes
        return self.__class__(self.hyper_params + hyp_list.hyper_params)

    def __radd__(self, hyp_list):
        if hyp_list == 0:
            # for when sum is called over list of HyperParameterLists
            return self
        return self.__class__(hyp_list.hyper_params + self.hyper_params)

    def _short_repr(self):
        # simpler representation used when printing kernels
        return ", ".join(
            map("{0}".format, [hyp._short_repr() for hyp in self.hyper_params])
        )

    def __repr__(self):
        return (
            "{0}(".format(self.__class__.__name__)
            + ",\n\t\t   ".join(map("{0}".format, self.hyper_params))
            + ")"
        )

    def set_bounds(self, bounds):
        # used to turn params inactive to active or vice-versa
        bounds = self._bkd._la_atleast1d(bounds)
        if bounds.shape[0] == 2:
            bounds = self._bkd._la_repeat(bounds, self.nvars())
        if bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_bounds(bounds[cnt : cnt + 2*hyp.nvars()])
            cnt += 2*hyp.nvars()


class CombinedHyperParameter(HyperParameter):
    # Some times it is more intuitive for the user to pass to separate
    # hyperparameters but the code requires them to be treated
    # as a single hyperparameter, e.g. when set_active_opt_params
    # that requires both user hyperparameters must trigger an action
    # like updating of an internal variable not common to all hyperparameter
    # classes
    def __init__(self, hyper_params: list):
        self.hyper_params = hyper_params
        self._bkd = self.hyper_params[0]._bkd
        self.bounds = self._bkd._la_vstack(
            [hyp.bounds for hyp in self.hyper_params]
        )

    def nvars(self):
        return sum([hyp.nvars() for hyp in self.hyper_params])

    def nactive_vars(self):
        return sum([hyp.nactive_vars() for hyp in self.hyper_params])

    def set_active_opt_params(self, active_params):
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_active_opt_params(
                active_params[cnt : cnt + hyp.nactive_vars()]
            )
            cnt += hyp.nactive_vars()

    def set_bounds(self, bounds):
        # used to turn params inactive to active or vice-versa
        bounds = self._bkd._la_atleast1d(bounds)
        if bounds.shape[0] == 2:
            bounds = self._bkd._la_repeat(bounds, self.nvars())
        if bounds.shape[0] != 2 * self.nvars():
            msg = "bounds shape {0} inconsistent with 2*nvars={1}".format(
                self.bounds.shape, 2 * self.nvars()
            )
            raise ValueError(msg)
        
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_bounds(bounds[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()
        

    def get_active_opt_params(self):
        return self._bkd._la_hstack(
            [hyp.get_active_opt_params() for hyp in self.hyper_params]
        )

    def get_active_opt_bounds(self):
        return self._bkd._la_vstack(
            [hyp.get_active_opt_bounds() for hyp in self.hyper_params]
        )

    def get_bounds(self):
        return self._bkd._la_vstack(
            [hyp.get_bounds() for hyp in self.hyper_params]
        )

    def get_values(self):
        return self._bkd._la_hstack(
            [hyp.get_values() for hyp in self.hyper_params]
        )

    def set_values(self, values):
        cnt = 0
        for hyp in self.hyper_params:
            hyp.set_values(values[cnt : cnt + hyp.nvars()])
            cnt += hyp.nvars()
