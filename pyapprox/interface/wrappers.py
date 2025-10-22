import os
from multiprocessing.pool import ThreadPool, Pool

from pyapprox.util.backends.template import Array
from pyapprox.interface.model import (
    Model,
    expand_samples_from_indices,
    ModelWorkTracker,
    ModelDataBase,
)


def create_active_set_variable_model(
    model: Model,
    nvars: int,
    inactive_var_values: Array,
    active_var_indices: Array,
) -> Model:
    """
    Create a model wrapper that only accepts a subset of the model variables.

    This function dynamically creates a wrapper model (`ActiveSetVariableModel`) that restricts the input variables to a subset of the original model's variables. The wrapper ensures that only the active variables are considered during computations, while inactive variables are fixed to specified values.

    Parameters
    ----------
    model : Model
        The original model to be wrapped. Must be an instance of the `Model` class.
    nvars : int
        The total number of variables in the original model. This is used for error checking and validation.
    inactive_var_values : Array
        A 2D array specifying the fixed values for the inactive variables. The shape of this array must be `(num_inactive_vars, num_samples)`.
    active_var_indices : Array
        A 1D array specifying the indices of the active variables in the original model. These indices must be within the range `[0, nvars)`.

    Returns
    -------
    Model
        A wrapped model (`ActiveSetVariableModel`) that behaves like the original model but operates only on the active variables. The wrapper passes all other attributes and methods to the original model.

    Raises
    ------
    ValueError
        If `model` is not an instance of the `Model` class.
    AssertionError
        If the dimensions of `inactive_var_values` are invalid, or if the active variable indices are out of bounds.

    Notes
    -----
    - The wrapper uses the backend specified by the original model (`model._bkd`) for computations.
    - The wrapper dynamically inherits from the type of the original model (`type(model)`), ensuring compatibility with the original model's interface.
    - The wrapper provides methods for evaluating the model, computing Jacobians, and applying Jacobians and Hessians, while restricting operations to the active variables.
    """

    class ActiveSetVariableModel(type(model)):
        r"""
        Create a model wrapper that only accepts a subset of the model variables.
        """

        def __init__(
            self,
            model: Model,
            nvars: int,
            inactive_var_values: Array,
            active_var_indices: Array,
        ):
            self._bkd = model._bkd
            # nvars can de determined from inputs but making it
            # necessary allows for better error checking
            if not isinstance(model, Model):
                raise ValueError("model must be an instance of Model")
            self._model = model
            assert inactive_var_values.ndim == 2
            self._inactive_var_values = self._bkd.asarray(
                inactive_var_values, dtype=int
            )
            self._active_var_indices = self._bkd.asarray(
                active_var_indices, dtype=int
            )
            assert (
                self._active_var_indices.shape[0]
                + self._inactive_var_values.shape[0]
                == nvars
            )
            self._norignal_vars = nvars
            assert self._bkd.all(
                self._active_var_indices < self._norignal_vars
            )
            self._inactive_var_indices = self._bkd.delete(
                self._bkd.arange(self._norignal_vars, dtype=int),
                active_var_indices,
            )
            self._add_model_attributes()

        def unwrapped_model(self) -> Model:
            # return the unwrapped model
            return self._model

        # def apply_jacobian_implemented(self) -> bool:
        #     return self._base_model.apply_jacobian_implemented()

        # def jacobian_implemented(self) -> bool:
        #     return self._base_model.jacobian_implemented()

        # def apply_hessian_implemented(self) -> bool:
        #     return (
        #         self._base_model.apply_hessian_implemented()
        #         or self._base_model.hessian_implemented()
        #     )

        # def nqoi(self) -> int:
        #     return self._model.nqoi()

        def nvars(self) -> int:
            return self._active_var_indices.shape[0]

        def _expand_samples(self, reduced_samples):
            return expand_samples_from_indices(
                reduced_samples,
                self._active_var_indices,
                self._inactive_var_indices,
                self._inactive_var_values,
                self._bkd,
            )

        def _values(self, reduced_samples: Array) -> Array:
            samples = self._expand_samples(reduced_samples)
            return self._model(samples)

        def _jacobian(self, reduced_samples: Array) -> Array:
            samples = self._expand_samples(reduced_samples)
            jac = self._model.jacobian(samples)
            return jac[:, self._active_var_indices]

        def _apply_jacobian(self, reduced_samples: Array, vec: Array) -> Array:
            samples = self._expand_samples(reduced_samples)
            # set inactive entries of vec to zero when peforming
            # matvec product so they do not contribute to sum
            expanded_vec = self._bkd.zeros((self._norignal_vars, 1))
            expanded_vec[self._active_var_indices] = vec
            return self._model.apply_jacobian(samples, expanded_vec)

        def apply_hessian(self, reduced_samples: Array, vec: Array) -> Array:
            samples = self._expand_samples(reduced_samples)
            # set inactive entries of vec to zero when peforming
            # matvec product  so they do not contribute to sum
            expanded_vec = self._bkd.zeros((self._norignal_vars, 1))
            expanded_vec[self._active_var_indices] = vec
            return self._model.apply_hessian(samples, expanded_vec)[
                self._active_var_indices
            ]

        def noriginal_vars(self) -> int:
            return self._norignal_vars

        def __repr__(self) -> str:
            return "{0}(model={1})".format(
                self.__class__.__name__, self._model
            )

        def _add_model_attributes(self):
            for name in dir(self._model):
                if name.startswith("__"):
                    continue
                if hasattr(self, name):
                    # if we have overwridden the attribute do not
                    # use model attr
                    continue
                setattr(self, name, getattr(self._model, name))

    wrapped_model = ActiveSetVariableModel(
        model, nvars, inactive_var_values, active_var_indices
    )

    # Redirect all attribute/method calls to the wrapped instance
    return wrapped_model


class ScipyModelWrapper:
    def __init__(self, model):
        """
        Create a API that takes a sample as a 1D Array and returns
        the objects needed by scipy optimizers. E.g.
        jac will return Array
        even when model accepts and returns arrays associated with a
        different backend. This function is intended for use with scipy
        optimizers and so does not need to allow access to other
        # attributes model may have (unlike ActiveSetVariableModel for example)
        """
        self._bkd = model._bkd
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "hessian_implemented",
            "apply_hessian_implemented",
            "weighted_hessian_implemented",
            "apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def jacobian_implemented(self) -> bool:
        return (
            self._model.jacobian_implemented()
            or self._model.apply_jacobian_implemented()
        )

    def _check_sample(self, sample: Array) -> Array:
        if sample.ndim != 1:
            raise ValueError(
                "sample must be a 1D array but has shape {0}".format(
                    sample.shape
                )
            )
        return self._bkd.asarray(sample)

    def __call__(self, sample: Array) -> Array:
        # use copy to avoid warning:
        # The given NumPy array is not writable ...
        sample = self._check_sample(np.copy(sample))
        vals = self._model(sample[:, None])
        if vals.shape[0] == 1:
            return vals[0]
        return self._bkd.to_numpy(vals)

    def jac(self, sample: Array) -> Array:
        sample = self._check_sample(sample)
        if self._model.jacobian_implemented():
            jac = self._model.jacobian(sample[:, None])
            jac = self._bkd.detach(jac)
        else:
            jacs = []
            for ii in range(self._model.nvars()):
                vec = self._bkd.zeros((self._model.nvars(), 1))
                vec[ii] = 1.0
                jacs.append(
                    self._bkd.detach(
                        self._model.apply_jacobian(sample[:, None], vec)[0]
                    )
                )
            jac = self._bkd.to_numpy(self._bkd.stack(jacs, axis=1))

        if jac.shape[0] == 1:
            return jac[0]
        return self._bkd.to_numpy(jac)

    def hess(self, sample: Array) -> Array:
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(self._model.hessian(sample[:, None]))

    def hessp(self, sample: Array, vec: Array) -> Array:
        sample = self._check_sample(sample)
        if vec.ndim != 1:
            raise ValueError("vec must be 1D array")
        # dtype=self._bkd.double_type() is needed because
        # scipy passes an int8 vector to hessp to check size of hvp
        return self._bkd.to_numpy(
            self._model.apply_hessian(
                sample[:, None],
                self._bkd.asarray(vec[:, None], dtype=self._bkd.double_type()),
            )
        )

    def weighted_hess(self, sample: Array, weights: Array) -> Array:
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(
            self._model.weighted_hessian(
                sample[:, None], self._bkd.asarray(weights)[:, None]
            )
        )

    def __repr__(self) -> str:
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class ChangeModelSignWrapper(Model):
    """
    Change the sign of the values, jacobians etc returned by a model.

    This function is intended for use with scipy
    optimizers and so does not need to allow access to other
      attributes model may have (unlike ActiveSetVariableModel for example)
    """

    def __init__(self, model: Model):
        super().__init__(model._bkd)
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "jacobian_implemented",
            "hessian_implemented",
            "apply_hessian_implemented",
            "weighted_hessian_implemented",
            "apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._model.nvars()

    def _values(self, samples: Array) -> Array:
        vals = -self._model(samples)
        return vals

    def _jacobian(self, sample: Array) -> Array:
        return -self._model.jacobian(sample)

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        return -self._model.apply_jacobian(sample, vec)

    def _hessian(self, sample: Array) -> Array:
        return -self._model.hessian(sample)

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return -self._model.apply_hessian(sample, vec)

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class PoolModelWrapper(Model):
    r"""
    Wrap a Model so that it can be evaluated at multiple samples
    in parallel using multiprocessing.Pool.

    For now just supports parallelizing values, i.g. gradient computations
    are not yet supported
    """

    def __init__(self, model: Model, nprocs: int, assert_omp: bool = True):
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        super().__init__(model._bkd)
        if assert_omp and nprocs > 1:
            if (
                "OMP_NUM_THREADS" not in os.environ
                or not int(os.environ["OMP_NUM_THREADS"]) == 1
            ):
                raise Exception(
                    "User set assert_omp=True but OMP_NUM_THREADS "
                    "has not been set to 1. Run script with "
                    "OMP_NUM_THREADS=1 python script.py"
                )

        self._model = model
        self._nprocs = nprocs
        # overwrite model work_tracker with one that shares memory
        # between processes.
        self._model._work_tracker = ModelWorkTracker(
            model._bkd, self._nprocs > 1
        )

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._model.nvars()

    def _values(self, samples: Array) -> Array:
        if self._nprocs == 1:
            return self._model(samples)
        pool = Pool(self._nprocs)
        # call _values (instead of __call__) so times are not recorded twice)
        # once by the wrapper and once by model.__call__
        result = pool.map(
            self._model._values,
            [(samples[:, ii : ii + 1]) for ii in range(samples.shape[1])],
        )
        pool.close()
        return self._bkd.vstack(result)

    def model(self) -> Model:
        return self._model

    def work_tracker(self) -> ModelWorkTracker:
        # self._work_tracker as it will contain different data than
        # self._model._work_tracker. For a set of samples passed to
        # __call__, the former reports the total time / nsamples. The later
        # reports the time taken by each model without the overhead of
        # multiprocessing pickling everything
        return self._work_tracker

    def model_database(self) -> ModelDataBase:
        # multiprocessing makes copies of the model which all update
        # their own database. Need to make them share memory like for
        # work_tracker. However, this will slow down computations alot
        # (for cheap models)
        raise NotImplementedError("database not support for PoolModelWrapper")

    def activate_model_data_base(self):
        raise NotImplementedError("database not support for PoolModelWrapper")
