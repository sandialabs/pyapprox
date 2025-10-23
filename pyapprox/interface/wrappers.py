import os
from multiprocessing.pool import ThreadPool, Pool

import numpy as np

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
    Model: ActiveSetVariableModel
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

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyMixin as bkd
    >>> from pyapprox.interface.model import Model
    >>> from pyapprox.interface.wrappers import create_active_set_variable_model

    >>> # Define a custom model
    >>> class CustomModel(Model):
    ...     def nqoi(self):
    ...         return 1
    ...     def nvars(self):
    ...         return 3
    ...     def _values(self, x):
    ...         return (((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2) + x[2])[:, None]
    ...     def _jacobian(self, x):
    ...         return bkd.array([[2 * (x[0] - 1), 2 * (x[1] - 2.5), x[2] * 0 + 1]])
    ...     def foo(self):
    ...         return "Hello"

    >>> # Instantiate the custom model
    >>> model = CustomModel(backend=bkd)

    >>> # Define inactive variable indices
    >>> inactive_ids = bkd.array([0, 2])

    >>> # Wrap the model with active set variable model
    >>> wrapped_model = create_active_set_variable_model(
    ...     model, 3, bkd.array([2.0])[:, None], inactive_ids
    ... )

    >>> # Define input sample
    >>> sample = np.array([1.0, 2.0, 3.0])[:, None]

    >>> # Evaluate the original model
    >>> print(model(sample))
    [[3.25]]

    >>> # Evaluate the wrapped model
    >>> print(wrapped_model(sample[inactive_ids]))
    [[3.25]]

    >>> # Call a custom method from the wrapped model
    >>> wrapped_model.foo()
    'Hello'
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

        def model(self) -> Model:
            # return the unwrapped model
            return self._model

        def nvars(self) -> int:
            return self._active_var_indices.shape[0]

        def _expand_samples(self, reduced_samples: Array) -> Array:
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
            # Redirect all attribute/method calls to the wrapped instance
            overridden_attr_names = [
                "model",
                "nvars",
                "_expand_samples",
                "_values",
                "_new_values",
                "_jacobian",
                "_apply_jacobian",
                "apply_hessian",
                "noriginal_vars",
                # the following are not defined explicitly here but are
                # defined in model. However, they must not point to model.fun
                "_check_sample_shape",
                "_check_samples_shape",
                "jacobian",
                "apply_jacobian",
            ]
            for name in dir(self._model):
                if name.startswith("__"):
                    continue
                if name in overridden_attr_names:
                    # if we have overwridden the attribute do not
                    # use model attr
                    continue
                setattr(self, name, getattr(self._model, name))

    wrapped_model = ActiveSetVariableModel(
        model, nvars, inactive_var_values, active_var_indices
    )

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
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi: int
            The number of quantities of interest.
        """
        return self._model.nqoi()

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars: int
            The number of variables.
        """
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


def create_pool_model(
    model: Model, nprocs: int, assert_omp: bool = True
) -> Model:
    """
    Create a model wrapper that enables parallel evaluation of samples using multiprocessing.

    This function wraps a `Model` instance in a `PoolModelWrapper` class, allowing the model to evaluate multiple samples in parallel using Python's `multiprocessing.Pool`. Parallelization is currently limited to evaluating values; Jacobian computations are not yet supported in parallel.

    Parameters
    ----------
    model : Model
        The original model to be wrapped. Must be an instance of the `Model` class.
    nprocs : int
        The number of processes to use for parallel evaluation. If `nprocs` is set to 1, the model will run sequentially.
    assert_omp : bool, optional
        If `True`, ensures that the environment variable `OMP_NUM_THREADS` is set to 1 when `nprocs > 1`. This prevents OpenMP from using multiple threads, which can interfere with multiprocessing. Defaults to `True`.

    Returns
    -------
    Model
        A wrapped model (`PoolModelWrapper`) that supports parallel evaluation of samples.

    Raises
    ------
    ValueError
        If `model` is not an instance of the `Model` class.
    Exception
        If `assert_omp=True` and `OMP_NUM_THREADS` is not set to 1 in the environment when `nprocs > 1`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyMixin as bkd
    >>> from pyapprox.interface.model import Model
    >>> from pyapprox.interface.wrappers import create_pool_model
    >>> class CustomModel(Model):
    ...     def nqoi(self):
    ...          return 1
    ...     def nvars(self):
    ...          return 2
    ...     def _values(self, samples):
    ...         return self._bkd.sum(samples, axis=0)[:, None]
    ...     def foo(self):
    ...         return "Hello"
    >>> model = CustomModel(backend=bkd)
    >>> wrapped_model = create_pool_model(model, nprocs=2, assert_omp=False)
    >>> samples = np.array([[1, 2], [3, 4]])
    >>> print(wrapped_model(samples))
    [[4]
     [6]]
    >>> wrapped_model.foo()
    'Hello'
    """

    class PoolModelWrapper(Model):
        r"""
        Wrap a Model so that it can be evaluated at multiple samples
        in parallel using multiprocessing.Pool.

        For now just supports parallelizing values, i.e. paralelle jacobaian
        computations are not yet supported. Only one jacobian can be requested
        at a time.
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
            self._add_model_attributes()

        def nqoi(self) -> int:
            return self._model.nqoi()

        def nvars(self) -> int:
            """
            Return the number of variables in the model.

            Returns
            -------
            nvars: int
                The number of variables.
            """
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
            raise NotImplementedError(
                "database not support for PoolModelWrapper"
            )

        def activate_model_data_base(self):
            raise NotImplementedError(
                "database not support for PoolModelWrapper"
            )

        def _add_model_attributes(self):
            # Redirect all attribute/method calls to the wrapped instance
            overridden_attr_names = [
                "nqoi",  # needed so object is not abstract. calls model.nqoi
                "nvars",  # needed so  object is not abstract model.nvars
                "_values",
                "model",
                "work_tracker",
                "model_database",
                "activate_model_data_base",
            ]
            for name in dir(self._model):
                if name.startswith("__"):
                    continue
                if name in overridden_attr_names:
                    # if we have overwridden the attribute do not
                    # use model attr
                    continue
                setattr(self, name, getattr(self._model, name))

        def __repr__(self) -> str:
            return "{0}(model={1})".format(
                self.__class__.__name__, self._model
            )

    wrapped_model = PoolModelWrapper(model, nprocs, assert_omp)

    return wrapped_model
