import os
import inspect
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

        # must overide public versions of jacobian, apply_jacobian
        # apply_hessian
        # so that work tracker and data base of self._model
        # are updated correctly
        def jacobian(self, reduced_samples: Array) -> Array:
            samples = self._expand_samples(reduced_samples)
            jac = self._model.jacobian(samples)
            return jac[:, self._active_var_indices]

        def apply_jacobian(self, reduced_samples: Array, vec: Array) -> Array:
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
            return "{0}(model={1}, nactive_vars={2})".format(
                self.__class__.__name__, self._model, self.nvars()
            )

        def _public_function_names(self, cls):
            funs = inspect.getmembers(Model, predicate=inspect.isfunction)
            names = [name for name, func in funs if not name.startswith("_")]
            return names

        def _add_model_attributes(self):
            # Get list of public member functions in self._model not in
            # Model(ABC) class
            # private functions are assumed to start with _, e.g. def _fun(self)

            # First get names of all public functions from Model(ABC)
            abstract_model_public_fun_names = self._public_function_names(
                Model
            )
            # Second get names of all public functions from self._model
            model_public_fun_names = self._public_function_names(self._model)
            # Take the set difference
            specialized_model_public_funs = [
                name
                for name in model_public_fun_names
                if name not in abstract_model_public_fun_names
            ]
            # Add attributes to this model that are in self._model but not in
            # Model(ABC)
            for name in specialized_model_public_funs:
                setattr(self, name, getattr(self._model, name))

            # Add attributes that need to be extracted from Model(ABC) class
            # for example so that self._model data base is updated correctly
            other_attributes = [
                "nqoi",
                "jacobian_implemented",
                "apply_jacobian_implemented",
                "apply_hessian_implemented",
            ]
            for name in other_attributes:
                setattr(self, name, getattr(self._model, name))

        def hessian_implemented(self) -> int:
            return False

        def weighted_hessian_implemented(self) -> int:
            return False

        def apply_weighted_hessian_implemented(self) -> int:
            return False

        def work_tracker(self):
            raise AttributeError(
                "must call the wrapped models version of this function"
            )

        def model_database(self):
            raise AttributeError(
                "must call the wrapped models version of this function"
            )

    wrapped_model = ActiveSetVariableModel(
        model, nvars, inactive_var_values, active_var_indices
    )

    return wrapped_model


class ScipyModelWrapper:
    """
    Wrapper for models to interface with SciPy optimizers.

    This class provides an API that takes a sample as a 1D array and returns
    objects needed by SciPy optimizers. It ensures compatibility between models
    that use different backends and SciPy optimizers. The wrapper handles
    conversion of inputs and outputs to NumPy arrays and provides methods for
    evaluating the model, its Jacobian, Hessian, and Hessian-vector products.

    Parameters
    ----------
    model : Model
        The model to be wrapped. Must be derived from the `Model` class.
    """

    def __init__(self, model):
        """
        Initialize the wrapper for the given model.

        Parameters
        ----------
        model : Model
            The model to be wrapped. Must be derived from the `Model` class.

        Raises
        ------
        ValueError
            If the provided model is not derived from the `Model` class.
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
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian or the apply Jacobian method is implemented, False otherwise.
        """
        return (
            self._model.jacobian_implemented()
            or self._model.apply_jacobian_implemented()
        )

    def _check_sample(self, sample: Array) -> Array:
        """
        Validate the input sample and convert it to the backend array.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.

        Returns
        -------
        sample : Array
            Validated and converted sample.

        Raises
        ------
        ValueError
            If the sample is not a 1D array.
        """
        if sample.ndim != 1:
            raise ValueError(
                "sample must be a 1D array but has shape {0}".format(
                    sample.shape
                )
            )
        return self._bkd.asarray(sample)

    def __call__(self, sample: Array) -> Array:
        """
        Evaluate the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.

        Returns
        -------
        values : Array
            Model evaluations at the given sample.
        """
        sample = self._check_sample(np.copy(sample))
        vals = self._model(sample[:, None])
        if vals.shape[0] == 1:
            return vals[0]
        return self._bkd.to_numpy(vals)

    def jac(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.

        Returns
        -------
        jacobian : Array
            Jacobian matrix of the model at the given sample.

        Notes
        -----
        If the model does not implement the Jacobian method, the Jacobian is computed
        using the apply Jacobian method.
        """
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
        """
        Compute the Hessian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.

        Returns
        -------
        hessian : Array
            Hessian matrix of the model at the given sample.
        """
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(self._model.hessian(sample[:, None]))

    def hessp(self, sample: Array, vec: Array) -> Array:
        """
        Compute the Hessian-vector product at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.
        vec : Array
            Vector for the Hessian-vector product. Must be a 1D array.

        Returns
        -------
        hessp : Array
            Hessian-vector product at the given sample.

        Raises
        ------
        ValueError
            If `vec` is not a 1D array.
        """
        sample = self._check_sample(sample)
        if vec.ndim != 1:
            raise ValueError("vec must be 1D array")
        return self._bkd.to_numpy(
            self._model.apply_hessian(
                sample[:, None],
                self._bkd.asarray(vec[:, None], dtype=self._bkd.double_type()),
            )
        )

    def weighted_hess(self, sample: Array, weights: Array) -> Array:
        """
        Compute the weighted Hessian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 1D array.
        weights : Array
            Weights for the weighted Hessian.

        Returns
        -------
        weighted_hessian : Array
            Weighted Hessian matrix of the model at the given sample.
        """
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(
            self._model.weighted_hessian(
                sample[:, None], self._bkd.asarray(weights)[:, None]
            )
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the wrapped model.
        """
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class ChangeModelSignWrapper(Model):
    """
    Wrapper to change the sign of values, Jacobians, and Hessians returned by a model.

    This class wraps a given model and modifies its behavior by negating the values,
    Jacobians, Hessians, and Hessian-vector products returned by the model. It is
    intended for objectives that should be maximized but must be negated so they can be minimzed.

    Parameters
    ----------
    model : Model
        The model to be wrapped. Must be derived from the `Model` class.
    """

    def __init__(self, model: Model):
        """
        Initialize the wrapper for the given model.

        Parameters
        ----------
        model : Model
            The model to be wrapped. Must be derived from the `Model` class.

        Raises
        ------
        ValueError
            If the provided model is not derived from the `Model` class.
        """
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
        nqoi : int
            The number of quantities of interest.
        """
        return self._model.nqoi()

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            The number of variables.
        """
        return self._model.nvars()

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model at the given samples and negate the returned values.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Negated values returned by the model.
        """
        vals = -self._model(samples)
        return vals

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the model at the given sample and negate the returned values.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        jacobian : Array
            Negated Jacobian matrix returned by the model.
        """
        return -self._model.jacobian(sample)

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Jacobian to a vector and negate the returned values.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the Jacobian to.

        Returns
        -------
        applied_jacobian : Array
            Negated result of applying the Jacobian to the vector.
        """
        return -self._model.apply_jacobian(sample, vec)

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the model at the given sample and negate the returned values.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        hessian : Array
            Negated Hessian matrix returned by the model.
        """
        return -self._model.hessian(sample)

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Hessian to a vector and negate the returned values.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the Hessian to.

        Returns
        -------
        applied_hessian : Array
            Negated result of applying the Hessian to the vector.
        """
        return -self._model.apply_hessian(sample, vec)

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns
        -------
        repr : str
            String representation of the class, including the wrapped model.
        """
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

        def _public_function_names(self, cls):
            funs = inspect.getmembers(Model, predicate=inspect.isfunction)
            names = [name for name, func in funs if not name.startswith("_")]
            return names

        def _add_model_attributes(self):
            # Redirect all public attribute/method calls to the wrapped
            # instance except the following
            # private functions are assumed to start with _, e.g. def _fun(self)
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
                if name.startswith("_"):
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


def create_active_set_qoi_model(
    model: Model, active_qoi_indices: Array
) -> Model:
    class ActiveSetQoIModel(type(model)):
        r"""
        Create a model wrapper that only accepts a subset of the model variables.

        Notes
        -----
        work_tracker and database will update model passed in as argument here
        """

        def __init__(
            self,
            model: Model,
            active_qoi_indices: Array,
        ):
            # nvars can de determined from inputs but making it
            # necessary allows for better error checking
            if not isinstance(model, Model):
                raise ValueError("model must be an instance of Model")
            self._model = model
            Model.__init__(self, model._bkd)
            if self._bkd.max(active_qoi_indices) >= model.nqoi():
                raise ValueError("active qoi indices not found")
            self._active_qoi_indices = self._bkd.asarray(
                active_qoi_indices, dtype=int
            )
            self._add_model_attributes()

        def _public_function_names(self, cls):
            funs = inspect.getmembers(Model, predicate=inspect.isfunction)
            names = [name for name, func in funs if not name.startswith("_")]
            return names

        def _add_model_attributes(self):
            # Get list of public member functions in self._model not in
            # Model(ABC) class
            # private functions are assumed to start with _, e.g. def _fun(self)

            # First get names of all public functions from Model(ABC)
            abstract_model_public_fun_names = self._public_function_names(
                Model
            )
            # Second get names of all public functions from self._model
            model_public_fun_names = self._public_function_names(self._model)
            # Take the set difference
            specialized_model_public_funs = [
                name
                for name in model_public_fun_names
                if name not in abstract_model_public_fun_names
            ]
            # Add attributes to this model that are in self._model but not in
            # Model(ABC)
            for name in specialized_model_public_funs:
                setattr(self, name, getattr(self._model, name))

            # Add attributes that need to be extracted from Model(ABC) class
            # for example so that self._model data base is updated correctly
            other_attributes = [
                "nvars",
                "jacobian_implemented",
                "apply_jacobian_implemented",
            ]
            for name in other_attributes:
                setattr(self, name, getattr(self._model, name))

        def nqoi(self) -> int:
            return self._active_qoi_indices.shape[0]

        def _values(self, samples: Array) -> Array:
            vals = self._model(samples)[:, self._active_qoi_indices]
            return vals

        def jacobian(self, samples: Array) -> Array:
            jac = self._model.jacobian(samples)
            return jac[self._active_qoi_indices, :]

        def apply_jacobian(self, samples: Array, vec: Array) -> Array:
            return self._model.apply_jacobian(samples, vec)[
                self._active_qoi_indices
            ]

        def __repr__(self) -> str:
            return "{0}(model={1}, nactive_qoi={2})".format(
                self.__class__.__name__, self._model, self.nqoi()
            )

        # TODO: technically I could enable these
        # but for now it is easier not to do so unless a user requests it
        def hessian_implemented(self) -> int:
            return False

        def apply_hessian_implemented(self) -> int:
            return False

        def weighted_hessian_implemented(self) -> int:
            return False

        def apply_weighted_hessian_implemented(self) -> int:
            return False

        def work_tracker(self):
            raise AttributeError(
                "must call the wrapped models version of this function"
            )

        def model_database(self):
            raise AttributeError(
                "must call the wrapped models version of this function"
            )

    wrapped_model = ActiveSetQoIModel(model, active_qoi_indices)

    return wrapped_model
