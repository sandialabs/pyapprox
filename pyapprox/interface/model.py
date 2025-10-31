import requests
import os
import subprocess
import signal
import time
import glob
import tempfile
from abc import ABC, abstractmethod
import multiprocessing
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Union
import matplotlib

import numpy as np
import umbridge
import matplotlib.pyplot as plt

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.util.misc import (
    get_all_sample_combinations,
    unique_matrix_row_indices,
)
from pyapprox.surrogates.affine.multiindex import anova_level_indices
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


class ModelWorkTracker:
    """
    A class for tracking the computational performance of model evaluations.

    The `ModelWorkTracker` class is designed to record and analyze the wall times for various model evaluations, such as values, Jacobians, Hessians, and their vector products. It supports both single-process and multi-process environments, allowing shared memory access for multiprocessing scenarios.

    Parameters
    ----------
    backend : BackendMixin, optional
        The backend used for array operations (e.g., `NumpyMixin`). Defaults to `NumpyMixin`.
    multiproc : bool, optional
        Whether to enable multiprocessing support. Defaults to `False`.
    """

    def __init__(
        self, backend: BackendMixin = NumpyMixin, multiproc: bool = False
    ):
        """
        Initialize the ModelWorkTracker instance.

        Parameters
        ----------
        backend : BackendMixin, optional
            The backend used for array operations (e.g., `NumpyMixin`). Defaults to `NumpyMixin`.
        multiproc : bool, optional
            Whether to enable multiprocessing support. Defaults to `False`.

        Notes
        -----
        - When `multiproc=True`, the `_wall_times` dictionary is created using `multiprocessing.Manager()` to allow shared memory access across processes.
        - Shared memory access can slow down code due to locking mechanisms, so it should only be used when necessary.
        """
        self._bkd = backend
        self.set_active(False)
        # use multiprocessing.Manager() so that the dictionary
        # can be updated by multiple processes when calling
        # multiprocessing.pool
        if multiproc:
            manager = multiprocessing.Manager()
            self._wall_times = manager.dict(
                {
                    "val": manager.list([]),
                    "jac": manager.list([]),
                    "jvp": manager.list([]),
                    "hess": manager.list([]),
                    "hvp": manager.list([]),
                    "whess": manager.list([]),
                    "whvp": manager.list([]),
                }
            )
            # Do not create work_tracker with shared memory
            # unless necessary (e.g. do not do for models that dont use pool)
            # as  multiprocessing.Manager().dict slows down code substantially
            # I think because of lock it places on data access
        else:
            self._wall_times = {
                "val": [],
                "jac": [],
                "jvp": [],
                "hess": [],
                "hvp": [],
                "whess": [],
                "whvp": [],
            }

    def set_active(self, active: bool):
        """
        Set whether tracking is active.

        Parameters
        ----------
        active : bool
            Whether tracking is active.
        """
        self._active = active

    def update(self, eval_name: str, times: Array):
        """
        Update the wall times for a specific type of evaluation.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").
        times : Array
            The wall times to add.

        Notes
        -----
        - Wall times are stored as lists of arrays for efficient concatenation.
        - This method only updates wall times if tracking is active.
        """
        # use list of arrays because hstacking arrays continually is slow
        # only hstack arrays when accessing wall_times
        if self._active:
            self._wall_times[eval_name].append(times)

    def average_wall_time(self, eval_name: str) -> float:
        """
        Compute the average wall time for a specific type of evaluation.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").

        Returns
        -------
        float
            The average wall time for the specified evaluation.

        Notes
        -----
        - Excludes failed evaluations (i.e., `np.nan` values) from the calculation.
        - Returns "?" if no evaluations have been performed for the specified type.
        """
        if self.nevaluations(eval_name) == 0:
            return "?"
        # call self.walltimes() so _wall_times lists are concatenated into
        # an array
        wall_times = self.wall_times()[eval_name]
        # exclude failures from calculation
        wall_times = wall_times[wall_times != np.nan]
        return self._bkd.mean(wall_times)

    def nevaluations(self, eval_name: str) -> int:
        """
        Return the number of evaluations for a specific type of evaluation.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").

        Returns
        -------
        int
            The number of evaluations for the specified type.
        """
        return self.wall_times()[eval_name].shape[0]

    def wall_times(self) -> dict:
        """
        Return the wall times for all types of evaluations.

        Returns
        -------
        dict
            A dictionary containing the wall times for each type of evaluation.

        Notes
        -----
        - Concatenates lists of arrays into a single array for each evaluation type.
        - If no wall times exist for a specific type, returns an empty array for that type.
        """
        wtimes = {}
        for key, item in self._wall_times.items():
            if len(self._wall_times[key]) > 0:
                # must loop over list item so can call hstack with torch
                # when item is a manger.list
                wtimes[key] = self._bkd.hstack([t for t in item])
            else:
                wtimes[key] = self._bkd.zeros((0,))
        return wtimes

    def __repr__(self) -> str:
        return "{0}(\n{1}\n\ttimes=({2})\n)".format(
            self.__class__.__name__,
            ", ".join(
                "n{0}={1}".format(name, self.nevaluations(name))
                for name in self._wall_times.keys()
            ),
            ", ".join(
                "{0}={1}".format(name, self.average_wall_time(name))
                for name in self._wall_times.keys()
            ),
        )


class ModelDataBase:
    """
    Model evaluation database.

    This class provides a database for storing and retrieving evaluation results
    for numerical models. It supports storing results for multiple types of evaluations
    (e.g., values, Jacobians, Hessians) and ensures efficient access to previously computed
    results using hashed sample identifiers.

    Parameters
    ----------
    backend : BackendMixin, optional
        The backend used for array operations (e.g., `NumpyMixin`). Defaults to `NumpyMixin`.
    """

    # TODO Add option to save results to file at certain frequency
    # TODO support different file systems, e.g. pickle, HDF5 etc.
    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the ModelDataBase instance.

        Parameters
        ----------
        backend : BackendMixin, optional
            The backend used for array operations (e.g., `NumpyMixin`). Defaults to `NumpyMixin`.
        """
        self._bkd = backend
        self._samples_dict = {}
        self._values_dict = {
            "val": {},
            "jac": {},
            "jvp": {},
            "hess": {},
            "hvp": {},
            "whess": {},
            "whvp": {},
        }
        self._samples = []
        self._active = False

    def activate(self):
        """
        Activate the use of the database.

        Notes
        -----
        Using a database introduces overhead, which can slow down computationally fast model evaluations.
        This method enables the use of the database for storing and retrieving evaluation results.
        """
        self._active = True

    def isactive(self) -> bool:
        """
        Check whether the database is active.

        Returns
        -------
        bool
            True if the database is active, False otherwise.
        """
        return self._active

    def _hash_sample(self, sample: Array) -> int:
        """
        Compute a hash for a given sample.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample to hash.

        Returns
        -------
        int
            The hash value for the sample.

        Notes
        -----
        - The hash is computed using the byte representation of the sample.
        - This method is used to uniquely identify samples in the database.
        """
        return hash(self._bkd.to_numpy(sample).tobytes())

    def _add_sample(self, eval_name: str, key, sample):
        """
        Add a sample to the database.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").
        key : int
            The hash key for the sample.
        sample : Array (nvars, 1)
            The sample to add.

        Notes
        -----
        - If the sample is not already in the database, it is added to `_samples_dict` and `_samples`.
        """
        if key not in self._samples_dict:
            self._samples_dict[key] = len(self._samples)
            self._samples.append(sample)

    def _add_values(self, eval_name: str, key: int, values: Array):
        """
        Add evaluation results to the database.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").
        key : int
            The hash key for the sample.
        values : Array
            The evaluation results to add.

        Notes
        -----
        - The evaluation results are stored in `_values_dict` under the specified `eval_name`.
        """
        # append values to a list
        self._values_dict[eval_name][key] = self._bkd.copy(values)

    def add_data(self, eval_name: str, samples: Array, values: Array):
        """
        Add evaluation results for multiple samples to the database.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").
        samples : Array (nvars, nsamples)
            The samples used for the evaluation.
        values : Array
            The evaluation results.

        Notes
        -----
        - If the database is not active, this method does nothing.
        - For evaluations other than "val", only one sample can be added at a time.

        Raises
        ------
        ValueError
            If more than one sample is added for evaluations other than "val".
        """
        if not self.isactive():
            return
        for ii in range(samples.shape[1]):
            key = self._hash_sample(samples[:, ii])
            self._add_sample(eval_name, key, samples[:, ii])
            if eval_name == "val":
                self._add_values(eval_name, key, values[ii, :])
            else:
                if samples.shape[1] != 1:
                    raise ValueError("Only supports adding one sample at time")
                self._add_values(eval_name, key, values)

    def get_data(self, eval_name: str, samples: Array) -> Tuple[List, List]:
        """
        Retrieve stored evaluation results for a set of samples.

        Parameters
        ----------
        eval_name : str
            The name of the evaluation (e.g., "val", "jac", "hess").
        samples : Array (nvars, nsamples)
            The samples for which to retrieve evaluation results.

        Returns
        -------
        stored_values : List
            The evaluation results for samples already in the database.
        stored_sample_idx : List
            The indices of samples already in the database.
        new_sample_idx : List
            The indices of samples not in the database.

        Notes
        -----
        - If the database is not active, all samples are treated as new samples.
        """
        if not self.isactive():
            return None, [], self._bkd.arange(samples.shape[1])
        new_sample_idx = []
        stored_sample_idx = []
        stored_values = []
        for ii in range(samples.shape[1]):
            key = self._hash_sample(samples[:, ii])
            values = self._values_dict[eval_name].get(key, None)
            if values is None:
                new_sample_idx.append(ii)
            else:
                stored_values.append(values)
                stored_sample_idx.append(ii)
        return stored_values, stored_sample_idx, new_sample_idx


class Model(ABC):
    """
    Abstract base class for evaluating a model.

    The `Model` class provides a framework for defining models that can evaluate values, Jacobians, Hessians, and other derivatives at a single sample. It requires the implementation of certain abstract methods and provides optional methods for advanced functionality.

    Required Methods
    ----------------
    - `nqoi() -> int`: Returns the number of quantities of interest (QoI) in the model.
    - `nvars() -> int`: Returns the number of variables in the model.
    - `_values(sample: Array) -> Array`: Evaluates the model's values at the given sample. This method must be implemented in subclasses.


    Optional Methods
    ----------------
    - `_jacobian(sample: Array) -> Array`: Computes the Jacobian matrix at the given sample.
    - `_apply_jacobian(sample: Array, vec: Array) -> Array`: Applies the Jacobian matrix to a vector at the given sample.
    - `_hessian(sample: Array) -> Array`: Computes the Hessian matrix at the given sample.
    - `_apply_hessian(sample: Array, vec: Array) -> Array`: Applies the Hessian matrix to a vector at the given sample.
    - `_weighted_hessian(sample: Array, weights: Array) -> Array`: Computes the weighted Hessian matrix at the given sample.
    - `_apply_weighted_hessian(sample: Array, vec: Array, weights: Array) -> Array`: Applies the weighted Hessian matrix to a vector and weights at the given sample.

    Optional Functionality Flags
    ----------------------------
    If the optional methods are implemented in a subclass, the corresponding flags must be set to `True`:
    - `jacobian_implemented() -> bool`: Returns `True` if `_jacobian` is implemented.
    - `apply_jacobian_implemented() -> bool`: Returns `True` if `_apply_jacobian` is implemented.
    - `hessian_implemented() -> bool`: Returns `True` if `_hessian` is implemented.
    - `apply_hessian_implemented() -> bool`: Returns `True` if `_apply_hessian` is implemented.
    - `weighted_hessian_implemented() -> bool`: Returns `True` if `_weighted_hessian` is implemented.
    - `apply_weighted_hessian_implemented() -> bool`: Returns `True` if `_apply_weighted_hessian` is implemented.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the Model instance.

        Parameters
        ----------
        backend : BackendMixin, optional
            The backend used for array operations (e.g., `NumpyMixin`). Defaults to `NumpyMixin`.
        """
        if not hasattr(backend, "isbackend"):
            raise ValueError("backend must be derived from LinAlgBase")
        self._bkd = backend
        self._work_tracker = ModelWorkTracker(self._bkd)
        self._database = ModelDataBase(self._bkd)

    def activate_model_data_base(self):
        """
        Activate the use of a database tracking evaluation meta data.

        Notes
        -----
        Using a database introduces overhead, which can slow down computationally fast model evaluations.
        Activating a database may corrupt auto-differentiation results if used.
        This method raises an error if some samples have already been requested before activating the database.

        Raises
        ------
        RuntimeError
            If samples have already been requested before activating the database.
        """
        self._database.activate()
        self._work_tracker.set_active(True)
        for key, item in self._work_tracker._wall_times.items():
            if len(item) > 0:
                raise RuntimeError(
                    "Activating model data base but some samples have"
                    "already been requested"
                )

    def set_model_history(
        self, database: ModelDataBase, work_tracker: ModelWorkTracker
    ):
        """
        Load a database and work tracker from a previous study to resume tracking
        of evaluation meta data.

        Parameters
        ----------
        database : ModelDataBase
            The database containing the history of model evaluations.
        work_tracker : ModelWorkTracker
            The work tracker containing computational performance metrics.

        Raises
        ------
        RuntimeError
            If the database and work tracker are inconsistent (i.e., they have different numbers of evaluations).
        """
        for key in work_tracker._wt_dict:
            if not work_tracker.nevaluations(key) != len(
                database._values_dict[key]
            ):
                raise RuntimeError(
                    "Database and worktracker must be consistent, but have "
                    "different numbers of evaluations"
                )
        self._database = database
        self._work_tracker = work_tracker

    def apply_jacobian_implemented(self) -> bool:
        """
        Check if the apply_jacobian function is implemented.

        Returns
        -------
        flag : bool
            True if apply_jacobian is implemented, False otherwise.
        """
        return False

    def jacobian_implemented(self) -> bool:
        """
        Check if the jacobian function is implemented.

        Returns
        -------
        flag : bool
            True if jacobian is implemented, False otherwise.
        """
        return False

    def apply_hessian_implemented(self) -> bool:
        """
        Check if the apply_hessian function is implemented.

        Returns
        -------
        flag : bool
            True if apply_hessian is implemented, False otherwise.
        """
        return False

    def hessian_implemented(self) -> bool:
        """
        Check if the hessian function is implemented.

        Returns
        -------
        flag : bool
            True if hessian is implemented, False otherwise.
        """
        return False

    def apply_weighted_hessian_implemented(self) -> bool:
        """
        Check if the apply_weighted_hessian function is implemented.
        The weighted hessian is typicall used when a model is a contstraint in
        an optimizer. The weights are typically lagrange multipliers.

        Returns
        -------
        flag : bool
            True if apply_weighted_hessian is implemented, False otherwise.
        """
        return False

    def weighted_hessian_implemented(self) -> bool:
        """
        Check if the weighted_hessian function is implemented.
        The weighted hessian is typicall used when a model is a contstraint in
        an optimizer. The weights are typically lagrange multipliers.

        Returns
        -------
        flag : bool
            True if weighted_hessian is implemented, False otherwise.
        """
        return False

    @abstractmethod
    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi: int
            The number of quantities of interest.
        """
        raise NotImplementedError

    @abstractmethod
    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars: int
            The number of variables.
        """
        raise NotImplementedError

    @abstractmethod
    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model's values at a set of samples.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            The input samples used to evaluate the model.

        Returns
        -------
        values : Array (nsamples, nqoi)
            The model outputs at each sample.
        """
        raise NotImplementedError("Must implement self._values")

    def _check_values_shape(self, samples: Array, vals: Array):
        """
        Check that the shape of the values returned by the model is correct.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
        The input samples used to evaluate the model.
        vals : Array (nsamples, nqoi)
        The values returned by the model.

        Raises
        ------
        RuntimeError
            If the shape of `vals` does not match `(nsamples, nqoi)`.
        """
        if vals.shape != (samples.shape[1], self.nqoi()):
            raise RuntimeError(
                "{0}: values had shape {1} but should have shape {2}".format(
                    self, vals.shape, (samples.shape[1], self.nqoi())
                )
            )

    def _new_values(self, samples: Array) -> Array:
        """
        Evaluate the model at new samples and update the database.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            The input samples used to evaluate the model.

        Returns
        -------
        values : Array (nsamples, nqoi)
            The model outputs at each sample.
        """
        t0 = time.time()
        vals = self._values(samples)
        t1 = time.time()
        nsamples = samples.shape[1]
        # Make assumption that each sample took the same time
        times = self._bkd.full((nsamples,), (t1 - t0) / nsamples)
        self._work_tracker.update("val", times)
        self._check_values_shape(samples, vals)
        self._database.add_data("val", samples, vals)
        return vals

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the model at a set of samples.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            The model inputs used to evaluate the model

        Returns
        -------
        values : Array (nsamples, nqoi)
            The model outputs returned by the model at each sample
        """
        if samples.shape[0] != self.nvars():
            raise ValueError(
                f"{self}: samples has the wrong number of rows. "
                f"Was {samples.shape[0]}, should be {self.nvars()}"
            )
        stored_values, stored_idx, new_idx = self._database.get_data(
            "val", samples
        )
        if len(stored_idx) == 0:
            return self._new_values(samples)

        if len(new_idx) == 0:
            return self._bkd.vstack(stored_values)

        vals = self._bkd.empty((samples.shape[1], self.nqoi()))
        new_samples = samples[:, new_idx]
        vals[new_idx] = self._new_values(new_samples)
        vals[stored_idx] = self._bkd.vstack(stored_values)
        return vals

    def _check_sample_shape(self, sample: Array):
        """
        Check that the shape of a single sample is correct.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The input sample.

        Raises
        ------
        ValueError
            If the shape of `sample` is not `(nvars, 1)`.
        """
        if sample.shape != (self.nvars(), 1):
            raise ValueError(
                "{0}: sample must have shape {1} but had shape {2}".format(
                    self, (self.nvars(), 1), sample.shape
                )
            )

    def _check_samples_shape(self, sample: Array):
        """
        Check that the shape of multiple samples is correct.

        Parameters
        ----------
        sample : Array (nvars, nsamples)
            The input samples.

        Raises
        ------
        ValueError
            If the number of rows in `sample` does not match `nvars`.
        """
        if sample.shape[0] != self.nvars():
            raise ValueError(
                "{0}: sample must have nrows={1} but had shape {2}".format(
                    self, self.nvars(), sample.shape
                )
            )

    def _check_vec_shape(self, sample: Array, vec: Array):
        """
        Check that the shape of a vector is consistent with the sample.

        Parameters
        ----------
        sample : Array (nvars, nsamples)
            The input samples.
        vec : Array (nvars, nsamples)
            The vector to check.

        Raises
        ------
        ValueError
            If `vec` is not a 2D array or its shape is inconsistent with `sample`.
        """
        if vec.ndim != 2:
            raise ValueError(
                "{0}: vec is not a 2D array, has shape {1}".format(
                    self, vec.shape
                )
            )
        if sample.shape[0] != vec.shape[0]:
            raise ValueError(
                "{0}: sample.shape {1} and vec.shape {2} are inconsistent".format(
                    self, sample.shape, vec.shape
                )
            )

    def _check_weights_shape(self, weights: Array):
        """
        Check that the shape of weights is correct.

        Parameters
        ----------
        weights : Array (nqoi, nsamples)
            The weights to check.

        Raises
        ------
        ValueError
            If `weights` is not a 2D array or its shape is inconsistent with `nqoi`.
        """
        if weights.ndim != 2:
            raise ValueError(
                "weights is not a 2D array, has shape {0}".format(
                    weights.shape
                )
            )
        if weights.shape[0] != self.nqoi():
            raise ValueError("weights has the wrong shape")

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian matrix at a single sample.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian.

        Returns
        -------
        jac : Array (nqoi, nvars)
            The Jacobian matrix.


        Notes
        -------
        Default to using autograd to compute Jacobian.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.jacobian_implemented is False
        by default
        """
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError(
                f"{self._bkd.__name__} does not support autodiff"
            )
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[0], sample[:, 0]
        )

    def _check_jacobian_shape(self, jac: Array, sample: Array):
        """ "
        Check that the shape of the Jacobian matrix is correct.

        Parameters
        ----------
        jac : Array (nqoi, nvars)
            The Jacobian matrix.
        sample : Array (nvars, 1)
            The sample at which the Jacobian was computed.
        """
        if jac.shape != (self.nqoi(), sample.shape[0]):
            raise RuntimeError(
                "{0} Jacobian returned by _jacobian has shape {1}"
                " but must be {2}".format(
                    self, jac.shape, (self.nqoi(), sample.shape[0])
                )
            )

    def jacobian(self, sample: Array) -> Array:
        """
        Evaluate the jacobian of the model at a sample.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian

        Returns
        -------
        jac : Array (nqoi, nvars)
            The Jacobian matrix
        """
        # Note if calling this function thousands of times
        # using a trivial model and jacobian the cost of the checks
        # and database updates will cause code to be substantially
        # slower than using self._bkd.jacobian (if it is supported)
        if not self.jacobian_implemented():
            raise NotImplementedError(f"{self} _jacobian not implemented")

        self._check_sample_shape(sample)
        jac, stored_idx, new_idx = self._database.get_data("jac", sample)
        if len(stored_idx) == 1:
            self._check_jacobian_shape(jac[0], sample)
            return jac[0]
        t0 = time.time()
        jac = self._jacobian(sample)
        t1 = time.time()
        times = self._bkd.array([t1 - t0])
        self._work_tracker.update("jac", times)
        self._check_jacobian_shape(jac, sample)
        self._database.add_data("jac", sample, jac)
        return jac

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the Jacobian vector product using a user-provided function.

        Default behavior uses autograd to compute the Jacobian vector product.
        The user must ensure that all methods invoked by `__call__` are differentiable.
        By default, `apply_jacobian_implemented` is set to `False`.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian vector product.
        vec : Array (nvars, 1)
            The vector to multiply with the Jacobian.

        Returns
        -------
        result : Array (nqoi, 1)
            The result of the Jacobian vector product.
        """
        if not self._bkd.jvp_implemented():
            raise NotImplementedError
        return self._bkd.jvp(
            lambda x: self._values(x[:, None])[0], sample[:, 0], vec[:, 0]
        )

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the matrix vector product of the Jacobian with a vector.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian

        vec : np.narray (nvars, 1)
            The vector

        Returns
        -------
        result : Array (nqoi, 1)
            The dot product of the Jacobian with the vector
        """
        if (
            not self.apply_jacobian_implemented()
            and not self.jacobian_implemented()
        ):
            raise RuntimeError(
                f"{self}: apply_jacobian and jacobian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self.apply_jacobian_implemented():
            jvp, stored_idx, new_idx = self._database.get_data("jvp", sample)
            if len(stored_idx) == 1:
                return jvp[0]
            t0 = time.time()
            jvp = self._apply_jacobian(sample, vec)
            t1 = time.time()
            times = self._bkd.array([t1 - t0])
            self._work_tracker.update("jvp", times)
            self._database.add_data("jvp", sample, jvp)
            return jvp
        return self.jacobian(sample) @ vec

    def work_tracker(self) -> ModelWorkTracker:
        """
        Return the work tracker associated with the model.

        Returns
        -------
        work_tracker : ModelWorkTracker
            The work tracker containing computational performance metrics.
        """
        return self._work_tracker

    def model_database(self) -> ModelDataBase:
        """
        Return the database associated with the model.

        Returns
        -------
        database : ModelDataBase
            The database containing the history of model evaluations.
        """
        return self._database

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the Hessian vector product using a user-provided function.

        Default behavior uses autograd to compute the Hessian vector product.
        The user must ensure that all methods invoked by `__call__` are differentiable.
        By default, `apply_hessian_implemented` is set to `False`.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Hessian vector product.
        vec : Array (nvars, 1)
            The vector to multiply with the Hessian.

        Returns
        -------
        result : Array (nvars, 1)
            The result of the Hessian vector product.
        """
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            lambda x: self.__call__(x[:, None])[0],
            sample[:, 0],
            vec[:, 0],
        )[:, None]

    def _check_hvp_shape(self, hvp: Array, sample: Array):
        """
        Check that the shape of the Hessian vector product is correct.

        Parameters
        ----------
        hvp : Array (nvars, 1)
            The Hessian vector product.
        sample : Array (nvars, 1)
            The sample at which the Hessian vector product was computed.
        """
        if hvp.shape != (sample.shape[0], 1):
            raise RuntimeError(
                f"{self}:"
                "Hessian vector product returned by _apply_hessian "
                "has the wrong shape. "
                "was {0} but must be {1}".format(
                    hvp.shape, (sample.shape[0], 1)
                )
            )

    def apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the matrix vector product of the Hessian with a vector.

        Parameters
        ----------
        sample : array (nvars, 1)
            The sample at which to compute the Hessian

        vec : array (nvars, 1)
            The vector

        Returns
        -------
        result : array (nvars, 1)
            The dot product of the Hessian with the vector
        """
        if (
            not self.apply_hessian_implemented()
            and not self.hessian_implemented()
        ):
            raise RuntimeError("apply_hessian and hessian are not implemented")
        if self.nqoi() > 1:
            raise ValueError("apply_hessian cannot be used when nqoi > 1")
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self.apply_hessian_implemented():
            hvp, stored_idx, new_idx = self._database.get_data("hvp", sample)
            if len(stored_idx) == 1:
                return hvp[0]
            t0 = time.time()
            hvp = self._apply_hessian(sample, vec)
            t1 = time.time()
            self._check_hvp_shape(hvp, sample)
            times = self._bkd.array([t1 - t0])
            self._work_tracker.update("hvp", times)
            self._database.add_data("hvp", sample, hvp)
            return hvp
        return self.hessian(sample)[0] @ vec

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian matrix using a user-provided function.

        Default behavior uses autograd to compute the Hessian matrix.
        The user must ensure that all methods invoked by `__call__` are differentiable.
        By default, `hessian_implemented` is set to `False`.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Hessian.

        Returns
        -------
        hess : Array (nqoi, nvars, nvars)
            The Hessian matrix.
        """
        if not self._bkd.hessian_implemented():
            raise NotImplementedError
        return self._bkd.hessian(
            lambda x: self.__call__(x[:, None])[0], sample[:, 0]
        )[None]

    def _check_hessian_shape(self, hess: Array, sample: Array):
        """
        Check that the shape of the Hessian matrix is correct.

        Parameters
        ----------
        hess : Array (nqoi, nvars, nvars)
            The Hessian matrix.
        sample : Array (nvars, 1)
            The sample at which the Hessian was computed.
        """
        if hess.shape != (self.nqoi(), sample.shape[0], sample.shape[0]):
            raise RuntimeError(
                "Hessian returned by _hessian has the wrong shape. "
                "was {0} but must be {1}".format(
                    hess.shape, (self.nqoi(), sample.shape[0], sample.shape[0])
                )
            )

    def hessian(self, sample: Array) -> Array:
        """
        Evaluate the hessian of the model at a sample.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian

        Returns
        -------
        hess : Array (nqoi, nvars, nvars)
            The Jacobian matrix
        """
        if not self.hessian_implemented():
            raise NotImplementedError("Hessian not implemented")
        if not self.hessian_implemented() and self.nqoi() > 1:
            raise ValueError("apply_hessian cannot be used when nqoi > 1")

        self._check_sample_shape(sample)

        hess, stored_idx, new_idx = self._database.get_data("hess", sample)
        if len(stored_idx) == 1:
            return hess[0]
        t0 = time.time()
        hess = self._hessian(sample)
        t1 = time.time()
        times = self._bkd.array([t1 - t0])
        self._work_tracker.update("hess", times)
        self._database.add_data("hess", sample, hess)
        self._check_hessian_shape(hess, sample)
        return hess

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nqoi={2})".format(
            self.__class__.__name__, self.nvars(), self.nqoi()
        )

    def apply_weighted_jacobian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Compute the matrix vector product of the Jacobian,
        of a weighted sum of the QoI, with a vector.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian

        vec : np.narray (nvars, 1)
            The vector

        weights: array (nqoi, 1)
            Weights defining combination of quantities of interest

        Returns
        -------
        result : float
            The dot product of the weighted Jacobian with the vector
        """
        if (
            not self.apply_jacobian_implemented()
            and not self.jacobian_implemented()
        ):
            raise RuntimeError(
                "apply_jacobian and jacobian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        self._check_weights_shape(weights)
        if self.apply_jacobian_implemented():
            return self.apply_jacobian(sample, vec).T @ weights
        return (self.jacobian(sample) @ vec).T @ weights

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        User provided function to compute the weighted Hessian vector product.

        Default to using autograd to compute Jacobian vector product.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.apply_weighed_hessian is False
        by default
        """
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            lambda x: self.__call__(x[:, None])[0] @ weights[:, 0],
            sample[:, 0],
            vec[:, 0],
        )

    def apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Compute the matrix vector product of the Hessian,
        of a weighted combination of the QoI, with a vector.

        Parameters
        ----------
        sample : array (nvars, 1)
            The sample at which to compute the Hessian

        vec : array (nvars, 1)
            The vector

        weights: array (nqoi, 1)
            Weights defining combination of quantities of interest

        Returns
        -------
        result : array (nvars, 1)
            The dot product of the weighted Hessian with the vector
        """
        if (
            not self.apply_weighted_hessian_implemented()
            and not self.hessian_implemented()
            and not self.weighted_hessian_implemented()
        ):
            raise RuntimeError(
                "apply_weighted_hessian and hessian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        self._check_weights_shape(weights)
        if self.apply_weighted_hessian_implemented():
            whvp, stored_idx, new_idx = self._database.get_data("whvp", sample)
            if len(stored_idx) == 1:
                return whvp[0]
            t0 = time.time()
            whvp = self._apply_weighted_hessian(sample, vec, weights)
            t1 = time.time()
            times = self._bkd.array([t1 - t0])
            self._work_tracker.update("whvp", times)
            self._database.add_data("whvp", sample, whvp)
            return whvp
        if self.weighted_hessian_implemented():
            return self.weighted_hessian(sample, weights) @ vec
        return (weights.T @ (self.hessian(sample) @ vec[:, 0])).T

    def _weighted_hessian(self, sample: Array, weights: Array) -> Array:
        """Compute the weighted Hessian matrix.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the weighted Hessian.
        weights : Array (nqoi, 1)
            The weights defining the combination of quantities of interest.

        Returns
        -------
        weighted_hessian : Array (nvars, nvars)
            The weighted Hessian matrix.
        """
        raise NotImplementedError

    def weighted_hessian(self, sample: Array, weights: Array) -> Array:
        """
        Compute the weighted Hessian matrix.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the weighted Hessian.
        weights : Array (nqoi, 1)
            The weights defining the combination of quantities of interest.

        Returns
        -------
        weighted_hessian : Array (nvars, nvars)
            The weighted Hessian matrix.
        """
        if (
            not self.weighted_hessian_implemented()
            and not self.hessian_implemented()
        ):
            raise RuntimeError(
                "weighted_hessian and hessian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_weights_shape(weights)
        if self.weighted_hessian_implemented():
            whess, stored_idx, new_idx = self._database.get_data(
                "whess", sample
            )
            if len(stored_idx) == 1:
                return whess[0]
            t0 = time.time()
            whess = self._weighted_hessian(sample, weights)
            t1 = time.time()
            times = self._bkd.array([t1 - t0])
            self._work_tracker.update("whess", times)
            self._database.add_data("whess", sample, whess)
            return whess
        hess = self.hessian(sample)
        return self._bkd.einsum("i,ijk->jk", weights[:, 0], hess)

    def _check_apply(
        self,
        sample: Array,
        symb: str,
        fun: callable,
        apply_fun: callable,
        fd_eps: Array = None,
        direction: Array = None,
        relative: bool = True,
        disp: bool = False,
        args=[],
    ) -> Array:
        """
        Compare the result of an apply function with finite difference approximations.

        This function computes directional gradients using finite differences and compares them to the gradients computed by the provided `apply_fun`. It is useful for verifying the correctness of `apply_jacobian` or `apply_hessian`.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the gradients.
        symb : str
            A symbol representing the type of gradient being checked (e.g., "J" for Jacobian, "H" for Hessian).
        fun : callable
            The function used to compute the values at the sample.
        apply_fun : callable
            The function used to compute the directional gradient.
        fd_eps : Array, optional
            The finite difference step sizes. Defaults to logarithmically spaced values.
        direction : Array, optional
            The direction vector for computing directional gradients. Defaults to a random normalized vector.
        relative : bool, optional
            Whether to compute relative errors. Defaults to True.
        disp : bool, optional
            Whether to display the errors during computation. Defaults to False.
        args : list, optional
            Additional arguments passed to `fun` and `apply_fun`.

        Returns
        -------
        errors : Array
            The computed errors between finite difference gradients and `apply_fun` gradients.
        """
        if sample.ndim != 2:
            raise ValueError(
                "sample with shape {0} must be 2D array".format(sample.shape)
            )
        if fd_eps is None:
            fd_eps = self._bkd.flip(self._bkd.logspace(-13, 0, 14))
        if direction is None:
            nvars = sample.shape[0]
            direction = np.random.normal(0, 1, (nvars, 1))
            direction /= np.linalg.norm(direction)
            direction = self._bkd.asarray(direction)

        row_format = "{:<12} {:<25} {:<25} {:<25}"
        headers = [
            "Eps",
            "norm({0}v)".format(symb),
            "norm({0}v_fd)".format(symb),
            "Rel. Errors" if relative else "Abs. Errors",
        ]
        if disp:
            print(row_format.format(*headers))
        row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
        errors = []
        val = fun(sample, *args)
        directional_grad = apply_fun(sample, direction, *args)
        for ii in range(fd_eps.shape[0]):
            sample_perturbed = self._bkd.copy(sample) + fd_eps[ii] * direction
            perturbed_val = fun(sample_perturbed, *args)
            fd_directional_grad = (perturbed_val - val) / fd_eps[ii]
            errors.append(
                self._bkd.norm(
                    fd_directional_grad.reshape(directional_grad.shape)
                    - directional_grad
                )
            )
            if relative:
                errors[-1] /= self._bkd.norm(directional_grad)
            if disp:
                print(
                    row_format.format(
                        fd_eps[ii],
                        self._bkd.norm(directional_grad),
                        self._bkd.norm(fd_directional_grad),
                        errors[ii],
                    )
                )
        return self._bkd.asarray(errors)

    def check_apply_jacobian(
        self,
        sample: Array,
        fd_eps: Array = None,
        direction: Array = None,
        relative: bool = True,
        disp: bool = False,
    ):
        """
        Compare `apply_jacobian` with finite difference approximations.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian.
        fd_eps : Array, optional
            The finite difference step sizes. Defaults to logarithmically spaced values.
        direction : Array, optional
            The direction vector for computing directional gradients. Defaults to a random normalized vector.
        relative : bool, optional
            Whether to compute relative errors. Defaults to True.
        disp : bool, optional
            Whether to display the errors during computation. Defaults to False.

        Returns
        -------
        errors : Array
            The computed errors between finite difference gradients and `apply_jacobian` gradients.

        """
        if (
            not self.apply_jacobian_implemented()
            and not self.jacobian_implemented()
        ):
            print(self)
            raise RuntimeError(
                "Cannot check apply_jacobian because it not implemented"
            )
        return self._check_apply(
            sample,
            "J",
            self,
            self.apply_jacobian,
            fd_eps,
            direction,
            relative,
            disp,
        )

    def _weighted_jacobian(self, sample: Array, weights: Array) -> Array:
        """
        Compute the weighted Jacobian matrix. only used by check_apply_hessian

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian.
        weights : Array (nqoi, 1)
            The weights defining the combination of quantities of interest.

        Returns
        -------
        weighted_jacobian : Array (1, nvars)
            The weighted Jacobian matrix.
        """
        return weights.T @ self.jacobian(sample)

    def _jacobian_from_apply_jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian matrix using `apply_jacobian`.

        This function computes the Jacobian by applying `apply_jacobian` to unit vectors. It is used internally for checking `apply_hessian`.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian.

        Returns
        -------
        jacobian : Array (nqoi, nvars)
            The Jacobian matrix.
        """
        if self.jacobian_implemented():
            return self.jacobian(sample)
        nvars = sample.shape[0]
        actions = []
        for ii in range(nvars):
            vec = self._bkd.zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(self.apply_jacobian(sample, vec))
        return self._bkd.hstack(actions)

    def check_apply_hessian(
        self,
        sample: Array,
        fd_eps: Array = None,
        direction: Array = None,
        relative: bool = True,
        disp: bool = False,
        weights: bool = None,
    ):
        """
        Compare `apply_hessian` or `apply_weighted_hessian` with finite difference approximations.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Hessian.
        fd_eps : Array, optional
            The finite difference step sizes. Defaults to logarithmically spaced values.
        direction : Array, optional
            The direction vector for computing directional gradients. Defaults to a random normalized vector.
        relative : bool, optional
            Whether to compute relative errors. Defaults to True.
        disp : bool, optional
            Whether to display the errors during computation. Defaults to False.
        weights : Array, optional
            The weights defining the combination of quantities of interest. If `None`, checks `apply_hessian`.

        Returns
        -------
        errors : Array
            The computed errors between finite difference gradients and `apply_hessian` or `apply_weighted_hessian` gradients.
        """
        if weights is None:
            if (
                not self.apply_hessian_implemented()
                and not self.hessian_implemented()
            ):
                raise RuntimeError(
                    "Cannot check apply_hessian because it is not implemented"
                )
            return self._check_apply(
                sample,
                "H",
                self._jacobian_from_apply_jacobian,
                self.apply_hessian,
                fd_eps,
                direction,
                relative,
                disp,
            )

        if (
            not self.apply_weighted_hessian_implemented()
            and not self.hessian_implemented()
            and not self.weighted_hessian_implemented()
        ):
            raise RuntimeError(
                "Cannot check apply_weighted_hessian because not implemented"
            )
        return self._check_apply(
            sample,
            "H",
            self._weighted_jacobian,
            self.apply_weighted_hessian,
            fd_eps,
            direction,
            relative,
            disp,
            (weights,),
        )

    def approx_jacobian(
        self, sample: Array, eps: float = np.sqrt(np.finfo(float).eps)
    ) -> Array:
        """
        Compute the Jacobian matrix using finite differences.

        Parameters
        ----------
        sample : Array (nvars, 1)
            The sample at which to compute the Jacobian.
        eps : float, optional
            The perturbation size for finite differences. Defaults to the square root of machine epsilon.

        Returns
        -------
        jac : Array (nqoi, nvars)
            The Jacobian matrix computed using finite differences.
        """
        self._check_sample_shape(sample)
        nvars = sample.shape[0]
        val = self(sample)
        nqoi = val.shape[1]
        jac = self._bkd.zeros([nqoi, nvars])
        dx = self._bkd.zeros((nvars, 1))
        for ii in range(nvars):
            dx[ii] = eps
            val_perturbed = self(sample + dx)
            jac[:, ii] = (val_perturbed - val) / eps
            dx[ii] = 0.0
        return jac

    def _plot_surface_1d(
        self,
        ax: matplotlib.axes.Axes,
        qoi: int,
        plot_limits: tuple,
        npts_1d: int,
        **kwargs,
    ):
        """
        Plot a 1D surface of the model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the surface.
        qoi : int
            The quantity of interest to plot.
        plot_limits : tuple
            The limits of the plot (min, max).
        npts_1d : int
            The number of points to use for the plot.
        **kwargs : dict
            Additional arguments passed to the plot function.
        """
        plot_xx = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        ax.plot(plot_xx[0], self.__call__(plot_xx), **kwargs)

    def meshgrid_samples(
        self, plot_limits: Array, npts_1d: Union[Array, int] = 51
    ) -> Array:
        """
        Generate a meshgrid of samples for 2D plotting.

        Parameters
        ----------
        plot_limits : Array (2, 2)
            The limits of the plot for each variable.
        npts_1d : int or Array, optional
            The number of points to use for each variable. Defaults to 51.

        Returns
        -------
        X : Array
            The meshgrid for the first variable.
        Y : Array
            The meshgrid for the second variable.
        pts : Array
            The flattened meshgrid samples.
        """
        if self.nvars() != 2:
            raise RuntimeError(f"nvars = {self.nvars()} but must be 2")
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        return X, Y, pts

    def _plot_surface_2d(
        self,
        ax: matplotlib.axes.Axes,
        qoi: int,
        plot_limits: Array,
        npts_1d: Array,
        **kwargs,
    ):
        """
        Plot a 2D surface of the model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the surface (must use 3D projection).
        qoi : int
            The quantity of interest to plot.
        plot_limits : Array (4,)
            The limits of the plot for each variable.
        npts_1d : int or Array
            The number of points to use for each variable.
        **kwargs : dict
            Additional arguments passed to the plot_surface function.

        Returns
        -------
        surface : matplotlib.surface.Surface
            The plotted surface.
        """
        if ax.name != "3d":
            raise ValueError("ax must use 3d projection")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        # X = self._bkd.to_numpy(X)
        # Y = self._bkd.to_numpy(Y)
        # Z = self._bkd.to_numpy(Z)
        return ax.plot_surface(X, Y, Z, **kwargs)

    def plot_surface(
        self,
        ax: matplotlib.axes.Axes,
        plot_limits: Array,
        qoi: int = 0,
        npts_1d: Array = 51,
        **kwargs,
    ):
        """
        Plot the surface of the model for 1D or 2D inputs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the surface.
        plot_limits : Array
            The limits of the plot for each variable.
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        npts_1d : int or list, optional
            The number of points to use for each variable. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the plotting functions.

        """
        if self.nvars() > 3:
            raise RuntimeError("Cannot plot indices when nvars >= 3.")

        if not isinstance(npts_1d, list):
            npts_1d = [npts_1d] * self.nvars()

        if len(npts_1d) != self.nvars():
            raise ValueError("npts_1d must be a list")

        plot_surface_funs = {
            1: self._plot_surface_1d,
            2: self._plot_surface_2d,
        }
        plot_surface_funs[self.nvars()](
            ax, qoi, plot_limits, npts_1d, **kwargs
        )

    def get_plot_axis(
        self, figsize: Tuple = (8, 6), surface: bool = False
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Get the plot axis for 1D or 2D surfaces.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure. Defaults to (8, 6).
        surface : bool, optional
            Whether to use a 3D axis for surface plots. Defaults to False.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axis object.
        """
        if self.nvars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def plot_contours(
        self,
        ax: matplotlib.axes.Axes,
        plot_limits: Array,
        qoi: int = 0,
        npts_1d: Array = 51,
        **kwargs,
    ):
        """
        Plot contours of the model for 2D inputs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the contours.
        plot_limits : Array (2, 2)
            The limits of the plot for each variable.
        qoi : int, optional
            The quantity of interest to plot. Defaults to 0.
        npts_1d : int or Array, optional
            The number of points to use for each variable. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the contourf function.

        Returns
        -------
        contour : matplotlib.contour.QuadContourSet
            The plotted contours.

        Raises
        ------
        ValueError
            If `nvars != 2`.
        """
        if self.nvars() != 2:
            raise ValueError("Can only plot contours for 2D functions")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        return ax.contourf(X, Y, Z, **kwargs)

    def _2d_cross_section_values(
        self, nominal_sample: Array, id1: int, id2: int, pts: Array
    ) -> Array:
        """
        Compute the model values for a 2D cross-section.

        This function evaluates the model at a set of points in a 2D cross-section defined by two variable indices.

        Parameters
        ----------
        nominal_sample : Array (nvars, 1)
            The nominal sample used as a baseline for the cross-section.
        id1 : int
            The index of the first variable defining the cross-section.
        id2 : int
            The index of the second variable defining the cross-section.
        pts : Array (2, npts)
            The points in the 2D cross-section.

        Returns
        -------
        vals : Array (npts, nqoi)
            The model values at the points in the cross-section.
        """
        samples = []
        for pt in pts.T:
            sample = self._bkd.copy(nominal_sample)
            sample[[id1, id2], 0] = pt
            samples.append(sample)
        np.set_printoptions(precision=16)
        samples = self._bkd.hstack(samples)
        vals = self(samples)
        return vals

    def _1d_cross_section_values(
        self, nominal_sample: Array, id1: int, pts: Array
    ) -> Array:
        """
        Compute the model values for a 1D cross-section.

        This function evaluates the model at a set of points in a 1D cross-section defined by a single variable index.

        Parameters
        ----------
        nominal_sample : Array (nvars, 1)
            The nominal sample used as a baseline for the cross-section.
        id1 : int
            The index of the variable defining the cross-section.
        pts : Array (1, npts)
            The points in the 1D cross-section.

        Returns
        -------
        vals : Array (npts, nqoi)
            The model values at the points in the cross-section.
        """
        samples = self._bkd.tile(nominal_sample, (1, pts.shape[1]))
        samples[id1] = pts[0]
        return self(samples)

    def plot_cross_section(
        self,
        nominal_sample: Array,
        bounds: Array,
        ax,
        id1: int,
        id2: int,
        npts_1d=51,
        **kwargs,
    ):
        """
        Plot a 2D cross-section of the model.

        Parameters
        ----------
        nominal_sample : Array (nvars, 1)
            The nominal sample used as a baseline for the cross-section.
        bounds : Array (nvars, 2)
            The bounds for each variable.
        ax : matplotlib.axes.Axes
            The axis on which to plot the cross-section.
        id1 : int
            The index of the first variable defining the cross-section.
        id2 : int
            The index of the second variable defining the cross-section.
        npts_1d : int, optional
            The number of points to use for each variable. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the contourf function.

        Returns
        -------
        im : matplotlib.contour.QuadContourSet
            The plotted cross-section.
        """
        plot_limits = self._bkd.hstack((bounds[id1], bounds[id2]))
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        active_pt = self._bkd.copy(nominal_sample[[id1, id2], 0])
        Z = self._bkd.reshape(
            self._2d_cross_section_values(nominal_sample, id1, id2, pts)[:, 0],
            X.shape,
        )
        im = ax.contourf(X, Y, Z, **kwargs)
        ax.plot(*active_pt, "ko", ms=20)
        return im

    def get_all_variable_pairs(self) -> Array:
        """
        Get all pairs of variables for cross-section plotting.

        Returns
        -------
        variable_pairs : Array (n_pairs, 2)
            An array of variable pairs, where each row contains two indices representing a pair of variables.

        Notes
        -----
        - The first column of `variable_pairs` varies fastest to ensure lower triangular matrix plotting.
        """
        variable_pairs = self._bkd.asarray(
            anova_level_indices(self.nvars(), 2)
        )
        # make first column values vary fastest so we plot lower triangular
        # matrix of subplots
        variable_pairs[:, 0], variable_pairs[:, 1] = (
            self._bkd.copy(variable_pairs[:, 1]),
            self._bkd.copy(variable_pairs[:, 0]),
        )
        return variable_pairs

    def plot_cross_sections(
        self,
        nominal_sample: Array,
        bounds: Array,
        variable_pairs: List[Tuple[int, int]] = None,
        npts_1d=51,
        **kwargs,
    ):
        """
        Plot cross-sections of the model for all variable pairs.

        Parameters
        ----------
        nominal_sample : Array (nvars, 1)
            The nominal sample used as a baseline for the cross-sections.
        bounds : Array (nvars, 2)
            The bounds for each variable.
        variable_pairs : List[Tuple[int, int]], optional
            The pairs of variables to plot. If `None`, all pairs are plotted. Defaults to `None`.
        npts_1d : int, optional
            The number of points to use for each variable. Defaults to 51.
        **kwargs : dict
            Additional arguments passed to the plotting functions.

        Returns
        -------
        axs : numpy.ndarray
            The array of axes used for plotting.
        ims : list
            The list of plotted cross-sections.
        """
        if nominal_sample.shape != (self.nvars(), 1):
            raise ValueError(
                f"nominal_sample must have shape {(self.nvars(), 1)}"
            )
        if bounds.shape != (nominal_sample.shape[0], 2):
            raise ValueError(
                f"{bounds.shape=} must be {(nominal_sample.shape[0], 2)}"
            )
        if variable_pairs is None:
            # define all 2d cross sections
            variable_pairs = self.get_all_variable_pairs()
            # add 1d cross sections
            variable_pairs = self._bkd.vstack(
                (
                    self._bkd.array([[ii, ii] for ii in range(self.nvars())]),
                    variable_pairs,
                )
            )
        if variable_pairs.shape[1] != 2:
            raise ValueError("Variable pairs has the wrong shape")
        nfig_rows, nfig_cols = self._bkd.max(variable_pairs + 1, axis=0)
        fig, axs = plt.subplots(nfig_rows, nfig_cols, sharex="col")
        if nfig_rows == 1:
            axs = [axs]
        for ax_row in axs:
            for ax in ax_row:
                ax.axis("off")
        ims = []
        for ii, pair in enumerate(variable_pairs):
            print(f"plotting cross section {pair}")
            if pair[0] == pair[1]:
                plot_xx = self._bkd.linspace(*bounds[pair[0]], npts_1d)[
                    None, :
                ]
                im = axs[pair[0]][pair[1]].plot(
                    plot_xx[0],
                    self._1d_cross_section_values(
                        nominal_sample, pair[0], plot_xx
                    ),
                )
            else:
                im = self.plot_cross_section(
                    nominal_sample,
                    bounds,
                    axs[pair[0]][pair[1]],
                    pair[0],
                    pair[1],
                    npts_1d,
                    **kwargs,
                )
            ims.append(im)
        return axs, ims


class SingleSampleModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _evaluate(self, sample):
        """
        Evaluat the model at a single sample

        Parameters
        ----------
        sample: Array (nvars, 1)
            The sample use to evaluate the model

        Returns
        -------
        values : Array (1, nqoi)
            The model outputs returned by the model when evaluated
            at the sample
        """
        raise NotImplementedError

    def _new_values(self, samples: Array) -> Array:
        nvars, nsamples = samples.shape
        t0 = time.time()
        values_0 = self._evaluate(samples[:, :1])
        t1 = time.time()
        times = [t1 - t0]
        if values_0.ndim != 2 or values_0.shape[0] != 1:
            msg = "values returned by self._model has the wrong shape."
            msg += " shape is {0} but must be 2D array with single row".format(
                values_0.shape
            )
            raise ValueError(msg)
        values = self._bkd.empty((nsamples, self.nqoi()))
        values[0, :] = values_0
        for ii in range(1, nsamples):
            t0 = time.time()
            values[ii, :] = self._evaluate(samples[:, ii : ii + 1])
            t1 = time.time()
            times.append(t1 - t0)
        self._work_tracker.update("val", self._bkd.array(times))
        self._database.add_data("val", samples, values)
        return values

    def _values(self, samples: Array):
        stored_vals, stored_idx, new_idx = self._database.get_data(
            "val", samples
        )
        if len(stored_idx) == 0:
            return self._new_values(samples)

        if len(new_idx) == 0:
            return self._bkd.vstack(stored_vals)

        vals = self._bkd.empty((samples.shape[1], self.nqoi()))
        new_samples = samples[:, new_idx]
        vals[new_idx] = self._new_values(new_samples)
        vals[stored_idx] = self._bkd.vstack(stored_vals)
        return vals

    def __call__(self, samples: Array) -> Array:
        # overwrite call from model so not to count model evaluation twice
        self._check_samples_shape(samples)
        vals = self._values(samples)
        self._check_values_shape(samples, vals)
        return vals


class ModelFromCallable(Model):
    """
    A flexible wrapper for creating a model from user-defined callable functions.

    The `ModelFromCallable` class allows users to define a model programmatically by providing custom functions for evaluating values, Jacobians, Hessians, and other derivatives. This class is particularly useful for scenarios where the model's behavior is defined mathematically or algorithmically and needs to conform to the `Model` interface.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        function: callable,
        jacobian: callable = None,
        apply_jacobian: callable = None,
        apply_hessian: callable = None,
        hessian: callable = None,
        apply_weighted_hessian: callable = None,
        weighted_hessian: callable = None,
        sample_ndim: int = 2,
        values_ndim: int = 2,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the ModelFromCallable instance.

        Parameters
        ----------
        nqoi : int
            The number of quantities of interest (QoI) in the model.
        nvars : int
            The number of variables in the model.
        function : callable
            The user-defined function for evaluating the model's values.
        jacobian : callable, optional
            The user-defined function for computing the Jacobian matrix.
        apply_jacobian : callable, optional
            The user-defined function for applying the Jacobian matrix to a vector.
        apply_hessian : callable, optional
            The user-defined function for applying the Hessian matrix to a vector.
        hessian : callable, optional
            The user-defined function for computing the Hessian matrix.
        apply_weighted_hessian : callable, optional
            The user-defined function for applying the weighted Hessian matrix to a vector.
        weighted_hessian : callable, optional
            The user-defined function for computing the weighted Hessian matrix.
        sample_ndim : int, optional
            The dimension of the array accepted by the user-defined function. Must be either 1 or 2. Defaults to 2.
        values_ndim : int, optional
            The dimension of the array returned by the user-defined function. Must be 0, 1, or 2. Defaults to 2.
        backend : BackendMixin, optional
            The backend used for array operations. Defaults to NumpyMixin.

        Raises
        ------
        ValueError
            If any of the provided functions are not callable.
        """
        self._nqoi = nqoi
        self._nvars = nvars
        super().__init__(backend=backend)

        self._apply_jacobian_implemented = False
        self._jacobian_implemented = False
        self._apply_hessian_implemented = False
        self._hessian_implemented = False
        self._weighted_hessian_implemented = False
        self._apply_weighted_hessian_implemented = False

        if not callable(function):
            raise ValueError("function must be callable")
        self._user_function = function
        if jacobian is not None:
            if not callable(jacobian):
                raise ValueError("jacobian must be callable")
            self._user_jacobian = jacobian
            self._jacobian_implemented = True
        if apply_jacobian is not None:
            if not callable(apply_jacobian):
                raise ValueError("apply_jacobian must be callable")
            self._user_apply_jacobian = apply_jacobian
            self._apply_jacobian_implemented = True
        if apply_hessian is not None:
            if not callable(apply_hessian):
                raise ValueError("apply_hessian must be callable")
            self._user_apply_hessian = apply_hessian
            self._apply_hessian_implemented = True
        if hessian is not None:
            if not callable(hessian):
                raise ValueError("hessian must be callable")
            self._user_hessian = hessian
            self._hessian_implemented = True
        if apply_weighted_hessian is not None:
            if not callable(apply_weighted_hessian):
                raise ValueError("apply_weighed_hessian must be callable")
            self._user_apply_weighted_hessian = apply_weighted_hessian
            self._apply_weighted_hessian_implemented = True
        if weighted_hessian is not None:
            if not callable(weighted_hessian):
                raise ValueError("weighed_hessian must be callable")
            self._user_weighted_hessian = weighted_hessian
            self._weighted_hessian_implemented = True

        self._sample_ndim = sample_ndim
        self._values_ndim = values_ndim

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def apply_jacobian_implemented(self) -> bool:
        return self._apply_jacobian_implemented

    def jacobian_implemented(self) -> bool:
        return self._jacobian_implemented

    def apply_hessian_implemented(self) -> bool:
        return self._apply_hessian_implemented

    def hessian_implemented(self) -> bool:
        return self._hessian_implemented

    def apply_weighted_hessian_implemented(self) -> bool:
        return self._apply_weighted_hessian_implemented

    def weighted_hessian_implemented(self) -> bool:
        return self._weighted_hessian_implemented

    def _jacobian(self, sample: Array) -> Array:
        return self._eval_fun(self._user_jacobian, sample)

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        return self._eval_fun(self._user_apply_jacobian, sample, vec)

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._eval_fun(self._user_apply_hessian, sample, vec)

    def _hessian(self, sample: Array) -> Array:
        return self._eval_fun(self._user_hessian, sample)

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        return self._eval_fun(
            self._user_apply_weighted_hessian, sample, vec, weights
        )

    def _weighted_hessian(self, sample: Array, weights: Array) -> Array:
        return self._eval_fun(self._user_weighted_hessian, sample, weights)


class ModelFromVectorizedCallable(ModelFromCallable):
    """
    A wrapper for creating a model from user-defined vectorized callable functions.

    The `ModelFromVectorizedCallable` class extends `ModelFromCallable` and assumes that the user-defined functions can accept multiple samples simultaneously. This allows for efficient evaluation of values, Jacobians, Hessians, and other derivatives for a batch of samples, leveraging vectorized operations.

    Parameters
    ----------
    nqoi : int
        The number of quantities of interest (QoI) in the model.
    nvars : int
        The number of variables in the model.
    function : callable
        The user-defined function for evaluating the model's values. This function must accept a 2D array of samples (shape `(nvars, nsamples)`) and return a 2D array of values (shape `(nqoi, nsamples)`).
    jacobian : callable, optional
        The user-defined function for computing the Jacobian matrix. This function must accept a 2D array of samples and return a 3D array representing the Jacobian matrices for all samples (shape `(nqoi, nvars, nsamples)`).
    apply_jacobian : callable, optional
        The user-defined function for applying the Jacobian matrix to a vector. This function must accept a 2D array of samples and a 2D array of vectors, and return the result of the matrix-vector multiplication for all samples (shape `(nqoi, nsamples)`).
    apply_hessian : callable, optional
        The user-defined function for applying the Hessian matrix to a vector. This function must accept a 2D array of samples and a 2D array of vectors, and return the result of the matrix-vector multiplication for all samples (shape `(nqoi, nsamples)`).
    hessian : callable, optional
        The user-defined function for computing the Hessian matrix. This function must accept a 2D array of samples and return a 4D array representing the Hessian matrices for all samples (shape `(nqoi, nvars, nvars, nsamples)`).
    apply_weighted_hessian : callable, optional
        The user-defined function for applying the weighted Hessian matrix to a vector. This function must accept a 2D array of samples, a 2D array of vectors, and a 2D array of weights, and return the result of the matrix-vector multiplication for all samples (shape `(nqoi, nsamples)`).
    weighted_hessian : callable, optional
        The user-defined function for computing the weighted Hessian matrix. This function must accept a 2D array of samples and a 2D array of weights, and return a 4D array representing the weighted Hessian matrices for all samples (shape `(nqoi, nvars, nvars, nsamples)`).
    values_ndim : int, optional
        The dimension of the array returned by the user-defined function. Must be 0, 1, or 2. Defaults to 2.
    backend : BackendMixin, optional
        The backend used for array operations. Defaults to `NumpyMixin`.

    Notes
    -----
    - This class assumes that all user-defined functions are vectorized, meaning they can process multiple samples simultaneously.
    - The `function` parameter is required, while other derivative-related functions (`jacobian`, `apply_jacobian`, etc.) are optional.
    - The `sample_ndim` parameter is fixed to 2, as this class always expects 2D arrays of samples.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        function: callable,
        jacobian: callable = None,
        apply_jacobian: callable = None,
        apply_hessian: callable = None,
        hessian: callable = None,
        apply_weighted_hessian: callable = None,
        weighted_hessian: callable = None,
        values_ndim: int = 2,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__(
            nqoi,
            nvars,
            function,
            jacobian,
            apply_jacobian,
            apply_hessian,
            hessian,
            apply_weighted_hessian,
            weighted_hessian,
            2,
            values_ndim,
            backend,
        )

    def _eval_fun(self, fun: callable, sample: Array, *args) -> Array:
        if self._sample_ndim == 2:
            return fun(sample, *args)
        return fun(sample[:, 0], *args)

    def _values(self, samples: Array) -> Array:
        values = self._user_function(samples)
        if self._values_ndim != values.ndim:
            raise RuntimeError("function returned values with the wrong ndim")
        if values.shape[0] != samples.shape[1]:
            raise RuntimeError("function returned values with the wrong shape")
        if self._values_ndim == 1:
            values = values[:, None]
        return values


class ModelFromSingleSampleCallable(SingleSampleModelMixin, ModelFromCallable):
    r"""
    A wrapper for creating a model from user-defined callable functions that operate on single samples.

    The `ModelFromSingleSampleCallable` class extends `ModelFromCallable` and assumes that the user-defined functions operate on single samples (1D arrays). This class is designed for scenarios where the model computations are performed on individual samples rather than batches of samples.

    Parameters
    ----------
    nqoi : int
        The number of quantities of interest (QoI) in the model.
    nvars : int
        The number of variables in the model.
    function : callable
        The user-defined function for evaluating the model's values. This function must accept a 1D array representing a single sample (shape `(nvars,)`) and return a 1D array of values (shape `(nqoi,)`).
    jacobian : callable, optional
        The user-defined function for computing the Jacobian matrix. This function must accept a 1D array representing a single sample and return a 2D array representing the Jacobian matrix (shape `(nqoi, nvars)`).
    apply_jacobian : callable, optional
        The user-defined function for applying the Jacobian matrix to a vector. This function must accept a 1D array representing a single sample and a 1D array representing a vector, and return a 1D array of results (shape `(nqoi,)`).
    hessian : callable, optional
        The user-defined function for computing the Hessian matrix. This function must accept a 1D array representing a single sample and return a 3D array representing the Hessian matrix (shape `(nqoi, nvars, nvars)`).
    apply_hessian : callable, optional
        The user-defined function for applying the Hessian matrix to a vector. This function must accept a 1D array representing a single sample and a 1D array representing a vector, and return a 1D array of results (shape `(nqoi,)`).
    sample_ndim : int, optional
        The dimension of the array accepted by the user-defined function. Must be 1, as this class operates on single samples. Defaults to 1.
    values_ndim : int, optional
        The dimension of the array returned by the user-defined function. Must be 0, 1, or 2. Defaults to 1.
    backend : BackendMixin, optional
        The backend used for array operations. Defaults to `NumpyMixin`.

    Notes
    -----
    - This class assumes that all user-defined functions operate on single samples (1D arrays).
    - The `function` parameter is required, while other derivative-related functions (`jacobian`, `apply_jacobian`, etc.) are optional.
    - The `sample_ndim` parameter is fixed to 1, as this class always expects 1D arrays of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyMixin as bkd
    >>> from pyapprox.interface.model import ModelFromSingleSampleCallable
    >>> # Define the number of variables
    >>> nvars = 3
    >>> # Instantiate the model
    >>> model = ModelFromSingleSampleCallable(
    ...     nqoi=1,
    ...     nvars=nvars,
    ...     function=lambda x: bkd.hstack(
    ...         [1 * ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2)],
    ...     ),
    ...     jacobian=lambda x: bkd.stack(
    ...         [1 * bkd.array([2 * (x[0] - 1), 2 * (x[1] - 2.5), 0])],
    ...         axis=0,
    ...     ),
    ...     apply_jacobian=lambda x, v: bkd.asarray(
    ...         [1 * (2 * (x[0] - 1) * v[0] + 2 * (x[1] - 2.5) * v[1])]
    ...     ),
    ...     hessian=lambda x: bkd.stack([bkd.diag(bkd.array([2.0, 2, 0]))]),
    ...     apply_hessian=lambda x, v: bkd.diag(bkd.array([2.0, 2, 0])) @ v,
    ...     sample_ndim=1,
    ...     values_ndim=1,
    ...     backend=bkd,
    ... )
    >>> # Define a single sample
    >>> sample = bkd.array([1.0, 2.5, 0.0])[:, None]
    >>> # Evaluate the model
    >>> print(model(sample))
    [[0.]]
    >>> # Evaluate the Jacobian
    >>> print(model._jacobian(sample))
    [[0. 0. 0.]]
    >>> # Apply the Jacobian
    >>> vector = bkd.array([1.0, 1.0, 1.0])
    >>> print(model._apply_jacobian(sample, vector))
    [0.]
    >>> # Evaluate the Hessian
    >>> print(model._hessian(sample))
    [[[2. 0. 0.]
      [0. 2. 0.]
      [0. 0. 0.]]]
    >>> # Apply the Hessian
    >>> print(model._apply_hessian(sample, vector))
    [2. 2. 0.]
    """

    def _eval_fun(self, fun: callable, sample: Array, *args) -> Array:
        if self._sample_ndim == 2:
            return fun(sample, *args)
        return fun(sample[:, 0], *args)

    def _evaluate(self, sample: Array) -> Array:
        values = self._eval_fun(self._user_function, sample)
        if self._values_ndim != values.ndim:
            raise RuntimeError(
                "function returned values with the wrong ndim. "
                "Was {0} must be {1}".format(values.ndim, self._values_ndim)
            )
        if self._values_ndim != 2:
            return self._bkd.atleast2d(values)
        return values


class SingleSampleModel(SingleSampleModelMixin, Model):
    pass


class UmbridgeModel(Model):
    """
    Wrapper for UM-Bridge models.

    This class provides an interface for evaluating UM-Bridge models at multiple samples.
    It supports single-threaded and parallel evaluation, as well as methods for computing
    Jacobians, Hessians, and Hessian-vector products. UM-Bridge (UM-Bridge: the UQ and Model Bridge)
    is a unified interface for numerical models accessible from virtually any programming language
    or framework. It is primarily intended for coupling advanced models (e.g., simulations of complex
    physical processes) to advanced statistical or optimization methods.

    Parameters
    ----------
    umb_model : umbridge.HTTPModel
        The UM-Bridge model to be wrapped.
    config : dict, optional
        Configuration dictionary for the UM-Bridge model. Default is an empty dictionary.
    nprocs : int, optional
        Number of processes for parallel evaluation. Default is 1.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, umb_model, config={}, nprocs=1, backend=NumpyMixin):
        """
        Initialize the UM-Bridge model wrapper.

        Parameters
        ----------
        umb_model : umbridge.HTTPModel
            The UM-Bridge model to be wrapped.
        config : dict, optional
            Configuration dictionary for the UM-Bridge model. Default is an empty dictionary.
        nprocs : int, optional
            Number of processes for parallel evaluation. Default is 1.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.

        Raises
        ------
        ValueError
            If the provided model is not an instance of `umbridge.HTTPModel`.

        Notes
        -----
        On Linux and macOS, the PID of the process using port 4242 can be checked using:
        `lsof -i :4242`.
        """
        super().__init__(backend=backend)
        if not isinstance(umb_model, umbridge.HTTPModel):
            raise ValueError("model is not an umbridge.HTTPModel")
        self._model = umb_model
        self._config = config
        self._nprocs = nprocs
        self._nmodel_evaluations = 0

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi : int
            The number of quantities of interest.
        """
        return self._model.get_output_sizes(self._config)[0]

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            The number of variables.
        """
        return self._model.get_input_sizes(self._config)[0]

    def apply_jacobian_implemented(self) -> bool:
        """
        Check if the apply Jacobian method is implemented in the model.

        Returns
        -------
        apply_jacobian_implemented : bool
            True if the apply Jacobian method is implemented, False otherwise.
        """
        return self._model.supports_apply_jacobian()

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian method is implemented in the model.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian method is implemented, False otherwise.
        """
        return self._model.supports_gradient()

    def apply_hessian_implemented(self) -> bool:
        """
        Check if the apply Hessian method is implemented in the model.

        Returns
        -------
        apply_hessian_implemented : bool
            True if the apply Hessian method is implemented, False otherwise.
        """
        return self._model.supports_apply_hessian()

    def _check_sample(self, sample):
        """
        Validate the input sample and convert it to the required format.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.

        Returns
        -------
        parameters : List[List[float]]
            Converted sample in the required format.

        Raises
        ------
        ValueError
            If the sample is not a 2D array.
        """
        if sample.ndim != 2:
            raise ValueError(
                "sample is not a 2D array, has shape {0}".format(sample.shape)
            )
        return [sample[:, 0].tolist()]

    def _jacobian(self, sample):
        """
        Compute the Jacobian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.

        Returns
        -------
        jacobian : Array
            Jacobian matrix of the model at the given sample.

        Notes
        -----
        The Jacobian is computed using the `gradient` method of the UM-Bridge model.
        """
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.gradient(0, 0, parameters, [1.0], config=self._config)
        ).T

    def _apply_jacobian(self, sample, vec):
        """
        Apply the Jacobian to a vector.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.
        vec : Array
            Vector to apply the Jacobian to.

        Returns
        -------
        applied_jacobian : Array
            Result of applying the Jacobian to the vector.
        """
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.apply_jacobian(
                None, None, parameters, vec, config=self._config
            )
        )

    def _apply_hessian(self, sample, vec):
        """
        Apply the Hessian to a vector.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.
        vec : Array
            Vector to apply the Hessian to.

        Returns
        -------
        applied_hessian : Array
            Result of applying the Hessian to the vector.
        """
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.apply_hessian(
                None, None, None, parameters, vec, None, config=self._config
            )
        )

    def _evaluate_single_thread(self, sample, sample_id):
        """
        Evaluate the model for a single sample in a single thread.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.
        sample_id : int
            Identifier for the sample.

        Returns
        -------
        result : Array
            Model evaluation result for the given sample.
        """
        parameters = self._check_sample(sample)
        return self._model(parameters, config=self._config)[0]

    def _evaluate_parallel(self, samples):
        """
        Evaluate the model for multiple samples in parallel.

        Parameters
        ----------
        samples : Array
            Input samples as a 2D array.

        Returns
        -------
        results : List[Array]
            List of model evaluation results for the given samples.
        """
        pool = ThreadPool(self._nprocs)
        nsamples = samples.shape[1]
        results = pool.starmap(
            self._evaluate_single_thread,
            [
                (samples[:, ii : ii + 1], self._nmodel_evaluations + ii)
                for ii in range(nsamples)
            ],
        )
        pool.close()
        self._nmodel_evaluations += nsamples
        return results

    def _evaluate_serial(self, samples):
        """
        Evaluate the model for multiple samples serially.

        Parameters
        ----------
        samples : Array
            Input samples as a 2D array.

        Returns
        -------
        results : List[Array]
            List of model evaluation results for the given samples.
        """
        results = []
        nsamples = samples.shape[1]
        for ii in range(nsamples):
            results.append(
                self._evaluate_single_thread(
                    samples[:, ii : ii + 1], self._nmodel_evaluations + ii
                )
            )
        self._nmodel_evaluations += nsamples
        return results

    def _values(self, samples):
        """
        Evaluate the model at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as a 2D array.

        Returns
        -------
        values : Array
            Model evaluations at the given samples.
        """
        if self._nprocs > 1:
            return self._bkd.asarray(self._evaluate_parallel(samples))
        return self._bkd.asarray(self._evaluate_serial(samples))

    @staticmethod
    def start_server(
        run_server_string,
        url="http://localhost:4242",
        out=None,
        max_connection_time=20,
    ):
        """
        Start the UM-Bridge server.

        Parameters
        ----------
        run_server_string : str
            Command to start the server.
        url : str, optional
            URL of the server. Default is "http://localhost:4242".
        out : file-like object, optional
            Output stream for the server logs. Default is `os.devnull`.
        max_connection_time : int, optional
            Maximum time to wait for the server to start. Default is 20 seconds.

        Returns
        -------
        process : subprocess.Popen
            Process object for the server.
        out : file-like object
            Output stream for the server logs.

        Raises
        ------
        RuntimeError
            If the server cannot be started within the maximum connection time.

        Notes
        -----
        On Linux and macOS, the PID of the process using port 4242 can be checked using:
        `lsof -i :4242`.
        """
        if out is None:
            out = open(os.devnull, "w")
        process = subprocess.Popen(
            run_server_string,
            shell=True,
            stdout=out,
            stderr=out,
            preexec_fn=os.setsid,
        )
        t0 = time.time()
        print("Starting server using {0}".format(run_server_string))
        while True:
            try:
                requests.get(os.path.join(url, "Info"))
                print("Server running")
                break
            except requests.exceptions.ConnectionError:
                if time.time() - t0 > max_connection_time:
                    UmbridgeModel.kill_server(process, out)
                    raise RuntimeError("Could not connect to server") from None
        return process, out

    @staticmethod
    def kill_server(process, out=None):
        """
        Kill the UM-Bridge server.

        Parameters
        ----------
        process : subprocess.Popen
            Process object for the server.
        out : file-like object, optional
            Output stream for the server logs.

        Returns
        -------
        None
        """
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        if out is not None:
            out.close()


class UmbridgeIOModel(UmbridgeModel):
    """
    Wrapper for UM-Bridge models that require separate directories for each model run.

    This class extends `UmbridgeModel` to handle models that require the creation
    of separate directories for each model run. It enables loading and writing of files
    during model evaluation.

    Parameters
    ----------
    umb_model : umbridge.HTTPModel
        The UM-Bridge model to be wrapped.
    config : dict, optional
        Configuration dictionary for the UM-Bridge model. Default is an empty dictionary.
    nprocs : int, optional
        Number of processes for parallel evaluation. Default is 1.
    outdir_basename : str, optional
        Base name for the output directories created for each model run. Default is "modelresults".
    backend : BackendMixin, optional
    """

    def __init__(
        self,
        umb_model,
        config={},
        nprocs=1,
        outdir_basename="modelresults",
        backend=NumpyMixin,
    ):
        """
        Initialize the UM-Bridge model wrapper with directory creation support.

        Parameters
        ----------
        umb_model : umbridge.HTTPModel
            The UM-Bridge model to be wrapped.
        config : dict, optional
            Configuration dictionary for the UM-Bridge model. Default is an empty dictionary.
        nprocs : int, optional
            Number of processes for parallel evaluation. Default is 1.
        outdir_basename : str, optional
            Base name for the output directories created for each model run. Default is "modelresults".
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(umb_model, config, nprocs, backend=backend)
        self._outdir_basename = outdir_basename

    def _evaluate_single_thread(self, sample, sample_id):
        """
        Evaluate the model for a single sample in a single thread, creating a separate directory for the run.

        Parameters
        ----------
        sample : Array
            Input sample as a 2D array.
        sample_id : int
            Identifier for the sample.

        Returns
        -------
        result : Array
            Model evaluation result for the given sample.

        Notes
        -----
        A separate directory is created for each model run using the `outdir_basename`
        and the sample ID.
        """
        parameters = self._check_sample(sample)
        config = self._config.copy()
        config["outdir_basename"] = os.path.join(
            self._outdir_basename, "wdir-{0}".format(sample_id)
        )
        return self._model(parameters, config=config)[0]


class UmbridgeIOModelEnsemble(UmbridgeModel):
    """
    Wrapper for UM-Bridge models with multiple configurations that require separate directories for each model run.

    This class extends `UmbridgeModel` to handle models with multiple configurations
    that require the creation of separate directories for each model run. It enables
    loading and writing of files during model evaluation.

    Parameters
    ----------
    umb_model : umbridge.HTTPModel
        The UM-Bridge model to be wrapped.
    model_configs : dict, optional
        Dictionary containing multiple configurations for the UM-Bridge model. Default is an empty dictionary.
    nprocs : int, optional
        Number of processes for parallel evaluation. Default is 1.
    outdir_basename : str, optional
        Base name for the output directories created for each model run. Default is "modelresults".
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        umb_model,
        model_configs={},
        nprocs=1,
        outdir_basename="modelresults",
        backend=NumpyMixin,
    ):
        """
        Initialize the UM-Bridge model ensemble wrapper with directory creation support.

        Parameters
        ----------
        umb_model : umbridge.HTTPModel
            The UM-Bridge model to be wrapped.
        model_configs : dict, optional
            Dictionary containing multiple configurations for the UM-Bridge model. Default is an empty dictionary.
        nprocs : int, optional
            Number of processes for parallel evaluation. Default is 1.
        outdir_basename : str, optional
            Base name for the output directories created for each model run. Default is "modelresults".
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(umb_model, None, nprocs, backend=backend)
        self._outdir_basename = outdir_basename
        self._model_configs = model_configs

    def _evaluate_single_thread(self, full_sample, sample_id):
        """
        Evaluate the model for a single sample in a single thread, using the specified configuration and creating a separate directory for the run.

        Parameters
        ----------
        full_sample : Array
            Input sample as a 2D array. The last entry in the sample specifies the model configuration ID.
        sample_id : int
            Identifier for the sample.

        Returns
        -------
        result : Array
            Model evaluation result for the given sample.

        Raises
        ------
        ValueError
            If the sample shape does not match the expected input size for the specified model configuration.

        Notes
        -----
        A separate directory is created for each model run using the `outdir_basename`
        and the sample ID. The model configuration ID is extracted from the last entry
        in the sample.
        """
        sample, model_id = full_sample[:-1], int(full_sample[-1, 0])
        parameters = self._check_sample(sample)
        config = self._model_configs[model_id].copy()
        config["outdir_basename"] = os.path.join(
            self._outdir_basename, "wdir-{0}".format(sample_id)
        )
        if sample.shape[0] != self._model.get_input_sizes(config)[0]:
            raise ValueError(
                "sample must contain model id but shape {0}!={1}".format(
                    sample.shape[0], self._model.get_input_sizes(config)[0]
                )
            )
        return self._model(parameters, config=config)[0]


class IOModel(SingleSampleModelMixin, Model):
    """
    Base class for models that require loading and/or writing of files.

    This class provides an interface for models that use system calls and file I/O
    to run simulations, creating separate directories for each run if required. It
    supports temporary directories, soft linking of input files, and saving samples
    and values to files.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest (QoI) in the model.
    nvars : int
        Number of variables in the model.
    infilenames : List[str]
        List of input filenames required by the model.
    outdir_basename : str, optional
        Base name for the output directories created for each model run. Default is None.
    save : str, optional
        Specifies the saving behavior for temporary files. Must be one of:
        ["full", "limited", "no"]. Default is "no".
    datafilename : str, optional
        Name of the file to save samples and values. Required if `save` is not "no".
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        infilenames: List[str],
        outdir_basename: str = None,
        save: str = "no",
        datafilename: str = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the IOModel.

        Parameters
        ----------
        nqoi : int
            Number of quantities of interest (QoI) in the model.
        nvars : int
            Number of variables in the model.
        infilenames : List[str]
            List of input filenames required by the model.
        outdir_basename : str, optional
            Base name for the output directories created for each model run. Default is None.
        save : str, optional
            Specifies the saving behavior for temporary files. Must be one of:
            ["full", "limited", "no"]. Default is "no".
        datafilename : str, optional
            Name of the file to save samples and values. Required if save != "no".
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.

        Raises
        ------
        ValueError
            If `save` is not one of ["full", "limited", "no"].
        ValueError
            If `outdir_basename` is None and `save` is not "no".
        ValueError
            If `datafilename` is provided when `save` is "no".
        ValueError
            If `datafilename` is not provided when `save` is not "no".
        """
        super().__init__(backend=backend)
        self._nqoi = nqoi
        self._nvars = nvars
        self._infilenames = infilenames
        save_values = ["full", "limited", "no"]
        if save not in save_values:
            raise ValueError("save must be in {0}".format(save_values))
        if outdir_basename is None and save != "no":
            msg = (
                "You are requesting temporary files but save not set to 'no'."
            )
            raise ValueError(msg)
        if save == "no" and datafilename is not None:
            raise ValueError("datafilename provided even though save='no'")
        if save != "no" and datafilename is None:
            raise ValueError("datafilename must be provided if save != 'no'")
        self._outdir_basename = outdir_basename
        self._save = save
        self._datafilename = datafilename
        self._nmodel_evaluations = 0

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi : int
            The number of quantities of interest.
        """
        return self._nqoi

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            The number of variables.
        """
        return self._nvars

    def _create_outdir(self):
        """
        Create a directory for the model run.

        Returns
        -------
        outdirname : str
            Path to the created output directory.
        tmpdir : tempfile.TemporaryDirectory or None
            Temporary directory object if `outdir_basename` is None, otherwise None.

        Raises
        ------
        RuntimeError
            If the output directory already exists.
        """
        if self._outdir_basename is None:
            tmpdir = tempfile.TemporaryDirectory()
            outdirname = tmpdir.name
            return outdirname, tmpdir

        ext = "wdir-{0}".format(self._nmodel_evaluations)
        outdirname = os.path.join(self._outdir_basename, ext)
        if os.path.exists(outdirname):
            raise RuntimeError(
                "Tried to create {0} but it already exists".format(outdirname)
            )
        os.makedirs(outdirname)
        return outdirname, None

    def _link_files(self, outdirname):
        """
        Create soft links to input files in the output directory.

        Parameters
        ----------
        outdirname : str
            Path to the output directory.

        Returns
        -------
        linked_filenames : List[str]
            List of paths to the linked input files.

        Raises
        ------
        Exception
            If a file already exists in the output directory and cannot be linked.
        """
        linked_filenames = []
        for filename_w_src_path in self._infilenames:
            filename = os.path.basename(filename_w_src_path)
            filename_w_target_path = os.path.join(outdirname, filename)
            if not os.path.exists(filename_w_target_path):
                os.symlink(filename_w_src_path, filename_w_target_path)
            else:
                msg = "{0} exists in {1} so cannot create soft link".format(
                    filename, outdirname
                )
                raise Exception(msg)
            linked_filenames.append(filename_w_target_path)
        return linked_filenames

    def _cleanup_outdir(self, outdirname):
        """
        Delete all files in the output directory.

        Parameters
        ----------
        outdirname : str
            Path to the output directory.

        Returns
        -------
        None
        """
        filenames_to_delete = glob.glob(os.path.join(outdirname, "*"))
        for filename in filenames_to_delete:
            os.remove(filename)

    def _save_samples_and_values(self, sample, values, outdirname):
        """
        Save the sample and values to a file in the output directory.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        values : Array
            Model evaluation results as an array.
        outdirname : str
            Path to the output directory.

        Returns
        -------
        None
        """
        if self._datafilename is not None:
            filename = os.path.join(outdirname, self._datafilename)
            np.savez(filename, sample=sample, values=values)

    def _process_outdir(self, sample, values, outdirname, tmpdir):
        """
        Process the output directory based on the saving behavior.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        values : Array
            Model evaluation results as an array.
        outdirname : str
            Path to the output directory.
        tmpdir : tempfile.TemporaryDirectory or None
            Temporary directory object if `outdir_basename` is None, otherwise None.

        Returns
        -------
        None

        Notes
        -----
        - If `save` is "no", the temporary directory is cleaned up.
        - If `save` is "limited", the output directory is cleaned up after saving the sample and values.
        - If `save` is "full", the sample and values are saved without cleaning up the output directory.
        """
        if self._save == "no":
            if tmpdir is not None:
                tmpdir.cleanup()
            return
        if self._save == "limited":
            self._cleanup_outdir(outdirname)
        self._save_samples_and_values(sample, values, outdirname)
        if tmpdir is not None:
            tmpdir.cleanup()

    @abstractmethod
    def _run(
        self, sample: Array, linked_filenames: List[str], outdirname: str
    ) -> Array:
        """
        Abstract method to run the model using the provided sample and input files.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        linked_filenames : List[str]
            List of paths to the linked input files.
        outdirname : str
            Path to the output directory.

        Returns
        -------
        values : Array
            Model evaluation results.
        """
        raise NotImplementedError

    def _evaluate(self, sample: Array):
        """
        Evaluate the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        values : Array
            Model evaluation results.

        Notes
        -----
        This method creates a separate directory for the model run, links the input files,
        runs the model, and processes the output directory based on the saving behavior.
        """
        outdirname, tmpdir = self._create_outdir()
        self._nmodel_evaluations += 1
        linked_filenames = self._link_files(outdirname)
        values = self._bkd.asarray(
            self._run(sample, linked_filenames, outdirname)
        )
        self._process_outdir(sample, values, outdirname, tmpdir)
        return values


class SerialIOModel(IOModel):
    """
    Serial IO model for running simulations using shell commands.

    This class extends `IOModel` to handle models that require running simulations
    using shell commands. It supports saving parameters to a file, executing the shell
    command, and loading results from an output file.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest (QoI) in the model.
    nvars : int
        Number of variables in the model.
    infilenames : List[str]
        List of input filenames required by the model.
    shell_command : str
        Shell command to execute the model simulation.
    params_filename : str, optional
        Name of the file to save the input parameters. Default is "params.in".
    results_filename : str, optional
        Name of the file to load the simulation results. Default is "results.out".
    outdir_basename : str, optional
        Base name for the output directories created for each model run. Default is None.
    save : str, optional
        Specifies the saving behavior for temporary files. Must be one of:
        ["full", "limited", "no"]. Default is "no".
    datafilename : str, optional
        Name of the file to save samples and values. Required if `save` is not "no".
    verbosity : int, optional
        Verbosity level for shell command execution. Default is 0.
        - 0: No output.
        - 1: Save output to a file.
        - 2: Print output to the console.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        infilenames: List[str],
        shell_command: str,
        params_filename: str = "params.in",
        results_filename: str = "results.out",
        outdir_basename: str = None,
        save: str = "no",
        datafilename: str = None,
        verbosity: int = 0,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the serial IO model.

        Parameters
        ----------
        nqoi : int
            Number of quantities of interest (QoI) in the model.
        nvars : int
            Number of variables in the model.
        infilenames : List[str]
            List of input filenames required by the model.
        shell_command : str
            Shell command to execute the model simulation.
        params_filename : str, optional
            Name of the file to save the input parameters. Default is "params.in".
        results_filename : str, optional
            Name of the file to load the simulation results. Default is "results.out".
        outdir_basename : str, optional
            Base name for the output directories created for each model run. Default is None.
        save : str, optional
            Specifies the saving behavior for temporary files. Must be one of:
            ["full", "limited", "no"]. Default is "no".
        datafilename : str, optional
            Name of the file to save samples and values. Required if `save` is not "no".
        verbosity : int, optional
            Verbosity level for shell command execution. Default is 0.
            - 0: No output.
            - 1: Save output to a file.
            - 2: Print output to the console.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.

        Raises
        ------
        ValueError
            If `save` is not one of ["full", "limited", "no"].
        ValueError
            If `outdir_basename` is None and `save` is not "no".
        ValueError
            If `datafilename` is provided when `save` is "no".
        ValueError
            If `datafilename` is not provided when `save` is not "no".
        """
        super().__init__(
            nqoi,
            nvars,
            infilenames,
            outdir_basename,
            save,
            datafilename,
            backend=backend,
        )
        self._shell_command = shell_command
        self._results_filename = results_filename
        self._params_filename = params_filename
        self._verbosity = verbosity

    def _run_shell_command(self):
        """
        Execute the shell command to run the model simulation.

        Returns
        -------
        None

        Notes
        -----
        The verbosity level determines the behavior of the shell command execution:
        - 0: No output.
        - 1: Save output to a file named "shell_command.out".
        - 2: Print output to the console.
        """
        if self._verbosity == 0:
            subprocess.check_output(self._shell_command, shell=True, env=None)
        elif self._verbosity == 1:
            filename = "shell_command.out"
            with open(filename, "w") as f:
                subprocess.call(
                    self._shell_command,
                    shell=True,
                    stdout=f,
                    stderr=f,
                    env=None,
                )
        else:
            subprocess.call(self._shell_command, shell=True, env=None)

    def _run(
        self, sample: Array, linked_filenames: List[str], outdirname: str
    ) -> Array:
        """
        Run the model simulation using the provided sample and input files.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        linked_filenames : List[str]
            List of paths to the linked input files.
        outdirname : str
            Path to the output directory.

        Returns
        -------
        values : Array
            Model evaluation results.

        Notes
        -----
        - The input sample is saved to a file named `params_filename` in the output directory.
        - The shell command is executed to run the simulation.
        - The results are loaded from a file named `results_filename` in the output directory.
        """
        curdirname = os.getcwd()
        os.chdir(outdirname)
        np.savetxt(self._params_filename, sample)
        self._run_shell_command()
        vals = np.loadtxt(self._results_filename, usecols=[0])
        os.chdir(curdirname)
        return self._bkd.atleast2d(self._bkd.asarray(vals))


class AsyncIOModel(SerialIOModel):
    """
    Asynchronous IO model for running simulations using shell commands.

    This class extends `SerialIOModel` to handle models that require asynchronous
    execution of simulations using shell commands. It supports dispatching samples
    to separate directories, tracking running processes, and loading results asynchronously.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest (QoI) in the model.
    nvars : int
        Number of variables in the model.
    infilenames : List[str]
        List of input filenames required by the model.
    shell_command : str
        Shell command to execute the model simulation.
    params_filename : str, optional
        Name of the file to save the input parameters. Default is "params.in".
    results_filename : str, optional
        Name of the file to load the simulation results. Default is "results.out".
    outdir_basename : str, optional
        Base name for the output directories created for each model run. Default is None.
    save : str, optional
        Specifies the saving behavior for temporary files. Must be one of:
        ["full", "limited", "no"]. Default is "no".
    datafilename : str, optional
        Name of the file to save samples and values. Required if `save` is not "no".
    verbosity : int, optional
        Verbosity level for shell command execution. Default is 0.
        - 0: No output.
        - 1: Save output to a file.
        - 2: Print output to the console.
    nprocs : int, optional
        Number of processes for parallel evaluation. Default is 1.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        infilenames: List[str],
        shell_command: str,
        params_filename: str = "params.in",
        results_filename: str = "results.out",
        outdir_basename: str = None,
        save: str = "no",
        datafilename: str = None,
        verbosity: int = 0,
        nprocs: int = 1,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the asynchronous IO model.

        Parameters
        ----------
        nqoi : int
            Number of quantities of interest (QoI) in the model.
        nvars : int
            Number of variables in the model.
        infilenames : List[str]
            List of input filenames required by the model.
        shell_command : str
            Shell command to execute the model simulation.
        params_filename : str, optional
            Name of the file to save the input parameters. Default is "params.in".
        results_filename : str, optional
            Name of the file to load the simulation results. Default is "results.out".
        outdir_basename : str, optional
            Base name for the output directories created for each model run. Default is None.
        save : str, optional
            Specifies the saving behavior for temporary files. Must be one of:
            ["full", "limited", "no"]. Default is "no".
        datafilename : str, optional
            Name of the file to save samples and values. Required if `save` is not "no".
        verbosity : int, optional
            Verbosity level for shell command execution. Default is 0.
            - 0: No output.
            - 1: Save output to a file.
            - 2: Print output to the console.
        nprocs : int, optional
            Number of processes for parallel evaluation. Default is 1.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._nprocs = nprocs
        super().__init__(
            nqoi,
            nvars,
            infilenames,
            shell_command,
            params_filename,
            results_filename,
            outdir_basename,
            save,
            datafilename,
            verbosity,
            backend=backend,
        )

    def _run_shell_command(self):
        """
        Execute the shell command to run the model simulation asynchronously.

        Returns
        -------
        proc : subprocess.Popen
            Process object for the shell command.
        writefile : file-like object
            File object for capturing the shell command output.

        Notes
        -----
        The verbosity level determines the behavior of the shell command execution:
        - 0: No output (redirected to `/dev/null`).
        - 1: Save output to a file named "shell_command.out".
        - 2: Print output to the console.
        """
        if self._verbosity == 0:
            writefile = open(os.devnull, "w")
        else:
            filename = "shell_command.out"
            writefile = open(filename, "w")
        proc = subprocess.Popen(
            self._shell_command,
            shell=True,
            stdout=writefile,
            stderr=writefile,
            env=None,
        )
        return proc, writefile

    def _dispatch_sample(self, sample: Array):
        """
        Dispatch a sample for evaluation, creating a separate directory for the run.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Notes
        -----
        This method creates a separate directory for the model run, links the input files,
        saves the sample to a file, and dispatches the shell command for execution.
        """
        outdirname, tmpdir = self._create_outdir()
        curdirname = os.getcwd()
        t0 = time.time()
        self._link_files(outdirname)
        os.chdir(outdirname)
        np.savetxt(self._params_filename, sample)
        proc, writefile = self._run_shell_command()
        os.chdir(curdirname)
        self._running_workdirs[self._nmodel_evaluations] = (
            outdirname,
            tmpdir,
            proc,
            writefile,
            t0,
        )
        self._nmodel_evaluations += 1

    def _newly_completed_sample_ids(self) -> List:
        """
        Identify sample IDs for completed simulations.

        Returns
        -------
        completed_sample_ids : List[int]
            List of sample IDs for completed simulations.
        """
        completed_sample_ids = []
        for sample_id, item in self._running_workdirs.items():
            proc = item[2]
            if proc.poll() is not None:
                completed_sample_ids.append(sample_id)
        return completed_sample_ids

    def _load_sample_result(self, sample_id: int) -> Array:
        """
        Load the results for a completed sample.

        Parameters
        ----------
        sample_id : int
            Identifier for the sample.

        Returns
        -------
        sample : Array
            Input sample as an array.
        values : Array
            Model evaluation results.
        walltime : float
            Wall time for the sample evaluation.

        Raises
        ------
        RuntimeError
            If the results file does not contain the expected number of quantities of interest (QoI).

        Notes
        -----
        If the results file does not exist, the returned values are filled with NaN.
        """
        outdirname, tmpdir, proc, writefile, t0 = self._running_workdirs[
            sample_id
        ]
        sample = np.loadtxt(os.path.join(outdirname, self._params_filename))
        if os.path.exists(os.path.join(outdirname, self._results_filename)):
            values = np.loadtxt(
                os.path.join(outdirname, self._results_filename), usecols=[0]
            )
            if values.shape[0] != self.nqoi():
                raise RuntimeError(
                    "Values returned in {0} had the incorrect shape".format(
                        self._results_filename
                    )
                )
            walltime = time.time() - t0
        else:
            if self._verbosity > 0:
                print(f"Sample {sample_id} did not return a result")
            values = self._bkd.full((self.nqoi(),), np.nan)
            walltime = np.nan
        writefile.close()
        self._process_outdir(sample, values, outdirname, tmpdir)
        return sample, values, walltime

    def _close_completed_threads(self):
        """
        Close completed threads and process their results.

        Returns
        -------
        None

        Notes
        -----
        This method identifies completed threads, loads their results, and processes
        the output directories.
        """
        completed_sample_ids = self._newly_completed_sample_ids()
        curdirname = os.getcwd()
        for sample_id in completed_sample_ids:
            sample, values, walltime = self._load_sample_result(sample_id)
            self._completed_vals.append(values)
            self._completed_sample_ids.append(sample_id)
            self._completed_wall_times.append(walltime)
        for sample_id in completed_sample_ids:
            del self._running_workdirs[sample_id]
        os.chdir(curdirname)

    def _prepare_values(self, samples: Array) -> Array:
        """
        Prepare the model evaluation results.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Model evaluation results, sorted in the order of the input samples.

        Notes
        -----
        Asynchronous calls may result in samples being completed out of order.
        This method ensures that the results are returned in the correct order.
        """
        sorted_idx = self._bkd.argsort(
            self._bkd.array(self._completed_sample_ids, dtype=int)
        )
        self._work_tracker.update(
            "val", self._bkd.asarray(self._completed_wall_times)[sorted_idx]
        )
        vals = self._bkd.asarray(np.array(self._completed_vals))[sorted_idx]
        self._database.add_data("val", samples, vals)
        return vals

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model at the given samples asynchronously.

        Parameters
        ----------
        samples : Array
            Input samples as a 2D array.

        Returns
        -------
        values : Array
            Model evaluation results.

        Notes
        -----
        This method dispatches samples for evaluation, tracks running processes,
        and loads results asynchronously.
        """
        self._running_workdirs = dict()
        self._completed_vals = []
        self._completed_sample_ids = []
        self._completed_wall_times = []
        sample_id = 0
        while True:
            if (
                len(self._running_workdirs) < self._nprocs
                and sample_id < samples.shape[1]
            ):
                self._dispatch_sample(samples[:, sample_id])
                sample_id += 1
            self._close_completed_threads()
            if len(self._completed_vals) == samples.shape[1]:
                break
        if len(self._completed_vals) != samples.shape[1]:
            raise RuntimeError("This should not happen")
        return self._prepare_values(samples)

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the model at the given samples, using cached results if available.

        Parameters
        ----------
        samples : Array
            Input samples as a 2D array.

        Returns
        -------
        values : Array
            Model evaluation results.
        """
        stored_values, stored_idx, new_idx = self._database.get_data(
            "val", samples
        )
        vals = self._bkd.empty((samples.shape[1], self.nqoi()))
        if len(new_idx) > 0:
            new_samples = samples[:, new_idx]
            vals[new_idx] = self._values(new_samples)
        if len(stored_idx) > 0:
            vals[stored_idx] = self._bkd.vstack(stored_values)
        return vals


class MultiIndexModelEnsemble(ABC):
    """
    Multi-index model ensemble.

    This abstract base class defines an ensemble of models organized in a multi-dimensional hierarchy.
    Each model is identified by a unique `model_id` within the bounds specified by `index_bounds`.
    The class provides methods for managing models, splitting samples by model IDs, and combining
    results from multiple models.

    Parameters
    ----------
    index_bounds : List[int]
        List specifying the bounds for each refinement variable. Each entry must be greater than 0.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self, index_bounds: List[int], backend: BackendMixin = NumpyMixin
    ):
        """
        Initialize the multi-index model ensemble.

        Parameters
        ----------
        index_bounds : List[int]
            List specifying the bounds for each refinement variable. Each entry must be greater than 0.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.

        Raises
        ------
        ValueError
            If any entry in `index_bounds` is less than 1.

        Notes
        -----
        The `index_bounds` specify the range of refinement variables, where both the lower and upper bounds
        are included.
        """
        self._bkd = backend
        self._models = dict()
        self._index_bounds = self._bkd.asarray(index_bounds, dtype=int)
        if self._index_bounds.min() < 1:
            raise ValueError("index bounds must have entries > 0")

    def _hash_model_id(self, model_id: Array) -> int:
        """
        Compute a hash for the given `model_id`.

        Parameters
        ----------
        model_id : Array
            Array representing the model ID.

        Returns
        -------
        hash_value : int
            Hash value for the given `model_id`.
        """
        return hash(self._bkd.to_numpy(model_id).tobytes())

    def get_model(self, model_id: Array) -> Model:
        """
        Retrieve a model corresponding to the given `model_id`.

        Parameters
        ----------
        model_id : Array
            Array representing the model ID.

        Returns
        -------
        model : Model
            The model corresponding to the given `model_id`.

        Raises
        ------
        ValueError
            If `model_id` does not match the number of refinement variables.
        ValueError
            If `model_id` exceeds the bounds specified in `index_bounds`.
        RuntimeError
            If `setup_model` does not return a valid `Model`.

        Notes
        -----
        If the model corresponding to `model_id` does not exist, it is created using the `setup_model` method.
        """
        if model_id.shape != (self.nrefinement_vars(),):
            raise ValueError("model_id does not match nrefinement_vars")
        if self._bkd.any(model_id > self._index_bounds):
            raise ValueError("model_id exceeds index_bounds")
        key = self._hash_model_id(model_id)
        if key in self._models:
            return self._models[key]
        model = self.setup_model(model_id)
        if not isinstance(model, Model):
            raise RuntimeError("setup_model did not return a Model")
        self._models[key] = model
        return model

    @abstractmethod
    def setup_model(self, model_id: Array):
        """
        Abstract method to set up a model for the given `model_id`.

        Parameters
        ----------
        model_id : Array
            Array representing the model ID.

        Returns
        -------
        model : Model
            The model corresponding to the given `model_id`.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def nrefinement_vars(self) -> int:
        """
        Return the number of refinement variables in the ensemble.

        Returns
        -------
        nrefinement_vars : int
            Number of refinement variables in the ensemble.
        """
        return len(self._index_bounds)

    def nmodels(self) -> int:
        """
        Return the total number of models in the ensemble.

        Returns
        -------
        nmodels : int
            Total number of models in the ensemble.
        """
        return self._bkd.prod(self._index_bounds)

    def split_ensemble_samples(
        self, ensemble_samples: Array
    ) -> Tuple[Array, List[Array], List[Array]]:
        """
        Split ensemble samples into model IDs and samples for each model.

        Parameters
        ----------
        ensemble_samples : Array
            Array of shape `(nrefinement_vars + nvars, nsamples)` containing the ensemble samples.

        Returns
        -------
        unique_model_ids : Array
            Array of shape `(nrefinement_vars, nmodels)` containing the unique model IDs.
        samples_per_model : List[Array]
            List of arrays containing the samples for each model.
        sample_idx_per_unique_model : List[Array]
            List of arrays containing the indices of samples for each unique model.

        Notes
        -----
        This method splits the ensemble samples into the refinement variables (`model_ids`)
        and the samples for each model.
        """
        samples = ensemble_samples[: self.nrefinement_vars()]
        sample_model_ids = self._bkd.asarray(
            ensemble_samples[-self.nrefinement_vars() :], dtype=int
        )
        unique_model_idx, sample_idx_per_unique_model = (
            unique_matrix_row_indices(sample_model_ids.T, bkd=self._bkd)
        )
        unique_model_ids = sample_model_ids[:, unique_model_idx]
        samples_per_model = []
        for ii in range(unique_model_ids.shape[1]):
            samples_per_model.append(
                samples[:, sample_idx_per_unique_model[ii]]
            )
        return unique_model_ids, samples_per_model, sample_idx_per_unique_model

    def combine_values(
        self, values_per_model: List[Array], sample_idx_per_model: List[Array]
    ) -> Array:
        """
        Combine values from multiple models into a single array.

        Parameters
        ----------
        values_per_model : List[Array]
            List of arrays containing the values computed by each model.
        sample_idx_per_model : List[Array]
            List of arrays containing the indices of samples for each model.

        Returns
        -------
        ensemble_values : Array
            Array containing the combined values, reordered to match the original sample order.

        Notes
        -----
        This method combines the values computed by multiple models and reorders them
        to match the order of the input samples provided to `split_ensemble_samples`.
        """
        ensemble_values = self._bkd.vstack(values_per_model)[
            self._bkd.hstack(sample_idx_per_model)
        ]
        return ensemble_values

    def highest_fidelity_model(self) -> Model:
        """
        Retrieve the highest fidelity model in the ensemble.

        Returns
        -------
        highest_fidelity_model : Model
            The highest fidelity model in the ensemble.

        Notes
        -----
        The highest fidelity model corresponds to the maximum values of `index_bounds`.
        """
        return self.get_model(self._index_bounds)


class DenseMatrixLinearModel(Model):
    """
    Dense matrix linear model.

    This class implements a linear model defined by a dense matrix and an optional
    vector. The model computes values, Jacobians, and Hessians based on the provided
    matrix and vector.

    The model is defined as:

    .. math::
        f(x) = Mx + v

    where:

    - :math:`M`: Dense matrix of shape `(nqoi, nvars)`.
    - :math:`x`: Input sample vector of shape `(nvars, 1)`.
    - :math:`v`: Optional vector of shape `(nqoi, 1)`.

    Parameters
    ----------
    matrix : Array
        Dense matrix defining the linear transformation.
    vec : Array, optional
        Vector to be added to the linear transformation. Default is a zero vector.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        matrix: Array,
        vec: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the dense matrix linear model.

        Parameters
        ----------
        matrix : Array
            Dense matrix defining the linear transformation.
        vec : Array, optional
            Vector to be added to the linear transformation. Default is a zero vector.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._nqoi, self._nvars = matrix.shape
        self._matrix = matrix
        if vec is None:
            vec = backend.zeros((self.nqoi(), 1))
        self._vec = vec
        super().__init__(backend=backend)

    def jacobian_implemented(self):
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True, since the Jacobian is implemented.
        """
        return True

    def apply_jacobian_implemented(self):
        """
        Check if the apply Jacobian method is implemented.

        Returns
        -------
        apply_jacobian_implemented : bool
            True, since the apply Jacobian method is implemented.
        """
        return True

    def hessian_implemented(self):
        """
        Check if the Hessian is implemented.

        Returns
        -------
        hessian_implemented : bool
            True, since the Hessian is implemented.
        """
        return True

    def apply_hessian_implemented(self):
        """
        Check if the apply Hessian method is implemented.

        Returns
        -------
        apply_hessian_implemented : bool
            True, since the apply Hessian method is implemented.
        """
        return True

    def apply_weighted_hessian_implemented(self):
        """
        Check if the apply weighted Hessian method is implemented.

        Returns
        -------
        apply_weighted_hessian_implemented : bool
            True, since the apply weighted Hessian method is implemented.
        """
        return True

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            Number of variables in the model.
        """
        return self._nvars

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi : int
            Number of quantities of interest in the model.
        """
        return self._nqoi

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Model evaluations at the given samples.

        Notes
        -----
        The model computes:

        .. math::
            f(x) = Mx + v
        """
        return (self._matrix @ (samples) + self._vec).T

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the model.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        jacobian : Array
            Jacobian matrix of the model.

        Notes
        -----
        The Jacobian is constant and equal to the matrix :math:`M`.
        """
        return self._matrix

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Jacobian to a vector.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the Jacobian to.

        Returns
        -------
        applied_jacobian : Array
            Result of applying the Jacobian to the vector.

        Notes
        -----
        The result is computed as:

        .. math::
            Jv = Mv
        """
        return self._matrix @ vec

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the model.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        hessian : Array
            Hessian matrix of the model.

        Notes
        -----
        The Hessian is zero for a linear model.
        """
        return self._bkd.zeros((self.nqoi(), self.nvars(), self.nvars()))

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Hessian to a vector.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the Hessian to.

        Returns
        -------
        applied_hessian : Array
            Result of applying the Hessian to the vector.

        Notes
        -----
        The result is zero for a linear model.
        """
        return self._bkd.zeros((self.nvars(), 1))

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Apply the weighted Hessian to a vector.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the weighted Hessian to.
        weights : Array
            Weights for the weighted Hessian.

        Returns
        -------
        applied_weighted_hessian : Array
            Result of applying the weighted Hessian to the vector.

        Notes
        -----
        The result is zero for a linear model.
        """
        return self._bkd.zeros((self.nvars(), 1))

    def matrix(self) -> Array:
        """
        Return the dense matrix defining the linear transformation.

        Returns
        -------
        matrix : Array
            Dense matrix defining the linear transformation.
        """
        return self._matrix

    def vector(self) -> Array:
        """
        Return the vector added to the linear transformation.

        Returns
        -------
        vector : Array
            Vector added to the linear transformation.
        """
        return self._vec


class QuadraticMatrixModel(Model):
    """
    Quadratic matrix model.

    This class implements a model defined by a quadratic transformation of a dense matrix.
    The model computes values, Jacobians, and Hessians based on the provided matrix.

    The model is defined as:

    .. math::
        f(x) = (Mx)^2

    where:

    - :math:`M`: Dense matrix of shape `(nqoi, nvars)`.
    - :math:`x`: Input sample vector of shape `(nvars, 1)`.

    Parameters
    ----------
    matrix : Array
        Dense matrix defining the quadratic transformation.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, matrix: Array, backend=NumpyMixin):
        """
        Initialize the quadratic matrix model.

        Parameters
        ----------
        matrix : Array
            Dense matrix defining the quadratic transformation.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._matrix = matrix
        super().__init__(backend)

    def jacobian_implemented(self):
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True, since the Jacobian is implemented.
        """
        return True

    def hessian_implemented(self):
        """
        Check if the Hessian is implemented.

        Returns
        -------
        hessian_implemented : bool
            True, since the Hessian is implemented.
        """
        return True

    def apply_hessian_implemented(self):
        """
        Check if the apply Hessian method is implemented.

        Returns
        -------
        apply_hessian_implemented : bool
            True, since the apply Hessian method is implemented.
        """
        return True

    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            Number of variables in the model.
        """
        return self._matrix.shape[1]

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi : int
            Number of quantities of interest in the model.
        """
        return self._matrix.shape[0]

    def _values(self, samples: Array) -> Array:
        """
        Evaluate the model at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Model evaluations at the given samples.

        Notes
        -----
        The model computes:

        .. math::
            f(x) = (Mx)^2
        """
        return (self._matrix @ (samples)).T ** 2

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        jacobian : Array
            Jacobian matrix of the model.

        Notes
        -----
        The Jacobian is computed as:

        .. math::
            J = 2 (Mx) M
        """
        return 2 * (self._matrix @ sample) * self._matrix

    def _hessian(self, sample: Array) -> Array:
        """
        Compute the Hessian of the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        hessian : Array
            Hessian matrix of the model.

        Notes
        -----
        The Hessian is computed as:

        .. math::
            H = 2 M_i^T M_i

        where :math:`M_i` is the :math:`i`-th row of the matrix :math:`M`.
        """
        return 2 * self._bkd.stack(
            [row[:, None] * row[None, :] for row in self._matrix], axis=0
        )


class CostFunction:
    """
    Base class for cost functions.

    This class defines a generic cost function interface that can be used to compute
    the cost associated with a model or subspace index. Derived classes must implement
    specific cost computation logic.
    """

    def set_nrefinement_vars(self, nrefinement_vars: int):
        """
        Set the number of refinement variables.

        Parameters
        ----------
        nrefinement_vars : int
            Number of refinement variables.

        Returns
        -------
        None
        """
        self._nrefinement_vars = nrefinement_vars

    def __call__(self, subspace_index: Array) -> float:
        """
        Compute the cost for the given subspace index.

        Parameters
        ----------
        subspace_index : Array
            Subspace index for which the cost is computed.

        Returns
        -------
        cost : float
            Cost associated with the given subspace index.

        Raises
        ------
        RuntimeError
            If `set_nrefinement_vars()` has not been called.
        """
        if not hasattr(self, "_nrefinement_vars"):
            raise RuntimeError("must call set_nrefinement_vars()")
        return 1


class ModelListCostFunction(CostFunction):
    """
    Cost function for a list of models indexed by an integer.

    This class implements a cost function for a list of models, where each model
    is indexed by an integer. The cost for each model is specified in the `costs`
    array.

    Parameters
    ----------
    costs : Array
        Array containing the cost of each model.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, costs: Array, backend: BackendMixin = NumpyMixin):
        """
        Initialize the model list cost function.

        Parameters
        ----------
        costs : Array
            Array containing the cost of each model.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._bkd = backend
        self._costs = self._bkd.asarray(costs)
        self._nmodels = self._costs.shape[0]
        self.set_nrefinement_vars(1)

    def __call__(self, subspace_index: int) -> float:
        """
        Compute the cost for the given subspace index.

        Parameters
        ----------
        subspace_index : int
            Index of the model for which the cost is computed.

        Returns
        -------
        cost : float
            Cost associated with the given subspace index.

        Raises
        ------
        ValueError
            If `subspace_index` is greater than or equal to the number of models.
        """
        if subspace_index >= self._nmodels:
            raise ValueError("subspace_index >= nmodels")
        return self._costs[subspace_index]

    def cost_per_model(self) -> Array:
        """
        Return the cost of each model.

        Returns
        -------
        costs : Array
            Array containing the cost of each model.
        """
        return self._costs


class AdjointModel(SingleSampleModel, ABC):
    """
    Base class for adjoint models.

    This abstract base class defines an interface for models that implement adjoint
    equations to compute gradient information. Derived classes must implement methods
    for forward solving, setting parameters, evaluating functionals, and computing
    adjoint-based gradients and Hessians.

    Parameters
    ----------
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the adjoint model.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the backend supports Jacobian computation, False otherwise.
        """
        return self._bkd.jacobian_implemented()

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) in the model.

        Returns
        -------
        nqoi : int
            Number of quantities of interest in the model.
        """
        return self._functional.nqoi()

    @abstractmethod
    def nvars(self) -> int:
        """
        Return the number of variables in the model.

        Returns
        -------
        nvars : int
            Number of variables in the model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def _fwd_solve(self):
        """
        Perform the forward solve.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_solve(self, sample):
        """
        Perform the forward solve for the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def set_param(self, param: Array):
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_functional(self) -> Array:
        """
        Evaluate the functional.

        Returns
        -------
        functional_values : Array
            Values of the functional.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def _evaluate(self, sample: Array) -> Array:
        """
        Evaluate the model at the given sample.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        values : Array
            Model evaluation results.

        Notes
        -----
        This method sets the model parameters, performs the forward solve, and evaluates the functional.
        """
        self.set_param(sample[:, 0])
        self._fwd_solve()
        return self._eval_functional()

    @abstractmethod
    def _jacobian_from_adjoint(self) -> Array:
        """
        Compute the Jacobian using adjoint equations.

        Returns
        -------
        jacobian : Array
            Jacobian matrix computed using adjoint equations.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the model at the given sample using adjoint equations.

        Parameters
        ----------
        sample : Array
            Input sample as an array.

        Returns
        -------
        jacobian : Array
            Jacobian matrix computed using adjoint equations.
        """
        self.set_param(sample[:, 0])
        return self._jacobian_from_adjoint()

    def _apply_hessian_from_adjoint(self, vec: Array) -> Array:
        """
        Compute the Hessian-vector product using adjoint equations.

        Parameters
        ----------
        vec : Array
            Vector to apply the Hessian to.

        Returns
        -------
        applied_hessian : Array
            Result of applying the Hessian to the vector.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError("_hessian_from_adjoint not implemented")

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply the Hessian to a vector using adjoint equations.

        Parameters
        ----------
        sample : Array
            Input sample as an array.
        vec : Array
            Vector to apply the Hessian to.

        Returns
        -------
        applied_hessian : Array
            Result of applying the Hessian to the vector.
        """
        self.set_param(sample[:, 0])
        return self._apply_hessian_from_adjoint(vec[:, 0])[:, None]


class ScalarElementwiseFunction(ABC):
    """
    Abstract base class for scalar elementwise functions.

    This class defines an interface for scalar functions that operate elementwise
    on input samples. It supports computation of values, first derivatives, second
    derivatives, and third derivatives, along with methods for checking the correctness
    of derivatives using finite differences.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the input samples.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, ndim: int, backend: BackendMixin = NumpyMixin):
        """
        Initialize the scalar elementwise function.

        Parameters
        ----------
        ndim : int
            Number of dimensions of the input samples.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._ndim = ndim
        self._bkd = backend

    def _check_samples(self, samples: Array):
        """
        Validate the input samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Raises
        ------
        ValueError
            If the samples do not have the expected number of dimensions.
        """
        if samples.ndim != self._ndim:
            raise ValueError(
                "samples has the wrong dimension. Samples has shape "
                f"{samples.shape} but should have ndim={self._ndim}"
            )

    def _check_values(self, samples: Array, values: Array):
        """
        Validate the output values.

        Parameters
        ----------
        samples : Array
            Input samples as an array.
        values : Array
            Output values as an array.

        Raises
        ------
        ValueError
            If the values do not match the shape of the samples.
        """
        if samples.shape != values.shape:
            raise ValueError(
                f"values has the shape {values.shape}"
                f"but should be {samples.shape}"
            )

    @abstractmethod
    def _values(self, samples: Array) -> Array:
        """
        Compute the values of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Values of the function at the given samples.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def _first_derivative(self, samples: Array) -> Array:
        """
        Compute the first derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        first_derivative : Array
            First derivative of the function at the given samples.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def _second_derivative(self, samples: Array) -> Array:
        """
        Compute the second derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        second_derivative : Array
            Second derivative of the function at the given samples.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def _third_derivative(self, samples: Array) -> Array:
        """
        Compute the third derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        third_derivative : Array
            Third derivative of the function at the given samples.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a derived class.
        """
        raise NotImplementedError

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        values : Array
            Values of the function at the given samples.

        Notes
        -----
        This method validates the input samples, computes the values, and validates the output values.
        """
        self._check_samples(samples)
        vals = self._values(samples)
        self._check_values(samples, vals)
        return vals

    def first_derivative_implemented(self) -> bool:
        """
        Check if the first derivative is implemented.

        Returns
        -------
        first_derivative_implemented : bool
            True if the backend supports Jacobian computation, False otherwise.
        """
        return self._bkd.jacobian_implemented()

    def first_derivative(self, samples: Array) -> Array:
        """
        Compute the first derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        first_derivative : Array
            First derivative of the function at the given samples.

        Notes
        -----
        This method validates the input samples, computes the first derivative, and validates the output values.
        """
        self._check_samples(samples)
        vals = self._first_derivative(samples)
        self._check_values(samples, vals)
        return vals

    def second_derivative_implemented(self) -> bool:
        """
        Check if the second derivative is implemented.

        Returns
        -------
        second_derivative_implemented : bool
            True if the backend supports Hessian computation, False otherwise.
        """
        return self._bkd.hessian_implemented()

    def second_derivative(self, samples: Array) -> Array:
        """
        Compute the second derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        second_derivative : Array
            Second derivative of the function at the given samples.

        Notes
        -----
        This method validates the input samples, computes the second derivative, and validates the output values.
        """
        self._check_samples(samples)
        vals = self._second_derivative(samples)
        self._check_values(samples, vals)
        return vals

    def third_derivative_implemented(self) -> bool:
        """
        Check if the third derivative is implemented.

        Returns
        -------
        third_derivative_implemented : bool
            True if the backend supports Hessian computation, False otherwise.
        """
        return self._bkd.hessian_implemented()

    def third_derivative(self, samples: Array) -> Array:
        """
        Compute the third derivative of the function at the given samples.

        Parameters
        ----------
        samples : Array
            Input samples as an array.

        Returns
        -------
        third_derivative : Array
            Third derivative of the function at the given samples.

        Notes
        -----
        This method validates the input samples, computes the third derivative, and validates the output values.
        """
        self._check_samples(samples)
        vals = self._third_derivative(samples)
        self._check_values(samples, vals)
        return vals

    def _check_derivative(
        self,
        samples: Array,
        fun: callable,
        grad: callable,
        symb: str,
        fd_eps: Array = None,
        relative: bool = True,
        disp: bool = False,
    ):
        """
        Check the correctness of a derivative using finite differences.

        Parameters
        ----------
        samples : Array
            Input samples as an array.
        fun : callable
            Function to evaluate.
        grad : callable
            Derivative function to evaluate.
        symb : str
            Symbol representing the derivative (e.g., "f", "g", "h").
        fd_eps : Array, optional
            Array of finite difference step sizes. Default is logarithmically spaced values.
        relative : bool, optional
            Whether to compute relative errors. Default is True.
        disp : bool, optional
            Whether to display the errors. Default is False.

        Returns
        -------
        errors : Array
            Array of errors between the finite difference approximation and the computed derivative.

        Notes
        -----
        This method computes finite difference approximations of the derivative and compares them
        to the computed derivative, returning the errors.
        """
        if fd_eps is None:
            fd_eps = self._bkd.flip(self._bkd.logspace(-13, 0, 14))

        vals = fun(samples)
        grad = grad(samples)
        errors = []

        row_format = "{:<12} {:<25} {:<25} {:<25}"
        headers = [
            "Eps",
            "norm(grad {0})".format(symb),
            "norm(fd {0})".format(symb),
            "Rel. Errors" if relative else "Abs. Errors",
        ]
        if disp:
            print(row_format.format(*headers))
        row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
        for ii in range(fd_eps.shape[0]):
            perturbed_samples = samples + fd_eps[ii]
            perturbed_vals = fun(perturbed_samples)
            fd = (perturbed_vals - vals) / fd_eps[ii]
            errors.append(self._bkd.norm(fd - grad))
            if disp:
                print(
                    row_format.format(
                        fd_eps[ii],
                        self._bkd.norm(grad),
                        self._bkd.norm(fd),
                        errors[ii],
                    )
                )
        return self._bkd.asarray(errors)

    def check_first_derivative(
        self,
        samples: Array,
        fd_eps: Array = None,
        relative: bool = True,
        disp: bool = False,
    ):
        """
        Check the correctness of the first derivative using finite differences.

        Parameters
        ----------
        samples : Array
            Input samples as an array.
        fd_eps : Array, optional
            Array of finite difference step sizes. Default is logarithmically spaced values.
        relative : bool, optional
            Whether to compute relative errors. Default is True.
        disp : bool, optional
            Whether to display the errors. Default is False.

        Returns
        -------
        errors : Array
            Array of errors between the finite difference approximation and the computed first derivative.
        """
        return self._check_derivative(
            samples,
            self.__call__,
            self._first_derivative,
            "f",
            fd_eps,
            relative,
            disp,
        )

    def check_second_derivative(
        self,
        samples: Array,
        fd_eps: Array = None,
        relative: bool = True,
        disp: bool = False,
    ):
        """
        Check the correctness of the second derivative using finite differences.

        Parameters
        ----------
        samples : Array
            Input samples as an array.
        fd_eps : Array, optional
            Array of finite difference step sizes. Default is logarithmically spaced values.
        relative : bool, optional
            Whether to compute relative errors. Default is True.
        disp : bool, optional
            Whether to display the errors. Default is False.

        Returns
        -------
        errors : Array
            Array of errors between the finite difference approximation and the computed second derivative.
        """
        return self._check_derivative(
            samples,
            self._first_derivative,
            self._second_derivative,
            "g",
            fd_eps,
            relative,
            disp,
        )

    def check_third_derivative(
        self,
        samples: Array,
        fd_eps: Array = None,
        relative: bool = True,
        disp: bool = False,
    ):
        """
        Check the correctness of the third derivative using finite differences.

        Parameters
        ----------
        samples : Array
            Input samples as an array.
        fd_eps : Array, optional
            Array of finite difference step sizes. Default is logarithmically spaced values.
        relative : bool, optional
            Whether to compute relative errors. Default is True.
        disp : bool, optional
            Whether to display the errors. Default is False.

        Returns
        -------
        errors : Array
            Array of errors between the finite difference approximation and the computed third derivative.
        """
        return self._check_derivative(
            samples,
            self._second_derivative,
            self._third_derivative,
            "h",
            fd_eps,
            relative,
            disp,
        )


def expand_samples_from_indices(
    reduced_samples: Array,
    active_var_indices: Array,
    inactive_var_indices: Array,
    inactive_var_values: Array,
    bkd: BackendMixin = NumpyMixin,
):
    """
    Expand reduced samples into full samples by incorporating inactive variable values.

    This function takes a set of reduced samples (samples corresponding to active variables only) and expands them into full samples by combining them with fixed values for inactive variables. The expanded samples are arranged such that the active and inactive variables are placed in their original positions.

    Parameters
    ----------
    reduced_samples : Array
        A 2D array of samples corresponding to active variables only. The shape of this array must be `(num_active_vars, num_samples)`.
    active_var_indices : Array
        A 1D array specifying the indices of the active variables in the full set of variables.
    inactive_var_indices : Array
        A 1D array specifying the indices of the inactive variables in the full set of variables.
    inactive_var_values : Array
        A 2D array specifying the fixed values for the inactive variables. The shape of this array must be `(num_inactive_vars, num_samples)`.
    bkd : BackendMixin, optional
        A backend for array operations, such as `NumpyMixin`. Defaults to `NumpyMixin`.

    Returns
    -------
    Array
        A 2D array of expanded samples, where the active and inactive variables are placed in their original positions. The shape of this array is `(num_total_vars, num_samples)`.

    Raises
    ------
    AssertionError
        If `reduced_samples` is not a 2D array.
    """
    assert reduced_samples.ndim == 2
    raw_samples = get_all_sample_combinations(
        inactive_var_values, reduced_samples, bkd
    )
    samples = bkd.empty(raw_samples.shape)
    samples[inactive_var_indices, :] = raw_samples[
        : inactive_var_indices.shape[0]
    ]
    samples[active_var_indices, :] = raw_samples[
        inactive_var_indices.shape[0] :
    ]
    return samples
