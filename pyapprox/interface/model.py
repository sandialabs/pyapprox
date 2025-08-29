import requests
import os
import subprocess
import signal
import time
import glob
import tempfile
from abc import ABC, abstractmethod
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from typing import List, Tuple, Union

import numpy as np
import umbridge
import matplotlib.pyplot as plt

from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.util.misc import (
    get_all_sample_combinations,
    unique_matrix_row_indices,
)
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


class ModelWorkTracker:
    def __init__(
        self, backend: BackendMixin = NumpyMixin, multiproc: bool = False
    ):
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
        self._active = active

    def update(self, eval_name: str, times: Array):
        # use list of arrays because hstacking arrays continually is slow
        # only hstack arrays when accessing wall_times
        if self._active:
            self._wall_times[eval_name].append(times)

    def average_wall_time(self, eval_name: str) -> float:
        if self.nevaluations(eval_name) == 0:
            return "?"
        # call self.walltimes() so _wall_times lists are concatenated into
        # an array
        wall_times = self.wall_times()[eval_name]
        # exclude failures from calculation
        wall_times = wall_times[wall_times != np.nan]
        return self._bkd.mean(wall_times)

    def nevaluations(self, eval_name: str) -> int:
        return self.wall_times()[eval_name].shape[0]

    def wall_times(self) -> dict:
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
    # TODO Add option to save results to file at certain frequency
    # TODO support different file systems, e.g. pickle, HDF5 etc.
    def __init__(self, backend: BackendMixin = NumpyMixin):
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
        Using a database has a overhead which can slow down computationally
        fast model evaluations.
        Use this function to activate the use of a database
        """
        self._active = True

    def isactive(self) -> bool:
        return self._active

    def _hash_sample(self, sample: Array) -> int:
        return hash(self._bkd.to_numpy(sample).tobytes())

    def _add_sample(self, eval_name: str, key, sample):
        if key not in self._samples_dict:
            self._samples_dict[key] = len(self._samples)
            self._samples.append(sample)

    def _add_values(self, eval_name: str, key: int, values: Array):
        # append values to a list
        self._values_dict[eval_name][key] = self._bkd.copy(values)

    def add_data(self, eval_name: str, samples: Array, values: Array):
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
    Evaluate a model at a single sample.

    Required functions:
    _values is required

    Optional functions:
    _jacobian, _apply_jacobian, _hessian _apply_hessian

    If optional functions are implemented set the corresponding functions to return True
    to True:
    jacobian_implemented,
    apply_jacobian_implemented,
    apply_hessian_implemented,
    apply_weighted_hessian_implemented
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        if not hasattr(backend, "isbackend"):
            raise ValueError("backend must be derived from LinAlgBase")
        self._bkd = backend
        self._work_tracker = ModelWorkTracker(self._bkd)
        self._database = ModelDataBase(self._bkd)

    def activate_model_data_base(self):
        """
        Using a database has a overhead which can slow down computationally
        fast model evaluations. Also using a database will corrupt auto
        differentiation results if used.
        Use this function to activate the use of a database.
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
        "Load in a database and worktracker from a previous study"
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
        return False

    def jacobian_implemented(self) -> bool:
        return False

    def apply_hessian_implemented(self) -> bool:
        return False

    def hessian_implemented(self) -> bool:
        return False

    def apply_weighted_hessian_implemented(self) -> bool:
        return False

    def weighted_hessian_implemented(self) -> bool:
        return False

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _values(self, samples: Array) -> Array:
        raise NotImplementedError("Must implement self._values")

    def _check_values_shape(self, samples: Array, vals: Array):
        if vals.shape != (samples.shape[1], self.nqoi()):
            raise RuntimeError(
                "{0}: values had shape {1} but should have shape {2}".format(
                    self, vals.shape, (samples.shape[1], self.nqoi())
                )
            )

    def _new_values(self, samples: Array) -> Array:
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
        if sample.shape != (self.nvars(), 1):
            raise ValueError(
                "sample must have shape {0} but had shape {1}".format(
                    (self.nvars(), 1), sample.shape
                )
            )

    def _check_samples_shape(self, sample: Array):
        if sample.shape[0] != self.nvars():
            raise ValueError(
                "sample must have nrows={0} but had shape {1}".format(
                    self.nvars(), sample.shape
                )
            )

    def _check_vec_shape(self, sample: Array, vec: Array):
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
        User provided function to compute the Jacobian.

        Default to using autograd to compute Jacobian.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.jacobian_implemented is False
        by default
        """
        if not self._bkd.jacobian_implemented():
            raise NotImplementedError
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[0], sample[:, 0]
        )

    def _check_jacobian_shape(self, jac: Array, sample: Array):
        if jac.shape != (self.nqoi(), sample.shape[0]):
            raise RuntimeError(
                "Jacobian returned by _jacobian has shape {0}"
                " but must be {1}".format(
                    jac.shape, (self.nqoi(), sample.shape[0])
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
        User provided function to compute the Jacobian vector product.

        Default to using autograd to compute Jacobian vector product.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.apply_jacobian_implemented is
        False by default
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
        return self._work_tracker

    def model_database(self) -> ModelDataBase:
        return self._database

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        User provided function to compute the Hessian vector product.

        Default to using autograd to compute Jacobian vector product.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.apply_hessian_implemented is False
        by default
        """
        if not self._bkd.hvp_implemented():
            raise NotImplementedError
        return self._bkd.hvp(
            lambda x: self.__call__(x[:, None])[0],
            sample[:, 0],
            vec[:, 0],
        )[:, None]

    def _check_hvp_shape(self, hvp: Array, sample: Array):
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
        User provided function to compute the Hessian.

        Default to using autograd to compute Jacobian vector product.
        However, the user must ensure that all methods envocked by __call__
        are differentiable. This is why self.hessian_implemented is False
        by default
        """
        if not self._bkd.hessian_implemented():
            raise NotImplementedError
        return self._bkd.hessian(
            lambda x: self.__call__(x[:, None])[0], sample[:, 0]
        )[None]

    def _check_hessian_shape(self, hess: Array, sample: Array):
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
        raise NotImplementedError

    def weighted_hessian(self, sample: Array, weights: Array) -> Array:
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
        Compare apply_jacobian with finite difference.
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
        # only used by check_apply_hessian
        return weights.T @ self.jacobian(sample)

    def _jacobian_from_apply_jacobian(self, sample: Array) -> Array:
        # Only use when checking apply_hessian.
        # It computes jacobian from apply jacobian which is necessary
        # to check apply_hessian. We do not want to support this for their
        # user as it can lead to large numbers of apply_jacobians
        # in typical model analyses.
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
        Compare apply_hessian with finite difference.
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

    def _plot_surface_1d(self, ax, qoi, plot_limits, npts_1d, **kwargs):
        plot_xx = self._bkd.linspace(*plot_limits, npts_1d[0])[None, :]
        ax.plot(plot_xx[0], self.__call__(plot_xx), **kwargs)

    def meshgrid_samples(
        self, plot_limits: Array, npts_1d: Union[Array, int] = 51
    ) -> Array:
        if self.nvars() != 2:
            raise RuntimeError(f"nvars = {self.nvars()} but must be 2")
        X, Y, pts = get_meshgrid_samples(plot_limits, npts_1d, bkd=self._bkd)
        return X, Y, pts

    def _plot_surface_2d(self, ax, qoi, plot_limits, npts_1d, **kwargs):
        if ax.name != "3d":
            raise ValueError("ax must use 3d projection")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        # X = self._bkd.to_numpy(X)
        # Y = self._bkd.to_numpy(Y)
        # Z = self._bkd.to_numpy(Z)
        return ax.plot_surface(X, Y, Z, **kwargs)

    def plot_surface(self, ax, plot_limits, qoi=0, npts_1d=51, **kwargs):
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

    def get_plot_axis(self, figsize=(8, 6), surface=False):
        if self.nvars() < 3 and not surface:
            fig = plt.figure(figsize=figsize)
            return fig, fig.gca()
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(111, projection="3d")

    def plot_contours(self, ax, plot_limits, qoi=0, npts_1d=51, **kwargs):
        if self.nvars() != 2:
            raise ValueError("Can only plot contours for 2D functions")
        X, Y, pts = self.meshgrid_samples(plot_limits, npts_1d)
        vals = self.__call__(pts)
        Z = self._bkd.reshape(vals[:, qoi], X.shape)
        return ax.contourf(X, Y, Z, **kwargs)


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
        Parameters
        ----------
        samples_ndim : integer
            The dimension of the Array accepted by function in [1, 2]

        values_ndim : integer
            The dimension of the Array returned by function in [0, 1, 2]
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


class ScipyModelWrapper:
    def __init__(self, model):
        """
        Create a API that takes a sample as a 1D Array and returns
        the objects needed by scipy optimizers. E.g.
        jac will return Array
        even when model accepts and returns arrays associated with a
        different backend
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
        return self._bkd.to_numpy(
            self._model.apply_hessian(
                sample[:, None], self._bkd.asarray(vec[:, None])
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


class UmbridgeModelWrapper(Model):
    def __init__(self, umb_model, config={}, nprocs=1, backend=NumpyMixin):
        """
        Evaluate an umbridge model at multiple samples

        Notes:
        Sometimes port can be left open. On linux and osx the PID of the
        process using the port 4242 can be checked using lsof -i :4242
        """
        super().__init__(backend=backend)
        if not isinstance(umb_model, umbridge.HTTPModel):
            raise ValueError("model is not an umbridge.HTTPModel")
        self._model = umb_model
        self._config = config
        self._nprocs = nprocs
        self._nmodel_evaluations = 0

    def nqoi(self) -> int:
        return self._model.get_output_sizes(self._config)[0]

    def nvars(self) -> int:
        return self._model.get_input_sizes(self._config)[0]

    def apply_jacobian_implemented(self) -> bool:
        return self._model.supports_apply_jacobian()

    def jacobian_implemented(self) -> bool:
        return self._model.supports_gradient()

    def apply_hessian_implemented(self) -> bool:
        return self._model.supports_apply_hessian()

    def _check_sample(self, sample):
        if sample.ndim != 2:
            raise ValueError(
                "sample is not a 2D array, has shape {0}".format(sample.shape)
            )
        return [sample[:, 0].tolist()]

    def _jacobian(self, sample):
        # self._model.gradient computes the v * Jac
        # umbridge models accept a list of lists. Each sub list represents
        # a subset of the total model parameters. Here we just assume
        # that there is only one sublist
        # in_wrt specifies which sublist to take the gradient with respect to
        # because we assume only one sublist inWrt=0
        # out_wrt specifies which output sublist to take the gradient of
        # because we assume only one sublist outWrt=0
        # sens is vector v and applies a constant to each sublist of outputs
        # we want jacobian so set sens to [1]
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.gradient(0, 0, parameters, [1.0], config=self._config)
        ).T

    def _apply_jacobian(self, sample, vec):
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.apply_jacobian(
                None, None, parameters, vec, config=self._config
            )
        )

    def _apply_hessian(self, sample, vec):
        parameters = self._check_sample(sample)
        return self._bkd.asarray(
            self._model.apply_hessian(
                None, None, None, parameters, vec, None, config=self._config
            )
        )

    def _evaluate_single_thread(self, sample, sample_id):
        parameters = self._check_sample(sample)
        return self._model(parameters, config=self._config)[0]

    def _evaluate_parallel(self, samples):
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
                    UmbridgeModelWrapper.kill_server(process, out)
                    raise RuntimeError("Could not connect to server") from None
        return process, out

    @staticmethod
    def kill_server(process, out=None):
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        if out is not None:
            out.close()


class UmbridgeIOModelWrapper(UmbridgeModelWrapper):
    def __init__(
        self,
        umb_model,
        config={},
        nprocs=1,
        outdir_basename="modelresults",
        backend=NumpyMixin,
    ):
        """
        Evaluate an umbridge model that wraps models that require
        creation of separate directories for each model run to enable
        loading and writing of files
        """
        super().__init__(umb_model, config, nprocs, backend=backend)
        self._outdir_basename = outdir_basename

    def _evaluate_single_thread(self, sample, sample_id):
        parameters = self._check_sample(sample)
        config = self._config.copy()
        config["outdir_basename"] = os.path.join(
            self._outdir_basename, "wdir-{0}".format(sample_id)
        )
        return self._model(parameters, config=config)[0]


class UmbridgeIOModelEnsembleWrapper(UmbridgeModelWrapper):
    def __init__(
        self,
        umb_model,
        model_configs={},
        nprocs=1,
        outdir_basename="modelresults",
        backend=NumpyMixin,
    ):
        """
        Evaluate an umbridge model with multiple configs that wraps models
        that require
        creation of separate directories for each model run to enable
        loading and writing of files
        """
        super().__init__(umb_model, None, nprocs, backend=backend)
        self._outdir_basename = outdir_basename
        self._model_configs = model_configs

    def _evaluate_single_thread(self, full_sample, sample_id):
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
        Base class for models that require loading and or writing of files
        """
        super().__init__(backend=backend)
        self._nqoi = nqoi
        self._nvars = nvars
        self._infilenames = infilenames
        save_values = ["full", "limited", "no"]
        if save not in save_values:
            raise ValueError("save must be in {0}".format(save_values))
        if outdir_basename is None and save != "no":
            msg = " You are requesting temporary files but save not set to no"
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
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def _create_outdir(self):
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
        filenames_to_delete = glob.glob(os.path.join(outdirname, "*"))
        for filename in filenames_to_delete:
            os.remove(filename)

    def _save_samples_and_values(self, sample, values, outdirname):
        if self._datafilename is not None:
            filename = os.path.join(outdirname, self._datafilename)
            np.savez(filename, sample=sample, values=values)

    def _process_outdir(self, sample, values, outdirname, tmpdir):
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
        raise NotImplementedError

    def _evaluate(self, sample: Array):
        outdirname, tmpdir = self._create_outdir()
        self._nmodel_evaluations += 1
        linked_filenames = self._link_files(outdirname)
        values = self._bkd.asarray(
            self._run(sample, linked_filenames, outdirname)
        )
        self._process_outdir(sample, values, outdirname, tmpdir)
        return values


class SerialIOModel(IOModel):
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
        curdirname = os.getcwd()
        os.chdir(outdirname)
        np.savetxt(self._params_filename, sample)
        self._run_shell_command()
        vals = np.loadtxt(self._results_filename, usecols=[0])
        os.chdir(curdirname)
        return self._bkd.atleast2d(self._bkd.asarray(vals))


class AsyncIOModel(SerialIOModel):
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
        outdirname, tmpdir = self._create_outdir()
        curdirname = os.getcwd()
        t0 = time.time()
        self._link_files(outdirname)
        os.chdir(outdirname)
        np.savetxt(self._params_filename, sample)
        proc, writefile = self._run_shell_command()
        os.chdir(curdirname)
        # store proc_id and outdirname for this sample
        # note proc_id is equal to self._nmodel_evaluations at the time
        # sample is dispatched
        self._running_workdirs[self._nmodel_evaluations] = (
            outdirname,
            tmpdir,
            proc,
            writefile,
            t0,
        )
        self._nmodel_evaluations += 1

    def _newly_completed_sample_ids(self) -> List:
        completed_sample_ids = []
        for sample_id, item in self._running_workdirs.items():
            proc = item[2]
            if proc.poll() is not None:
                completed_sample_ids.append(sample_id)
        return completed_sample_ids

    def _load_sample_result(self, sample_id: int) -> Array:
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
                    "Values returned in {0} had the incorrect shape".foramt(
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
        # sort values so that they are returned in the order samples
        # was given to call. Asnchornous call will usually result
        # in samples being completed out of order.
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


class ActiveSetVariableModel(Model):
    r"""
    Create a model wrapper that only accepts a subset of the model variables.
    """

    def __init__(
        self,
        model,
        nvars,
        inactive_var_values,
        active_var_indices,
        base_model=None,
        backend=NumpyMixin,
    ):
        super().__init__(backend=model._bkd)
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
        assert self._bkd.all(self._active_var_indices < self._norignal_vars)
        self._inactive_var_indices = self._bkd.delete(
            self._bkd.arange(self._norignal_vars, dtype=int),
            active_var_indices,
        )
        if base_model is None:
            base_model = model
        self._base_model = base_model

    def apply_jacobian_implemented(self) -> bool:
        return self._base_model.apply_jacobian_implemented()

    def jacobian_implemented(self) -> bool:
        return self._base_model.jacobian_implemented()

    def apply_hessian_implemented(self) -> bool:
        return (
            self._base_model.apply_hessian_implemented()
            or self._base_model.hessian_implemented()
        )

    def nqoi(self) -> int:
        return self._model.nqoi()

    def nvars(self) -> int:
        return self._active_var_indices.shape[0]

    @staticmethod
    def _expand_samples_from_indices(
        reduced_samples,
        active_var_indices,
        inactive_var_indices,
        inactive_var_values,
        bkd=NumpyMixin,
    ):
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

    def _expand_samples(self, reduced_samples):
        return self._expand_samples_from_indices(
            reduced_samples,
            self._active_var_indices,
            self._inactive_var_indices,
            self._inactive_var_values,
            self._bkd,
        )

    def _values(self, reduced_samples):
        samples = self._expand_samples(reduced_samples)
        return self._model(samples)

    def _jacobian(self, reduced_samples):
        samples = self._expand_samples(reduced_samples)
        jac = self._model.jacobian(samples)
        return jac[:, self._active_var_indices]

    def _apply_jacobian(self, reduced_samples, vec):
        samples = self._expand_samples(reduced_samples)
        # set inactive entries of vec to zero when peforming
        # matvec product so they do not contribute to sum
        expanded_vec = self._bkd.zeros((self._norignal_vars, 1))
        expanded_vec[self._active_var_indices] = vec
        return self._model.apply_jacobian(samples, expanded_vec)

    def _apply_hessian(self, reduced_samples, vec):
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

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class ChangeModelSignWrapper(Model):
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


class MultiIndexModelEnsemble(ABC):
    def __init__(
        self, index_bounds: List[int], backend: BackendMixin = NumpyMixin
    ):
        self._bkd = backend
        self._models = dict()
        self._index_bounds = self._bkd.asarray(index_bounds, dtype=int)
        # index bounds supports values in [0, N]
        # Note upper and lower bounds are both included
        if self._index_bounds.min() < 1:
            raise ValueError("index bounds must have entries > 0")

    def _hash_model_id(self, model_id: Array) -> int:
        return hash(self._bkd.to_numpy(model_id).tobytes())

    def get_model(self, model_id: Array) -> Model:
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
        raise NotImplementedError

    def nrefinement_vars(self) -> int:
        return len(self._index_bounds)

    def nmodels(self) -> int:
        return self._bkd.prod(self._index_bounds)

    def split_ensemble_samples(
        self, ensemble_samples: Array
    ) -> Tuple[Array, List[Array], List[Array]]:
        # split into samples and model_ids
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
    ):
        # values reordered to match the order the samples entered
        # split_ensemble_samples
        ensemble_values = self._bkd.vstack(values_per_model)[
            self._bkd.hstack(sample_idx_per_model)
        ]
        return ensemble_values

    def highest_fidelity_model(self):
        return self.get_model(self._index_bounds)


class DenseMatrixLinearModel(Model):
    def __init__(
        self,
        matrix: Array,
        vec: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        self._nqoi, self._nvars = matrix.shape
        self._matrix = matrix
        if vec is None:
            vec = backend.zeros((self.nqoi(), 1))
        self._vec = vec
        super().__init__(backend=backend)

    def jacobian_implemented(self):
        return True

    def apply_jacobian_implemented(self):
        return True

    def hessian_implemented(self):
        return True

    def apply_hessian_implemented(self):
        return True

    def apply_weighted_hessian_implemented(self):
        return True

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def _values(self, samples: Array) -> Array:
        return (self._matrix @ (samples) + self._vec).T

    def _jacobian(self, sample: Array) -> Array:
        return self._matrix

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        return self._matrix @ vec

    def _hessian(self, sample: Array) -> Array:
        return self._bkd.zeros((self.nqoi(), self.nvars(), self.nvars()))

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return self._bkd.zeros((self.nvars(), 1))

    def _apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(), 1))

    def matrix(self) -> Array:
        return self._matrix

    def vector(self) -> Array:
        return self._vec


class QuadraticMatrixModel(Model):
    def __init__(self, matrix: Array, backend=NumpyMixin):
        self._matrix = matrix
        super().__init__(backend)

    def jacobian_implemented(self):
        return True

    def hessian_implemented(self):
        return True

    def apply_hessian_implemented(self):
        return True

    # def apply_weighted_hessian_implemented(self):
    #     return True

    def nvars(self) -> int:
        return self._matrix.shape[1]

    def nqoi(self) -> int:
        return self._matrix.shape[0]

    def _values(self, samples: Array) -> Array:
        return (self._matrix @ (samples)).T ** 2

    def _jacobian(self, sample: Array) -> Array:
        # warning jacobian will be zero when sample is zero
        return 2 * (self._matrix @ sample) * self._matrix

    def _hessian(self, sample: Array) -> Array:
        return 2 * self._bkd.stack(
            [row[:, None] * row[None, :] for row in self._matrix], axis=0
        )


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


class CostFunction:
    def set_nrefinement_vars(self, nrefinement_vars: int):
        self._nrefinement_vars = nrefinement_vars

    def __call__(self, subspace_index: Array) -> float:
        if not hasattr(self, "_nrefinement_vars"):
            raise RuntimeError("must call set_nrefinement_vars()")
        return 1


class ModelListCostFunction(CostFunction):
    def __init__(self, costs: Array, backend: BackendMixin = NumpyMixin):
        "Cost function for a list of models indexed by an integer"
        self._bkd = backend
        self._costs = self._bkd.asarray(costs)
        self._nmodels = self._costs.shape[0]
        self.set_nrefinement_vars(1)

    def __call__(self, subspace_index: int) -> float:
        if subspace_index >= self._nmodels:
            raise ValueError("subspace_index >= nmodels")
        return self._costs[subspace_index]

    def cost_per_model(self) -> Array:
        return self._costs


class AdjointModel(SingleSampleModel, ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def nqoi(self) -> int:
        return self._functional.nqoi()

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _fwd_solve(self):
        raise NotImplementedError

    @abstractmethod
    def forward_solve(self, sample):
        raise NotImplementedError

    @abstractmethod
    def set_param(self, param: Array):
        raise NotImplementedError

    @abstractmethod
    def _eval_functional(self) -> Array:
        raise NotImplementedError

    def _evaluate(self, sample: Array) -> Array:
        self.set_param(sample[:, 0])
        self._fwd_solve()
        return self._eval_functional()

    @abstractmethod
    def _jacobian_from_adjoint(self) -> Array:
        raise NotImplementedError

    def _jacobian(self, sample: Array) -> Array:
        self.set_param(sample[:, 0])
        return self._jacobian_from_adjoint()

    def _apply_hessian_from_adjoint(self, vec: Array) -> Array:
        raise NotImplementedError("_hessian_from_adjoint not implemented")

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        self.set_param(sample[:, 0])
        return self._apply_hessian_from_adjoint(vec[:, 0])[:, None]


class ScalarElementwiseFunction(ABC):
    def __init__(self, ndim: int, backend: BackendMixin = NumpyMixin):
        self._ndim = ndim
        self._bkd = backend

    def _check_samples(self, samples: Array):
        if samples.ndim != self._ndim:
            raise ValueError(
                "samples has the wrong dimension. Samples has shape "
                f"{samples.shape} but should have ndim={self._ndim}"
            )

    def _check_values(self, samples: Array, values: Array):
        if samples.shape != values.shape:
            raise ValueError(
                f"values has the shape {values.shape}"
                f"but should be {samples.shape}"
            )

    @abstractmethod
    def _values(self, samples: Array) -> Array:
        raise NotImplementedError

    def _first_derivative(self, samples: Array) -> Array:
        raise NotImplementedError

    def _second_derivative(self, samples: Array) -> Array:
        raise NotImplementedError

    def _third_derivative(self, samples: Array) -> Array:
        raise NotImplementedError

    def __call__(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._values(samples)
        self._check_values(samples, vals)
        return vals

    def first_derivative_implemented(self):
        return self._bkd.jacobian_implemented()

    def first_derivative(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._first_derivative(samples)
        self._check_values(samples, vals)
        return vals

    def second_derivative_implemented(self):
        return self._bkd.hessian_implemented()

    def second_derivative(self, samples: Array) -> Array:
        self._check_samples(samples)
        vals = self._second_derivative(samples)
        self._check_values(samples, vals)
        return vals

    def third_derivative_implemented(self):
        return self._bkd.hessian_implemented()

    def third_derivative(self, samples: Array) -> Array:
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
        return self._check_derivative(
            samples,
            self._second_derivative,
            self._third_derivative,
            "h",
            fd_eps,
            relative,
            disp,
        )
