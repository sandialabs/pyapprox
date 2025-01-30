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

import numpy as np
import umbridge

from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class ModelWorkTracker:
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend
        # use multiprocessing.Manager() so that the dictionary
        # can be updated by multiple processes when calling
        # multiprocessing.pool
        self._wall_times = multiprocessing.Manager().dict(
            {
                "val": self._bkd.empty((0,)),
                "jac": self._bkd.empty((0,)),
                "jvp": self._bkd.empty((0,)),
                "hess": self._bkd.empty((0,)),
                "hvp": self._bkd.empty((0,)),
                "whess": self._bkd.empty((0,)),
                "whvp": self._bkd.empty((0,))
            }
        )

    def update(self, eval_name: str, times: Array):
        self._wall_times[eval_name] = self._bkd.hstack(
            (self._wall_times[eval_name], times)
        )

    def average_wall_time(self, eval_name: str) -> float:
        return self._bkd.mean(self._wall_times[eval_name])

    def nevaluations(self, eval_name: str) -> int:
        return self._wall_times[eval_name].shape[0]

    def wall_times(self) -> dict:
        return self._wall_times

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
            )
        )


class Model(ABC):
    """
    Evaluate a model at a single sample.

    Required functions:
    _values is required

    Optional functions:
    _jacobian, _apply_jacobian, _hessian _apply_hessian

    If optional functions are implemented set the corresponding flag
    to True:
    _jacobian_implemented,
    _apply_jacobian_implemented,
    _apply_hessian_implemented,
    _apply_weighted_hessian_implemented
    """

    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

        # TODO: Make variables below functions that can be overwritten
        self._apply_jacobian_implemented = False
        self._jacobian_implemented = False
        self._apply_hessian_implemented = False
        self._hessian_implemented = False
        self._weighted_hessian_implemented = False
        self._apply_weighted_hessian_implemented = False

        self._work_tracker = ModelWorkTracker(self._bkd)

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _values(self, samples: Array) -> Array:
        raise NotImplementedError("Must implement self._values")

    def _check_values_shape(self, samples: Array, vals: Array):
        if vals.shape != (samples.shape[1], self.nqoi()):
            raise RuntimeError(
                "values had shape {0} but should have shape {1}".format(
                    vals.shape, (samples.shape[1], self.nqoi())
                )
            )

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the model at a set of samples.

        Parameters
        ----------
        samples : np.ndarray (nvars, nsamples)
            The model inputs used to evaluate the model

        Returns
        -------
        values : np.ndarray (nsamples, nqoi)
            The model outputs returned by the model at each sample
        """
        t0 = time.time()
        vals = self._values(samples)
        t1 = time.time()
        nsamples = samples.shape[1]
        # Make assumption that each sample took the same time
        times = self._bkd.full((nsamples,), (t1-t0) / nsamples)
        self._work_tracker.update("val", times)
        self._check_values_shape(samples, vals)
        return vals

    def _check_sample_shape(self, sample: Array):
        if sample.ndim != 2:
            raise ValueError(
                "sample is not a 2D array, has shape {0}".format(sample.shape)
            )
        if sample.shape[1] != 1:
            raise ValueError(
                "sample is not a 2D array with 1 column, has shape {0}".format(
                    sample.shape
                )
            )

    def _check_vec_shape(self, sample: Array, vec: Array):
        if vec.ndim != 2:
            raise ValueError(
                "vec is not a 2D array, has shape {0}".format(vec.shape)
            )
        if sample.shape[0] != vec.shape[0]:
            raise ValueError(
                "sample.shape {0} and vec.shape {1} are inconsistent".format(
                    sample.shape, vec.shape
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
        raise NotImplementedError

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
        sample : np.ndarray (nvars, 1)
            The sample at which to compute the Jacobian

        Returns
        -------
        jac : np.ndarray (nqoi, nvars)
            The Jacobian matrix
        """
        if (
            not self._jacobian_implemented
        ):
            raise NotImplementedError("_jacobian not implemented")
        self._check_sample_shape(sample)
        t0 = time.time()
        jac = self._jacobian(sample)
        t1 = time.time()
        times = self._bkd.array([t1-t0])
        self._work_tracker.update("jac", times)
        self._check_jacobian_shape(jac, sample)
        return jac

    def _apply_jacobian(self, sample: Array, vec: Array) -> Array:
        raise NotImplementedError

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the matrix vector product of the Jacobian with a vector.

        Parameters
        ----------
        sample : np.ndarray (nvars, 1)
            The sample at which to compute the Jacobian

        vec : np.narray (nvars, 1)
            The vector

        Returns
        -------
        result : np.ndarray (nqoi, 1)
            The dot product of the Jacobian with the vector
        """
        if (
            not self._apply_jacobian_implemented
            and not self._jacobian_implemented
        ):
            raise RuntimeError(
                "apply_jacobian and jacobian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self._apply_jacobian_implemented:
            t0 = time.time()
            jvp = self._apply_jacobian(sample, vec)
            t1 = time.time()
            times = self._bkd.array([t1-t0])
            self._work_tracker.update("jvp", times)
            return jvp
        return self.jacobian(sample) @ vec

    def work_tracker(self) -> ModelWorkTracker:
        return self._work_tracker

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        raise NotImplementedError

    def apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Compute the matrix vector product of the Hessian with a vector.

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
            The dot product of the Hessian with the vector
        """
        if (
            not self._apply_hessian_implemented
            and not self._hessian_implemented
        ):
            raise RuntimeError("apply_hessian and hessian are not implemented")
        if self.nqoi() > 1:
            raise ValueError("apply_hessian cannot be used when nqoi > 1")
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self._apply_hessian_implemented:
            t0 = time.time()
            hvp = self._apply_hessian(sample, vec)
            t1 = time.time()
            times = self._bkd.array([t1-t0])
            self._work_tracker.update("hvp", times)
            return hvp
        return self.hessian(sample)[0] @ vec

    def _hessian(self, sample: Array) -> Array:
        raise NotImplementedError

    def _check_hessian_shape(self, hess: Array, sample: Array):
        if hess.shape != (self.nqoi(), sample.shape[0], sample.shape[0]):
            raise RuntimeError(
                "Hessian returned by _hessian has the wrong shape. "
                "was {0} but must be {1}".format(
                    hess.shape,
                    (self.nqoi(), sample.shape[0], sample.shape[0])
                )
            )

    def hessian(self, sample: Array) -> Array:
        """
        Evaluate the hessian of the model at a sample.

        Parameters
        ----------
        sample : np.ndarray (nvars, 1)
            The sample at which to compute the Jacobian

        Returns
        -------
        hess : np.ndarray (nqoi, nvars, nvars)
            The Jacobian matrix
        """
        if not self._hessian_implemented:
            raise NotImplementedError("Hessian not implemented")
        if not self._hessian_implemented and self.nqoi() > 1:
            raise ValueError("apply_hessian cannot be used when nqoi > 1")

        self._check_sample_shape(sample)
        t0 = time.time()
        hess = self._hessian(sample)
        t1 = time.time()
        times = self._bkd.array([t1-t0])
        self._work_tracker.update("hess", times)
        self._check_hessian_shape(hess, sample)
        return hess

    def __repr__(self) -> str:
        return "{0}()".format(self.__class__.__name__)

    def apply_weighted_jacobian(
            self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Compute the matrix vector product of the Jacobian,
        of a weighted sum of the QoI, with a vector.

        Parameters
        ----------
        sample : np.ndarray (nvars, 1)
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
            not self._apply_jacobian_implemented
            and not self._jacobian_implemented
        ):
            raise RuntimeError(
                "apply_jacobian and jacobian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        self._check_weights_shape(weights)
        if self._apply_jacobian_implemented:
            return self.apply_jacobian(sample, vec).T @ weights
        return (self.jacobian(sample) @ vec).T @ weights

    def _apply_weighted_hessian(
            self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        raise NotImplementedError

    def apply_weighted_hessian(
            self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Compute the matrix vector product of the Hessian,
        of a weighted combinatino of the QoI, with a vector.

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
            not self._apply_weighted_hessian_implemented
            and not self._hessian_implemented
            and not self._weighted_hessian_implemented
        ):
            raise RuntimeError(
                "apply_weighted_hessian and hessian are not implemented"
            )
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self._apply_weighted_hessian_implemented:
            t0 = time.time()
            hvp = self._apply_weighted_hessian(sample, vec, weights)
            t1 = time.time()
            times = self._bkd.array([t1-t0])
            self._work_tracker.update("whvp", times)
            return hvp
        if self._weighted_hessian_implemented:
            return self.weighted_hessian(sample, weights) @ vec
        return weights.T @ (self.hessian(sample) @ vec[:, 0])

    def _weighted_hessian(self, sample: Array, weights: Array) -> Array:
        raise NotImplementedError

    def weighted_hessian(self, sample: Array, weights: Array) -> Array:
        if (
            not self._weighted_hessian_implemented
            and not self._hessian_implemented
        ):
            raise RuntimeError(
                "weighted_hessian and hessian are not implemented"
            )
        self._check_sample_shape(sample)
        if self._weighted_hessian_implemented:
            t0 = time.time()
            hess = self._weighted_hessian(sample, weights)
            t1 = time.time()
            times = self._bkd.array([t1-t0])
            self._work_tracker.update("whess", times)
            return hess
        hess = self.hessian(sample)
        return self._bkd.einsum("il,ijk->jkl", weights, hess)[..., 0]

    def _check_apply(
        self,
        sample,
        symb,
        fun,
        apply_fun,
        fd_eps=None,
        direction=None,
        relative=True,
        disp=False,
        args=[],
    ):
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
        self, sample: Array, fd_eps: Array = None, direction: Array = None,
        relative: bool = True, disp: bool = False
    ):
        """
        Compare apply_jacobian with finite difference.
        """
        if (
            not self._apply_jacobian_implemented
            and not self._jacobian_implemented
        ):
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
        if self._jacobian_implemented:
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
                not self._apply_hessian_implemented
                and not self._hessian_implemented
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
            not self._apply_weighted_hessian_implemented
            and not self._hessian_implemented
            and not self._weighted_hessian_implemented
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


class SingleSampleModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _evaluate(self, sample):
        """
        Evaluat the model at a single sample

        Parameters
        ----------
        sample: np.ndarray (nvars, 1)
            The sample use to evaluate the model

        Returns
        -------
        values : np.ndarray (1, nqoi)
            The model outputs returned by the model when evaluated
            at the sample
        """
        raise NotImplementedError

    def _values(self, samples):
        nvars, nsamples = samples.shape
        t0 = time.time()
        values_0 = self._evaluate(samples[:, :1])
        t1 = time.time()
        times = [t1-t0]
        if values_0.ndim != 2 or values_0.shape[0] != 1:
            msg = "values returned by self._model has the wrong shape."
            msg += " shape is {0} but must be 2D array with single row".format(
                values_0.shape
            )
            raise ValueError(msg)
        nqoi = values_0.shape[1]
        values = self._bkd.empty((nsamples, nqoi))
        values[0, :] = values_0
        for ii in range(1, nsamples):
            t0 = time.time()
            values[ii, :] = self._evaluate(samples[:, ii : ii + 1])
            t1 = time.time()
            times.append(t1-t0)
        self._work_tracker.update("val", self._bkd.array(times))
        return values


class ModelFromCallable(Model):
    def __init__(
        self,
        nqoi: int,
        function: callable,
        jacobian: callable = None,
        apply_jacobian: callable = None,
        apply_hessian: callable = None,
        hessian: callable = None,
        apply_weighted_hessian: callable = None,
        sample_ndim: int = 2,
        values_ndim: int = 2,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        """
        Parameters
        ----------
        samples_ndim : integer
            The dimension of the np.ndarray accepted by function in [1, 2]

        values_ndim : integer
            The dimension of the np.ndarray returned by function in [0, 1, 2]
        """
        super().__init__(backend=backend)
        self._nqoi = nqoi
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
        self._sample_ndim = sample_ndim
        self._values_ndim = values_ndim

    def nqoi(self):
        return self._nqoi

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


class ModelFromVectorizedCallable(ModelFromCallable):
    def __init__(
        self,
        nqoi: int,
        function: callable,
        jacobian: callable = None,
        apply_jacobian: callable = None,
        apply_hessian: callable = None,
        hessian: callable = None,
        apply_weighted_hessian: callable = None,
        values_ndim: int = 2,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(
            nqoi,
            function,
            jacobian,
            apply_jacobian,
            apply_hessian,
            hessian,
            apply_weighted_hessian,
            2,
            values_ndim,
            backend,
        )

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
            raise RuntimeError("function returned values with the wrong ndim")
        if self._values_ndim != 2:
            return self._bkd.atleast2d(values)
        return values


class SingleSampleModel(SingleSampleModelMixin, Model):
    pass


class ScipyModelWrapper:
    def __init__(self, model):
        """
        Create a API that takes a sample as a 1D np.ndarray and returns
        the objects needed by scipy optimizers. E.g.
        jac will return np.ndarray
        even when model accepts and returns arrays associated with a
        different backend
        """
        self._bkd = model._bkd
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "_jacobian_implemented",
            "_hessian_implemented",
            "_apply_hessian_implemented",
            "_weighted_hessian_implemented",
            "_apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, self._model.__dict__[attr])

    def _check_sample(self, sample):
        if sample.ndim != 1:
            raise ValueError(
                "sample must be a 1D array but has shape {0}".format(
                    sample.shape
                )
            )
        return self._bkd.asarray(sample)

    def __call__(self, sample):
        # use copy to avoid warning:
        # The given NumPy array is not writable ...
        sample = self._check_sample(np.copy(sample))
        vals = self._model(sample[:, None])
        if vals.shape[0] == 1:
            return vals[0]
        return self._bkd.to_numpy(vals)

    def jac(self, sample):
        sample = self._check_sample(sample)
        jac = self._model.jacobian(sample[:, None])
        if jac.shape[0] == 1:
            return jac[0]
        return self._bkd.to_numpy(jac)

    def hess(self, sample):
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(self._model.hessian(sample[:, None]))

    def hessp(self, sample, vec):
        sample = self._check_sample(sample)
        if vec.ndim != 1:
            raise ValueError("vec must be 1D array")
        return self._bkd.to_numpy(
            self._model.apply_hessian(
                sample[:, None], self._bkd.asarray(vec[:, None])
            )
        )

    def weighted_hess(self, sample, weights):
        sample = self._check_sample(sample)
        return self._bkd.to_numpy(
            self._model.weighted_hessian(
                sample[:, None], self._bkd.asarray(weights)[:, None]
            )
        )

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class UmbridgeModelWrapper(Model):
    def __init__(
        self, umb_model, config={}, nprocs=1, backend=NumpyLinAlgMixin
    ):
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
        self._jacobian_implemented = self._model.supports_gradient()
        self._apply_jacobian_implemented = (
            self._model.supports_apply_jacobian()
        )
        self._apply_hessian_implemented = self._model.supports_apply_hessian()
        self._nmodel_evaluations = 0

    def nqoi(self):
        return self._model.get_output_sizes()[0]

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
        self._model.apply_jacobian(
            None, None, parameters, vec, config=self._config
        )

    def _apply_hessian(self, sample, vec):
        parameters = self._check_sample(sample)
        self._model.apply_hessian(
            None, None, None, parameters, vec, None, config=self._config
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
        return self._bkd.array(self._evaluate_serial(samples))

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
        backend=NumpyLinAlgMixin,
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
        backend=NumpyLinAlgMixin,
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
        nqoi,
        infilenames,
        outdir_basename=None,
        save="no",
        datafilename=None,
        backend=NumpyLinAlgMixin,
    ):
        """
        Base class for models that require loading and or writing of files
        """
        super().__init__(backend=backend)
        self._nqoi = nqoi
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

    def nqoi(self):
        return self._nqoi

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

    def _save_samples_and_values(self, sample, values, outdirname, tmpfile):
        filename = os.path.join(outdirname, self._datafilename)
        np.savez(filename, sample=sample, values=values)

    def _process_outdir(self, sample, values, outdirname, tmpfile):
        if tmpfile is not None:
            tmpfile.cleanup()
            return
        if self._save == "limited":
            self._cleanup_outdir(outdirname)
        self._save_samples_and_values(sample, values, outdirname, tmpfile)

    @abstractmethod
    def _run(self, sample, linked_filenames, outdirname):
        raise NotImplementedError

    def _evaluate(self, sample):
        outdirname, tmpfile = self._create_outdir()
        self._nmodel_evaluations += 1
        linked_filenames = self._link_files(outdirname)
        values = self._bkd.asarray(
            self._run(sample, linked_filenames, outdirname)
        )
        self._process_outdir(sample, values, outdirname, tmpfile)
        return values


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
        backend=NumpyLinAlgMixin,
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
        self._nvars = nvars
        assert self._bkd.all(self._active_var_indices < self._nvars)
        self._inactive_var_indices = self._bkd.delete(
            self._bkd.arange(self._nvars, dtype=int), active_var_indices
        )
        if base_model is None:
            base_model = model
        self._base_model = base_model

        self._jacobian_implemented = self._base_model._jacobian_implemented
        self._apply_jacobian_implemented = (
            self._base_model._apply_jacobian_implemented
        )
        self._apply_hessian_implemented = (
            self._base_model._apply_hessian_implemented
            or self._base_model._hessian_implemented
        )

    def nqoi(self):
        return self._model.nqoi()

    @staticmethod
    def _expand_samples_from_indices(
        reduced_samples,
        active_var_indices,
        inactive_var_indices,
        inactive_var_values,
        bkd=NumpyLinAlgMixin,
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
        expanded_vec = self._bkd.zeros((self._nvars, 1))
        expanded_vec[self._active_var_indices] = vec
        return self._model.apply_jacobian(samples, expanded_vec)

    def _apply_hessian(self, reduced_samples, vec):
        samples = self._expand_samples(reduced_samples)
        # set inactive entries of vec to zero when peforming
        # matvec product  so they do not contribute to sum
        expanded_vec = self._bkd.zeros((self._nvars, 1))
        expanded_vec[self._active_var_indices] = vec
        return self._model.apply_hessian(
            samples, expanded_vec
        )[self._active_var_indices]

    def nactive_vars(self):
        return len(self._active_var_indices)

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class ChangeModelSignWrapper(Model):
    def __init__(self, model):
        super().__init__(model._bkd)
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model
        for attr in [
            "_jacobian_implemented",
            "_apply_jacobian_implemented",
            "_hessian_implemented",
            "_apply_hessian_implemented",
            "_apply_weighted_hessian_implemented",
        ]:
            setattr(self, attr, self._model.__dict__[attr])

    def nqoi(self):
        return self._model.nqoi()

    def _values(self, samples):
        vals = -self._model(samples)
        return vals

    def _jacobian(self, sample):
        return -self._model.jacobian(sample)

    def _apply_jacobian(self, sample, vec):
        return -self._model.apply_jacobian(sample, vec)

    def _hessian(self, sample):
        return -self._model.hessian(sample)

    def _apply_hessian(self, sample, vec):
        return -self._model.hessian(sample, vec)

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
        print(model)
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        super().__init__(model._bkd)
        if assert_omp and nprocs > 1:
            if ('OMP_NUM_THREADS' not in os.environ or
                    not int(os.environ['OMP_NUM_THREADS']) == 1):
                msg = "User set assert_omp=True but OMP_NUM_THREADS "
                "has not been set to 1. Run script with "
                "OMP_NUM_THREADS=1 python script.py"
            raise Exception(msg)

        self._model = model
        self._nprocs = nprocs

    def nqoi(self) -> int:
        return self._model.nqoi()

    def _values(self, samples: Array) -> Array:
        pool = Pool(self._nprocs)
        result = pool.map(
            self._model,
            [(samples[:, ii:ii+1]) for ii in range(samples.shape[1])])
        pool.close()
        return self._bkd.vstack(result)

    def model(self) -> Model:
        return self._model

    def work_tracker(self) -> ModelWorkTracker:
        # do not call self._work_tracker as it will contain incorrect
        # information, must return self._model._work_tracker instead
        return self._model._work_tracker
