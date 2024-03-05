from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """
    Evaluate a model at a single sample.

    __call__ is required

    _jacobian, _apply_jacobian, apply_hessian are optional.
    If they are implemented set _jacobian_implemented,
    _apply_jacobian_implemented, _apply_hessian_implemented
    to True
    """
    def __init__(self):
        self._apply_jacobian_implemented = False
        self._jacobian_implemented = False
        self._apply_hessian_implemented = False
        self._hessian_implemented = False

    @abstractmethod
    def __call__(self, samples):
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
        raise NotImplementedError("Must implement self.__call__")

    def _check_sample_shape(self, sample):
        if sample.ndim != 2:
            raise ValueError(
                "sample is not a 2D array, has shape {0}".format(sample.shape))
        if sample.shape[1] != 1:
            raise ValueError(
                "sample is not a 2D array with 1 column, has shape {0}".format(
                    sample.shape))

    def _check_vec_shape(self, sample, vec):
        if vec.ndim != 2:
            raise ValueError(
                "vec is not a 2D array, has shape {0}".format(vec.shape))
        if sample.shape[0] != vec.shape[0]:
            raise ValueError(
                "sample.shape {0} and vec.shape {1} are inconsistent".format(
                    sample.shape, vec.shape))

    def _jacobian(self, sample):
        raise NotImplementedError

    def jacobian(self, sample):
        """
        Evaluate the jacobian of the model at a set of sample.

        Parameters
        ----------
        sample : np.ndarray (nvars, 1)
            The sample at which to compute the Jacobian

        Returns
        -------
        jac : np.ndarray (nqoi, nvars)
            The Jacobian matrix
        """
        if (not self._jacobian_implemented and
                not self._apply_jacobian_implemented):
            raise NotImplementedError(
                "_jacobian and _apply_jacobian are not implemented")
        self._check_sample_shape(sample)
        if self._jacobian_implemented:
            return self._jacobian(sample)
        actions = []
        nvars = sample.shape[0]
        for ii in range(nvars):
            vec = np.zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(self._apply_jacobian(sample, vec))
        return np.hstack(actions)

    def _apply_jacobian(self, sample, vec):
        raise NotImplementedError

    def apply_jacobian(self, sample, vec):
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
        if not self._apply_jacobian and not self._jacobian_implemented:
            raise RuntimeError(
                "apply_jacobian and jacobian are not implemented")
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        if self._jacobian_implemented:
            return self.jacobian(sample) @ vec
        return self._apply_jacobian(sample, vec)

    def _apply_hessian(self, sample, vec):
        raise NotImplementedError

    def apply_hessian(self, sample, vec):
        """
        Compute the matrix vector product of the Hessian with a vector.

        Parameters
        ----------
        sample : np.ndarray (nvars, 1)
            The sample at which to compute the Hessian

        vec : np.narray (nvars, 1
            The vector

        Returns
        -------
        result : np.ndarray (nvars, 1)
            The dot product of the Hessian with the vector
        """
        if not self._apply_hessian:
            raise RuntimeError(
                "apply_hessian not implemented")
        self._check_sample_shape(sample)
        self._check_vec_shape(sample, vec)
        return self._apply_hessian(sample, vec)

    def _hessian(self, sample):
        raise NotImplementedError

    def hessian(self, sample):
        if (not self._apply_hessian_implemented and
                not self._hessian_implemented):
            raise NotImplementedError("Hessian not implemented")
        self._check_sample_shape(sample)
        if self._hessian_implemented:
            return self._hessian(sample)
        actions = []
        nvars = sample.shape[0]
        for ii in range(nvars):
            vec = np.zeros((nvars, 1))
            vec[ii] = 1.0
            actions.append(self._apply_hessian(sample, vec))
        return np.hstack(actions)

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)

    def _check_apply(self, sample, symb, fun, apply_fun, fd_eps=None,
                     direction=None, relative=True, disp=False):
        if sample.ndim != 2:
            raise ValueError(
                "sample with shape {0} must be 2D array".format(sample.shape))
        if fd_eps is None:
            fd_eps = np.logspace(-13, 0, 14)[::-1]
        if direction is None:
            nvars = sample.shape[0]
            direction = np.random.normal(0, 1, (nvars, 1))
            direction /= np.linalg.norm(direction)

        row_format = "{:<12} {:<25} {:<25} {:<25}"
        headers = [
            "Eps", "norm({0}v)".format(symb), "norm({0}v_fd)".format(symb),
            "Rel. Errors" if relative else "Abs. Errors"]
        if disp:
            print(row_format.format(*headers))
        row_format = "{:<12.2e} {:<25} {:<25} {:<25}"
        errors = []
        val = fun(sample)
        directional_grad = apply_fun(sample, direction)
        for ii in range(fd_eps.shape[0]):
            sample_perturbed = sample.copy()+fd_eps[ii]*direction
            perturbed_val = fun(sample_perturbed)
            fd_directional_grad = (perturbed_val-val)/fd_eps[ii]
            errors.append(np.linalg.norm(
                fd_directional_grad.reshape(directional_grad.shape) -
                directional_grad))
            if relative:
                errors[-1] /= np.linalg.norm(directional_grad)
            if disp:
                print(row_format.format(
                    fd_eps[ii], np.linalg.norm(directional_grad),
                    np.linalg.norm(fd_directional_grad), errors[ii]))
        return np.array(errors)

    def check_apply_jacobian(self, sample, fd_eps=None, direction=None,
                             relative=True, disp=False):
        """
        Compare apply_jacobian with finite difference.
        """
        if not self._apply_jacobian and not self._jacobian_implemented:
            raise RuntimeError(
                "Cannot check apply_jacobian because it is not implemented")
        return self._check_apply(
            sample, "J", self, self.apply_jacobian, fd_eps, direction,
            relative, disp)

    def check_apply_hessian(self, sample, fd_eps=None, direction=None,
                            relative=True, disp=False):
        """
        Compare apply_hessian with finite difference.
        """
        if not self._apply_hessian:
            raise RuntimeError(
                "Cannot check apply_hessian because it is not implemented")
        return self._check_apply(
            sample, "H", self.jacobian, self.apply_hessian, fd_eps,
            direction, relative, disp)


class SingleSampleModel(Model):
    @abstractmethod
    def _evaluate(self, sample):
        """
        Evaluat the model at a single sample

        Parameters
        ----------
        sample: np.ndarray (nvars)
            The sample use to evaluate the model

        Returns
        -------
        values : np.ndarray (nqoi)
            The model outputs returned by the model when evaluated
            at the sample
        """
        raise NotImplementedError

    def __call__(self, samples):
        assert samples.ndim == 2
        nvars, nsamples = samples.shape
        values_0 = self._evaluate(samples[:, :1])
        if values_0.ndim != 2 or values_0.shape[0] != 1:
            msg = "values returned by self._model has the wrong shape."
            msg += " shape is {0} but must be 2D array with single row".format(
                values_0.shape)
            raise ValueError(msg)
        nqoi = values_0.shape[1]
        values = np.empty((nsamples, nqoi), float)
        values[0, :] = values_0
        for ii in range(1, nsamples):
            values[ii, :] = self._model(samples[:, ii:ii+1])
        return values


class ModelFromCallable(SingleSampleModel):
    def __init__(self, function, apply_jacobian=None, apply_hessian=None):
        super().__init__()
        if not callable(function):
            raise ValueError("function must be callable")
        self._function = function
        if apply_jacobian is not None:
            if not callable(apply_jacobian):
                raise ValueError("apply_jacobian must be callable")
            self._apply_jacobian = apply_jacobian
            self._apply_jacobian_implemented = True
        if apply_hessian is not None:
            if not callable(apply_jacobian):
                raise ValueError("apply_hessian must be callable")
            self._apply_hessian = apply_hessian
            self._apply_hessian_implemented = True

    def _evaluate(self, sample):
        return self._function(sample)


class ScipyModelWrapper():
    def __init__(self, model):
        """
        Create a API that takes a sample as a 1D array and returns
        a scalar
        """
        if not issubclass(model.__class__, Model):
            raise ValueError("model must be derived from Model")
        self._model = model

    def _check_sample(self, sample):
        if sample.ndim != 1:
            raise ValueError(
                "sample must be a 1D array but has shape {0}".format(
                    sample.shape))

    def __call__(self, sample, return_grad=False):
        self._check_sample(sample)
        vals = self._model(sample[:, None])[0]
        if vals.shape[0] != 1:
            raise ValueError("model does not return a scalar")
        vals = vals[0]
        if not return_grad:
            return vals
        return vals, self.jacobian(sample)

    def jac(self, sample):
        self._check_sample(sample)
        return self._model.jacobian(sample[:, None])

    def hess(self, sample):
        self._check_sample(sample)
        return self._model.hessian(sample[:, None])

    def hessp(self, sample, vec):
        self._check_sample(sample)
        if (vec.ndim != 1 or vec.shape[0] != sample.shape[0]):
            raise ValueError(
                "vec shape {0} and sample shape {1} are inconsistent".format(
                    vec.shape, sample.shape))
        return self._model.apply_hessian(sample[:, None], vec[:, None])
