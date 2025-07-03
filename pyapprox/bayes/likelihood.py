from abc import abstractmethod

import numpy as np

from pyapprox.util.linalg import (
    inverse_of_cholesky_factor,
    log_determinant_from_cholesky_factor,
    diag_of_mat_mat_product,
)
from pyapprox.interface.model import Model, ChangeModelSignWrapper
from pyapprox.variables.joint import JointVariable
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.optimization.minimize import Optimizer
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer


class LogLikelihood(Model):
    """
        A base class for computing the log-likelihood of a model given
    observations.

    This class is designed to be extended by specific implementations of
    log-likelihood computations for different models.
    While likelihoods are just probability distributions, using them to
    solve inverse problems, requires evaluating the log PDF
    distribution, e.g. :math:`p(x|s)` for many different realization of
    the shape parameters :math:`s`  of the distribution,
    e.g. the mean and covariance are shape parameters of a
    Gaussian distribution. The term shapes is also used loosely and may
    also include parameters of a distribution often called scales by scipy.
    This is in contrast the PDF of PyApprox's random variables which
    are evaluated at the independent variable of the PDF :math:`x`.
    Consequently, loglikeihood functions are written separately to allow
    computations to be fast for multiple shape parameters, where as
    variables are designed to be fast when being evaluated for the
    independent variable.
    Moreover, some likelihoods fix certain shape parameters and only
    allows sampling of a subset of the shape parameters, e.g.
    Gaussian Likelihoods used to estiamte an unknown mean for a fixed
    covariance

    Attributes
    ----------
    _obs : Array
        Stores the observations provided to the model.
    _nobs : int
        Number of observations.
    _tile_obs : bool
        Flag indicating whether to tile observations for certain computations.
    backend : BackendMixin
        Computational backend used for numerical operations
        (e.g., Numpy, TensorFlow).
    """

    def __init__(
        self, tile_obs: bool = True, backend: BackendMixin = NumpyMixin
    ):
        """
        Initializes the LogLikelihood class.

        Parameters
        ----------
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to NumpyMixin.
        """
        super().__init__(backend=backend)
        self._set_tile_obs(tile_obs)

    def _set_tile_obs(self, tile_obs: bool):
        """
        Sets the flag indicating whether to tile observations.

        Parameters
        ----------
        tile_obs : bool
            If True, observations will be tiled for certain computations.
        """
        self._tile_obs = tile_obs

    def _parse_obs(self, obs: Array, shapes: Array) -> Array:
        """
        Parses and pre-processes observations and shapes before log-likelihood
        computation.

        Parameters
        ----------
        obs : Array
            Observations provided to the model.
        shapes : Array
            Predicted shapes or values from the model.

        Returns
        -------
        tuple
            Processed observations and shapes.

        Raises
        ------
        RuntimeError
            If `_setup()` or `_set_tile_obs()` has not been called.
        ValueError
            If the shapes' dimensions do not match the observations'
            dimensions.
        """
        if not hasattr(self, "_tile_obs"):
            raise RuntimeError("must call _set_tile_obs()")
        if self._tile_obs:
            # stack vals for each obs vertically
            return (
                self._bkd.repeat(self._obs, shapes.shape[1], axis=1),
                self._bkd.tile(shapes, (1, self._obs.shape[1])),
            )
        if shapes.shape != self._obs.shape:
            msg = "shapes shape {0} does not match self._obs shape {1}".format(
                shapes.shape, self._obs.shape
            )
            raise ValueError(msg)
        return (self._obs, shapes)

    @abstractmethod
    def nvars(self) -> int:
        """
        Returns the number of unknown shape parameters

        Returns
        -------
        nobs: int
            Number of shape parameters
        """
        raise NotImplementedError

    @abstractmethod
    def nobs(self) -> int:
        """
        Returns the number of observations.

        Returns
        -------
        nobs: int
            Number of observations.
        """
        raise NotImplementedError

    def nqoi(self) -> int:
        return 1

    def set_observations(self, obs: Array):
        """
        Sets the observations for the model.

        Parameters
        ----------
        obs : Array
            Observations provided to the model. Must be a 2D array with
            shape (nobs, 1).

        Raises
        ------
        ValueError
            If the observations do not have the expected shape.
        """
        self._nobs = obs.shape[0]
        if obs.ndim != 2 or obs.shape[0] != self.nobs():
            raise ValueError(
                "obs has the wrong shape {0} should be {1}".format(
                    obs.shape, (self.nobs(), 1)
                )
            )
        self._obs = obs

    def _check_shapes(self, shapes: Array):
        """
        Validates the dimensions of the shapes array.

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Muliple realizations of the shape parameters of the likelihood
            distribution

        Raises
        ------
        ValueError
            If the shapes' dimensions do not match the number of observations.
        """
        if shapes.shape[0] != self.nobs():
            raise ValueError(
                "shapes has the wrong shape {0}".format(shapes.shape)
                + " nobs is {0}".format(self.nobs())
            )

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        """
        Computes the log-likelihood from muliple realizations of the shape
        parameters of the likelihood distribution

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Muliple realizations of the shape parameters of the likelihood
            distribution


        Returns
        -------
        values : Array (nsamples, 1)
            Log-likelihood values.
        """
        self._check_shapes(shapes)
        return self._loglike(shapes)

    def _values(self, shapes: Array) -> Array:
        return self._loglike_from_shapes(shapes)

    @abstractmethod
    def _rvs(self, shapes: Array) -> Array:
        """
        Abstract method for generating observations based on the
        likelihood distribution's shapes.

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def rvs(self, shapes: Array) -> Array:
        """
        Generate observations based on the
        likelihood distribution's shapes.

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        if shapes.ndim != 2 or shapes.shape[0] != self.nvars():
            raise ValueError(
                "shapes has the wrong shape. "
                f"Shape was {shapes.shape} but should should be 2D array with "
                f"{self.nvars()} rows"
            )
        nsamples = shapes.shape[1]
        obs = self._rvs(shapes)
        if obs.shape != (self.nobs(), nsamples):
            raise RuntimeError(
                "self._rvs returned obs with the wrong shape. "
                "Shape was {0} but should be {1}".format(
                    obs.shape, (nsamples, self.nobs())
                )
            )
        return obs


class ModelBasedLogLikelihoodMixin:
    """
    A mixin class for computing log-likelihoods based on a model that maps
    parameters to observations.

    This class provides methods for interacting with the model,
    computing Jacobians, Hessians, with respect to the shape parameters,
    and applying these derivatives to vectors
    """

    def model(self) -> Model:
        """
        Return the model used to map the parameters to the observations

        Returns
        -------
        model : Model
            The model used to map the parameters to the observations
        """
        return self._model

    def nvars(self) -> int:
        """
        Returns the number of model parameters which parameterize
        the unknown shapes of the likelihood (which are the qoi of the model)

        Returns
        -------
        nobs: int
            Number of shape parameters
        """
        return self._model.nvars()

    def nobs(self) -> int:
        return self._model.nqoi()

    def __repr__(self) -> str:
        return "{0}(nobs={1}, model={2})".format(
            self.__class__.__name__, self.nobs(), self._model
        )

    def _rvs(self, model_parameter_samples: Array) -> Array:
        return super()._rvs(self._model(model_parameter_samples).T)

    def _values(self, model_parameter_samples: Array):
        return self._loglike_from_shapes(
            self._model(model_parameter_samples).T
        )

    def _jacobian(self, model_parameter_sample: Array) -> Array:
        pred_obs = self._model(model_parameter_sample).T
        residual = self._obs - pred_obs
        L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
        return L_inv_res.T @ self._noise_cov_sqrt_inv_apply(
            self._model.jacobian(model_parameter_sample)
        )

    def _apply_jacobian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        pred_obs = self._model(model_parameter_sample).T
        residual = self._obs - pred_obs
        L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
        return L_inv_res.T @ self._noise_cov_sqrt_inv_apply(
            self._model.apply_jacobian(model_parameter_sample, vec)
        )

    def _apply_hessian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        pred_obs = self._model(model_parameter_sample).T
        # use adjoint method to compute hvp with
        # objective f(s,z) = -1/2 s.T @ s
        residual = self._obs - pred_obs
        # solve forward equation for state s
        # c(s,z) = Gs - obs + model(z) = 0, G=wnoise_chol_inv
        state = self._noise_cov_sqrt_inv_apply(residual)
        # solve adjoint solution
        # lambda = -inv(c_s.T)f_s = -inv(G.T)s
        lamda = self._noise_cov_sqrt_inv_apply_transpose(state)
        tmp1 = self._noise_cov_sqrt_inv_apply(
            self._model.jacobian(model_parameter_sample)
        )
        # Hvp = c_z.T @ p + L_zs @ w + L_zz @ v
        # w = inv(c_s)@ c_z @ v = inv(G) @ model.jacobian(z) @ v
        # L_ss = -I, L_zs = 0, L_sz = 0, L_zz = lamda.T @ model.hessian(z)
        # p = -inv(G.T) @ w
        return -tmp1.T @ (tmp1 @ vec) + self._model.apply_weighted_hessian(
            model_parameter_sample, vec, lamda
        )


class GaussianLogLikelihood(LogLikelihood):
    """
    GaussianLikelihood that requires the application of the square-root
    of the known noise covariance matrix to vectors
    """

    def _parse_obs(self, obs: Array, shapes: Array) -> Array:
        if not hasattr(self, "_loglike_const"):
            raise RuntimeError("must call _setup()")
        return super()._parse_obs(obs, shapes)

    def jacobian_implemented(self) -> bool:
        return self._model.jacobian_implemented()

    def apply_jacobian_implemented(self) -> bool:
        return self._model.apply_jacobian_implemented()

    def apply_hessian_implemented(self) -> bool:
        return self._model.hessian_implemented()

    def set_design_weights(self, design_weights: Array) -> Array:
        """
        Set the design weights that change the noise covariance to change
        the impact of each observations on the likelihood value.

        Parameters
        ----------
        design_weights : Array (nobs,)
            Design weights used to scale the noise covariance matrix.
        """
        self._design_weights = design_weights
        self._noise_chol = self._bkd.cholesky(self._noise_cov)
        self._noise_chol_inv = inverse_of_cholesky_factor(
            self._noise_chol, self._bkd
        )
        self._wnoise_chol_inv = self._noise_chol_inv * self._bkd.sqrt(
            self._design_weights
        )
        # Todo avoid inverting and just apply weights is possible
        self._wnoise_chol = inverse_of_cholesky_factor(
            self._wnoise_chol_inv, self._bkd
        )
        self._log_wnoise_cov_det = log_determinant_from_cholesky_factor(
            self._wnoise_chol, self._bkd
        )
        self._loglike_const = (
            self._nobs * np.log(2 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov: Array):
        """
        Set up the noise covariance matrix and related properties.

        Parameters
        ----------
        noise_cov : Array (nobs, nobs)
            Dense noise covariance matrix.

        Raises
        ------
        ValueError
            If the noise covariance matrix is not square.
        """
        if noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]:
            raise ValueError(
                "noise_cov has the wrong shape {0}".format(noise_cov.shape)
            )
        self._noise_cov = noise_cov
        self._nobs = noise_cov.shape[0]
        self._last_L_inv_res = None
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def nobs(self) -> int:
        return self._nobs

    def _make_noisy(self, shapes: Array, noise: Array) -> Array:
        """
        Add noise to noiseless samples of the unknown mean (shapes).

        Parameters
        ----------
        shapes : Array (nobs, nsamples)
            Noiseless shapes.
        noise : Array (nobs, nsamples)
            Noise to be added.

        Returns
        -------
        noisy_shapes : Array (nobs, nsamples)
            Noisy shapes.

        Raises
        ------
        ValueError
            If the shapes of noiseless shapes and noise do not match.
        """
        if shapes.shape != noise.shape:
            msg = "shapes of shapes {0} and noise {1} must match".format(
                shapes.shape, noise.shape
            )
            raise ValueError(msg)
        return shapes + noise

    def _noise_cov_sqrt_apply(self, vecs: Array) -> Array:
        """
        Apply the square root of the noise covariance matrix to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise covariance square root is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol @ vecs

    def _noise_cov_sqrt_apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the tranpose of the square root of the noise covariance matrix
        to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the tranpose.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol.T @ vecs

    def _noise_cov_sqrt_inv_apply(self, vecs: Array) -> Array:
        """
        Apply the inverse of the square root of the noise covariance matrix
        to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise inverse is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol_inv @ vecs

    def _noise_cov_sqrt_inv_apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the transpose of the inverse of the square root of the
        noise covariance matrix to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise inverse transpose is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol_inv.T @ vecs

    def _sample_noise(self, nsamples: int) -> Array:
        """
        Sample noise from the Gaussian distribution.

        Parameters
        ----------
        nsamples : int
            Number of noise samples to generate.

        Returns
        -------
        noise : Array (nobs, nsamples)
            Noise samples.
        """
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self.nobs()))
        ).T
        return self._noise_cov_sqrt_apply(normal_samples)

    def _loglike_many(self, many_obs: Array, shapes: Array) -> Array:
        """
        Compute the log-likelihood after observations have been processed
        and repeated if necessary to speed up compuation.

        Parameters
        ----------
        many_obs : Array
            Multiple observations.
        shapes : Array
            Predicted shapes or values.

        Returns
        -------
        Array
            Log-likelihood values for each observation.
        """
        residual = many_obs - shapes
        L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
        vals = diag_of_mat_mat_product(L_inv_res.T, L_inv_res, bkd=self._bkd)
        return (-0.5 * (vals + self._loglike_const))[:, None]

    def _loglike(self, shapes: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        return self._loglike_many(*self._parse_obs(self._obs, shapes))

    def noise_covariance(self) -> Array:
        """
        Return the weighted noise covariance matrix.

        Returns
        -------
        noise_cov : Array (nobs, nobs)
            Weighted noise covariance matrix.
        """
        return self._wnoise_chol @ self._wnoise_chol.T

    def _rvs(self, shapes: Array) -> Array:
        return self._make_noisy(shapes, self._sample_noise(shapes.shape[1]))


class IndependentGaussianLogLikelihood(GaussianLogLikelihood):
    """
    A likelihood function associated with adding independent
    but not necessarily identically distributed Gaussian noise.
    Evaluating this is faster than GaussianLikelihood if noise covariance
    is a diagonal matrix
    """

    def set_design_weights(self, design_weights: Array):
        self._design_weights = design_weights
        assert self._bkd.all(design_weights >= 0), design_weights
        self._noise_cov_inv_diag = 1 / (self._noise_cov_diag)
        self._wnoise_cov_inv_diag = (
            self._noise_cov_inv_diag * self._design_weights
        )
        self._wnoise_std_diag = self._bkd.sqrt(1 / self._wnoise_cov_inv_diag)
        self._log_wnoise_cov_det = (
            2 * self._bkd.log(self._wnoise_std_diag).sum()
        )
        self._r_noise_cov_inv = None
        self._loglike_const = (
            self._nobs * np.log(2 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov_diag: Array):
        if noise_cov_diag.ndim != 2 or noise_cov_diag.shape[1] != 1:
            raise ValueError(
                "noise_cov_diag has the wrong shape {0}".format(
                    noise_cov_diag.shape
                )
            )
        self._nobs = noise_cov_diag.shape[0]
        self._noise_cov_diag = noise_cov_diag
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def _noise_cov_sqrt_apply(self, vecs: Array) -> Array:
        return self._wnoise_std_diag * vecs

    def _noise_cov_sqrt_apply_transpose(self, vecs: Array) -> Array:
        return self._noise_cov_sqrt_apply(vecs)

    def _noise_cov_sqrt_inv_apply(self, vecs: Array) -> Array:
        return self._bkd.sqrt(self._wnoise_cov_inv_diag) * vecs

    def _noise_cov_sqrt_inv_apply_transpose(self, vecs: Array) -> Array:
        return self._noise_cov_sqrt_inv_apply(vecs)

    def noise_covariance(self) -> Array:
        return self._bkd.diag(1 / self._wnoise_cov_inv_diag[:, 0])

    def __repr__(self) -> str:
        return "{0}(nobs={1}, sigma={2})".format(
            self.__class__.__name__,
            self.nobs(),
            self._noise_std_diag[:, 0],
        )


class ModelBasedGaussianLogLikelihood(
    ModelBasedLogLikelihoodMixin, GaussianLogLikelihood
):
    """
    A likelihood function for adding correlated Gaussian noise to observations,
    using a model that maps parameters to observations.
    """

    def __init__(self, model: Model, noise_cov: Array, tile_obs: bool = True):
        """
        Initialize the ModelBasedGaussianLogLikelihood class.

        Parameters
        ----------
        model : Model
            The model used to map parameters to observations.

        noise_cov : Array (nobs, nobs)
            Dense noise covariance matrix.

        tile_obs : bool, optional
            If True, observations will be tiled for certain computations.
            Defaults to True.
        """
        super().__init__(tile_obs=tile_obs, backend=model._bkd)
        self._model = model
        self._setup(noise_cov)


class ModelBasedIndependentGaussianLogLikelihood(
    ModelBasedLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    """
    A likelihood function for adding independent (but not necessarily
    identically distributed) Gaussian noise to observations, using a model
    that maps parameters to observations.

    Evaluating this is faster than GaussianLikelihood if noise covariance
    is a diagonal matrix
    """

    def __init__(
        self, model: Model, noise_cov_diag: Array, tile_obs: bool = True
    ):
        """
        Initialize the ModelBasedIndependentGaussianLogLikelihood class.

        Parameters
        ----------
        model : Model
            The model used to map parameters to observations.

        noise_cov_diag : Array (nobs,)
            Diagonal noise covariance matrix.

        tile_obs : bool, optional
            If True, observations will be tiled for certain computations.
            Defaults to True.
        """
        super().__init__(tile_obs, backend=model._bkd)
        self._model = model
        self._setup(noise_cov_diag)


class IndependentExponentialLogLikelihood(LogLikelihood):
    r"""
    Exponetially distributed variable with unknown rate parameters
    which determines the rate at which events occur over time.
    Observations are times between consecutive events
    """

    def __init__(
        self,
        nvars: int,
        tile_obs: bool = True,
        backend: BackendMixin = NumpyMixin,
    ):
        self._nvars = nvars
        super().__init__(tile_obs, backend)

    def nvars(self) -> int:
        return self._nvars

    def nobs(self) -> int:
        return self._nvars

    def _loglike(self, shapes: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        return self._loglike_many(*self._parse_obs(self._obs, shapes))

    def _loglike_many(self, many_obs: Array, shapes: Array) -> Array:
        return (
            self._bkd.log(shapes).sum(axis=0) - (many_obs * shapes).sum(axis=0)
        )[:, None]

    def _make_noisy(self, shapes: Array, noise: Array) -> Array:
        # obs = noise/shapes if using model noise/model(samples)
        # given exponentially distributed noise with unit mean
        # return realizations exponentially disributed with means
        # specified by noiseless_shapes
        if shapes.shape != noise.shape:
            msg = f"sizes of shapes {shapes.shape}"
            f"and noise {noise.shape} must match"
            raise ValueError(msg)
        return noise / shapes

    def _sample_noise(self, nsamples: int) -> Array:
        # sample noise so that it has unit mean
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsamples
        exponential_samples = self._bkd.asarray(
            np.random.exponential(1, (nsamples, self.nobs()))
        ).T
        return exponential_samples

    def _rvs(self, shapes: Array) -> Array:
        return self._make_noisy(shapes, self._sample_noise(shapes.shape[1]))


class ModelBasedIndependentExponentialLogLikelihood(
    ModelBasedLogLikelihoodMixin, IndependentExponentialLogLikelihood
):
    def __init__(self, model: Model, tile_obs: bool = True):
        super().__init__(model.nvars(), tile_obs, backend=model._bkd)
        self._model = model

    def nobs(self) -> int:
        return self._model.nqoi()


class LogUnNormalizedPosterior(Model):
    def __init__(self, loglike: LogLikelihood, prior: JointVariable):
        self._loglike = loglike
        self._prior = prior
        super().__init__(loglike._bkd)

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._loglike.nvars()

    def jacobian_implemented(self) -> bool:
        return (
            self._prior.pdf_jacobian_implemented()
            and self._loglike.jacobian_implemented()
        )

    def hessian_implemented(self) -> bool:
        return (
            self._prior.pdf_hessian_implemented()
            and self._loglike.hessian_implemented()
        )

    def apply_hessian_implemented(self) -> bool:
        return (
            self._prior.pdf_hessian_implemented()
            and self._loglike.apply_hessian_implemented()
        )

    def _values(self, samples: Array) -> Array:
        return self._loglike(samples) + self._prior.logpdf(samples)

    def _jacobian(self, sample: Array) -> Array:
        return self._loglike.jacobian(sample) + self._prior.logpdf_jacobian(
            sample
        )

    def _hessian(self, sample: Array) -> Array:
        return self._loglike.hessian(sample) + self._prior.logpdf_hessian(
            sample
        )

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return (
            self._loglike.apply_hessian(sample, vec)
            + self._prior.logpdf_hessian(sample) @ vec
        )

    def set_optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer
        self._optimizer.set_objective_function(ChangeModelSignWrapper(self))

    def default_optimizer(self) -> Optimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        # global_optimizer = ScipyConstrainedDifferentialEvolutionOptimizer(
        #     opts={"maxiter": 100}
        # )
        # # only set bounds for differential evolution because it is required
        # # by algorithm
        # global_optimizer.set_bounds(self._prior.truncated_ranges(1 - 1e-3))
        # optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        return local_optimizer

    def maximum_aposteriori_point(self, iterate: Array = None) -> Array:
        if iterate is None:
            iterate = self._prior.mean()
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        res = self._optimizer.minimize(iterate)
        return res.x


class ExponentialQuarticLogLikelihoodModel(LogLikelihood):
    # likelihood function sometimes found in the literature
    # not a true likelihood so only implements a subset of
    # functions
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(False, backend=backend)
        self._a = 3.0

    def nvars(self) -> int:
        return 2

    def _values(self, x: Array) -> Array:
        value = -(0.1 * x[0] ** 4 + 0.5 * (2.0 * x[1] - x[0] ** 2) ** 2)
        return value[:, None]

    def _jacobian(self, x: Array):
        grad = -self._bkd.array(
            [
                12.0 / 5.0 * x[0] ** 3 - 4.0 * x[0] * x[1],
                4.0 * x[1] - 2.0 * x[0] ** 2,
            ]
        )
        return grad

    def jacobian_implemented(self) -> bool:
        return True

    def _loglike(self, shapes: Array) -> Array:
        raise NotImplementedError

    def _make_noisy(self, noiseless_obs: Array, noise, Array) -> Array:
        raise NotImplementedError


class BernoulliLogLikelihood(LogLikelihood):
    def _loglike(self, many_theta: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        # single vector realization of obs forms one column
        # different realizations in different colums
        log_vals = self._bkd.sum(
            self._bkd.log(many_theta).T * self._obs.T
            + self._bkd.log(1.0 - many_theta.T) * (1.0 - self._obs.T),
            axis=1,
        )
        return log_vals

    def nvars(self) -> int:
        return 1

    def _make_noisy(self) -> Array:
        raise NotImplementedError(
            "Make noisy only relevant to certain likelihoods."
            "TODO remove this function with more general function"
        )

    def _values(self, samples: Array) -> Array:
        return self._loglike(samples)[:, None]

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[:, 0], sample[:, 0]
        )


class MultinomialLogLikelihood(LogLikelihood):
    def __init__(
        self, noptions: int, ntrials: int, backend: BackendMixin = NumpyMixin
    ):
        super().__init__(False, backend)
        self._noptions = noptions
        self._ntrials = ntrials

    def _loglike(self, many_theta: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        # single vector realization of obs forms one column
        # different realizations in different colums
        # here we flatten multinomial observations self._obs
        # such that
        # [
        #     [x11,...,x1K],
        #     [x21,...,x2K]
        # ] -> [x11,...,x1K, x21,...,x2K]
        if self._obs.shape[1] != self.nvars():
            raise RuntimeError("Obs has the wrong shape")
        # log(factorial(N) = gammaln(N+1)
        const = self._nobs * self._bkd.gammaln(self._ntrials + 1)
        log_vals = []
        for ii in range(self._obs.shape[0]):
            log_vals.append(
                self._bkd.sum(
                    self._bkd.log(many_theta).T * self._obs[ii : ii + 1],
                    axis=1,
                )
                - self._bkd.sum(self._bkd.gammaln(self._obs[ii] + 1))
            )
        return self._bkd.sum(self._bkd.stack(log_vals, axis=0), axis=0) + const

    def nvars(self) -> int:
        return self._noptions

    def _make_noisy(self) -> Array:
        raise NotImplementedError(
            "Make noisy only relevant to certain likelihoods."
            "TODO remove this function with more general function"
        )

    def _values(self, samples: Array) -> Array:
        return self._loglike(samples)[:, None]

    def __repr__(self) -> str:
        return "{0}(noptions={1})".format(
            self.__class__.__name__, self.nvars()
        )

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[:, 0], sample[:, 0]
        )
