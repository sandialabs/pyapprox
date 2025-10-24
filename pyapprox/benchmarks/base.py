from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union

from pyapprox.interface.model import Model, MultiIndexModelEnsemble
from pyapprox.interface.wrappers import ChangeModelSignWrapper
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.joint import JointVariable, DesignVariable
from pyapprox.surrogates.affine.basis import (
    FixedGaussianTensorProductQuadratureRuleFromVariable,
)
from scipy.optimize import LinearConstraint
from pyapprox.optimization.minimize import Constraint
from pyapprox.inference.likelihood import LogLikelihood


class SingleModelBenchmark(ABC):
    """
    Base class for single-model benchmarks.

    This class defines the structure for benchmarks where derived classes must
    specify the uncertain variable and the model. The base class provides
    utility methods to access these components and their properties.

    Attributes:
        _bkd (BackendMixin): Backend used for numerical computations.
        _prior (JointVariable): Random variable parameterizing model
                                   uncertainty.
        _model (Model): Model representing the relationship between uncertain
                        variables and quantities of interest (QoI).
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the base class with a specified backend for computations.

        Args:
            backend (BackendMixin): Backend for numerical computations

        Raises:
            ValueError: If required attributes (_prior, _model) are not set.
        """
        self._bkd = backend
        self._set_prior()
        self._set_model()

        # Validate that required attributes are set
        self._validate_attributes()

    @abstractmethod
    def _set_model(self):
        """
        Abstract method to set the model.

        Derived classes must implement this method to define the model.

        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_prior(self):
        """
        Abstract method to set the uncertain prior.

        Derived classes must implement this method to define the uncertain prior.

        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError

    def _validate_attributes(self):
        """
        Validate that the required attributes (_prior, _model) are set.

        Raises:
            ValueError: If any required attributes are not set.
        """
        if not hasattr(self, "_prior") or self._prior is None:
            raise ValueError(
                "The '_prior' attribute must be set using '_set_prior'."
            )
        if not hasattr(self, "_model") or self._model is None:
            raise ValueError(
                "The '_model' attribute must be set using '_set_model'."
            )

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns:
            int: Number of uncertain variables.
        """
        return self._prior.nvars()

    def prior(self) -> "JointVariable":
        """
        Return the random variable parameterizing the model uncertainty.

        Returns:
            JointVariable: The uncertain variable.
        """
        return self._prior

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI) of all models.

        Returns:
            int: Number of quantities of interest (QoI).
        """
        return self._model.nqoi()

    def model(self) -> "Model":
        """
        Return the model.

        Returns:
            Model: The model.
        """
        return self._model

    def __repr__(self) -> str:
        """
        Return a string representation of the class.

        Returns:
            str: Class name.
        """
        return "{0}".format(self.__class__.__name__)


class MultiModelBenchmark(ABC):
    # Unordered set of models
    def __init__(self, backend: BackendMixin):
        self._bkd = backend
        self._set_prior()
        self._set_models()

    def nvars(self) -> int:
        """Return the number of uncertain variables."""
        return self._prior.nvars()

    @abstractmethod
    def nmodels(self) -> int:
        """Return the number of models."""
        raise NotImplementedError

    def prior(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        return self._prior

    @abstractmethod
    def nqoi(self) -> int:
        """Return the number of quantities of interest (QoI) of all models."""
        raise NotImplementedError

    @abstractmethod
    def _set_models(self):
        raise NotImplementedError

    def models(self) -> List[Model]:
        return self._models


class MultiIndexModelBenchmark(MultiModelBenchmark):
    # Ordered set of models in a multi-dimensional hierarchy
    def models(self) -> MultiIndexModelEnsemble:
        return self._models

    def nmodels(self) -> int:
        return self._models.nmodels()


class ACVBenchmark(MultiModelBenchmark):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)
        self._set_quadrature_samples_weights()
        self._flatten_funs()

    @abstractmethod
    def costs(self) -> Array:
        """Return the cost of each model."""
        raise NotImplementedError

    def _set_quadrature_samples_weights(self):
        quad_rule = FixedGaussianTensorProductQuadratureRuleFromVariable(
            self.variable(), [21] * self.nvars()
        )
        self._quadx, self._quadw = quad_rule()
        self._quadw = self._quadw[:, 0]

    def _flat_fun_wrapper(self, ii, jj, xx) -> Array:
        if xx.ndim != 2 or xx.shape[0] != self.nvars():
            raise RuntimeError("xx has the wrong shape")
        return self._models[ii](xx)[:, jj]

    def _flatten_funs(self):
        self._flat_funs = []
        for ii in range(self.nmodels()):
            for jj in range(self.nqoi()):
                self._flat_funs.append(partial(self._flat_fun_wrapper, ii, jj))

    def _mean(self) -> Array:
        # overide this function if you know the means exactly
        means = self._bkd.array(
            [f(self._quadx).dot(self._quadw) for f in self._flat_funs]
        )
        return self._bkd.reshape(means, (self.nmodels(), self.nqoi()))

    def mean(self) -> Array:
        """
        Return the means of the QoI of each model.

        Returns
        -------
        means : Array (nmodels, nqoi)
            The means of each model
        """
        means = self._mean()
        if (
            means.ndim != 2
            or means.shape[0] != self.nmodels()
            or means.shape[1] != self.nqoi()
        ):
            raise RuntimeError("_mean() not implemented correctly")
        return means

    def _covariance(self) -> Array:
        # overide this function if you know the means exactly
        means = [f(self._quadx) @ self._quadw for f in self._flat_funs]

        cov = self._bkd.empty(
            (self.nmodels() * self.nqoi(), self.nmodels() * self.nqoi())
        )
        ii = 0
        for fi, mi in zip(self._flat_funs, means):
            jj = 0
            for fj, mj in zip(self._flat_funs, means):
                cov[ii, jj] = (
                    (fi(self._quadx) - mi) * (fj(self._quadx) - mj)
                ) @ (self._quadw)
                jj += 1
            ii += 1
        return cov

    def covariance(self) -> Array:
        """
        The covariance between the qoi of each model

        Returns
        -------
        cov = Array (nmodels*nqoi, nmodels*nqoi)
            The covariance treating functions concatinating the qoi
            of each model f0, f1, f2
        """
        cov = self._covariance()
        if (
            cov.ndim != 2
            or cov.shape[0] != self.nmodels() * self.nqoi()
            or cov.shape[1] != self.nmodels() * self.nqoi()
        ):
            raise RuntimeError("_covariance() not implemented correctly")
        return cov

    def _V_fun_entry(self, jj, kk, ll, means, flat_covs, xx):
        idx1 = jj * self.nqoi() + kk
        idx2 = jj * self.nqoi() + ll
        return (self._flat_funs[idx1](xx) - means[idx1]) * (
            self._flat_funs[idx2](xx) - means[idx2]
        ) - flat_covs[jj][kk * self.nqoi() + ll]

    def _V_fun(self, jj1, kk1, ll1, jj2, kk2, ll2, means, flat_covs, xx):
        return self._V_fun_entry(
            jj1, kk1, ll1, means, flat_covs, xx
        ) * self._V_fun_entry(jj2, kk2, ll2, means, flat_covs, xx)

    def _B_fun(self, ii, jj, kk, ll, means, flat_covs, xx):
        return (self._flat_funs[ii](xx) - means[ii]) * self._V_fun_entry(
            jj, kk, ll, means, flat_covs, xx
        )

    def _flat_covs(self) -> List[Array]:
        cov = self.covariance()
        # store covariance only between the QoI of a model with QoI of the same
        # model
        flat_covs = []
        for ii in range(self.nmodels()):
            flat_covs.append([])
            for jj in range(self.nqoi()):
                for kk in range(self.nqoi()):
                    flat_covs[ii].append(
                        cov[ii * self.nqoi() + jj][ii * self.nqoi() + kk]
                    )
        return flat_covs

    def covariance_of_centered_values_kronker_product(self) -> Array:
        r"""
        The W matrix used to compute the covariance between the
        Kroneker product of centered (mean is subtracted off) values.

        Returns
        -------
        res : Array (nmodels*nqoi**2, nmodels*nqoi**2)
            The covariance :math:`Cov[(f_i-\mathbb{E}[f_i])^{\otimes^2}, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.mean().flatten()
        flat_covs = self._flat_covs()

        est_cov = self._bkd.empty(
            (
                self.nmodels() * self.nqoi() ** 2,
                self.nmodels() * self.nqoi() ** 2,
            )
        )
        cnt1 = 0
        for jj1 in range(self.nmodels()):
            for kk1 in range(self.nqoi()):
                for ll1 in range(self.nqoi()):
                    cnt2 = 0
                    for jj2 in range(self.nmodels()):
                        for kk2 in range(self.nqoi()):
                            for ll2 in range(self.nqoi()):
                                quad_cov = self._V_fun(
                                    jj1,
                                    kk1,
                                    ll1,
                                    jj2,
                                    kk2,
                                    ll2,
                                    means,
                                    flat_covs,
                                    self._quadx,
                                ).dot(self._quadw)
                                est_cov[cnt1, cnt2] = quad_cov.item()
                                cnt2 += 1
                    cnt1 += 1
        return self._bkd.array(est_cov)

    def covariance_of_mean_and_variance_estimators(self) -> Array:
        r"""
        The B matrix used to compute the covariance between mean and variance
        estimators.

        Returns
        -------
        res : Array (nmodels*nqoi, nmodels*nqoi**2)
            The covariance :math:`Cov[f_i, (f_j-\mathbb{E}[f_j])^{\otimes^2}]`
        """
        means = self.mean().flatten()
        flat_covs = self._flat_covs()
        est_cov = self._bkd.empty(
            (self.nmodels() * self.nqoi(), self.nmodels() * self.nqoi() ** 2)
        )
        for ii in range(len(self._flat_funs)):
            cnt = 0
            for jj in range(self.nmodels()):
                for kk in range(self.nqoi()):
                    for ll in range(self.nqoi()):
                        quad_cov = self._B_fun(
                            ii, jj, kk, ll, means, flat_covs, self._quadx
                        ).dot(self._quadw)
                        est_cov[ii, cnt] = quad_cov.item()
                        cnt += 1
        return self._bkd.array(est_cov)

    def __repr__(self) -> str:
        return "{0}(nmodels={1}, nqoi={2})".format(
            self.__class__.__name__, self.nmodels(), self.nqoi()
        )

    def models(self) -> List[Model]:
        return self._models


class OptimizationBenchmark(ABC):
    """
    Base class for optimization benchmarks.

    This class defines the structure for benchmarks where derived classes must
    specify the objective model and the design variable. The base class ensures
    that the objective model is created only once during initialization to allow
    internal model variables to persist (e.g., `objective.work_tracker`).

    Attributes:
        _bkd (BackendMixin): Backend used for numerical computations.
        _objective (Model): The objective model representing the optimization
                            goal.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the base class with a specified backend for computations.

        Args:
            backend (BackendMixin): Backend for numerical computations.

        Raises:
            ValueError: If the `_objective` attribute is not set.
        """
        self._bkd = backend

        # Use _set_objective to ensure the model is created only once
        self._set_objective()

        # Validate that the objective is set
        self._validate_objective()

    @abstractmethod
    def _set_objective(self):
        """
        Abstract method to set the objective model.

        Derived classes must implement this method to define the objective
        model.

        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError

    def _validate_objective(self):
        """
        Validate that the `_objective` attribute is set.

        Raises:
            ValueError: If the `_objective` attribute is not set.
        """
        if not hasattr(self, "_objective") or self._objective is None:
            raise ValueError(
                "The '_objective' attribute must be set using "
                "'_set_objective'."
            )

    def objective(self) -> Model:
        """
        Return the objective model.

        Returns:
            Model: The objective model.
        """
        return self._objective

    @abstractmethod
    def design_variable(self) -> DesignVariable:
        """
        Return the design variable.

        Derived classes must implement this method to define the design
        variable.

        Returns:
            DesignVariable: The design variable.
        """
        raise NotImplementedError


class ConstrainedOptimizationBenchmark(OptimizationBenchmark):
    """
    Base class for constrained optimization benchmarks.

    This class extends the `OptimizationBenchmark` class to include constraints
    and additional methods for defining optimal and initial iterates. Derived
    classes must specify the constraints, optimal iterate, and initial iterate.

    Attributes:
        _bkd (BackendMixin): Backend used for numerical computations.
        _objective (Model): The objective model representing the optimization goal.
        _constraints (List[Union[Constraint, LinearConstraint]]): Constraints
            applied to the optimization problem.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the constrained optimization benchmark.

        Args:
            backend (BackendMixin): Backend for numerical computations.

        Raises:
            ValueError: If required attributes (_objective, _constraints) are
                        not set.
        """
        super().__init__(backend)

        # Use _set_constraints to ensure constraints are created only once
        self._set_constraints()

        # Validate that constraints are set
        self._validate_constraints()

    @abstractmethod
    def _set_constraints(self):
        """
        Abstract method to set the constraints.

        Derived classes must implement this method to define the constraints.
        """
        raise NotImplementedError

    def _validate_constraints(self):
        """
        Validate that the `_constraints` attribute is set.

        Raises:
            ValueError: If the `_constraints` attribute is not set.
        """
        if not hasattr(self, "_constraints") or self._constraints is None:
            raise ValueError(
                "The '_constraints' attribute must be set using "
                "'_set_constraints'."
            )

    def constraints(self) -> List[Union["Constraint", "LinearConstraint"]]:
        """
        Return the constraints applied to the optimization problem.

        Returns:
            List[Union[Constraint, LinearConstraint]]: The constraints.
        """
        return self._constraints

    @abstractmethod
    def optimal_iterate(self) -> Array:
        """
        Return the optimal iterate.

        Derived classes must implement this method to define the optimal
        iterate.

        Returns:
            Array: The optimal iterate.
        """
        raise NotImplementedError

    @abstractmethod
    def init_iterate(self) -> Array:
        """
        Return the initial iterate.

        Derived classes must implement this method to define the initial
        iterate passed to the optimizer.

        Returns:
            Array: The initial iterate.
        """
        raise NotImplementedError


class ConstrainedUncertainOptimizationBenchmark(
    ConstrainedOptimizationBenchmark
):
    """
    Base class for constrained optimization benchmarks with uncertainty.

    This class extends `ConstrainedOptimizationBenchmark` to include methods
    for handling uncertainty, such as defining a prior and specifying the
    indices of design variables in a combined uncertain and design variable
    array.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the constrained uncertain optimization benchmark.

        Args:
            backend (BackendMixin): Backend for numerical computations
        """
        super().__init__(backend)

        # Validate that the prior is set correctly
        self._validate_prior()

    @abstractmethod
    def prior(self) -> JointVariable:
        """
        Return the prior random variable.

        Derived classes must implement this method to define the prior, which
        represents uncertainty in the optimization problem.

        Returns:
            JointVariable: The prior random variable.
        """
        raise NotImplementedError

    @abstractmethod
    def design_var_indices(self) -> Array:
        """
        Return the indices of design variables.

        Derived classes must implement this method to specify the positions of
        design variables in a combined uncertain and design variable array.

        Returns:
            Array: Array of indices for design variables.
        """
        raise NotImplementedError

    def _validate_prior(self):
        """
        Validate that the `prior` method is implemented correctly.

        Raises:
            ValueError: If the `prior` method does not return a valid
                        JointVariable.
        """
        prior = self.prior()
        if not isinstance(prior, JointVariable):
            raise ValueError(
                "The `prior` method must return an instance of JointVariable."
            )


class SingleModelBayesianInferenceBenchmark:
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend
        self._set_prior()
        self._set_obs_model()

    def nvars(self) -> int:
        """Return the number of uncertain variables."""
        return self._prior.nvars()

    def prior(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        return self._prior

    def nobservations(self) -> int:
        """Return the number of observations."""
        return self._obs_model.nqoi()

    @abstractmethod
    def _set_obs_model(self):
        raise NotImplementedError

    @abstractmethod
    def _set_prior(self):
        raise NotImplementedError

    def observation_model(self) -> Model:
        """Return the model"""
        return self._obs_model

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)

    @abstractmethod
    def loglike(self) -> LogLikelihood:
        raise NotImplementedError

    def negloglike(self) -> Model:
        return ChangeModelSignWrapper(self.loglike())

    @abstractmethod
    def observation_generating_parameters(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def observations(self) -> Array:
        raise NotImplementedError


# TODO make bencmhar base class then derive different types of benchmarks from
# that class to better reuse code
class OperatorBenchmark(ABC):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._bkd = backend
        # must use set_model here to make sure model
        # is only created once and not everytime self.objective is called
        # so that internal model variables persist, e.g. _work_tracker
        self._set_model()

    @abstractmethod
    def prior(self) -> JointVariable:
        """Return the random variable parameterizing the model uncertainty."""
        raise NotImplementedError

    @abstractmethod
    def _set_model(self) -> Model:
        raise NotImplementedError

    def model(self) -> Model:
        """Return the model"""
        return self._model

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class SingleModelBayesianOEDBenchmark(ABC):
    """
    Base class for Bayesian Optimal Experimental Design (OED) benchmarks.

    This class defines the structure for Bayesian OED benchmarks where derived
    classes must specify the prior and observation model.
    The base class provides utility methods to access these components and
    their properties.

    Attributes:
        _bkd (BackendMixin): Backend used for numerical computations.
        _prior (Model): The prior model representing uncertainty in the
                        parameters.
        _obs_model (Model): The observation model representing the relationship
                            between parameters and observations.
        _pred_model (Model): The prediction model representing the relationship
                             between parameters and predictions.
    """

    def __init__(self, backend: "BackendMixin" = "NumpyMixin"):
        """
        Initialize the base class with a specified backend for numerical
        computations.

        Args:
            backend (BackendMixin): Backend used for numerical computations.
            Defaults to NumpyMixin.

        Raises:
            ValueError: If any of the required attributes
            (_prior, _obs_model, _pred_model) are not set.
        """
        self._bkd = backend
        self._set_prior()
        self._set_obs_model()
        self._set_pred_model()

        # Validate that required attributes are set
        self._validate_attributes()

    @abstractmethod
    def _set_prior(self) -> "Model":
        """
        Abstract method to set the prior model.

        Derived classes must implement this method to define the prior model.

        Returns:
            Model: The prior model.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_obs_model(self) -> "Model":
        """
        Abstract method to set the observation model.

        Derived classes must implement this method to define the observation
        model.

        Returns:
            Model: The observation model.
        """
        raise NotImplementedError

    def _validate_attributes(self):
        """
        Validate that the required attributes (_prior, _obs_model, _pred_model) are set.

        Raises:
            ValueError: If any of the required attributes are not set.
        """
        if not hasattr(self, "_prior") or self._prior is None:
            raise ValueError(
                "The '_prior' attribute must be set by the derived class."
            )
        if not hasattr(self, "_obs_model") or self._obs_model is None:
            raise ValueError(
                "The '_obs_model' attribute must be set by the derived class."
            )

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the prior model.

        Returns:
            int: Number of uncertain variables.
        """
        return self._prior.nvars()

    def prior(self) -> "JointVariable":
        """
        Return the random variable parameterizing the model uncertainty.

        Returns:
            JointVariable: The prior random variable.
        """
        return self._prior

    def nobservations(self) -> int:
        """
        Return the number of observations in the observation model.

        Returns:
            int: Number of observations.
        """
        return self._obs_model.nqoi()

    def observation_model(self) -> "Model":
        """
        Return the observation model.

        Returns:
            Model: The observation model.
        """
        return self._obs_model

    def __repr__(self) -> str:
        return "{0}".format(self.__class__.__name__)


class SingleModelBayesianGoalOrientedOEDBenchmark(
    SingleModelBayesianOEDBenchmark
):
    """
    Base class for goal-oriented Bayesian Optimal Experimental Design (OED)
    benchmarks.

    This class defines the structure for Bayesian OED benchmarks where derived
    classes must specify the prior, observation model, and prediction model.
    The base class provides utility methods to access these components and
    their properties.

    This class adds functionality to access the prediction model explicitly.
    It requires the derived class to set the `_pred_model` attribute using the
    `_set_pred_model` method.

    Attributes:
        _pred_model (Model): The prediction model representing the relationship
                             between parameters and predictions.
    """

    def __init__(self, backend: "BackendMixin" = "NumpyMixin"):
        """
        Initialize the extended class with a specified backend for numerical
        computations.

        Args:
            backend (BackendMixin): Backend used for numerical computations.
            Defaults to NumpyMixin.

        Raises:
            ValueError: If the `_pred_model` attribute is not set.
        """
        super().__init__(backend)

        # Validate that the prediction model is set
        self._validate_prediction_model()

    def _validate_prediction_model(self):
        """
        Validate that the `_pred_model` attribute is set.

        Raises:
            ValueError: If the `_pred_model` attribute is not set.
        """
        if not hasattr(self, "_pred_model") or self._pred_model is None:
            raise ValueError(
                "The '_pred_model' attribute must be set by the derived class "
                "using '_set_pred_model'."
            )

    def prediction_model(self) -> "Model":
        """
        Return the prediction model.

        Returns:
            Model: The prediction model.

        Raises:
            ValueError: If the `_pred_model` attribute is not set.
        """
        return self._pred_model

    def npredictions(self) -> int:
        """
        Return the number of predictions (quantities of interest) in the
        prediction model.

        Returns:
            int: The number of predictions (QoI) in the prediction model.
        """
        return self._pred_model.nqoi()
