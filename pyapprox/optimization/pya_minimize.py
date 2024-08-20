import numpy as np
from scipy.optimize import minimize as scipy_minimize
import scipy

from pyapprox.util.sys_utilities import package_available
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

if package_available("ROL"):
    has_ROL = True
    from pyapprox.optimization._rol_minimize import rol_minimize
else:
    has_ROL = False


def pyapprox_minimize(
    fun,
    x0,
    args=(),
    method="rol-trust-constr",
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options={},
    x_grad=None,
):
    options = options.copy()
    if x_grad is not None and "rol" not in method:
        # Fix this limitation
        msg = f"Method {method} does not currently support gradient checking"
        # raise Exception(msg)
        print(msg)

    if "rol" in method and has_ROL:
        if callback is not None:
            raise ValueError(f"Method {method} cannot use callbacks")
        if args != ():
            raise ValueError(f"Method {method} cannot use args")
        rol_methods = {"rol-trust-constr": None}
        if method in rol_methods:
            rol_method = rol_methods[method]
        else:
            raise ValueError(f"Method {method} not found")
        return rol_minimize(
            fun,
            x0,
            rol_method,
            jac,
            hess,
            hessp,
            bounds,
            constraints,
            tol,
            options,
            x_grad,
        )

    x0 = x0.squeeze()  # scipy only takes 1D np.ndarrays
    x0 = np.atleast_1d(x0)  # change scalars to np.ndarrays
    assert x0.ndim <= 1
    if method == "rol-trust-constr" and not has_ROL:
        print("ROL requested by not available switching to scipy.minimize")
        method = "trust-constr"

    if method == "trust-constr":
        if "ctol" in options:
            del options["ctol"]
        return scipy_minimize(
            fun,
            x0,
            args,
            method,
            jac,
            hess,
            hessp,
            bounds,
            constraints,
            tol,
            callback,
            options,
        )
    elif method == "slsqp":
        hess, hessp = None, None
        if "ctol" in options:
            del options["ctol"]
        if "gtol" in options:
            ftol = options["gtol"]
            del options["gtol"]
        options["ftol"] = ftol
        if "verbose" in options:
            verbose = options["verbose"]
            options["disp"] = verbose
            del options["verbose"]
        return scipy_minimize(
            fun,
            x0,
            args,
            method,
            jac,
            hess,
            hessp,
            bounds,
            constraints,
            tol,
            callback,
            options,
        )

    raise ValueError(f"Method {method} was not found")


from abc import ABC, abstractmethod
from pyapprox.interface.model import Model, ScipyModelWrapper, ActiveSetVariableModel
from scipy.optimize import Bounds, NonlinearConstraint, LinearConstraint


class OptimizationResult(dict):
    """
    The optimization result returned by optimizers. must contain at least
    the iterate and objective function value at the minima,
    which can be accessed via res.x and res.fun, respectively.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        return self.__class__.__name__ + (
            "(\n\t x[:, 0]={0}, \n\t fun={1}, \n\t attr={2})".format(
                self.x[:, 0], self.fun, list(self.keys())))


class ScipyOptimizationResult(OptimizationResult):
    def __init__(self, scipy_result, bkd):
        """
        Parameters
        ----------
        scipy_result : :py:class:`scipy.optimize.OptimizeResult`
            The result returned by scipy.minimize
        """
        super().__init__()
        for key, item in scipy_result.items():
            if isinstance(item, np.ndarray):
                self[key] = bkd.asarray(item)
            else:
                self[key] = item


class Constraint(Model):
    def __init__(self, model, bounds, keep_feasible=False, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        if not isinstance(model, Model):
            raise ValueError(
                "objective must be an instance of {0}".format(
                    "pyapprox.interface.model.Model"
                )
            )
        self._model = model
        if (
            not isinstance(bounds, np.ndarray)
            or bounds.ndim != 2
            or bounds.shape[1] != 2
        ):
            raise ValueError("bounds must be 2D np.ndarray with two columns")
        self._bounds = bounds
        self._keep_feasible = keep_feasible
        self._update_attributes()

    def _update_attributes(self):
        for attr in [
            "_apply_jacobian_implemented",
            "_jacobian_implemented",
            "_jacobian_implemented",
            "_hessian_implemented",
            "_apply_hessian_implemented",
            "jacobian",
            "apply_jacobian",
            "apply_hessian",
            "hessian",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def _check_sample(self, sample):
        if sample.shape[1] > 1:
            raise ValueError("Constraint can only be evaluated at one sample")

    def __call__(self, sample):
        self._check_sample(sample)
        return self._model(sample)

    def __repr__(self):
        return "{0}(model={1})".format(self.__class__.__name__, self._model)


class OptimizerIterateGenerator(ABC):
    def __init__(self, backend):
        self._bkd = backend

    @abstractmethod
    def __call__(self):
        raise NotImplementedError


class RandomUniformOptimzerIterateGenerator(OptimizerIterateGenerator):
    def __init__(self, nvars, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._bounds = None
        self._nvars = nvars
        self._numeric_upper_bound = 100

    def set_bounds(self, bounds):
        bounds = self._bkd.atleast1d(bounds)
        if bounds.shape[0] == 2:
            bounds = self._bkd.reshape(
                self._bkd.repeat(bounds, self._nvars), (self._nvars, 2)
            )
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("Bounds has the wrong shape")
        self._bounds = bounds

    def set_numeric_upper_bound(self, ub):
        self._numeric_upper_bound = ub

    def __call__(self):
        if self._bounds is None:
            raise RuntimeError(
                "must call set_bounds to generate random initial guess"
            )
        # convert bounds to numpy to use numpy random number generator
        bounds = self._bkd.to_numpy(self._bounds)
        bounds[bounds == -np.inf] = -self._numeric_upper_bound
        bounds[bounds == np.inf] = self._numeric_upper_bound
        return self._bkd.asarray(
            np.random.uniform(bounds[:, 0], bounds[:, 1]))[:, None]


class Optimizer(ABC):
    def __init__(self, objective=None, bounds=None, opts={}):
        self._bkd = None
        self._verbosity = 0
        self._objective = None
        self._bounds = None

        if objective is not None:
            self.set_objective_function(objective)
        if bounds is not None:
            self.set_bounds(bounds)
        self.set_options(**opts)

    def set_options(self, **opts):
        self._opts = opts

    @abstractmethod
    def _minimize(self, iterate):
        raise NotImplementedError

    def minimize(self, iterate):
        """
        Minimize the objective function.

        Parameters
        ----------
        iterate : array
             The initial guess used to start the optimizer

        Returns
        -------
        res : :py:class:`~pyapprox.sciml.OptimizationResult`
             The optimization result.
        """
        if self._objective is None:
            raise RuntimeError("Must call set_objective_function")
        if iterate.ndim != 2 or iterate.shape[1] != 1:
            raise ValueError("iterate must be a 2D array with one column.")
        result = self._minimize(iterate)
        if not isinstance(result, OptimizationResult):
            raise RuntimeError(
                "{0}.minimize did not return OptimizationResult".format(self)
            )
        return result

    def set_bounds(self, bounds):
        """
        Set the bounds of the design variables.

        Parameters
        ----------
        bounds : array (ndesign_vars, 2)
            The upper and lower bounds of each design variable
        """
        if (bounds.ndim != 2 or bounds.shape[1] != 2):
            raise ValueError("Bounds has the wrong shape")
        self._bounds = bounds

    def set_objective_function(self, objective):
        """
        Set the objective function.

        Parameters
        ----------
        objective_fun : callable
            Function that returns both the function value and gradient at an
            iterate with signature

            `objective_fun(x) -> (val, grad)`

            where `x` and `val` are 1D arrays with shape (ndesign_vars,) and
            `val` is a float.
        """
        if not isinstance(objective, Model):
            raise ValueError(
                "objective must be an instance of {0}".format(
                    "pyapprox.interface.model.Model"
                )
            )
        self._bkd = objective._bkd
        self._objective = objective

    def set_verbosity(self, verbosity):
        """
        Set the verbosity.

        Parameters
        ----------
        verbosity_flag : int, default 0
            0 = no output
            1 = final iteration
            2 = each iteration
            3 = each iteration, plus details
        """
        self._verbosity = verbosity

    def _is_iterate_within_bounds(self, iterate):
        if self._bounds is None:
            return True
        # convert bounds to numpy to use np.logical
        bounds = self._bkd.to_numpy(self._bounds)
        iterate = self._bkd.to_numpy(iterate)
        return np.logical_and(
            iterate >= bounds[:, 0],
            iterate <= bounds[:, 1]).all()

    def __repr__(self):
        return "{0}(verbosity={1})".format(
            self.__class__.__name__, self._verbosity
        )


class MultiStartOptimizer(Optimizer):
    def __init__(self, optimizer, ncandidates=1):
        """
        Find the smallest local optima associated with a set of
        initial guesses.

        Parameters
        ----------
        optimizer : :py:class:`~pyapprox.sciml.Optimizer`
            Optimizer to find each local minima

        ncandidates : int
            Number of initial guesses used to comptue local optima
        """
        self._ncandidates = ncandidates
        self._optimizer = optimizer
        self._bounds = None
        self._initial_interate_gen = None
        super().__init__(objective=optimizer._objective)

    def set_bounds(self, bounds):
        super().set_bounds(bounds)
        self._optimizer.set_bounds(bounds)

    def set_initial_iterate_generator(self, gen):
        if not isinstance(gen, OptimizerIterateGenerator):
            raise ValueError("gen is not an OptimizerIterateGenerator.")
        self._initial_interate_gen = gen

    def set_objective_function(self, objective):
        self._optimizer.set_objective_function(objective)
        super().set_objective_function(objective)

    def _minimize(self, x0_global, **kwargs):
        if self._initial_interate_gen is None:
            raise ValueError("Must call set_initial_iterate_generator")
        best_res = self._optimizer.minimize(x0_global)
        if self._verbosity > 1:
            print("it {0}: best objective {1}".format(0, best_res.fun))
        for ii in range(1, self._ncandidates):
            iterate = self._initial_interate_gen()
            res = self._optimizer.minimize(iterate, **kwargs)
            if res.fun < best_res.fun:
                best_res = res
            if self._verbosity > 1:
                print("it {0}: best objective {1}".format(ii+1, best_res.fun))
        if self._verbosity > 0:
            print("{0}\n\t {1}".format(self, best_res))
        return best_res

    def __repr__(self):
        return "{0}(optimizer={1}, ncandidates={2})".format(
            self.__class__.__name__, self._optimizer, self._ncandidates
        )


class ConstrainedOptimizer(Optimizer):
    def __init__(self, objective=None, constraints=[], bounds=None, opts={}):
        super().__init__(objective, bounds, opts)
        self.set_constraints(constraints)

    def set_constraints(self, constraints):
        for con in constraints:
            if not isinstance(con, Constraint) and not isinstance(
                con, LinearConstraint
            ):
                raise ValueError(
                    "constraint must be an instance of {0} or {1}".format(
                        "pyapprox.optimize.pya_minimize.Constraint",
                        "scipy.optimize.LinearConstraint",
                    )
                )
        self._constraints = self._convert_constraints(constraints)

    def _convert_constraints(self, constraints):
        return constraints


class ScipyConstrainedOptimizer(ConstrainedOptimizer):
    def _convert_constraints(self, constraints):
        scipy_constraints = []
        for _con in constraints:
            if isinstance(_con, LinearConstraint):
                scipy_constraints.append(_con)
                continue
            con = ScipyModelWrapper(_con)
            jac = con.jac if con._jacobian_implemented else "2-point"
            if con._apply_hessian_implemented or con._hessian_implemented:
                hess = con.hessp
            else:
                hess = scipy.optimize._hessian_update_strategy.BFGS()
            scipy_con = NonlinearConstraint(
                con, *_con._bounds.T, jac, hess, _con._keep_feasible
            )
            scipy_constraints.append(scipy_con)
        return scipy_constraints

    def _minimize(self, init_guess):
        opts = self._opts.copy()
        nvars = init_guess.shape[0]
        if self._bounds is None:
            bounds = Bounds(
                np.full((nvars,), -np.inf), np.full((nvars,), np.inf)
            )
        else:
            bounds = self._bounds
        objective = ScipyModelWrapper(self._objective)
        jac = objective.jac if objective._jacobian_implemented else None
        if objective._apply_hessian_implemented or objective._hessian_implemented:
            hessp = objective.hessp
        else:
            hessp = None

        method = opts.pop("method", "trust-constr")
        if method == "L-BFGS-B":
            if self._verbosity < 3:
                self._opts['iprint'] = self._verbosity-1
            else:
                self._opts['iprint'] = 200
        elif method == "trust-constr":
            self._opts["verbose"] = self._verbosity
        elif method == "slsqp":
            if self._verbosity > 0:
                self._opts["disp"]=True
                self._opts["iprint"] = self._verbosity

        scipy_result = scipy_minimize(
            objective,
            init_guess[:, 0],
            method=method,
            jac=jac,
            hessp=hessp,
            bounds=bounds,
            constraints=self._constraints,
            options=self._opts,
        )
        scipy_result.x = scipy_result.x[:, None]
        result = ScipyOptimizationResult(scipy_result, self._bkd)
        if self._verbosity > 1:
            print(result)
        return result


# TODO consider merging with multifidelity.stat
class SampleAverageStat(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def __call__(self, values, weights):
        raise NotImplementedError

    def jacobian(self, jac_values, weights):
        raise NotImplementedError

    def jacobian_apply(self, values, jv_values, weights, vec):
        raise NotImplementedError

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)


class SampleAverageMean(SampleAverageStat):
    def __call__(self, values, weights):
        # values.shape (nsamples, ncontraints)
        return (values.T @ weights).T

    def jacobian(self, values, jac_values, weights):
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        return self._bkd.einsum("ijk,i->jk", jac_values, weights[:, 0])

    def apply_jacobian(self, values, jv_values, weights):
        # jac_values.shape (nsamples, ncontraints)
        return (jv_values[..., 0].T @ weights[:, 0])[:, None]


class SampleAverageVariance(SampleAverageStat):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend=backend)
        self._mean_stat = SampleAverageMean()

    def _diff(self, values, weights):
        mean = self._mean_stat(values, weights).T
        return (values - mean[:, 0]).T

    def __call__(self, values, weights):
        # values.shape (nsamples, ncontraints)
        return (self._diff(values, weights) ** 2 @ weights).T

    def jacobian(self, values, jac_values, weights):
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)[None, :]
        tmp = jac_values - mean_jac
        tmp = 2 * self._diff(values, weights).T[..., None] * tmp
        return self._bkd.einsum("ijk,i->jk", tmp, weights[:, 0])

    def apply_jacobian(self, values, jv_values, weights):
        mean_jv = self._mean_stat.apply_jacobian(values, jv_values, weights)
        tmp = jv_values - mean_jv
        tmp = 2 * self._diff(values, weights).T * tmp[..., 0]
        return self._bkd.einsum("ij,i->j", tmp, weights[:, 0])


class SampleAverageStdev(SampleAverageVariance):
    def __call__(self, samples, weights):
        return self._bkd.sqrt(super().__call__(samples, weights))

    def jacobian(self, values, jac_values, weights):
        variance_jac = super().jacobian(values, jac_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super().__call__(values, weights).T))
        return tmp * variance_jac

    def apply_jacobian(self, values, jac_values, weights):
        variance_jv = super().apply_jacobian(values, jac_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super().__call__(values, weights).T))
        return tmp * variance_jv[:, None]


class SampleAverageMeanPlusStdev(SampleAverageStat):
    def __init__(self, safety_factor):
        self._mean_stat = SampleAverageMean()
        self._stdev_stat = SampleAverageStdev()
        self._safety_factor = safety_factor

    def __call__(self, values, weights):
        return self._mean_stat(
            values, weights
        ) + self._safety_factor * self._stdev_stat(values, weights)

    def jacobian(self, values, jac_values, weights):
        return self._mean_stat.jacobian(
            values, jac_values, weights
        ) + self._safety_factor * self._stdev_stat.jacobian(values, jac_values, weights)

    def apply_jacobian(self, values, jv_values, weights):
        return self._mean_stat.apply_jacobian(
            values, jv_values, weights
        ) + self._safety_factor * self._stdev_stat.apply_jacobian(
            values, jv_values, weights
        )


class SampleAverageEntropicRisk(SampleAverageStat):
    def __init__(self, alpha, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._alpha = alpha

    def __call__(self, values, weights):
        # values (nsamples, noutputs)
        return self._bkd.log(self._bkd.exp(self._alpha * values.T) @ weights).T / self._alpha

    def jacobian(self, values, jac_values, weights):
        # jac_values (nsamples, noutputs, nvars)
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        jac = (
            1
            / tmp
            * self._bkd.einsum(
                "ijk,i->jk", (exp_values[..., None] * jac_values), weights[:, 0]
            )
        )
        return jac

    def apply_jacobian(self, values, jv_values, weights):
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        jv = (
            1
            / tmp
            * self._bkd.einsum("ij,i->j", (exp_values * jv_values[..., 0]), weights[:, 0])[
                :, None
            ]
        )
        return jv


class SmoothLogBasedMaxFunction:
    def __init__(self, eps, threshold=None, backend=NumpyLinAlgMixin):
        self._bkd = backend
        self._eps = eps
        if threshold is None:
            threshold = 1e2
        self._thresh = threshold
        self._jacobian_implemented = True

    def _check_samples(self, samples):
        if samples.ndim != 2:
            raise ValueError("samples must be a 2D array")

    def __call__(self, samples):
        self._check_samples(samples)
        x = samples
        x_div_eps = x / self._eps
        # avoid overflow
        vals = self._bkd.zeros(x.shape)
        II = self._bkd.where((x_div_eps < self._thresh) & (x_div_eps > -self._thresh))
        vals[II] = x[II] + self._eps * self._bkd.log(1 + self._bkd.exp(-x_div_eps[II]))
        J = self._bkd.where(x_div_eps >= self._thresh)
        vals[J] = x[J]
        return vals

    def jacobians(self, samples):
        # samples (noutputs, nsamples)
        # jac_values (nsamples, noutputs, noutputs)
        # but only return (nsamples, noutputs) because jac for each sample
        # is just a diagonal matrix
        self._check_samples(samples)
        x = samples
        x_div_eps = x / self._eps
        # Avoid overflow.
        II = self._bkd.where((x_div_eps < self._thresh) & (x_div_eps > -self._thresh))
        jac = self._bkd.zeros((x_div_eps.shape))
        jac[II] = 1.0 / (1 + self._bkd.exp(-x_div_eps[II]))
        jac[x_div_eps >= self._thresh] = 1.0
        return jac[..., None]


class SampleAverageConditionalValueAtRisk(SampleAverageStat):
    def __init__(self, alpha, eps=1e-2, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        alpha = self._bkd.atleast1d(alpha)
        self._alpha = alpha
        self._max = SmoothLogBasedMaxFunction(eps, backend=self._bkd)
        self._t = None

    def set_value_at_risk(self, t):
        t = self._bkd.atleast1d(t)
        if t.shape[0] != self._alpha.shape[0]:
            msg = "VaR shape {0} and alpha shape {1} are inconsitent".format(
                t.shape, self._alpha.shape
            )
            raise ValueError(msg)
        if t.ndim != 1:
            raise ValueError("t must be a 1D array")
        self._t = t

    def __call__(self, values, weights):
        if values.shape[1] != self._t.shape[0]:
            raise ValueError("must specify a VaR for each QoI")
        return self._t + (self._max(values - self._t).T @ weights).T / (1 - self._alpha)

    def jacobian(self, values, jac_values, weights):
        # grad withe respect to parameters of x
        max_jac = self._max.jacobians(values - self._t)
        param_jac = self._bkd.einsum("ijk,i->jk", (max_jac * jac_values), weights[:, 0]) / (
            1 - self._alpha[:, None]
        )
        t_jac = 1 - self._bkd.einsum("ij,i->j", max_jac[..., 0], weights[:, 0]) / (
            1 - self._alpha
        )
        return self._bkd.hstack((param_jac, self._bkd.diag(t_jac)))


class SampleAverageConstraint(Constraint):
    def __init__(
        self,
        model,
        samples,
        weights,
        stat,
        design_bounds,
        nvars,
        design_indices,
        keep_feasible=False,
        backend=NumpyLinAlgMixin
    ):
        super().__init__(model, design_bounds, keep_feasible, backend=backend)
        if samples.ndim != 2 or weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("shapes of samples and/or weights are incorrect")
        if samples.shape[1] != weights.shape[0]:
            raise ValueError("samples and weights are inconsistent")
        self._weights = weights
        self._samples = samples
        self._stat = stat
        self._nvars = nvars
        self._design_indices = design_indices
        self._random_indices = self._bkd.delete(self._bkd.arange(nvars, dtype=int), design_indices)
        # warning self._joint_samples must be recomputed if self._samples
        # is changed.
        self._joint_samples = ActiveSetVariableModel._expand_samples_from_indices(
            self._samples,
            self._random_indices,
            self._design_indices,
            self._bkd.zeros((design_indices.shape[0], 1)),
        )

    def _update_attributes(self):
        self._jacobian_implemented = self._model._jacobian_implemented
        self._apply_jacobian_implemented = self._model._apply_jacobian_implemented
        self._hessian_implemented = False
        self._apply_hessian_implemented = False

    def _random_samples_at_design_sample(self, design_sample):
        # this is slow so only update design samples as self._samples is
        # always fixed
        # return ActiveSetVariableModel._expand_samples_from_indices(
        #     self._samples, self._random_indices, self._design_indices,
        #     design_sample)
        self._joint_samples[self._design_indices, :] = self._bkd.asarray(
            np.repeat(design_sample, self._samples.shape[1], axis=1)
        )
        return self._joint_samples

    def __call__(self, design_sample):
        self._check_sample(design_sample)
        samples = self._random_samples_at_design_sample(design_sample)
        values = self._model(samples)
        return self._stat(values, self._weights)

    def _jacobian(self, design_sample):
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple jacobians. Also how to take advantage of
        # adjoint methods that compute model values to then
        # compute jacobian
        values = self._model(samples)
        jac_values = self._bkd.array(
            [
                self._model.jacobian(sample[:, None])[:, self._design_indices]
                for sample in samples.T
            ]
        )
        return self._stat.jacobian(values, jac_values, self._weights)

    def _apply_jacobian(self, design_sample, vec):
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple apply_jacs
        expanded_vec = self._bkd.zeros((self._nvars, 1))
        expanded_vec[self._design_indices] = vec
        values = self._model(samples)
        jv_values = self._bkd.array(
            [
                self._model._apply_jacobian(sample[:, None], expanded_vec)
                for sample in samples.T
            ]
        )
        return self._stat.apply_jacobian(values, jv_values, self._weights)

    def __repr__(self):
        return "{0}(model={1}, stat={2})".format(
            self.__class__.__name__, self._model, self._stat
        )


class CVaRSampleAverageConstraint(SampleAverageConstraint):
    def __init__(
        self,
        model,
        samples,
        weights,
        stat,
        design_bounds,
        nvars,
        design_indices,
        keep_feasible=False,
    ):
        if not isinstance(stat, SampleAverageConditionalValueAtRisk):
            msg = "stat not instance of SampleAverageConditionalValueAtRisk"
            raise ValueError(msg)
        self._nconstraints = stat._alpha.shape[0]
        super().__init__(
            model,
            samples,
            weights,
            stat,
            design_bounds,
            nvars,
            design_indices,
            keep_feasible,
        )

    def __call__(self, design_sample):
        # assumes avar variable t is at the end of design_sample
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        return super().__call__(design_sample[: -self._nconstraints])

    def _jacobian(self, design_sample):
        self._stat.set_value_at_risk(design_sample[-self._nconstraints :, 0])
        jac = super()._jacobian(design_sample[: -self._nconstraints])
        return jac


class ObjectiveWithCVaRConstraints(Model):
    """
    When optimizing for CVaR additional variables t are introduced.
    This class wraps a function that does not take variables t
    and returns a jacobian that includes derivatives with respect to the
    variables t (they will be zero).

    Assumes samples consist of vstack(random_vars, t)
    """

    def __init__(self, model, ncvar_constraints):
        super().__init__()
        self._model = model
        self._ncvar_constraints = ncvar_constraints
        self._jacobian_implemented = self._model._jacobian_implemented
        self._apply_jacobian_implemented = self._model._apply_jacobian_implemented

    def __call__(self, design_samples):
        return self._model(design_samples[: -self._ncvar_constraints])

    def _apply_jacobian(self, design_sample, vec):
        return self._model.apply_jacobian(
            design_sample[: -self._ncvar_constraints], vec[: -self._ncvar_constraints]
        )

    def _jacobian(self, design_sample):
        jac = self._model.jacobian(design_sample[: -self._ncvar_constraints])
        return self._bkd.hstack((jac, self._bkd.zeros((jac.shape[0], self._ncvar_constraints))))


def approx_jacobian(func, x, epsilon=np.sqrt(np.finfo(float).eps)):
    x0 = np.asfarray(x)
    assert x0.ndim == 1 or x0.shape[1] == 1
    f0 = np.atleast_2d(func(x0))
    assert f0.shape[0] == 1
    f0 = f0[0, :]
    jac = np.zeros([len(f0), len(x0)])
    dx = np.zeros(x0.shape)
    for ii in range(len(x0)):
        dx[ii] = epsilon
        f1 = np.atleast_2d(func(x0 + dx))
        assert f1.shape[0] == 1
        f1 = f1[0, :]
        jac[:, ii] = (f1 - f0) / epsilon
        dx[ii] = 0.0
    return jac


def approx_hessian(jac_fun, x, epsilon=np.sqrt(np.finfo(float).eps)):
    return approx_jacobian(lambda y: jac_fun(y).T, x, epsilon)
