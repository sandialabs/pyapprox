from abc import abstractmethod, ABC
import pickle
import warnings

from pyapprox.interface.model import Model
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.minimize import (
    MultiStartOptimizer,
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.transforms import Transform, IdentityTransform
from pyapprox.util.hyperparameter import HyperParameterList
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer


class Surrogate(Model):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)

    @abstractmethod
    def nvars(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_samples(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_values(self):
        raise NotImplementedError


class Regressor(Surrogate):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._set_default_transforms()
        # canonical traning samples after transformation
        self._ctrain_samples = None
        # canonical traning values after transformation
        self._ctrain_values = None

        # TODO: consider adding transform also to OthogonalPolyBasis as
        # regressor should not have to know about constraints on
        # poly for instance Legendre samples must be in [-1, 1].
        # The regressor could apply another transform on top of this
        # but only for consistency with other regressors as
        # a second transform would not be necessary if poly
        # had its own.

        # TODO: Jacobian hessian and apply versions must be mapped from
        # canonical domain. Need to implement mappings in transforms.
        # For now make sure they are not called
        if not isinstance(self._out_trans, IdentityTransform):
            self._jacobian_implemented = False
            self._apply_jacobian_implemented = False
            self._hessian_implemented = False
            self._apply_hessian_implemented = False

    def _set_default_transforms(self):
        self.set_input_transform(IdentityTransform())
        self.set_output_transform(IdentityTransform())

    def set_input_transform(self, in_trans: Transform):
        if not isinstance(in_trans, Transform):
            raise ValueError("in_trans must be an instance of Transform")
        self._in_trans = in_trans

    def set_output_transform(self, out_trans: Transform):
        if not isinstance(out_trans, Transform):
            raise ValueError("out_trans must be an instance of Transform")
        self._out_trans = out_trans

    def _set_training_data(self, train_samples: Array, train_values: Array):
        if train_samples.shape[1] != train_values.shape[0]:
            raise ValueError(
                (
                    "Number of cols of samples {0} does not match"
                    + "number of rows of values"
                ).format(train_samples.shape[1], train_values.shape[0])
            )
        self._ctrain_samples = self._in_trans.map_to_canonical(train_samples)
        self._ctrain_values = self._out_trans.map_to_canonical(
            train_values.T
        ).T

    @abstractmethod
    def _fit(self, iterate: Array):
        raise NotImplementedError

    def fit(
        self, train_samples: Array, train_values: Array, iterate: Array = None
    ):
        """fit the regressor"""
        self._set_training_data(train_samples, train_values)
        self._fit(iterate)

    def save(self, filename):
        """
        To load, use pyapprox.sciml.network.load(filename)
        """
        pickle.dump(self, open(filename, "wb"))

    def __call__(self, samples) -> Array:
        return self._out_trans.map_from_canonical(super().__call__(samples))

    def nvars(self) -> int:
        return self._ctrain_samples.shape[0]

    def get_train_samples(self) -> Array:
        return self._in_trans.map_to_canonical(self._ctrain_samples)

    def get_train_values(self) -> Array:
        return self._out_trans.map_to_canonical(self._ctrain_values)


class AdaptiveRegressorMixin(ABC):
    @abstractmethod
    def step_samples(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def step_values(self, values: Array):
        raise NotImplementedError

    def step(self, fun: Model) -> bool:
        unique_samples = self.step_samples()
        if unique_samples is None:
            return False
        unique_values = fun(unique_samples)
        self.step_values(unique_values)
        return True

    def build(self, fun: Model):
        # step implements termination criteria
        while self.step(fun):
            pass


class OptimizedRegressor(Regressor):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._loss = None
        self._optimizer = None

    def set_loss(self, loss: LossFunction):
        if not isinstance(loss, LossFunction):
            raise ValueError(
                "loss {0} must be instance of LossFunction".format(loss)
            )
        self._loss = loss
        self._loss.set_model(self)
        self._optimizer.set_objective_function(loss)
        self._optimizer.set_bounds(self.hyp_list().get_active_opt_bounds())

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _fit(self, iterate: Array):
        if self.hyp_list().nactive_vars() == 0:
            warnings.warn("No active parameters so fit was not called")
            return
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        if iterate is None:
            # iterate = self._optimizer._initial_interate_gen()
            iterate = self.hyp_list().get_active_opt_params()[:, None]
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self.hyp_list().set_active_opt_params(active_opt_params)

    def hyperparam_jacobian(self, active_opt_params: Array) -> Array:
        if active_opt_params.ndim != 2 and active_opt_params.shape[1] != 1:
            raise ValueError(
                "active_opt_params must be a 2D array with one column"
            )
        return self._hyperparam_jacobian(active_opt_params)

    def _hyperparam_jacobian(self, active_opt_params: Array) -> Array:
        self.hyp_list().set_active_opt_params(active_opt_params[:, 0])
        return self.basis(self._ctrain_values)

    def _default_iterator_gen(self):
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            self._hyp_list.nactive_vars(), backend=self._bkd
        )
        iterate_gen.set_bounds(
            self._bkd.to_numpy(self._hyp_list.get_active_opt_bounds())
        )
        return iterate_gen

    def default_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        gtol: float = 1e-8,
        iterate_gen: OptimizerIterateGenerator = None,
    ) -> MultiStartOptimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        # L-BFGS-Bseems to require less iterations than trust-constr when
        # building GPs
        local_optimizer.set_options(
            gtol=gtol,
            maxiter=1000,
            method="L-BFGS-B",
            # method="trust-constr",
        )
        local_optimizer.set_verbosity(verbosity - 1)
        ms_optimizer = MultiStartOptimizer(
            local_optimizer, ncandidates=ncandidates
        )
        if iterate_gen is None:
            iterate_gen = self._default_iterator_gen()
        if not isinstance(iterate_gen, OptimizerIterateGenerator):
            raise ValueError(
                "iterate_gen must be an instance of OptimizerIterateGenerator"
            )
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(verbosity)
        return ms_optimizer


def QuadratureRule(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError


# def TensorProductQuadratureRule(QuadratureRule):
#     def __init__(self, basis):
#         if not isinstance(basis, TensorProductBasis):
#             raise ValueError("basis must be instance of MultiIndexBasis")
#         self._basis = basis
#         self._bkd = basis._bkd
#         self._samples = None
#         self._weights = None
#         self._compute_rule()

#     def _compute_rule(self):
#         samples_1d, weights_1d = [], []
#         for dd in range(self._basis.nvars()):
#             xx, ww = self._basis.univariate_quadrature(dd)
#             samples_1d.append(xx)
#             weights_1d.append(ww)
#         self._samples = self._bkd.cartesian_product(samples_1d)
#         self._weights = self._bkd.outer_product(weights_1d)

#     def __call__(self) -> Tuple[Array, Array]:
#         return self._samples, self._weights
