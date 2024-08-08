from abc import abstractmethod
import pickle

from pyapprox.interface.model import Model
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.pya_minimize import MultiStartOptimizer
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.transforms import Transform, IdentityTransform


class Regressor(Model):
    def __init__(self, backend=NumpyLinAlgMixin()):
        super().__init__(backend)
        self._in_trans = IdentityTransform()
        self._out_trans = IdentityTransform()
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

    def set_input_transform(self, in_trans: Transform):
        if not isinstance(in_trans, Transform):
            raise ValueError("in_trans must be an instance of Transform")
        self._in_trans = in_trans

    def set_output_transform(self, out_trans: Transform):
        if not isinstance(out_trans, Transform):
            raise ValueError("out_trans must be an instance of Transform")
        self._out_trans = out_trans

    def _set_training_data(self, train_samples, train_values):
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
    def _fit(self, iterate):
        raise NotImplementedError

    def fit(self, train_samples, train_values, iterate=None):
        """fit the regressor"""
        self._set_training_data(train_samples, train_values)
        self._fit(iterate)

    def save(self, filename):
        '''
        To load, use pyapprox.sciml.network.load(filename)
        '''
        pickle.dump(self, open(filename, 'wb'))

    def __call__(self, samples):
        return self._out_trans.map_from_canonical(super().__call__(samples))


class OptimizedRegressor(Regressor):
    def __init__(self, backend=NumpyLinAlgMixin()):
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
        self._optimizer.set_bounds(self.hyp_list.get_active_opt_bounds())

    def set_optimizer(self, optimizer: MultiStartOptimizer):
        if not isinstance(optimizer, MultiStartOptimizer):
            raise ValueError(
                "optimizer {0} must be instance of MultiStartOptimizer".format(
                    optimizer
                )
            )
        self._optimizer = optimizer

    def _fit(self, iterate):
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        if iterate is None:
            iterate = self._optimizer._initial_interate_gen()
        res = self._optimizer.minimize(iterate)
        active_opt_params = res.x[:, 0]
        self.hyp_list.set_active_opt_params(active_opt_params)
