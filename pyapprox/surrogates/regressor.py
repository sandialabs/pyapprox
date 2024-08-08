from abc import abstractmethod

from pyapprox.interface.model import Model
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.pya_minimize import MultiStartOptimizer
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class Regressor(Model):
    @abstractmethod
    def fit(self, train_samples, train_values):
        raise NotImplementedError

    @staticmethod
    def _check_training_data(train_samples, train_values):
        if train_samples.shape[1] != train_values.shape[0]:
            raise ValueError(
                (
                    "Number of cols of samples {0} does not match"
                    + "number of rows of values"
                ).format(train_samples.shape[1], train_values.shape[0])
            )
        return train_samples, train_values


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

    def fit(self, train_samples, train_values, init_iterate=None):
        """Fit the expansion by finding the optimal coefficients."""
        self._train_samples, self._train_values = self._check_training_data(
            train_samples, train_values
        )
        if self._optimizer is None:
            raise RuntimeError("must call set_optimizer")
        if init_iterate is None:
            init_iterate = self._optimizer._initial_interate_gen()
        res = self._optimizer.minimize(init_iterate)
        active_opt_params = res.x[:, 0]
        self.hyp_list.set_active_opt_params(active_opt_params)
