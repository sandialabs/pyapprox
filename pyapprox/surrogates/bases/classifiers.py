from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.surrogates.loss import LossFunction
from pyapprox.surrogates.regressor import OptimizedRegressor
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.optimization.pya_minimize import MultiStartOptimizer
from pyapprox.interface.model import Model


class CrossEntropyLoss(LossFunction):
    def set_model(self, model: Model):
        super().set_model(model)
        self._jacobian_implemented = model._jacobian_implemented

    def _loss_values(self, active_opt_params: Array) -> Array:
        self._check_model(self._model)
        self._model.hyp_list.set_active_opt_params(active_opt_params[:, 0])
        # always pass in _ctrain samples which is data after processing
        # by surrogate. Surrogates such as PCE may have additional
        # transformations under that operate on _ctrain_sampels
        prob = self._model(self._model._ctrain_samples)
        obs = self._model._ctrain_values
        return -self._bkd.atleast2d(
            self._bkd.mean(
                (
                    obs * self._bkd.log(prob)
                    + (1 - obs) * self._bkd.log(1 - prob)
                ),
                axis=0,
            )
        )

    def _jacobian(self, active_opt_params: Array) -> Array:
        self._model.hyp_list.set_active_opt_params(active_opt_params[:, 0])
        prob = self._model(self._model._ctrain_samples)
        obs = self._model._ctrain_values
        return -self._bkd.mean(
            (
                (obs - prob)
                / ((1 - prob) * prob)
                * self._model.hyperparam_jacobian(active_opt_params)
            ),
            axis=0,
        )[None, :]


class CrossEntropyLossLogisticRegression(CrossEntropyLoss):
    def __init__(self, penalty_weight: float = 1):
        self._penalty_weight = penalty_weight
        super().__init__()

    def _loss_values(self, active_opt_params: Array) -> Array:
        loss = super()._loss_values(active_opt_params)
        penalty = self._bkd.sum(active_opt_params ** 2)
        nsamples = self._model._ctrain_samples.shape[1]
        return loss + 0.5 * self._penalty_weight * penalty / nsamples

    def _jacobian(self, active_opt_params: Array) -> Array:
        self._model.hyp_list.set_active_opt_params(active_opt_params[:, 0])
        prob = self._model(self._model._ctrain_samples)
        obs = self._model._ctrain_values
        basis = self._model._bexp.basis(self._model._ctrain_samples)
        nsamples = self._model._ctrain_samples.shape[1]
        return (
            (basis.T @ (prob-obs)).T / nsamples
            + self._penalty_weight / nsamples * active_opt_params.T
        )

    def _apply_hessian(self, active_opt_params: Array, vec: Array) -> Array:
        basis = self._model._bexp.basis(self._model._ctrain_samples)
        prob = self._model(self._model._ctrain_samples)
        nsamples = self._model._ctrain_samples.shape[1]
        return (
            basis.T @ (prob * (1. - prob) * basis @ vec) / nsamples
            + self._penalty_weight / nsamples * vec
        )

    def set_model(self, model: Model):
        super().set_model(model)
        self.jacobian_implemented = True
        self._apply_hessian_implemented = True


class LogisticClassifier(OptimizedRegressor):
    def __init__(self, basisexp: PolynomialChaosExpansion):
        self._bexp = basisexp
        super().__init__(backend=basisexp._bkd)
        self.hyp_list = self._bexp.hyp_list
        self._jacobian_implemented = True
        self._apply_hessian_implemented = True

    def set_optimizer(
            self, optimizer: MultiStartOptimizer, penalty_weight: float = 1.
    ):
        # penalty needed if data classes are perfectly separated
        super().set_optimizer(optimizer)
        # loss = CrossEntropyLoss()
        loss = CrossEntropyLossLogisticRegression(penalty_weight)
        loss.set_model(self)
        self._optimizer.set_objective_function(loss)

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._bexp.nvars()

    def _values(self, samples: Array) -> Array:
        return 1.0 / (1.0 + self._bkd.exp(-self._bexp(samples)))

    def hyperparam_jacobian(self, active_opt_params: Array) -> Array:
        self._bexp.hyp_list.set_active_opt_params(active_opt_params[:, 0])
        exp_vals = self._bkd.exp(self._bexp(self._ctrain_samples))
        return (
            exp_vals
            / (1 + exp_vals) ** 2
            * self._bexp.basis(self._ctrain_samples)
        )

    def labels(self, samples: Array, threshold: float = 0.5) -> Array:
        # threshold in [0, 1]
        return self._bkd.asarray(self(samples) > threshold)
