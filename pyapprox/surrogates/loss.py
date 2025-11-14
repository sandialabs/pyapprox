from pyapprox.util.backends.template import Array
from pyapprox.interface.model import Model, SingleSampleModel


class LossFunction(SingleSampleModel):
    def __init__(self):
        super().__init__()
        self._bkd = None
        self._model = None

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._model.hyp_list().nactive_vars()

    def set_model(self, model: Model):
        self._bkd = model._bkd
        self._model = model

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def apply_hessian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _check_model(self, model: Model):
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        if (
            not hasattr(model, "_ctrain_samples")
            or model._ctrain_samples is None
        ):
            raise ValueError("model must have attribute _ctrain_samples")
        if (
            not hasattr(model, "_ctrain_values")
            or model._ctrain_values is None
        ):
            raise ValueError("model must have attribute _ctrain_values")

    def _jacobian(self, active_opt_params: Array) -> Array:
        if active_opt_params.ndim != 2 or active_opt_params.shape[1] != 1:
            raise ValueError("active_opt_params must be a 2D column array")
        jac = self._bkd.jacobian(
            lambda p: self._loss_values(p[:, None])[:, 0],
            active_opt_params[:, 0],
        )
        return jac

    def _apply_hessian(self, active_opt_params: Array, vec: Array) -> Array:
        if active_opt_params.ndim != 2 or active_opt_params.shape[1] != 1:
            raise ValueError("active_opt_params must be a 2D column array")
        hvp = self._bkd.hvp(
            lambda p: self._loss_values(p[:, None])[:, 0],
            active_opt_params[:, 0],
            vec[:, 0],
        )[:, None]
        return hvp

    def _evaluate(self, active_opt_params: Array) -> Array:
        if active_opt_params.ndim != 2 or active_opt_params.shape[1] != 1:
            raise ValueError("active_opt_params must be a 2D column array")
        return self._loss_values(active_opt_params)


class RMSELoss(LossFunction):
    def _loss_values(self, active_opt_params: Array) -> Array:
        self._check_model(self._model)
        self._model.hyp_list().set_active_opt_params(active_opt_params[:, 0])
        return self._bkd.atleast2d(
            self._bkd.mean(
                self._bkd.norm(
                    (
                        self._model(self._model._ctrain_samples)
                        - self._model._ctrain_values
                    ),
                    axis=1,
                )
            )
        )
