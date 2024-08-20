from pyapprox.interface.model import Model


class LossFunction(Model):
    def __init__(self):
        super().__init__()
        self._bkd = None
        self._model = None

    def set_model(self, model):
        self._bkd = model._bkd
        self._model = model
        # TODO check if analytical jacobian available and use
        # with chain rule to compute grad of objective but for
        # now assume that autograd must be used
        if not self._bkd.jacobian_implemented():
            # todo implement gradients via custom backprop
            # only requires slight modification of _core_jacobian
            # to be more efficient by storing certain info
            # when sweeping through the cores
            raise NotImplementedError(
                "Backend must support auto differentiation."
            )

    def _check_model(self, model):
        if (
            not hasattr(model, "_ctrain_samples") or
            model._ctrain_samples is None
        ):
            raise ValueError("model must have attribute _ctrain_samples")
        if not hasattr(model, "_ctrain_values") or model._ctrain_values is None:
            raise ValueError("model must have attribute _ctrain_values")


class RMSELoss(LossFunction):
    def __init__(self):
        super().__init__()
        self._jacobian_implemented = True

    def __call__(self, active_opt_params):
        self._check_model(self._model)
        self._model.hyp_list.set_active_opt_params(active_opt_params[:, 0])
        return self._bkd.atleast2d(
            self._bkd.mean(
                self._bkd.norm(
                    (
                        self._model(self._model._ctrain_samples) -
                        self._model._ctrain_values
                    ),
                    axis=1,
                )
            )
        )

    def _jacobian(self, active_opt_params):
        val, grad = self._bkd.grad(self.__call__, active_opt_params)
        # todo move detach to linalgmixin if needed
        for hyp in self._model.hyp_list.hyper_params:
            self._bkd.detach(hyp)
        return self._bkd.detach(grad).T
