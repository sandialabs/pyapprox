import umbridge
import numpy as np

from pyapprox.benchmarks.genz import GenzFunction


class GenzModel(umbridge.Model):
    def __init__(self):
        super().__init__("genz")
        self._model = GenzFunction()

    def get_input_sizes(self, config):
        return [config.get("nvars", 5)]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        sample = np.asarray(parameters).T
        assert sample.shape[0] == config.get("nvars", 5)
        self._model.set_coefficients(
            sample.shape[0],
            config.get("c_factor", 1),
            config.get("coef_type", "sqexp"),
            config.get("w_factor", 0.5))
        name = config.get("name", "oscillatory")
        val = self._model(name, sample)[0, 0]
        return [[val]]

    def supports_evaluate(self):
        return True

    def gradient(self, out_wrt, in_wrt, parameters, sens, config):
        sample = np.asarray(parameters).T
        self._model.set_coefficients(
            int(config.get("nvars", 5)),
            config.get("c_factor", 1.),
            config.get("coef_type", "sqexp"),
            config.get("w_factor", 0.5))
        name = config.get("name", "oscillatory")
        grad = np.atleast_1d(
            self._model(name, sample, return_grad=True)[1]).tolist()
        return grad

    def supports_gradient(self):
        return True


class GenzIntegral(umbridge.Model):
    def __init__(self):
        super().__init__("genz-integral")
        self._model = GenzFunction()

    def get_input_sizes(self, config):
        return [0]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        self._model.set_coefficients(
            int(config.get("nvars", 5)),
            config.get("c_factor", 1.),
            config.get("coef_type", "sqexp"),
            config.get("w_factor", 0.5))
        name = config.get("name", "oscillatory")
        val = np.atleast_1d(self._model.integrate(name))[0]
        return [[val]]

    def supports_evaluate(self):
        return True


if __name__ == "__main__":
    models = [GenzModel(), GenzIntegral()]
    umbridge.serve_models(models, 4242)
