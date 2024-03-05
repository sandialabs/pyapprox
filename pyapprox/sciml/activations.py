from abc import ABC, abstractmethod

from pyapprox.sciml.util._torch_wrappers import tanh, zeros, maximum, erf, exp


class Activation(ABC):
    @abstractmethod
    def _evaluate(self, values):
        raise NotImplementedError()

    def __call__(self, values):
        return self._evaluate(values)

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)


class TanhActivation(Activation):
    def _evaluate(self, values):
        return tanh(values)


class IdentityActivation(Activation):
    def _evaluate(self, values):
        return values


class RELUActivation(Activation):
    def _evaluate(self, values):
        return maximum(values, zeros(values.shape))


class GELUActivation(Activation):
    def _evaluate(self, values):
        return values * (1.0 + erf(values/1.41421356237))/2


class ELUActivation(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _evaluate(self, values):
        return values*(values > 0) + (
            self.alpha*(exp(values)-1)*(values < 0))
