from abc import ABC, abstractmethod
from pyapprox.sciml.util import LinAlgMixin, TorchLinAlgMixin


class Activation(ABC):
    def __init__(self, backend: LinAlgMixin = TorchLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def _evaluate(self, values):
        raise NotImplementedError()

    def __call__(self, values):
        return self._evaluate(values)

    def __repr__(self):
        return "{0}()".format(self.__class__.__name__)


class TanhActivation(Activation):
    def _evaluate(self, values):
        return self._bkd.tanh(values)


class IdentityActivation(Activation):
    def _evaluate(self, values):
        return values


class RELUActivation(Activation):
    def _evaluate(self, values):
        return self._bkd.maximum(values, self._bkd.zeros(values.shape))


class GELUActivation(Activation):
    def _evaluate(self, values):
        '''Use GELU approximation'''
        pi = self._bkd.arccos(-1)
        return 0.5*values*(1.0 + self._bkd.tanh(self._bkd.sqrt(2.0 / pi)) * (
                   values + 0.044715*values**3))


class ELUActivation(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _evaluate(self, values):
        return values*(values > 0) + (
            self.alpha*(self._bkd.exp(values)-1)*(values < 0))
