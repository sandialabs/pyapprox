import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt

from pyapprox.typing.interface.functions.plot.plot1d import Plotter1D
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestPlotter1D(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        # Define the 1D function and plot limits
        nqoi = 1
        nvars = 1
        self.function = FunctionFromCallable(
            nqoi=nqoi, nvars=nvars, fun=self.example_function, bkd=self.bkd()
        )
        self.plot_limits = [0, 10]
        self.plotter = Plotter1D(self.function, plot_limits=self.plot_limits)

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sin(x)
        """
        return self.bkd().sin(samples)

    def test_plot_valid(self) -> None:
        fig, ax = plt.subplots()
        result = self.plotter.plot(ax, npts_1d=100)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


# Derived test class for NumPy backend
class TestPlotter1DNumpy(TestPlotter1D[NDArray[Any]], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd
        return NumpyBkd()


# Derived test class for PyTorch
class TestPlotter1DTorch(TestPlotter1D[torch.Tensor], unittest.TestCase):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    # Base test class TestFunction must be typed on Generic[Array]
    # and the derived class must return Backend[torch.Tensor]
    def bkd(self) -> Backend[torch.Tensor]:  # -> TorchBkd:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
