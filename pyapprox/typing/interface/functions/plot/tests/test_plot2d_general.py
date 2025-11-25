import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.contour import QuadContourSet

from pyapprox.typing.interface.functions.function import FunctionFromCallable
from pyapprox.typing.interface.functions.plot.plot2d_rectangular import (
    meshgrid_samples,
)
from pyapprox.typing.interface.functions.plot.plot2d_general import (
    Plotter2DGeneralDomain,
)
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestPlotter2DGeneralDomain(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        # Define the 2D function
        nqoi = 1
        nvars = 2
        self.function = FunctionFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=self.example_function,
            bkd=self.bkd(),
        )
        self.plotter = Plotter2DGeneralDomain(self.function)

        # Generate scattered points in the domain
        self.points = self.generate_points()

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sin(x) * cos(y)
        """
        x, y = samples[0], samples[1]
        return self.bkd().stack(
            [self.bkd().sin(x) * self.bkd().cos(y)], axis=0
        )

    def generate_points(self) -> Array:
        """
        Generate scattered points in the domain.
        """
        X, Y, pts = meshgrid_samples(50, [-1, 1, -1, 1], self.bkd())
        return pts

    def test_plot_trisurf(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = self.plotter.plot_trisurf(ax, self.points, qoi=0)
        self.assertIsInstance(result, Axes3D)

    def test_plot_tricontour(self) -> None:
        fig, ax = plt.subplots()
        result = self.plotter.plot_tricontour(ax, self.points, qoi=0)
        self.assertIsInstance(result, QuadContourSet)


# Derived test class for NumPy backend
class TestPlotter2DGeneralDomainNumpy(
    TestPlotter2DGeneralDomain[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestPlotter2DGeneralDomainTorch(
    TestPlotter2DGeneralDomain[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
