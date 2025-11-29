import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.contour import QuadContourSet

from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.typing.interface.functions.plot.plot2d_rectangular import (
    Plotter2DRectangularDomain,
    meshgrid_samples,
)
from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase


class TestPlotter2DRectangularDomain(Generic[Array], AbstractTestCase):
    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def setUp(self) -> None:
        # Define the 2D function and plot limits
        nqoi = 1
        nvars = 2
        self.function = FunctionFromCallable(
            nqoi=nqoi,
            nvars=nvars,
            fun=self.example_function,
            bkd=self.bkd(),
        )
        self.plot_limits = [0, 10, 0, 10]
        self.plotter = Plotter2DRectangularDomain(
            self.function, self.plot_limits
        )

    def example_function(self, samples: Array) -> Array:
        """
        Example function: Z = sin(x) * cos(y)
        """
        x, y = samples[0], samples[1]
        return self.bkd().stack(
            [self.bkd().sin(x) * self.bkd().cos(y)], axis=0
        )

    def test_meshgrid_samples(self) -> None:
        X, Y, pts = meshgrid_samples(50, self.plotter._plot_limits, self.bkd())
        self.assertEqual(X.shape, (50, 50))
        self.assertEqual(Y.shape, (50, 50))
        self.assertEqual(pts.shape, (2, 2500))  # Flattened meshgrid samples
        self.bkd().assert_allclose(X[0, :], self.bkd().linspace(0, 10, 50))
        self.bkd().assert_allclose(Y[:, 0], self.bkd().linspace(0, 10, 50))

    def test_plot_surface(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = self.plotter.plot_surface(ax, qoi=0, npts_1d=50)
        self.assertIsInstance(result, Axes3D)

    def test_plot_contours(self) -> None:
        fig, ax = plt.subplots()
        result = self.plotter.plot_contours(ax, qoi=0, npts_1d=50)
        self.assertIsInstance(result, QuadContourSet)

    def test_plot(self) -> None:
        # Test 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = self.plotter.plot(ax, qoi=0, npts_1d=50)
        self.assertIsInstance(result, Axes3D)

        # Test 2D plot
        fig, ax = plt.subplots()
        result = self.plotter.plot(ax, qoi=0, npts_1d=50)
        self.assertIsInstance(result, QuadContourSet)


# Derived test class for NumPy backend
class TestPlotter2DRectangularDomainNumpy(
    TestPlotter2DRectangularDomain[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Derived test class for PyTorch backend
class TestPlotter2DRectangularDomainTorch(
    TestPlotter2DRectangularDomain[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
