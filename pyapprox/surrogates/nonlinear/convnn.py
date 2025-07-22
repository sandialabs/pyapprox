from typing import Tuple

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.hyperparameter import (
    HyperParameterList,
    HyperParameter,
    IdentityHyperParameterTransform,
)


class TwoDimensionalConvolution:
    def __init__(self, nfilters: int, backend: BackendMixin):
        self._bkd = backend
        self._nfilters = nfilters
        self._filter_dim = 3
        transform = IdentityHyperParameterTransform(backend=self._bkd)
        init_weights = self._bkd.asarray(
            np.random.normal(
                0, 0.1, (self._nfilters * self._filter_dim * self._filter_dim,)
            )
        )
        self._weights = HyperParameter(
            "weights",
            nfilters * self._filter_dim * self._filter_dim,
            init_weights,
            (-np.inf, np.inf),
            transform,
            fixed=False,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._weights])

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _iterate_subdomains(self, array: Array) -> Tuple[Array, int, int]:
        nrows, ncols = array.shape
        for ii in range(nrows - (self._filter_dim - 1)):
            for jj in range(ncols - (self._filter_dim - 1)):
                yield ii, jj, array[
                    ii : ii + self._filter_dim, jj : jj + self._filter_dim
                ]

    def __call__(self, array: Array) -> Array:
        nrows, ncols = array.shape
        values = self._bkd.empty(
            (
                self._nfilters,
                nrows - (self._filter_dim - 1),
                ncols - (self._filter_dim - 1),
            )
        )
        weights = self._bkd.reshape(
            self._weights.get_values(),
            (self._nfilters, self._filter_dim, self._filter_dim),
        )
        for ii, jj, subdomain in self._iterate_subdomains(array):
            # f: filters, r: rows, c: cols
            values[:, ii, jj] = self._bkd.einsum(
                "frc,rc->f", weights, subdomain
            )
        return values


class MaxPoolingLayer:
    def __init__(self, backend: BackendMixin):
        self._bkd = backend
        self._pool_dim = 2

    def _iterate_subdomains(self, array: Array) -> Tuple[Array, int, int]:
        nrows, ncols = array.shape
        for ii in range(nrows // self._pool_dim):
            for jj in range(ncols // self._pool_dim):
                yield ii, jj, array[
                    ii * self._pool_dim : (ii + 1) * self._pool_dim,
                    jj * self._pool_dim : (jj + 1) * self._pool_dim,
                ]

    def __call__(self, array: Array) -> Array:
        nrows, ncols = array.shape
        if nrows % self._filter_dim != 0 or ncols % self._filter_dim != 0:
            raise ValueError(
                "The number of rows or columns in the image are not "
                f"divisible by {self._filter_dim=}"
            )
        values = self._bkd.empty(
            (self._nfilters, nrows // self._pool_dim, ncols // self._pool_dim)
        )
        for ii, jj, subdomain in self._iterate_subdomains(array):
            values[:, ii, jj] = self._bkd.max(subdomain, axis=(1, 2))
        return values


from mnist import MNIST
from pyapprox.util.backends.torch import TorchMixin as bkd

conv = TwoDimensionalConvolution(8, bkd)
pool = MaxPoolingLayer(bkd)

# requires pip install torchvision
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

np.random.seed(1)
training_data = datasets.FashionMNIST(
    root="~/minst-fashion", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="~/minst-fahsion", train=False, download=True, transform=ToTensor()
)

image, label = training_data[0]
image = bkd.asarray(image)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
fig, axs = plt.subplots(1, 3, figsize=(3 * 8, 6), sharey=True)
fig.suptitle(labels_map[label])
[ax.axis("off") for ax in axs]
axs[0].imshow(image.squeeze(), cmap="gray")
axs[0].set_title("Image")

convolution = TwoDimensionalConvolution(1, bkd)
horizontal_sobel_filter = bkd.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
convolution.hyp_list().set_active_opt_params(horizontal_sobel_filter.flatten())
axs[1].imshow(convolution(image.squeeze())[0], cmap="gray")
axs[1].set_title("Horizontal Sobel Filter")

vertical_sobel_filter = bkd.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
convolution.hyp_list().set_active_opt_params(vertical_sobel_filter.flatten())
axs[2].imshow(convolution(image.squeeze())[0], cmap="gray")
axs[2].set_title("Vertical Sobel Filter")


class PlotConvolutionPooling:
    def __init__(self, backend: BackendMixin = bkd):
        self._bkd = backend
        # Create a 4x4 grid of pixel intensity values (0 to 255)
        self._pixel_values = bkd.asarray(
            np.random.uniform(0, 255 / 2, (4, 4)).astype(int)
            # np.random.permutation(
            #     np.linspace(0, 255, 16).reshape(4, 4).astype(int)
            # )
        )
        # Normalize pixel values to range [0, 1] for grayscale colors
        self._pixel_colors = self._pixel_values / 255.0
        self._fig, self._axs = plt.subplots(1, 2, figsize=(2 * 8, 6))

    def setup_axes(self):
        self._axs[0].set_xlim(0, 4)
        self._axs[0].set_ylim(0, 4)
        self._axs[0].set_aspect("equal")
        self._axs[0].axis("off")  # Turn off axes

        self._axs[1].set_xlim(0, 2)
        self._axs[1].set_ylim(0, 2)
        self._axs[1].set_aspect("equal")
        self._axs[1].axis("off")
        self.setup_rectangles()

    def plot_image(self):
        # Plot each box with its corresponding gray color and pixel intensity
        for ii in range(4):
            for jj in range(4):
                intensity = self._pixel_values[ii, jj]
                color = self._pixel_colors[ii, jj]
                # Draw the box
                self._axs[0].add_patch(
                    plt.Rectangle(
                        (jj, 3 - ii), 1, 1, color=str(bkd.to_numpy(color))
                    )
                )
                # Add the pixel intensity text
                self._axs[0].text(
                    jj + 0.5,
                    3 - ii + 0.5,
                    str(bkd.to_numpy(intensity)),
                    color="black" if color > 0.5 else "white",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

        for ii in range(2):
            for jj in range(2):
                # Draw the box
                self._axs[1].add_patch(
                    plt.Rectangle(
                        (jj, 1 - ii),
                        1,
                        1,
                        edgecolor="black",
                        fill=False,
                        linewidth=2,
                    )
                )
                self._axs[1].add_patch(
                    plt.Rectangle((jj, 1 - ii), 1, 1, color=str(0.5))
                )
                value = self.get_value(
                    self.get_subimage(self._pixel_values, ii, jj)
                )
                # Draw the box
                # Add the sum text
                self._axs[1].text(
                    jj + 0.5,
                    1 - ii + 0.5,
                    str(value),
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
        self._axs[0].add_patch(self._red_rectangle_0)
        self._axs[1].add_patch(self._red_rectangle_1)


class PlotConvolution(PlotConvolutionPooling):
    def __init__(self, filt, backend: BackendMixin = bkd):
        super().__init__(bkd)
        self._filter = filt

    def setup_rectangles(self):
        self._red_rectangle_0 = plt.Rectangle(
            (0, 1), 3, 3, fill=False, edgecolor="red", linewidth=2
        )
        self._red_rectangle_1 = plt.Rectangle(
            (0, 1), 1, 1, fill=False, edgecolor="red", linewidth=2
        )

    def get_subimage(self, image, ii, jj):
        return self._pixel_values[ii : ii + 3, jj : jj + 3]

    def get_value(self, subimage):
        return self._bkd.to_numpy(self._bkd.sum(self._filter * subimage))

    def update(self, it):
        self.setup_axes()
        self.plot_image()
        kk = it // 2
        ll = it % 2
        self._red_rectangle_0.set_xy([ll, 1 - kk])
        self._red_rectangle_1.set_xy([ll, 1 - kk])
        return self._red_rectangle_0, self._red_rectangle_1


class PlotMaxPooling(PlotConvolutionPooling):
    def setup_rectangles(self):
        self._red_rectangle_0 = plt.Rectangle(
            (0, 1), 2, 2, fill=False, edgecolor="red", linewidth=2
        )
        self._red_rectangle_1 = plt.Rectangle(
            (0, 1), 1, 1, fill=False, edgecolor="red", linewidth=2
        )

    def get_subimage(self, image, ii, jj):
        return self._pixel_values[2 * ii : 2 * (ii + 1), 2 * jj : 2 * (jj + 1)]

    def get_value(self, subimage):
        return self._bkd.to_numpy(self._bkd.max(subimage))

    def update(self, it):
        self.setup_axes()
        self.plot_image()
        kk = it // 2
        ll = it % 2
        self._red_rectangle_0.set_xy([2 * ll, 2 - 2 * kk])
        self._red_rectangle_1.set_xy([ll, 1 - kk])
        return self._red_rectangle_0, self._red_rectangle_1


conv_plot = PlotConvolution(horizontal_sobel_filter)

from matplotlib.animation import FuncAnimation

anim = FuncAnimation(
    conv_plot._fig, conv_plot.update, frames=range(4), interval=1000, blit=True
)
anim.save("conv.gif", dpi=100)

pool_plot = PlotMaxPooling(bkd)
anim = FuncAnimation(
    pool_plot._fig, pool_plot.update, frames=range(4), interval=1000, blit=True
)
anim.save("maxpooling.gif", dpi=100)

# Show the plot
plt.tight_layout()

# A pixel with a high value (bright white) in the filtered image corresponds to a strong in the original image.
# The horizontal Sobel filter detects horizontal edges vertical Sobel filter detects vertical ones.
