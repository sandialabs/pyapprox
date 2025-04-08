"""
Multivariate Piecewise Polynomial Interpolation
-----------------------------------------------
Piecewise polynomial interpolation can efficiently approximation functions
with low smoothness, e.g. piecewise continuous functions. Here we will use
piecewise quadratic polynomials to interpolate the discontinuous Genz
benchmark. The function interpolates functions on tensor-products of
1D equidistant grids. The number of points in the grid doubles with each level.
Here levels specifies the level of each 1D grid

"""

import numpy as np
import matplotlib.pyplot as plt
from pyapprox.benchmarks import GenzBenchmark
from pyapprox.surrogates.univariate.local import (
    setup_univariate_piecewise_polynomial_basis,
)
from pyapprox.surrogates.affine.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.affine.basisexp import TensorProductInterpolant

nvars = 2
benchmark = GenzBenchmark(name="discontinuous", nvars=nvars)

nnodes_1d = np.array([5] * nvars)
bounds = [0, 1]
basis_types = ["quadratic"] * nvars
bases_1d = [
    setup_univariate_piecewise_polynomial_basis(bt, bounds)
    for bt in basis_types
]
basis = TensorProductInterpolatingBasis(bases_1d)
interp = TensorProductInterpolant(basis)
basis.set_tensor_product_indices(nnodes_1d)

train_samples = basis.tensor_product_grid()
train_values = benchmark.model()(train_samples)
interp.fit(train_values)

fig, axs = interp.get_plot_axis(surface=True)
_ = interp.plot_surface(axs, [0, 1, 0, 1])
