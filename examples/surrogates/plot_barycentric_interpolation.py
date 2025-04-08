"""
Multivariate Lagrange Interpolation
-----------------------------------
For smooth function Lagrange polynomials can be used to
interpolate univariate functions evaluated on a tensor-product grid.
Pyapprox uses the Barycentric formulation of Lagrange interpolation
which is more efficient and stable than traditional Lagrange interpolation.

We must define the univariate grids that we will use to construct the
tensor product grid. While technically Lagrange interpolation can be used with
any 1D grids, it is better to use points well suited to polynomial
interpolation. Here we use the samples of a Gaussian quadrature rule.
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates.univariate.orthopoly import GaussQuadratureRule
from pyapprox.surrogates.univariate.lagrange import (
    UnivariateLagrangeBasis,
)
from pyapprox.surrogates.affine.basis import TensorProductInterpolatingBasis
from pyapprox.surrogates.affine.basisexp import TensorProductInterpolant
from pyapprox.interface.model import ModelFromSingleSampleCallable

degree = 10
marginals = [stats.uniform(-1, 2), stats.uniform(-1, 2)]
quad_rules_1d = [GaussQuadratureRule(marginal) for marginal in marginals]
bases_1d = [
    UnivariateLagrangeBasis(quad_rule, 1) for quad_rule in quad_rules_1d
]


# %%
# Now lets define the function we want to interpolate, e.g.
#:math:`f(\rv)=\rv_1^2+\rv_2^2`
def fun(samples):
    return np.sum(samples**2, axis=0)[:, None]


model = ModelFromSingleSampleCallable(1, 2, fun)

##%
# Now create the interpolant
nnodes_1d = np.array([5] * 2)
basis = TensorProductInterpolatingBasis(bases_1d)
interp = TensorProductInterpolant(basis)
basis.set_tensor_product_indices(nnodes_1d)

train_samples = basis.tensor_product_grid()
train_values = model(train_samples)
interp.fit(train_values)

# %% Now plot the interpolant
fig, axs = interp.get_plot_axis(surface=True)
_ = interp.plot_surface(axs, [-1, 1, -1, 1])

# %%
# Multivariate Lagrange interpolation can be used for any number of variables. However,
# the number of evaluations of the target function grows exponentially with
# the number of variables
