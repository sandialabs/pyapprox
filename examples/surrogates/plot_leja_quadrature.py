r"""
Univariate Leja Quadrature
--------------------------
Univariate Leja quadrature can be used to efficiently integrate
smooth one-dimensional functions with nested sequences. Leja quadrature rules are not as accurate as Gaussian Quadrature rules, a n-point Gauss rule will integrate all 2n-1 dgree polynomials and a n-point Leja quadrature rule will integrate all n-1 degree polynomials. However, unlike Gauss rules, Leja quadrature can be incremented one sample at a time, which is useful for constructing adaptive rules.

To generate a univariate quadrature rule for uniform random variables
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from pyapprox.surrogates.univariate.leja import (
    TwoPointChristoffelLejaQuadratureRule,
)
from pyapprox.analysis import visualize
from pyapprox.variables import marginals

nsamples = 21
marginal = marginals.ContinuousScipyMarginal(stats.beta(11, 11))
quad_rule = TwoPointChristoffelLejaQuadratureRule(marginal)
x_quad, w_quad = quad_rule(nsamples)

# %%
# For interest, we plot the quadrature rule against the PDF of the exponential
# variable
ax = plt.subplots(1, 1)[1]
visualize.plot_discrete_measure_1d(x_quad, w_quad, ax)
vrange = marginal.truncated_range(1 - 1e-6)
xx = np.linspace(vrange[0], vrange[1], 101)
_ = plt.fill_between(xx, 0 * xx, marginal.pdf(xx), alpha=0.3)

# %%
# As an example, we can use this quadrature rule to integrate :math:`\rv^2`
# with repsect to the uniform PDF on [-1, 1], i.e. 1/2
values = x_quad**2
integral = values @ w_quad
print(integral)

# %%
# The quadrature recovers the integral value of 1/3 to machine precision. Note
# unlike :class:`~numpy.polynomial.legendre.leggauss` we are integrating with
# resepect to the uniform PDF.

# %%
# The function is also capable of generating rules on different intervals.
# Just change the marginal, for example,
marginal = marginals.ContinuousScipyMarginal(stats.uniform(0, 2))
x_quad, w_quad = TwoPointChristoffelLejaQuadratureRule(marginal)(nsamples)
values = x_quad**2
integral = values @ w_quad
print(integral)
