"""
PyApprox : Sotware for model and data analysis

Find way to avoid importing anything from matplotlib, scipy and other big
pacakges. It increases overhead before function is run 
"""


name = "pyapprox"
# fast loads
from pyapprox.univariate_polynomials.orthonormal_polynomials import *
from pyapprox.univariate_polynomials.quadrature import *
from pyapprox.univariate_polynomials.leja_quadrature import *
from pyapprox.univariate_polynomials.orthonormal_recursions import *
from pyapprox.univariate_polynomials.numeric_orthonormal_recursions import *
from pyapprox.polynomial_chaos.multivariate_polynomials import *

from pyapprox.utilities import *
from pyapprox.random_variable_algebra import *
from pyapprox.models.wrappers import *
from pyapprox.indexing import *
from pyapprox.monomial import *
from pyapprox.quantile_regression import *
from pyapprox.low_discrepancy_sequences import *
from pyapprox.parameter_sweeps import *
# slower loads
from pyapprox.control_variate_monte_carlo import *
from pyapprox.adaptive_polynomial_chaos import *
from pyapprox.variable_transformations import *
from pyapprox.variables import *
from pyapprox.probability_measure_sampling import *
from pyapprox.adaptive_sparse_grid import *
from pyapprox.visualization import *
from pyapprox.optimization import *
from pyapprox.density import *
from pyapprox.arbitrary_polynomial_chaos import *
from pyapprox.multivariate_polynomials import *
from pyapprox.sensitivity_analysis import *
from pyapprox.gaussian_network import *
from pyapprox.gaussian_process import *
from pyapprox.approximate import *
from pyapprox.barycentric_interpolation import tensor_product_barycentric_lagrange_interpolation

import sys
from pyapprox.sys_utilities import package_available


PYA_DEV_AVAILABLE = package_available('pyapprox_dev.bayesian_inference') and sys.platform != 'win32'
