"""
PyApprox : Sotware for model and data analysis

Find way to avoid importing anything from matplotlib, scipy and other big
pacakges. It increases overhead before function is run 
"""


name = "pyapprox"
# fast loads
from pyapprox.utilities import *
from pyapprox.random_variable_algebra import *
from pyapprox.numerically_generate_orthonormal_polynomials_1d import *
from pyapprox.models.wrappers import *
from pyapprox.indexing import *
from pyapprox.monomial import *
from pyapprox.quantile_regression import *
from pyapprox.low_discrepancy_sequences import *
# slower loads
from pyapprox.control_variate_monte_carlo import *
from pyapprox.adaptive_polynomial_chaos import *
from pyapprox.variable_transformations import *
from pyapprox.variables import *
from pyapprox.probability_measure_sampling import *
from pyapprox.adaptive_sparse_grid import * 
from pyapprox.univariate_quadrature import * 
from pyapprox.visualization import *
from pyapprox.optimization import *
from pyapprox.density import *
from pyapprox.arbitrary_polynomial_chaos import *
from pyapprox.multivariate_polynomials import *
from pyapprox.sensitivity_analysis import *
from pyapprox.gaussian_network import *
from pyapprox.gaussian_process import *

