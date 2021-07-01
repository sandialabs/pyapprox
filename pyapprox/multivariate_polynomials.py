from warnings import warn
msg = "The module pyapprox.multivariate_polynomials has now moved to "
msg += "pyapprox.polynomial_chaos.multivariate_polynomials"
warn(msg, DeprecationWarning)
from pyapprox.polynomial_chaos.multivariate_polynomials import *
