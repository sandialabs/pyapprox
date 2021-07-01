from warnings import warn
msg = f"The module pyappprox.{__name__} has now moved to "
msg += f"pyappprox.polynomial_chaos.{__name__}"
warn(msg, DeprecationWarning)
from pyapprox.univariate_polynomials.quadrature import *
