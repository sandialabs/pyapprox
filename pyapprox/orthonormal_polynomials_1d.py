from warnings import warn
msg = f"The module pyappprox.{__name__} has now moved to "
msg += f"pyappprox.univariate_polynomials_chaos.orthonormal_polynomials"
warn(msg, DeprecationWarning)
from pyapprox.univariate_polynomials.orthonormal_polynomials import *
