"""Normal distribution utility functions for acquisition functions.

Provides standard normal CDF and PDF using the backend erf function,
avoiding scipy dependency in hot paths.
"""

import math

from pyapprox.util.backends.protocols import Array, Backend

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def normal_cdf(x: Array, bkd: Backend[Array]) -> Array:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Parameters
    ----------
    x : Array
        Input values.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        CDF values, same shape as x.
    """
    return 0.5 * (1.0 + bkd.erf(x / _SQRT2))


def normal_pdf(x: Array, bkd: Backend[Array]) -> Array:
    """Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi).

    Parameters
    ----------
    x : Array
        Input values.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        PDF values, same shape as x.
    """
    return bkd.exp(-0.5 * x**2) / _SQRT2PI
