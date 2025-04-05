import math
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

# FIXME: DOES STROUD ASSUME A WEIGHT OF w(x)=1/2**ndim


def CdD2(
    ndim: int, uniform_weight: bool = True, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Tuple[Array, Array]:
    """Arbitrary Dimensions, Degree 2, d+1 Points (Stroud)"""
    if not uniform_weight:
        V = 2.0**ndim
    else:
        V = 1.0
    x = bkd.empty((ndim + 1, ndim))
    w = bkd.empty(ndim + 1)
    for i in range(ndim + 1):
        for k in range(1, int(ndim) // 2 + 1):
            x[i, 2 * k - 2] = bkd.sqrt(2.0 / 3.0) * bkd.cos(
                2.0 * k * i * math.pi / (ndim + 1)
            )
            x[i, 2 * k - 1] = bkd.sqrt(2.0 / 3.0) * bkd.sin(
                2.0 * k * i * math.pi / (ndim + 1)
            )
        if ndim % 2 == 1:
            x[i, ndim - 1] = (-1.0) ** i / bkd.sqrt(3.0)
        w[i] = V / (ndim + 1.0)
    return x, w


def CdD3(
    ndim: int, uniform_weight: bool = True, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Tuple[Array, Array]:
    """Arbitrary Dimensions, Degree 3, 2d Points (Stroud)"""
    if not uniform_weight:
        V = 2.0**ndim
    else:
        V = 1.0
        x = bkd.empty((2 * ndim, ndim))
        w = bkd.empty(2 * ndim)
    for i in range(1, 2 * ndim + 1):
        for k in range(1, int(ndim) // 2 + 1):
            x[i - 1, 2 * k - 2] = bkd.sqrt(2.0 / 3.0) * bkd.cos(
                (2.0 * k - 1.0) * i * math.pi / (ndim)
            )
            x[i - 1, 2 * k - 1] = bkd.sqrt(2.0 / 3.0) * bkd.sin(
                (2.0 * k - 1.0) * i * math.pi / (ndim)
            )
        if ndim % 2 == 1:
            x[i - 1, ndim - 1] = (-1.0) ** i / bkd.sqrt(3.0)
        w[i - 1] = V / (2.0 * ndim)
    return x, w


def CdD5(
    ndim: int, uniform_weight: bool = True, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Tuple[Array, Array]:
    """
    Arbitrary Dimensions, Degree 5, 2^d+1 Points (Hammer and Stroud)
    """
    numPts = 2 * ndim**2 + 1
    r = bkd.sqrt(3.0 / 5.0)
    w0 = (25.0 * ndim**2 - 115.0 * ndim + 162.0) / 162.0
    w1 = (70 - 25 * ndim) / 162.0
    w2 = 25.0 / 324.0
    x = bkd.zeros((numPts, ndim))
    w = bkd.empty(numPts)
    i = 0
    x[i] = bkd.zeros(ndim)
    w[i] = w0
    i += 1
    for d in range(ndim):
        x[i, d] = r
        x[i + 1, d] = -r
        w[i] = w1
        w[i + 1] = w1
        i += 2

    for d1 in range(ndim - 1):
        for d2 in range(d1 + 1, ndim):
            if d1 != d2:
                x[i, d1] = r
                x[i, d2] = r

                x[i + 1, d1] = r
                x[i + 1, d2] = -r

                x[i + 2, d1] = -r
                x[i + 2, d2] = r

                x[i + 3, d1] = -r
                x[i + 3, d2] = -r

                w[i : i + 4] = w2
                i += 4

    if not uniform_weight:
        w *= 2**ndim
    return x, w


def get_cubature_rule(
    nvars: int,
    degree: int,
    uniform_weight: bool = True,
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Tuple[Array, Array]:
    cases = {2: CdD2, 3: CdD3, 5: CdD5}
    if degree not in cases:
        raise ValueError(f"(nvars, degree)={(nvars, degree)} not supported")

    x, w = cases[degree](nvars, uniform_weight, bkd)
    return x.T, w[:, None]
