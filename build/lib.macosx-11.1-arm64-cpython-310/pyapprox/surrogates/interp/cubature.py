import numpy as np
# FIXME: DOES STROUD ASSUME A WEIGHT OF w(x)=1/2**ndim


def CdD2(ndim, uniform_weight=True):
    """ Arbitrary Dimensions, Degree 2, d+1 Points (Stroud)
    """
    if not uniform_weight:
        V = 2.**ndim
    else:
        V = 1.
    x = np.empty((ndim+1, ndim), np.double)
    w = np.empty(ndim+1, np.double)
    for i in range(ndim+1):
        for k in range(1, int(ndim)//2+1):
            x[i, 2*k-2] = np.sqrt(2./3.)*np.cos(2.*k*i*np.pi/(ndim+1))
            x[i, 2*k-1] = np.sqrt(2./3.)*np.sin(2.*k*i*np.pi/(ndim+1))
        if ndim % 2 == 1:
            x[i, ndim-1] = (-1.)**i/np.sqrt(3.)
        w[i] = V/(ndim+1.)
    return x, w


def CdD3(ndim, uniform_weight=True):
    """ Arbitrary Dimensions, Degree 3, 2d Points (Stroud)
    """
    if not uniform_weight:
        V = 2.**ndim
    else:
        V = 1.
        x = np.empty((2*ndim, ndim), np.double)
        w = np.empty(2*ndim, np.double)
    for i in range(1, 2*ndim+1):
        for k in range(1, int(ndim)//2+1):
            x[i-1, 2*k-2] = np.sqrt(2./3.)*np.cos((2.*k-1.)*i*np.pi/(ndim))
            x[i-1, 2*k-1] = np.sqrt(2./3.)*np.sin((2.*k-1.)*i*np.pi/(ndim))
        if ndim % 2 == 1:
            x[i-1, ndim-1] = (-1.)**i/np.sqrt(3.)
        w[i-1] = V/(2.*ndim)
    return x, w


def CdD5(ndim, uniform_weight=True):
    """
    Arbitrary Dimensions, Degree 5, 2^d+1 Points (Hammer and Stroud)
    """
    numPts = 2*ndim**2+1
    r = np.sqrt(3./5.)
    w0 = (25.*ndim**2-115.*ndim+162.)/162.
    w1 = (70-25*ndim)/162.
    w2 = 25./324.
    x = np.zeros((numPts, ndim), np.double)
    w = np.empty(numPts, np.double)
    i = 0
    x[i] = np.zeros(ndim, np.double)
    w[i] = w0
    i += 1
    for d in range(ndim):
        x[i, d] = r
        x[i+1, d] = -r
        w[i] = w1
        w[i+1] = w1
        i += 2

    for d1 in range(ndim-1):
        for d2 in range(d1+1, ndim):
            if (d1 != d2):
                x[i, d1] = r
                x[i, d2] = r

                x[i+1, d1] = r
                x[i+1, d2] = -r

                x[i+2, d1] = -r
                x[i+2, d2] = r

                x[i+3, d1] = -r
                x[i+3, d2] = -r

                w[i:i+4] = w2
                i += 4

    if not uniform_weight:
        w *= 2**ndim
    return x, w


def get_cubature_rule(nvars, degree, uniform_weight=True):
    cases = {2: CdD2, 3: CdD3, 5: CdD5}
    if degree not in cases:
        raise ValueError(f"(nvars, degree)={(nvars, degree)} not supported")

    x, w = cases[degree](nvars, uniform_weight)
    return x.T, w[:, None]
