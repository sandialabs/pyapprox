"""Univariate basis values for testing the tensor-product basis kernels."""

from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd


def make_basis_kernel_data(
    nvars: int, max_level: int, nsamples: int
) -> Tuple[
    List[NDArray[Any]],
    List[NDArray[Any]],
    List[NDArray[Any]],
    NDArray[Any],
    NumpyBkd,
]:
    """Univariate Legendre values, first and second derivatives at random
    samples in ``[-0.9, 0.9]``, and hyperbolic indices, on NumPy.

    Returns ``(vals_1d, derivs_1d, hess_1d, indices, bkd)``.
    """
    bkd = NumpyBkd()
    np.random.seed(42)

    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)

    # Ensure bases have enough terms
    for dd in range(nvars):
        max_deg = int(np.max(bkd.to_numpy(indices[dd, :])))
        if bases_1d[dd].nterms() < max_deg + 1:
            bases_1d[dd].set_nterms(max_deg + 1)

    samples = bkd.asarray(np.random.uniform(-0.9, 0.9, (nvars, nsamples)))

    # Evaluate univariate bases
    vals_1d = [bases_1d[dd](samples[dd : dd + 1, :]) for dd in range(nvars)]
    derivs_1d = [
        bases_1d[dd].jacobian_batch(samples[dd : dd + 1, :]) for dd in range(nvars)
    ]
    hess_1d = [
        bases_1d[dd].hessian_batch(samples[dd : dd + 1, :]) for dd in range(nvars)
    ]

    return vals_1d, derivs_1d, hess_1d, indices, bkd
