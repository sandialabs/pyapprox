r"""Diagnostics for weighted least-squares operator learning.

Pure functions. They exist so the stability properties the method
relies on can be asserted rather than assumed: that the weighted design
is well conditioned, that the sample count matches the theory, and that
the basis is orthonormal under the measure actually being sampled.
"""

from __future__ import annotations

import math
from typing import Optional

from pyapprox.util.backends.protocols import Array, Backend


def weighted_gram(
    design_matrix: Array,
    weights: Optional[Array],
    bkd: Backend[Array],
) -> Array:
    r"""Return the empirical weighted Gram matrix.

    .. math::

        G = \frac{1}{M} \sum_i w_i\, \Phi(\hat f^i) \Phi(\hat f^i)^T

    Under the sampling measure this is an unbiased estimate of the
    identity, which is the property the whole stability argument rests
    on. Passing the weights and their reciprocal produce different
    matrices, only one of which converges to :math:`I`.

    Parameters
    ----------
    design_matrix : Array
        Basis values. Shape: (nsamples, nterms)
    weights : Array or None
        Sample weights :math:`w_i`. Shape: (nsamples,). None means
        unit weights, appropriate only for Monte Carlo sampling from
        the reference measure itself.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        The Gram matrix. Shape: (nterms, nterms)
    """
    nsamples = design_matrix.shape[0]
    if weights is None:
        scaled = design_matrix
    else:
        if weights.shape != (nsamples,):
            raise ValueError(
                f"weights has wrong shape {weights.shape}, "
                f"expected ({nsamples},)"
            )
        scaled = weights[:, None] * design_matrix
    return bkd.dot(design_matrix.T, scaled) / nsamples


def gram_condition_number(
    design_matrix: Array,
    weights: Optional[Array],
    bkd: Backend[Array],
) -> float:
    """Return the condition number of the weighted Gram matrix.

    The quantity swept in stability studies: induced sampling keeps it
    bounded as the basis grows, while Monte Carlo sampling does not.

    Parameters
    ----------
    design_matrix : Array
        Basis values. Shape: (nsamples, nterms)
    weights : Array or None
        Sample weights. Shape: (nsamples,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The 2-norm condition number.
    """
    return float(bkd.cond(weighted_gram(design_matrix, weights, bkd)))


def sample_complexity(nterms: int, delta: float, epsilon: float) -> int:
    r"""Return the samples needed for a stable weighted least squares fit.

    The near-optimal sampling bound

    .. math::

        M \ge \frac{N}{\delta^2}
              \left(\log\frac{2N}{\epsilon}\right)

    with :math:`\delta` the tolerated deviation of the Gram matrix from
    the identity and :math:`\epsilon` the failure probability. For an
    operator basis pass :math:`N_{\mathrm{eff}}`, not the full basis
    size: :math:`d_{\mathrm{out}}` cancels from the sampling weight and
    so does not enter the sample count.

    This is a sufficient condition proved through a matrix Chernoff
    bound, and it is loose: the constant is not sharp, so fewer samples
    may suffice for a given basis and measure. It is not an estimate of
    what is needed, and nothing here establishes that a smaller count
    is safe — measure :func:`gram_condition_number` on the sample
    actually drawn to find out.

    Parameters
    ----------
    nterms : int
        Basis size :math:`N`.
    delta : float
        Gram deviation tolerance, in (0, 1).
    epsilon : float
        Failure probability, in (0, 1).

    Returns
    -------
    int
        The required number of samples.
    """
    if nterms < 1:
        raise ValueError(f"nterms must be positive, got {nterms}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    return int(
        math.ceil(nterms / delta**2 * math.log(2.0 * nterms / epsilon))
    )


def christoffel_integral(
    basis_values: Array,
    quadrature_weights: Array,
    bkd: Backend[Array],
) -> float:
    r"""Return :math:`\int k_\Lambda \,d\rho`, which must equal one.

    The normalized Christoffel function is a probability density with
    respect to the reference measure, so its integral is one for any
    orthonormal basis and any index set. A value away from one means
    the basis is not orthonormal under the measure being integrated
    against, which invalidates the induced sampling weight.

    Parameters
    ----------
    basis_values : Array
        Basis at the quadrature points. Shape: (nquad, nterms)
    quadrature_weights : Array
        Quadrature weights summing to one. Shape: (nquad,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The integral of the normalized Christoffel function.
    """
    nterms = basis_values.shape[1]
    christoffel = bkd.sum(basis_values**2, axis=1) / nterms
    return float(bkd.sum(quadrature_weights * christoffel))


def bochner_error(
    predicted: Array,
    reference: Array,
    bkd: Backend[Array],
) -> float:
    r"""Return the relative Bochner error between coefficient sets.

    .. math::

        \frac{\left(\sum_i \|c^i - \tilde c^i\|_2^2\right)^{1/2}}
             {\left(\sum_i \|c^i\|_2^2\right)^{1/2}}

    Equals the relative error in the Bochner norm of the fields only
    when the output encoder is an isometry, which is the reason
    encoders declare that property.

    Parameters
    ----------
    predicted : Array
        Predicted output coefficients. Shape: (noutputs, nsamples)
    reference : Array
        Reference output coefficients. Shape: (noutputs, nsamples)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The relative error.
    """
    if predicted.shape != reference.shape:
        raise ValueError(
            f"predicted shape {predicted.shape} does not match "
            f"reference shape {reference.shape}"
        )
    numerator = float(bkd.sqrt(bkd.sum((predicted - reference) ** 2)))
    denominator = float(bkd.sqrt(bkd.sum(reference**2)))
    if denominator == 0.0:
        raise ValueError("reference coefficients are all zero")
    return numerator / denominator
