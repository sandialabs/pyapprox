r"""Diagnostics for weighted least-squares operator learning.

Pure functions. They exist so the stability properties the method
relies on can be asserted rather than assumed: that the weighted design
is well conditioned, that the sample count matches the theory, and that
the basis is orthonormal under the measure actually being sampled.
"""

from __future__ import annotations

import math
from typing import Optional

from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
    require_coefficient_error_is_field_error,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


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


def coefficient_error(
    encoder: FunctionEncoderProtocol[Array],
    predicted_fields: Array,
    reference_fields: Array,
    bkd: Backend[Array],
) -> float:
    r"""Return the relative error between two field sets, in coefficients.

    .. math::

        \frac{\left(\sum_i \|c^i - \tilde c^i\|_2^2\right)^{1/2}}
             {\left(\sum_i \|c^i\|_2^2\right)^{1/2}}

    on the coefficients ``encoder`` produces. Defined for any encoder,
    and it *approximates* the error in the field norm -- exactly when
    the encoder is an isometry, approximately otherwise, with a gap
    nothing here bounds. Use :func:`bochner_error` when the equality is
    the point; use this when a cheap comparison in the encoder's own
    coordinates is what is wanted, such as monitoring a fit over a
    nonlinear manifold.

    Parameters
    ----------
    encoder : FunctionEncoderProtocol[Array]
        The output encoder whose coefficients define the comparison.
    predicted_fields : Array
        Predicted output fields. Shape: (ngrid_out, nsamples)
    reference_fields : Array
        Reference output fields. Shape: (ngrid_out, nsamples)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The relative coefficient error.
    """
    if predicted_fields.shape != reference_fields.shape:
        raise ValueError(
            f"predicted shape {predicted_fields.shape} does not match "
            f"reference shape {reference_fields.shape}"
        )
    predicted = encoder.encode(predicted_fields)
    reference = encoder.encode(reference_fields)
    numerator = float(bkd.sqrt(bkd.sum((predicted - reference) ** 2)))
    denominator = float(bkd.sqrt(bkd.sum(reference**2)))
    if denominator == 0.0:
        raise ValueError("reference coefficients are all zero")
    return numerator / denominator


def bochner_error(
    encoder: FieldEncoderProtocol[Array],
    predicted_fields: Array,
    reference_fields: Array,
    bkd: Backend[Array],
) -> float:
    r"""Return the relative error in the Bochner norm of the fields.

    Computes :func:`coefficient_error` and asserts the condition under
    which that number *is* the Bochner error: the encoder must be an
    isometry, so that :math:`\|f\|_Y = \|\mathrm{encode}(f)\|_2`.

    **No opt-out.** The name states which quantity is returned, so
    returning the coefficient ratio for a non-isometric encoder would
    be reporting one thing under the name of another -- and silently,
    since the two agree closely in the easy cases and diverge exactly
    where it matters. A caller who wants the ratio anyway should ask
    :func:`coefficient_error` for it by name, which is honest about
    approximating rather than equalling.

    Parameters
    ----------
    encoder : FieldEncoderProtocol[Array]
        The output encoder, which must report itself an isometry.
    predicted_fields : Array
        Predicted output fields. Shape: (ngrid_out, nsamples)
    reference_fields : Array
        Reference output fields. Shape: (ngrid_out, nsamples)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The relative Bochner error.

    Raises
    ------
    ValueError
        If the encoder is not an isometry, if the shapes disagree, or
        if the reference is identically zero.
    """
    require_coefficient_error_is_field_error(encoder, "bochner_error")
    return coefficient_error(encoder, predicted_fields, reference_fields, bkd)


def field_error(
    metric: InnerProductProtocol[Array],
    predicted_fields: Array,
    reference_fields: Array,
    bkd: Backend[Array],
) -> float:
    r"""Return the relative error of two field sets, in the field norm.

    .. math::

        \frac{\left(\sum_i \|u^i - \tilde u^i\|_M^2\right)^{1/2}}
             {\left(\sum_i \|u^i\|_M^2\right)^{1/2}},
        \qquad \|v\|_M = \sqrt{v^T M v}

    measured on the fields as given, with no encode step. Three
    consequences separate this from :func:`coefficient_error` and
    :func:`bochner_error`, and together they are why it exists.

    It is **valid for any encoder, or none**, because it never round
    trips through one. Over a nonlinear manifold the decoder is not an
    isometry and a coefficient residual bounds nothing, so this is the
    only measure there that means what its name says. The coefficient
    number stays useful as a cheap proxy, and the ratio of the two is the
    empirical size of the gap between them.

    It measures the **full field, including any mean** a decoder adds
    back. A centering encoder subtracts the mean before projecting, so a
    coefficient residual describes only the fluctuation; where the mean
    carries most of the energy the two differ by a large factor, and the
    full-field number is the one the operator-learning literature
    reports.

    It weights by :math:`M` rather than counting nodes equally, which on
    a graded grid is not a refinement but the difference between a
    physical quantity and a grid artifact: for a unit-length domain
    :math:`\|1\|_M = 1`, while the Euclidean norm of the same field is
    :math:`\sqrt{n_{\mathrm{nodes}}}`. An error concentrated where nodes
    happen to be dense is overstated accordingly -- measured at 0.036
    against 0.013 on a quadratically graded grid.

    Samples aggregate in quadrature rather than as a mean of per-sample
    ratios, which would be dominated by whichever sample has the
    smallest norm.

    Parameters
    ----------
    metric : InnerProductProtocol[Array]
        The field-space inner product, typically
        ``domain.inner_product()``. Sparse FEM mass matrices are
        supported and keep the autograd graph.
    predicted_fields : Array
        Predicted output fields. Shape: (ngrid_out, nsamples)
    reference_fields : Array
        Reference output fields. Shape: (ngrid_out, nsamples)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    float
        The relative field-space error.

    Raises
    ------
    ValueError
        If the shapes disagree, if they do not match the metric, or if
        the reference is identically zero.
    """
    if predicted_fields.shape != reference_fields.shape:
        raise ValueError(
            f"predicted shape {predicted_fields.shape} does not match "
            f"reference shape {reference_fields.shape}"
        )
    if int(reference_fields.shape[0]) != metric.nstates():
        raise ValueError(
            f"fields have {int(reference_fields.shape[0])} rows but the "
            f"metric acts on {metric.nstates()} states"
        )
    residual = metric.norm(predicted_fields - reference_fields)
    reference = metric.norm(reference_fields)
    denominator = float(bkd.sqrt(bkd.sum(reference**2)))
    if denominator == 0.0:
        raise ValueError("reference fields are all zero")
    return float(bkd.sqrt(bkd.sum(residual**2))) / denominator
