r"""How many modes of a spectrum to keep.

Truncation is the one decision every empirical basis must make and the
one most easily made inconsistently. The same four policies were written
three times across this library -- once in the kernel-driven eigensolvers,
once in the PCA encoder, once in an external ROM builder -- and they
disagreed in ways no caller could see from the outside: on whether a
fraction is of energy or of variance, on whether "not enough terms" is
an error or a silent shortfall, on where the floor between a small mode
and a numerically-zero one sits.

Each policy here is a function from a spectrum to a count. They are
deliberately not methods on a KLE: the same choice applies to a kernel
spectrum, a snapshot spectrum, and a stored basis being reloaded, none
of which share a class.

**Spectra are eigenvalues, not singular values.** The two differ by a
square, so a fraction computed on the wrong one silently keeps too few
modes -- ``by_variance_fraction`` on singular values would ask for a
fraction of :math:`\sum s_i` where the energy is :math:`\sum s_i^2`.
Every function here takes eigenvalues, descending; convert with
``s**2`` at the call site, where it is visible.
"""

from typing import Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

# The scale below which an eigenvalue is rounding rather than variance.
# Shared with the rejection in ``eigensolvers`` so a basis truncated by
# ``by_numerical_rank`` is exactly one that passes that check.
_MACHINE_EPS = float(np.finfo(np.float64).eps)


def by_count(eig_vals: Array, nterms: int, bkd: Backend[Array]) -> int:
    """Keep exactly ``nterms`` modes.

    Trivial except for its validation, which is the point: an explicit
    count is the policy most likely to exceed what the spectrum holds,
    since the caller chose it without looking.

    Parameters
    ----------
    eig_vals : Array
        Shape ``(k,)``, descending, non-negative.
    nterms : int
        Number of modes wanted.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    int
        ``nterms``.

    Raises
    ------
    ValueError
        If ``nterms`` is not positive or exceeds ``len(eig_vals)``.
    """
    navailable = int(eig_vals.shape[0])
    if nterms < 1:
        raise ValueError(f"nterms={nterms} must be positive")
    if nterms > navailable:
        raise ValueError(
            f"nterms={nterms} exceeds the {navailable} modes the "
            "spectrum contains"
        )
    return nterms


def by_variance_fraction(
    eig_vals: Array, fraction: float, bkd: Backend[Array]
) -> int:
    r"""Keep the fewest modes carrying ``fraction`` of the total.

    The cumulative sum of eigenvalues is the captured variance, so this
    returns the smallest :math:`k` with

    .. math:: \sum_{i \le k} \lambda_i \ge f \sum_i \lambda_i.

    Parameters
    ----------
    eig_vals : Array
        Shape ``(k,)``, descending, non-negative.
    fraction : float
        In ``(0, 1]``. A fraction of 1.0 keeps every mode carrying
        variance.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    int
        The smallest count reaching ``fraction``, at least 1.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``(0, 1]``, or the spectrum carries
        no variance at all.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"fraction={fraction} must lie in (0, 1]"
        )
    total = bkd.to_float(bkd.sum(eig_vals))
    if total <= 0.0:
        raise ValueError(
            "the spectrum carries no variance, so no fraction of it can "
            "be reached"
        )
    ratios = bkd.to_numpy(bkd.cumsum(eig_vals)) / total
    # searchsorted finds the first index reaching the fraction in one
    # pass. The max() guards a fraction so small the first mode already
    # exceeds it, where the answer is 1 rather than 0 -- a basis of no
    # modes is not a basis.
    return max(1, int(np.searchsorted(ratios, fraction) + 1))


def by_numerical_rank(eig_vals: Array, bkd: Backend[Array]) -> int:
    """Keep every mode carrying variance rather than rounding error.

    A covariance operator is positive semi-definite, so in exact
    arithmetic its zero eigenvalues are zero. In floating point they are
    small positives and negatives indistinguishable from genuinely tiny
    modes by sign alone, so the cut is made at a magnitude scaled by the
    largest eigenvalue and the problem size.

    This is the policy to use when the caller does not know the rank --
    smooth kernels are severely rank deficient, and asking for more
    terms than the operator supplies yields columns of zeros.

    Parameters
    ----------
    eig_vals : Array
        Shape ``(k,)``, descending.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    int
        Count of eigenvalues above the threshold; 0 if the spectrum is
        empty of variance.
    """
    largest = bkd.to_float(bkd.max(eig_vals))
    if largest <= 0.0:
        return 0
    nvals = int(eig_vals.shape[0])
    tolerance = largest * nvals * _MACHINE_EPS
    return int((bkd.to_numpy(eig_vals) > tolerance).sum())


def by_eigenvalue_floor(
    eig_vals: Array, relative_floor: float, bkd: Backend[Array]
) -> int:
    r"""Keep modes whose eigenvalue exceeds ``relative_floor * lambda_max``.

    Differs from :func:`by_numerical_rank` in who chooses the threshold.
    That one uses machine precision, which is the right cut for "is this
    a mode at all". This takes the floor from the caller, for algorithms
    whose stability fails well before machine precision does -- the
    method of snapshots scales by :math:`\lambda^{-1/2}`, which amplifies
    a mode near the floor rather than merely keeping it.

    Parameters
    ----------
    eig_vals : Array
        Shape ``(k,)``, descending, non-negative.
    relative_floor : float
        Non-negative, relative to the largest eigenvalue.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    int
        Count above the floor.

    Raises
    ------
    ValueError
        If ``relative_floor`` is negative, or no mode clears it.
    """
    if relative_floor < 0.0:
        raise ValueError(
            f"relative_floor={relative_floor} must be non-negative"
        )
    largest = bkd.to_float(bkd.max(eig_vals))
    if largest <= 0.0:
        raise ValueError(
            "the spectrum carries no variance, so no mode clears any floor"
        )
    nkept = int(
        (bkd.to_numpy(eig_vals) > relative_floor * largest).sum()
    )
    if nkept == 0:
        raise ValueError(
            f"no eigenvalue exceeds relative_floor={relative_floor:.1e} "
            f"times the largest ({largest:.3e}); every mode would be "
            "discarded"
        )
    return nkept


def resolve_nterms(
    eig_vals: Array,
    bkd: Backend[Array],
    nterms: Optional[int] = None,
    variance_fraction: Optional[float] = None,
) -> int:
    """Apply whichever of the two caller-facing policies was given.

    The pair ``nterms`` / ``variance_fraction`` appears in several
    constructor signatures, always with the same rule: exactly one.
    Centralized so the error message is the same wherever it is hit, and
    so a caller cannot find that one entry point tolerates both while
    another rejects them.

    Parameters
    ----------
    eig_vals : Array
        Shape ``(k,)``, descending, non-negative.
    bkd : Backend[Array]
        Computational backend.
    nterms : int, optional
        An explicit count.
    variance_fraction : float, optional
        A fraction of the total variance.

    Returns
    -------
    int
        The resolved count.

    Raises
    ------
    ValueError
        If both or neither are given, or the chosen policy rejects.
    """
    if (nterms is None) == (variance_fraction is None):
        raise ValueError(
            "pass exactly one of nterms or variance_fraction; they are "
            "two ways of asking the same question and giving both leaves "
            "it undefined which one truncates"
        )
    if variance_fraction is not None:
        return by_variance_fraction(eig_vals, variance_fraction, bkd)
    # nterms is not None here: the check above rejected the case where
    # both are None, and variance_fraction was just excluded.
    return by_count(eig_vals, nterms if nterms is not None else 0, bkd)
