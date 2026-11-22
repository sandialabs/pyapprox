"""Tests for Lagrange dispatch: path consistency and picklability.

Verifies that the backend-selected implementations (Numba on NumPy,
torch.compile on Torch) match the backend-generic barycentric formula,
and that dispatched implementations survive pickle round-trips so that
objects storing them (e.g. sparse grid surrogates) can be used with
multiprocess tools.
"""

import pickle
from typing import Tuple

import numpy as np

from pyapprox.polynomials.univariate import (
    LagrangeBasis1D,
    LegendrePolynomial1D,
)
from pyapprox.polynomials.univariate.lagrange import compute_barycentric_weights
from pyapprox.polynomials.univariate.lagrange_dispatch import (
    _generic_lagrange_eval,
    _generic_lagrange_hessian,
    _generic_lagrange_jacobian,
    get_lagrange_eval_impl,
    get_lagrange_hessian_impl,
    get_lagrange_jacobian_impl,
)
from pyapprox.util.backends.protocols import Array, Backend


def _make_basis(bkd: Backend[Array], nterms: int = 7) -> LagrangeBasis1D[Array]:
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(nterms)
    basis = LagrangeBasis1D(bkd, poly.gauss_quadrature_rule)
    basis.set_nterms(nterms)
    return basis


def _make_dispatch_args(
    bkd: Backend[Array], nterms: int = 7, nsamples: int = 11
) -> Tuple[Array, Array, Array]:
    """Return (abscissa, samples, bary_weights) with samples hitting nodes."""
    basis = _make_basis(bkd, nterms)
    abscissa = bkd.flatten(basis.quadrature_rule()[0])
    bary_weights = compute_barycentric_weights(abscissa, bkd)
    # Include exact node hits to exercise the near-node code paths
    samples = bkd.hstack(
        [bkd.asarray(np.random.uniform(-1, 1, nsamples)), abscissa[::2]]
    )
    return abscissa, samples, bary_weights


class TestLagrangeDispatchConsistency:
    """Dispatched impls must match the backend-generic barycentric formula.

    On NumPy this compares the Numba kernel against the generic path; on
    Torch it compares the torch.compile path against the generic path.
    """

    def test_eval_matches_generic(self, bkd: Backend[Array]) -> None:
        abscissa, samples, bary_weights = _make_dispatch_args(bkd)
        impl = get_lagrange_eval_impl(bkd)
        result = impl(abscissa, samples, bary_weights, bkd)
        expected = _generic_lagrange_eval(abscissa, samples, bary_weights, bkd)
        bkd.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_jacobian_matches_generic(self, bkd: Backend[Array]) -> None:
        abscissa, samples, bary_weights = _make_dispatch_args(bkd)
        impl = get_lagrange_jacobian_impl(bkd)
        result = impl(abscissa, samples, bary_weights, bkd)
        expected = _generic_lagrange_jacobian(abscissa, samples, bary_weights, bkd)
        bkd.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_hessian_matches_generic(self, bkd: Backend[Array]) -> None:
        abscissa, samples, bary_weights = _make_dispatch_args(bkd)
        impl = get_lagrange_hessian_impl(bkd)
        result = impl(abscissa, samples, bary_weights, bkd)
        expected = _generic_lagrange_hessian(abscissa, samples, bary_weights, bkd)
        bkd.assert_allclose(result, expected, rtol=1e-10, atol=1e-8)


class TestLagrangeDispatchPickle:
    """Dispatched impls are module-level functions that pickle by reference."""

    def test_impls_pickle_by_reference(self, bkd: Backend[Array]) -> None:
        for getter in (
            get_lagrange_eval_impl,
            get_lagrange_jacobian_impl,
            get_lagrange_hessian_impl,
        ):
            impl = getter(bkd)
            assert pickle.loads(pickle.dumps(impl)) is impl

    def test_basis_pickle_roundtrip(self, bkd: Backend[Array]) -> None:
        basis = _make_basis(bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 9)))
        expected_vals = basis(samples)
        expected_jac = basis.jacobian_batch(samples)

        restored = pickle.loads(pickle.dumps(basis))
        # Pickling is exact, so the two sides differ only in how their
        # kernels execute. The torch impls are compiled, and a compiled
        # kernel that exceeds dynamo's recompile limit falls back to
        # eager, which contracts in a different order; 1e-14 is tight
        # enough that the resulting last-bit disagreement fails the test.
        bkd.assert_allclose(restored(samples), expected_vals, rtol=1e-12)
        bkd.assert_allclose(restored.jacobian_batch(samples), expected_jac, rtol=1e-12)
