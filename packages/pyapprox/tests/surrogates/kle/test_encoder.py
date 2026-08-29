"""Tests for reading a KLE basis as a reduction.

The encoder holds a KLE rather than copying its arrays, so the tests
that matter are about the two views agreeing: that encode really is the
projection onto the basis the expansion was built from, and that a
capability the wrapped object cannot support is absent rather than
present-and-failing.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
    StdDecodingEncoderProtocol,
)
from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.surrogates.kle.encoder import KLEEncoder
from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
)


def _centered(bkd, nstates=12, nsamples=8, seed=0):
    rng = np.random.RandomState(seed)
    data = bkd.array(rng.standard_normal((nstates, nsamples)))
    return data - bkd.reshape(bkd.mean(data, axis=1), (nstates, 1))


def _kle(bkd, nterms=4, **kwargs):
    return DataDrivenKLE(
        _centered(bkd), 0.0, False, nterms, None, bkd=bkd, **kwargs
    )


class TestProtocolConformance:
    def test_satisfies_the_encoder_protocols(self, bkd) -> None:
        enc = KLEEncoder(_kle(bkd))
        assert isinstance(enc, FunctionEncoderProtocol)
        assert isinstance(enc, StdDecodingEncoderProtocol)

    def test_dimensions_come_from_the_basis(self, bkd) -> None:
        enc = KLEEncoder(_kle(bkd, nterms=3))
        assert enc.full_dim() == 12
        assert enc.latent_dim() == 3
        assert enc.basis().shape == (12, 3)
        assert enc.mean().shape == (12, 1)

    def test_rejects_a_non_kle(self, bkd) -> None:
        with pytest.raises(TypeError, match="must satisfy KLEProtocol"):
            KLEEncoder(object())


class TestProjection:
    """encode must be the projection onto the wrapped basis."""

    def test_roundtrip_is_exact_at_full_rank(self, bkd) -> None:
        """Centering costs one mode, so full rank is nsamples - 1."""
        data = _centered(bkd)
        enc = KLEEncoder(_kle(bkd, nterms=7))
        bkd.assert_allclose(
            enc.decode(enc.encode(data)), data, atol=1e-12
        )

    def test_is_idempotent_when_truncated(self, bkd) -> None:
        """Truncation loses information, but re-encoding a decoded
        latent must return it: that is what makes it a projection
        rather than merely a linear map."""
        data = _centered(bkd)
        enc = KLEEncoder(_kle(bkd, nterms=4))
        latents = enc.encode(data)
        bkd.assert_allclose(
            enc.encode(enc.decode(latents)), latents, atol=1e-12
        )

    def test_truncation_error_decreases_with_more_modes(self, bkd) -> None:
        data = _centered(bkd)
        errors = []
        for nterms in (2, 4, 7):
            enc = KLEEncoder(_kle(bkd, nterms=nterms))
            residual = enc.decode(enc.encode(data)) - data
            errors.append(bkd.to_float(bkd.norm(residual)))
        assert errors[0] > errors[1] > errors[2]

    def test_decode_is_not_the_generative_call(self, bkd) -> None:
        """The two readings have different units: the generative side
        scales by sqrt(eigenvalue) because its argument is a
        standardized coefficient, not a coordinate."""
        kle = _kle(bkd)
        enc = KLEEncoder(kle)
        latents = bkd.ones((kle.nterms(), 1))
        assert not bool(
            bkd.allclose(enc.decode(latents), kle(latents))
        )


class TestMetric:
    def test_euclidean_metric_matches_no_metric(self, bkd) -> None:
        data = _centered(bkd)
        kle = _kle(bkd)
        bare = KLEEncoder(kle).encode(data)
        explicit = KLEEncoder(
            kle, EuclideanInnerProduct(12, bkd)
        ).encode(data)
        bkd.assert_allclose(bare, explicit, rtol=1e-12)

    def test_metric_changes_the_projection(self, bkd) -> None:
        """A weighted projection is a different question, not a
        rescaling of the same one."""
        data = _centered(bkd)
        kle = _kle(bkd)
        weights = bkd.asarray(np.linspace(0.5, 2.0, 12))
        weighted = KLEEncoder(
            kle, DiagonalInnerProduct(weights, bkd)
        ).encode(data)
        assert not bool(
            bkd.allclose(weighted, KLEEncoder(kle).encode(data))
        )

    def test_rejects_a_metric_of_the_wrong_size(self, bkd) -> None:
        with pytest.raises(ValueError, match="defined on"):
            KLEEncoder(_kle(bkd), EuclideanInnerProduct(5, bkd))


class TestLognormalHasNoEncoder:
    """The reason this is composition rather than more methods.

    A lognormal expansion decodes through exp(), which no linear
    projection inverts. Refusing at construction makes the absence a
    fact about which objects exist, rather than an isinstance check that
    passes over an object that raises on use.
    """

    def _lognormal(self, bkd):
        kle = _kle(bkd)
        return PrecomputedKLE(
            kle.eigenvalues(),
            kle.eigenvectors(),
            kle.mean_field(),
            use_log=True,
            bkd=bkd,
        )

    def test_refuses_construction(self, bkd) -> None:
        with pytest.raises(ValueError, match="exponentiates"):
            KLEEncoder(self._lognormal(bkd))

    def test_the_same_basis_without_the_flag_is_accepted(
        self, bkd
    ) -> None:
        """It is the exponentiation that is refused, not the basis."""
        kle = _kle(bkd)
        linear = PrecomputedKLE(
            kle.eigenvalues(),
            kle.eigenvectors(),
            kle.mean_field(),
            use_log=False,
            bkd=bkd,
        )
        assert KLEEncoder(linear).latent_dim() == kle.nterms()


class TestSharesRatherThanCopies:
    def test_basis_is_the_wrapped_one(self, bkd) -> None:
        kle = _kle(bkd)
        bkd.assert_allclose(
            KLEEncoder(kle).basis(), kle.eigenvectors(), atol=0.0
        )

    def test_wrapped_kle_is_reachable(self, bkd) -> None:
        kle = _kle(bkd)
        assert KLEEncoder(kle).kle() is kle


class TestPrecomputedSpectrum:
    def test_singular_values_are_the_root_of_the_eigenvalues(
        self, bkd
    ) -> None:
        kle = _kle(bkd)
        stored = PrecomputedKLE(
            kle.eigenvalues(), kle.eigenvectors(), kle.mean_field(),
            bkd=bkd,
        )
        bkd.assert_allclose(
            stored.singular_values() ** 2,
            stored.eigenvalues(),
            rtol=1e-12,
        )
