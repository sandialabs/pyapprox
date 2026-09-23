"""Tests for reading a KLE basis as a reduction.

The encoder holds a KLE rather than copying its arrays, so the tests
that matter are about the two views agreeing: that encode really is the
projection onto the basis the expansion was built from, and that a
capability the wrapped object cannot support is absent rather than
present-and-failing.
"""

import os

import numpy as np
import pytest
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
    StdDecodingEncoderProtocol,
)
from pyapprox.surrogates.kle.basis_sinks import (
    ArrayBasisSink,
    MemmapBasisSink,
)
from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.surrogates.kle.encoder import KLEEncoder, fit_kle_encoder
from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.surrogates.kle.protocols import KLEProtocol
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
)
from pyapprox.surrogates.operatorlearning.protocols import (
    FieldEncoderProtocol,
)
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    EuclideanInnerProduct,
    MassInnerProduct,
    m_orthonormality_drift,
)
from scipy.sparse import diags


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

    def test_satisfies_the_field_encoder_protocol(self, bkd) -> None:
        """The one that lets it into an operator surrogate.

        Without ``is_isometry`` this encoder was structurally excluded
        from least-squares operator learning despite being the canonical
        basis for it.
        """
        assert isinstance(KLEEncoder(_kle(bkd)), FieldEncoderProtocol)

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


class TestDecodeStd:
    """Variance propagation, and the assumption it rests on."""

    def _from_arrays(self, bkd, nstates=10, nterms=3):
        """An encoder over a basis given directly rather than fitted.

        The route for a basis that arrives as arrays -- reloaded, or
        computed elsewhere -- and the reason KLEEncoder needs no second
        constructor: PrecomputedKLE already is that adapter.
        """
        rng = np.random.RandomState(42)
        q, _ = np.linalg.qr(rng.standard_normal((nstates, nterms)))
        return q, KLEEncoder(
            PrecomputedKLE(
                bkd.ones((nterms,)),
                bkd.array(q),
                bkd.zeros((nstates,)),
                bkd=bkd,
            )
        )

    def test_matches_explicit_variance_propagation(self, bkd) -> None:
        q, enc = self._from_arrays(bkd)
        rng = np.random.RandomState(7)
        std_np = np.abs(rng.standard_normal((3, 5)))
        bkd.assert_allclose(
            enc.decode_std(bkd.array(std_np)),
            bkd.array(np.sqrt((q**2) @ (std_np**2))),
            atol=1e-12,
        )

    def test_zero_std_decodes_to_zero(self, bkd) -> None:
        _, enc = self._from_arrays(bkd)
        bkd.assert_allclose(
            enc.decode_std(bkd.zeros((3, 1))), bkd.zeros((10, 1)),
            atol=1e-14,
        )

    def test_is_nonnegative(self, bkd) -> None:
        _, enc = self._from_arrays(bkd)
        rng = np.random.RandomState(7)
        std = bkd.array(np.abs(rng.standard_normal((3, 5))))
        assert bool(bkd.all_bool(enc.decode_std(std) >= 0.0))

    def test_carries_no_mean_shift(self, bkd) -> None:
        """A standard deviation is a spread, not a location, so the
        mean must not enter -- unlike decode."""
        rng = np.random.RandomState(42)
        q, _ = np.linalg.qr(rng.standard_normal((10, 3)))
        std = bkd.array(np.abs(rng.standard_normal((3, 4))))
        results = []
        for mean_value in (0.0, 5.0):
            kle = PrecomputedKLE(
                bkd.ones((3,)),
                bkd.array(q),
                bkd.full((10,), mean_value),
                bkd=bkd,
            )
            results.append(KLEEncoder(kle).decode_std(std))
        bkd.assert_allclose(results[0], results[1], atol=0.0)


class TestSharesRatherThanCopies:
    def test_basis_is_the_wrapped_one(self, bkd) -> None:
        kle = _kle(bkd)
        bkd.assert_allclose(
            KLEEncoder(kle).basis(), kle.eigenvectors(), atol=0.0
        )

    def test_wrapped_kle_is_reachable(self, bkd) -> None:
        kle = _kle(bkd)
        assert KLEEncoder(kle).kle() is kle


class TestIsometry:
    r"""Whether a coefficient residual may be read as a field error.

    The property is computed from the basis and the metric rather than
    asserted, because ``KLEProtocol`` promises nothing about
    orthonormality and the encoder may be handed a metric the basis was
    not built in.
    """

    def test_m_pod_basis_is_an_isometry(self, bkd) -> None:
        weights = bkd.array(np.linspace(0.5, 3.0, 12))
        metric = DiagonalInnerProduct(weights, bkd)
        enc = fit_kle_encoder(
            _centered(bkd), bkd, latent_dim=4, metric=metric
        )
        assert enc.is_isometry()
        assert enc.orthonormality_drift() < 1e-12

    def test_euclidean_basis_is_an_isometry(self, bkd) -> None:
        assert KLEEncoder(_kle(bkd)).is_isometry()

    def test_arbitrary_basis_is_not(self, bkd) -> None:
        """A PrecomputedKLE may hold any array at all."""
        rng = np.random.RandomState(3)
        kle = PrecomputedKLE(
            bkd.array(np.array([4.0, 3.0, 2.0])),
            bkd.array(rng.standard_normal((12, 3))),
            bkd.array(np.zeros(12)),
            1.0,
            False,
            bkd,
        )
        enc = KLEEncoder(kle)
        assert not enc.is_isometry()
        assert enc.orthonormality_drift() > 1.0

    def test_mismatched_metric_is_not_an_isometry(self, bkd) -> None:
        """The failure the class docstring warns about, made detectable.

        A basis built Euclidean and paired with a weighted metric is not
        orthonormal in that metric, so encode is not a projection. Before
        ``is_isometry`` nothing reported this.
        """
        weights = bkd.array(np.linspace(0.5, 3.0, 12))
        enc = KLEEncoder(_kle(bkd), DiagonalInnerProduct(weights, bkd))
        assert not enc.is_isometry()

    def test_tolerance_is_honored(self, bkd) -> None:
        rng = np.random.RandomState(3)
        kle = PrecomputedKLE(
            bkd.array(np.array([4.0, 3.0, 2.0])),
            bkd.array(rng.standard_normal((12, 3))),
            bkd.array(np.zeros(12)),
            1.0,
            False,
            bkd,
        )
        assert KLEEncoder(kle, orthonormality_tol=1e3).is_isometry()

    def test_agrees_with_the_shared_drift_helper(self, bkd) -> None:
        """The encoder reports what ``m_orthonormality_drift`` computes.

        Pinned because the value must not drift from the free function
        that the sibling encoder and the KLE tests both use; one
        definition of orthonormality, checked the same way everywhere.
        """
        weights = bkd.array(np.linspace(0.5, 3.0, 12))
        metric = DiagonalInnerProduct(weights, bkd)
        enc = fit_kle_encoder(
            _centered(bkd), bkd, latent_dim=4, metric=metric
        )
        bkd.assert_allclose(
            bkd.asarray([enc.orthonormality_drift()]),
            bkd.asarray(
                [m_orthonormality_drift(enc.basis(), metric, bkd)]
            ),
        )


class TestFitKLEEncoder:
    """The one-call path, and the seams it leaves injectable."""

    def _data(self, bkd, nstates=12, nsamples=8):
        rng = np.random.RandomState(0)
        return bkd.array(rng.standard_normal((nstates, nsamples)))

    def test_matches_building_the_two_steps_by_hand(self, bkd) -> None:
        """A convenience, not a different computation."""
        data = self._data(bkd)
        by_hand = KLEEncoder(
            DataDrivenKLE(data, nterms=4, center=True, bkd=bkd)
        )
        bkd.assert_allclose(
            fit_kle_encoder(data, bkd, latent_dim=4).basis(),
            by_hand.basis(),
            rtol=1e-12,
        )

    def test_truncates_by_variance_fraction(self, bkd) -> None:
        data = self._data(bkd)
        assert (
            fit_kle_encoder(data, bkd, variance_fraction=0.5).latent_dim()
            < fit_kle_encoder(data, bkd).latent_dim()
        )

    def test_centers_by_default(self, bkd) -> None:
        """Unlike DataDrivenKLE: a reduction is almost always taken
        about the data's mean, while an expansion may be about anything."""
        data = self._data(bkd)
        enc = fit_kle_encoder(data, bkd, latent_dim=3)
        bkd.assert_allclose(
            enc.mean()[:, 0], bkd.mean(data, axis=1), rtol=1e-12
        )

    def test_metric_reaches_the_basis(self, bkd) -> None:
        """The gap that made a separate weighted encoder necessary."""
        data = self._data(bkd)
        weights = bkd.asarray(np.linspace(0.5, 2.0, 12))
        metric = DiagonalInnerProduct(weights, bkd)
        enc = fit_kle_encoder(data, bkd, latent_dim=3, metric=metric)
        assert m_orthonormality_drift(enc.basis(), metric, bkd) < 1e-10
        assert enc.metric() is metric

    def test_eigensolver_is_injectable(self, bkd) -> None:
        """Solver choice was unreachable through the old encoder."""
        data = self._data(bkd)
        default = fit_kle_encoder(data, bkd, latent_dim=3)
        gram = fit_kle_encoder(
            data, bkd, latent_dim=3,
            eigensolver=MethodOfSnapshotsSolver(bkd),
        )
        bkd.assert_allclose(
            gram.basis(), default.basis(), rtol=1e-6, atol=1e-8
        )

    def test_result_is_still_a_kle(self, bkd) -> None:
        """What makes the encoder storable by save_kle."""
        enc = fit_kle_encoder(self._data(bkd), bkd, latent_dim=3)
        assert isinstance(enc.kle(), KLEProtocol)


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


class TestDecodingAtSelectedStates:
    """``decode_at``, for plotting and probing without a full field.

    :math:`f = Vz + \\bar{f}` has no coupling across rows, so the field
    at a few points is exactly those rows of the basis against the
    coefficients -- never the whole field followed by a subscript.
    """

    def _encoder(self, bkd, nstates=400, nterms=5):
        raw = np.random.RandomState(0).standard_normal(
            (nstates, nterms)
        )
        basis = bkd.array(np.linalg.qr(raw)[0])
        values = bkd.array(np.logspace(0, -2, nterms))
        mean = bkd.array(
            np.random.RandomState(1).standard_normal(nstates)
        )
        return KLEEncoder(
            PrecomputedKLE(values, basis, mean, bkd=bkd)
        )

    def test_matches_decoding_then_subscripting(self, bkd) -> None:
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 3))
        )
        rows = [0, 311, 7, 1]
        full = encoder.decode(latents)
        bkd.assert_allclose(
            encoder.decode_at(latents, rows),
            full[bkd.asarray(rows, dtype=int), :],
            atol=0.0,
        )

    def test_the_order_given_is_the_order_returned(self, bkd) -> None:
        """So a caller can pass the ordering its plot wants."""
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 2))
        )
        full = encoder.decode(latents)
        got = encoder.decode_at(latents, [311, 0])
        bkd.assert_allclose(got[0], full[311], atol=0.0)
        bkd.assert_allclose(got[1], full[0], atol=0.0)

    def test_a_repeated_state_is_returned_twice(self, bkd) -> None:
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 2))
        )
        got = encoder.decode_at(latents, [9, 9])
        assert got.shape == (2, 2)
        bkd.assert_allclose(got[0], got[1], atol=0.0)

    def test_an_out_of_range_state_raises(self, bkd) -> None:
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 2))
        )
        with pytest.raises(ValueError, match="out of range"):
            encoder.decode_at(latents, [0, 400])

    def test_decoding_to_a_sink_matches_decoding(self, bkd) -> None:
        """Every field, written rather than returned.

        ``decode`` returns ``(full_dim, nsamples)``, which for a fine
        mesh and a full set of snapshots is larger than the data the
        basis came from; this writes the same values a block at a time.
        """
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 6))
        )
        written = encoder.decode_to_sink(
            latents, ArrayBasisSink(400, 6, bkd)
        )
        bkd.assert_allclose(
            written.to_array(), encoder.decode(latents), atol=0.0
        )

    @pytest.mark.parametrize("max_bytes", [1 << 8, 1 << 14, None])
    def test_the_block_size_does_not_change_the_fields(
        self, bkd, max_bytes
    ) -> None:
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 6))
        )
        written = encoder.decode_to_sink(
            latents, ArrayBasisSink(400, 6, bkd), max_bytes=max_bytes
        )
        bkd.assert_allclose(
            written.to_array(), encoder.decode(latents), atol=1e-14
        )

    def test_a_memmap_sink_holds_the_fields(
        self, bkd, tmp_path
    ) -> None:
        """The case the method exists for: fields larger than memory.

        ``rows`` on the result reads individual states back out, so a
        field written here can be plotted later without being
        reconstructed a second time.
        """
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 6))
        )
        written = encoder.decode_to_sink(
            latents,
            MemmapBasisSink(
                os.path.join(str(tmp_path), "fields.dat"), 400, 6, bkd
            ),
        )
        expected = encoder.decode(latents)
        bkd.assert_allclose(written.to_array(), expected, atol=0.0)
        bkd.assert_allclose(
            written.rows([0, 399, 7]),
            expected[bkd.asarray([0, 399, 7], dtype=int), :],
            atol=0.0,
        )

    def test_a_sink_sized_for_the_wrong_state_count_raises(
        self, bkd
    ) -> None:
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 6))
        )
        with pytest.raises(ValueError, match="states"):
            encoder.decode_to_sink(
                latents, ArrayBasisSink(401, 6, bkd)
            )

    def test_a_sink_sized_for_the_wrong_field_count_raises(
        self, bkd
    ) -> None:
        """Caught before any writing, naming fields rather than terms."""
        encoder = self._encoder(bkd)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, 6))
        )
        with pytest.raises(ValueError, match="fields to write"):
            encoder.decode_to_sink(
                latents, ArrayBasisSink(400, 4, bkd)
            )

    def test_latents_of_the_wrong_height_are_rejected(
        self, bkd
    ) -> None:
        encoder = self._encoder(bkd)
        wrong = bkd.array(
            np.random.RandomState(2).standard_normal((3, 6))
        )
        with pytest.raises(ValueError, match="terms"):
            encoder.decode_to_sink(wrong, ArrayBasisSink(400, 6, bkd))

    def test_decoding_to_a_sink_does_not_hold_every_field(
        self, numpy_bkd
    ) -> None:
        """The property the method exists for, asserted not assumed.

        The accuracy tests above all pass against an implementation
        that calls ``decode`` and writes the result in one go, so peak
        allocation is the only thing that tells the two apart. The
        sink's own storage is the intended output and is excluded.

        Numpy only: the torch allocator caches, so tracemalloc does not
        see tensor storage.
        """
        import tracemalloc

        bkd = numpy_bkd
        nstates, nfields = 4000, 20
        encoder = self._encoder(bkd, nstates=nstates)
        latents = bkd.array(
            np.random.RandomState(2).standard_normal((5, nfields))
        )
        budget = 1 << 12

        tracemalloc.start()
        try:
            encoder.decode_to_sink(
                latents,
                ArrayBasisSink(nstates, nfields, bkd),
                max_bytes=budget,
            )
            peak = int(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()

        sink_bytes = nstates * nfields * 8
        assert peak - sink_bytes < 16 * budget

    def test_1d_latents_are_rejected(self, bkd) -> None:
        encoder = self._encoder(bkd)
        with pytest.raises(ValueError, match="2D"):
            encoder.decode_at(bkd.array(np.zeros(5)), [0])


class TestTheDriftCheckWithAStreamedBasis:
    """Orthonormality is checkable without holding the basis.

    ``V^T M V`` contracts over rows, so it accumulates over row blocks
    whenever the metric acts on a block without reaching outside it.
    Both conditions are tested, including the one where they do not
    hold and the check must fall back rather than silently drop the
    terms straddling each boundary.
    """

    def _pair(self, bkd, tmp_path, metric=None, nstates=400, nterms=5):
        """The same basis, held as an array and as a memmap."""
        raw = np.random.RandomState(0).standard_normal(
            (nstates, nterms)
        )
        basis = bkd.array(np.linalg.qr(raw)[0])
        values = bkd.array(np.logspace(0, -2, nterms))
        mean = bkd.array(np.zeros(nstates))
        sink = MemmapBasisSink(
            os.path.join(str(tmp_path), "b.dat"), nstates, nterms, bkd
        )
        for start in range(0, nstates, 37):
            stop = min(start + 37, nstates)
            sink.write(slice(start, stop), basis[start:stop, :])
        streamed = sink.finalize()
        return (
            KLEEncoder(
                PrecomputedKLE(values, basis, mean, bkd=bkd), metric
            ),
            KLEEncoder(
                PrecomputedKLE(values, streamed, mean, bkd=bkd), metric
            ),
        )

    def test_a_streamed_basis_gives_the_same_drift(
        self, bkd, tmp_path
    ) -> None:
        resident, streamed = self._pair(bkd, tmp_path)
        assert streamed.orthonormality_drift() == pytest.approx(
            resident.orthonormality_drift(), abs=1e-14
        )
        assert streamed.is_isometry() == resident.is_isometry()

    def test_a_diagonal_metric_also_accumulates(
        self, bkd, tmp_path
    ) -> None:
        """Separable, so the block sums give the exact Gram."""
        weights = bkd.array(
            np.random.RandomState(4).uniform(0.5, 2.0, 400)
        )
        resident, streamed = self._pair(
            bkd, tmp_path, DiagonalInnerProduct(weights, bkd)
        )
        assert streamed.orthonormality_drift() == pytest.approx(
            resident.orthonormality_drift(), rel=1e-12
        )

    def test_a_coupled_metric_falls_back_rather_than_approximating(
        self, bkd, tmp_path
    ) -> None:
        """A mass matrix reaches outside the block, so blocks are wrong.

        The fallback materializes, which is what every expansion
        building its basis from a resident eigenproblem does anyway.
        What must not happen is a block-wise sum that drops the terms
        crossing each boundary and reports a different number.
        """
        nstates = 400
        mass = diags(
            [
                np.full(nstates - 1, 0.5),
                np.full(nstates, 2.0),
                np.full(nstates - 1, 0.5),
            ],
            [-1, 0, 1],
        )
        resident, streamed = self._pair(
            bkd, tmp_path, MassInnerProduct(mass, bkd)
        )
        assert streamed.orthonormality_drift() == pytest.approx(
            resident.orthonormality_drift(), rel=1e-12
        )
