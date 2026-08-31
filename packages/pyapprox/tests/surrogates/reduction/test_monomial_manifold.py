"""Tests for the polynomial manifold encoder.

The properties that matter are the ones the asymmetric encode/decode
pair is built on: encoding is exactly linear, the correction lives in
the complement of the basis so a round trip through the latent space is
exact, and the manifold beats a linear basis on curved data at the same
latent dimension. Latent dimensions are kept small so the greedy sweep,
which is quadratic in the candidate pool, stays fast.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
    StdDecodingEncoderProtocol,
)
from pyapprox.surrogates.reduction.feature_maps import (
    MonomialFeatureMap,
    SparseMonomialFeatureMap,
)
from pyapprox.surrogates.reduction.monomial_manifold import (
    MonomialManifoldEncoder,
    build_monomial_manifold_encoder,
)
from pyapprox.surrogates.reduction.protocols import (
    DecoderProtocol,
    LinearDecoderProtocol,
    ManifoldDecoderProtocol,
    SelfJacobianDecoderProtocol,
    is_linear_decoder,
    is_manifold_decoder,
    is_self_jacobian_decoder,
)


def _curved_data(bkd, nstates=25, nsnapshots=80, seed=0, curvature=0.3):
    """Snapshots on a curved two-dimensional manifold.

    The state is a quadratic function of two latent parameters, so it
    has linear rank 5 but is exactly representable by a quadratic
    manifold of latent dimension 2. This is the regime the correction
    exists for: a linear basis of two vectors cannot reproduce it.

    ``curvature`` scales the quadratic part below the linear one, so the
    two leading singular directions span the linear part rather than a
    mixture. That is the assumption the greedy construction makes -- the
    retained subspace carries the dominant behavior and the correction
    repairs what is left. At ``curvature=1`` all five directions have
    comparable energy, no two-dimensional subspace is dominant, and the
    method has correspondingly less to work with.
    """
    rng = np.random.RandomState(seed)
    t = rng.uniform(-1, 1, (2, nsnapshots))
    directions = np.linalg.qr(rng.normal(size=(nstates, 5)))[0]
    linear = np.outer(directions[:, 0], t[0]) + np.outer(
        directions[:, 1], t[1]
    )
    quadratic = (
        np.outer(directions[:, 2], t[0] ** 2)
        + np.outer(directions[:, 3], t[0] * t[1])
        + np.outer(directions[:, 4], t[1] ** 2)
    )
    return bkd.array(linear + curvature * quadratic)


def _linear_data(bkd, nstates=20, nsnapshots=60, rank=3, seed=1):
    """Snapshots lying exactly in a flat subspace of the given rank."""
    rng = np.random.RandomState(seed)
    basis = np.linalg.qr(rng.normal(size=(nstates, rank)))[0]
    return bkd.array(basis @ rng.normal(size=(rank, nsnapshots)))


def _relative_error(bkd, encoder, data):
    reconstructed = encoder.decode(encoder.encode(data))
    diff = reconstructed - data
    return float(
        bkd.to_numpy(bkd.sqrt(bkd.sum(diff * diff) / bkd.sum(data * data)))
    )


class TestFitAndReconstruct:
    """Fitting from snapshots, and the accuracy that buys."""

    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    def test_shapes(self, bkd, degrees) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=degrees
        )
        nterms = encoder.feature_map().nterms()
        assert encoder.latent_dim() == 2
        assert encoder.full_dim() == 25
        assert encoder.basis().shape == (25, 2)
        assert encoder.weights().shape == (25, nterms)
        assert encoder.mean().shape == (25, 1)
        assert encoder.encode(data).shape == (2, data.shape[1])
        assert encoder.decode(encoder.encode(data)).shape == data.shape

    def test_recovers_a_quadratic_manifold(self, bkd) -> None:
        # The data is exactly quadratic in two latent parameters, so a
        # quadratic manifold of latent dimension 2 reproduces it to the
        # accuracy of the regularized fit.
        data = _curved_data(bkd)
        manifold = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, gamma=1e-10
        )
        # Measured 1.1e-3: the residual after the quadratic correction
        # is the cubic and higher content the two retained directions
        # cannot express, not a defect of the fit.
        assert _relative_error(bkd, manifold, data) < 2e-3

    @pytest.mark.parametrize("degrees", [(0, 1, 2), (0, 1, 2, 3)])
    def test_beats_a_linear_basis_at_equal_latent_dim(
        self, bkd, degrees
    ) -> None:
        # The comparison the manifold exists to win: same latent
        # dimension, curved data. The linear reference is this encoder
        # with its correction zeroed, so only the correction differs.
        data = _curved_data(bkd)
        manifold = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=degrees, gamma=1e-10
        )
        linear = MonomialManifoldEncoder(
            manifold.basis(),
            bkd.zeros(manifold.weights().shape),
            manifold.feature_map(),
            manifold.mean(),
            bkd,
        )
        assert _relative_error(bkd, manifold, data) < 0.01 * _relative_error(
            bkd, linear, data
        )

    def test_low_degrees_are_not_redundant(self, bkd) -> None:
        # The claim that the decoder's own constant and linear parts
        # cover degrees 0 and 1: they do not, because the correction is
        # fitted in the complement of the basis, which neither reaches.
        data = _curved_data(bkd)
        errors = [
            _relative_error(
                bkd,
                MonomialManifoldEncoder.fit_from_data(
                    data, bkd, latent_dim=2, degrees=degrees, gamma=1e-10
                ),
                data,
            )
            for degrees in [(2,), (0, 2), (0, 1, 2)]
        ]
        band, with_constant, downward_closed = errors
        assert with_constant < band
        assert downward_closed < with_constant

    def test_cubic_is_at_least_as_good_as_quadratic(self, bkd) -> None:
        # A cubic map contains the quadratic terms, so with light
        # regularization it cannot fit the training data worse.
        data = _curved_data(bkd)
        errors = [
            _relative_error(
                bkd,
                MonomialManifoldEncoder.fit_from_data(
                    data, bkd, latent_dim=2, degrees=degrees, gamma=1e-10
                ),
                data,
            )
            for degrees in [(2,), (2, 3)]
        ]
        assert errors[1] <= errors[0] * 1.01

    def test_zero_weights_reproduce_the_linear_basis(self, bkd) -> None:
        # Ties the manifold to the linear reduction it generalizes: with
        # no correction, decode is exactly the projection onto V.
        data = _linear_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=3, degrees=(2,)
        )
        linear = MonomialManifoldEncoder(
            encoder.basis(),
            bkd.zeros(encoder.weights().shape),
            encoder.feature_map(),
            encoder.mean(),
            bkd,
        )
        codes = linear.encode(data)
        expected = (
            bkd.dot(encoder.basis(), codes) + encoder.mean()
        )
        bkd.assert_allclose(linear.decode(codes), expected)

    def test_flat_data_is_reconstructed_exactly(self, bkd) -> None:
        # Data in a rank-3 subspace needs no correction; a latent
        # dimension of 3 recovers it whether or not one is available.
        data = _linear_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=3, degrees=(2,), gamma=1e-12
        )
        assert _relative_error(bkd, encoder, data) < 1e-8


class TestEncodeDecodeContract:
    """The asymmetry between the linear encoder and the curved decoder."""

    def test_encode_is_linear(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,)
        )
        centered = data - encoder.mean()
        bkd.assert_allclose(
            encoder.encode(data), bkd.dot(encoder.basis().T, centered)
        )

    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    def test_encode_decode_round_trip_is_exact(self, bkd, degrees) -> None:
        # The correction lies in the complement of the basis, so V^T
        # annihilates it and encode(decode(z)) == z exactly, even though
        # decode(encode(s)) != s.
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=degrees
        )
        codes = bkd.array(
            np.random.RandomState(5).uniform(-0.5, 0.5, (2, 7))
        )
        bkd.assert_allclose(
            encoder.encode(encoder.decode(codes)), codes, atol=1e-9
        )

    def test_decode_adds_the_correction(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), gamma=1e-10
        )
        codes = bkd.array(
            np.random.RandomState(6).uniform(-0.5, 0.5, (2, 4))
        )
        expected = (
            bkd.dot(encoder.basis(), codes)
            + encoder.mean()
            + bkd.dot(
                encoder.weights(), encoder.feature_map()(codes)
            )
        )
        bkd.assert_allclose(encoder.decode(codes), expected)

    def test_is_not_an_isometry(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,)
        )
        assert encoder.is_isometry() is False


class TestDecodeJacobian:
    """The decoder tangent, which is state dependent."""

    @pytest.mark.parametrize("degrees", [(2,), (2, 3)])
    def test_matches_finite_differences(self, bkd, degrees) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=degrees
        )
        codes = bkd.array(np.array([[0.3], [-0.2]]))
        analytic = encoder.decode_jacobian(codes)

        eps = 1e-6
        columns = []
        for i in range(2):
            step = bkd.zeros((2, 1))
            step = bkd.copy(step)
            step[i, 0] = eps
            columns.append(
                (encoder.decode(codes + step) - encoder.decode(codes - step))
                / (2.0 * eps)
            )
        bkd.assert_allclose(analytic, bkd.hstack(columns), atol=1e-6)

    def test_varies_with_the_latent_coordinate(self, bkd) -> None:
        # A linear decoder has a constant tangent; a curved one does not.
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), gamma=1e-10
        )
        here = encoder.decode_jacobian(bkd.array(np.array([[0.4], [0.1]])))
        there = encoder.decode_jacobian(
            bkd.array(np.array([[-0.6], [0.5]]))
        )
        difference = float(
            bkd.to_numpy(bkd.sum(bkd.abs(here - there)))
        )
        assert difference > 1e-6

    def test_rejects_a_batch(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,)
        )
        with pytest.raises(ValueError, match="single latent coordinate"):
            encoder.decode_jacobian(bkd.zeros((2, 3)))


class TestProtocols:
    """Which capabilities the encoder declares."""

    def _encoder(self, bkd):
        return MonomialManifoldEncoder.fit_from_data(
            _curved_data(bkd), bkd, latent_dim=2, degrees=(2,)
        )

    def test_satisfies_the_decoder_protocols(self, bkd) -> None:
        encoder = self._encoder(bkd)
        assert isinstance(encoder, DecoderProtocol)
        assert isinstance(encoder, LinearDecoderProtocol)
        assert isinstance(encoder, ManifoldDecoderProtocol)
        assert isinstance(encoder, SelfJacobianDecoderProtocol)

    def test_typeguards_agree(self, bkd) -> None:
        encoder = self._encoder(bkd)
        assert is_self_jacobian_decoder(encoder)
        assert is_manifold_decoder(encoder)
        assert is_linear_decoder(encoder)

    def test_satisfies_the_function_encoder_protocol(self, bkd) -> None:
        # bkd/full_dim/latent_dim/encode/decode are all present.
        assert isinstance(self._encoder(bkd), FunctionEncoderProtocol)

    def test_declines_to_propagate_a_standard_deviation(self, bkd) -> None:
        # The linear propagation that protocol describes assumes a linear
        # decoder, which the correction makes false rather than
        # approximate, so the encoder must not declare it.
        assert not isinstance(
            self._encoder(bkd), StdDecodingEncoderProtocol
        )


class TestGreedySelection:
    """The basis is chosen, not taken in order."""

    def test_reports_the_selected_indices(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,)
        )
        selected = encoder.selected_indices()
        assert selected is not None
        assert len(selected) == 2
        assert len(set(selected)) == 2

    def test_candidate_pool_is_capped_by_the_rank(self, bkd) -> None:
        # A pool larger than the available rank is truncated rather than
        # erroring, so candidate_factor need not be tuned to the data.
        data = _linear_data(bkd, rank=3)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), ncandidates=1000
        )
        selected = encoder.selected_indices()
        assert selected is not None
        assert max(selected) < min(data.shape)

    def test_rejects_a_latent_dim_beyond_the_pool(self, bkd) -> None:
        data = _curved_data(bkd)
        with pytest.raises(ValueError, match="cannot exceed"):
            MonomialManifoldEncoder.fit_from_data(
                data, bkd, latent_dim=3, degrees=(2,), ncandidates=2
            )

    def test_records_the_fit_diagnostics(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), gamma=1e-8
        )
        assert encoder.fit_gamma() == 1e-8
        gram_cond = encoder.gram_cond()
        assert gram_cond is not None and gram_cond >= 1.0


class TestFeatureMapArgument:
    """A caller may supply the feature map instead of degrees."""

    def test_uses_a_supplied_map(self, bkd) -> None:
        data = _curved_data(bkd)
        fmap = MonomialFeatureMap(2, bkd, degrees=(2, 3))
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, feature_map=fmap
        )
        assert encoder.feature_map() is fmap

    def test_accepts_an_anisotropic_index_set(self, bkd) -> None:
        # Selection is driven by the index set, so a sparse map works
        # rather than failing part way through the greedy sweep.
        data = _curved_data(bkd)
        fmap = SparseMonomialFeatureMap(
            bkd.asarray(np.array([[2, 1], [0, 1]])), bkd
        )
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, feature_map=fmap
        )
        assert encoder.weights().shape[1] == 2

    def test_rejects_a_dimension_mismatch(self, bkd) -> None:
        data = _curved_data(bkd)
        fmap = MonomialFeatureMap(3, bkd, degrees=(2,))
        with pytest.raises(ValueError, match="must equal latent_dim"):
            MonomialManifoldEncoder.fit_from_data(
                data, bkd, latent_dim=2, feature_map=fmap
            )


class TestCenteringAndReuse:
    """Centering options and the injected SVD."""

    def test_uncentered_keeps_a_zero_mean(self, bkd) -> None:
        data = _curved_data(bkd)
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), center=False
        )
        bkd.assert_allclose(encoder.mean(), bkd.zeros((25, 1)))

    def test_supplied_mean_is_used(self, bkd) -> None:
        data = _curved_data(bkd)
        supplied = bkd.array(np.linspace(0.0, 1.0, 25))
        encoder = MonomialManifoldEncoder.fit_from_data(
            data, bkd, latent_dim=2, degrees=(2,), mean=supplied
        )
        bkd.assert_allclose(
            encoder.mean(), bkd.reshape(supplied, (25, 1))
        )

    def test_gamma_grid_selects_from_validation(self, bkd) -> None:
        data = _curved_data(bkd)
        validation = _curved_data(bkd, nsnapshots=30, seed=9)
        grid = [1e-10, 1e-4, 1.0]
        encoder = MonomialManifoldEncoder.fit_from_data(
            data,
            bkd,
            latent_dim=2,
            degrees=(2,),
            gamma_grid=grid,
            validation_data=validation,
        )
        assert encoder.fit_gamma() in grid

    def test_gamma_grid_requires_validation_data(self, bkd) -> None:
        data = _curved_data(bkd)
        with pytest.raises(ValueError, match="requires validation_data"):
            MonomialManifoldEncoder.fit_from_data(
                data, bkd, latent_dim=2, degrees=(2,), gamma_grid=[1e-6]
            )


class TestBuildFromTrajectories:
    """The trajectory-list convenience wrapper."""

    def test_matches_fitting_the_stacked_data(self, bkd) -> None:
        first = _curved_data(bkd, nsnapshots=40, seed=2)
        second = _curved_data(bkd, nsnapshots=40, seed=3)
        built = build_monomial_manifold_encoder(
            [first, second], bkd, latent_dim=2, degrees=(2,)
        )
        direct = MonomialManifoldEncoder.fit_from_data(
            bkd.hstack([first, second]), bkd, latent_dim=2, degrees=(2,)
        )
        bkd.assert_allclose(built.basis(), direct.basis())
        bkd.assert_allclose(built.weights(), direct.weights())
