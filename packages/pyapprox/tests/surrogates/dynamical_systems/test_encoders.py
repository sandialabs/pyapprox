"""Tests for IdentityEncoder and LinearEncoder."""

import numpy as np

from pyapprox.surrogates.dynamical_systems.encoders import (
    IdentityEncoder,
    LinearEncoder,
)
from pyapprox.surrogates.dynamical_systems.protocols import EncoderProtocol


class TestIdentityEncoder:
    def test_protocol_conformance(self, bkd):
        enc = IdentityEncoder(3, bkd)
        assert isinstance(enc, EncoderProtocol)

    def test_dimensions(self, bkd):
        enc = IdentityEncoder(5, bkd)
        assert enc.full_dim() == 5
        assert enc.latent_dim() == 5

    def test_round_trip(self, bkd):
        enc = IdentityEncoder(3, bkd)
        states = bkd.array(np.random.RandomState(0).randn(3, 10))
        encoded = enc.encode(states)
        decoded = enc.decode(encoded)
        bkd.assert_allclose(decoded, states)

    def test_encode_is_identity(self, bkd):
        enc = IdentityEncoder(2, bkd)
        states = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        bkd.assert_allclose(enc.encode(states), states)

    def test_encode_jacobian(self, bkd):
        enc = IdentityEncoder(3, bkd)
        jac = enc.encode_jacobian()
        bkd.assert_allclose(jac, bkd.eye(3))


class TestLinearEncoder:
    def test_protocol_conformance(self, bkd):
        P = bkd.array(np.eye(2, 3))
        enc = LinearEncoder(P, bkd)
        assert isinstance(enc, EncoderProtocol)

    def test_dimensions(self, bkd):
        P = bkd.array(np.eye(2, 4))
        enc = LinearEncoder(P, bkd)
        assert enc.full_dim() == 4
        assert enc.latent_dim() == 2

    def test_orthogonal_round_trip(self, bkd):
        rng = np.random.RandomState(0)
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        P = bkd.array(Q[:2, :])
        enc = LinearEncoder(P, bkd)
        states = bkd.array(rng.randn(3, 10))
        latents = enc.encode(states)
        assert latents.shape == (2, 10)
        reconstructed = enc.decode(latents)
        projected = enc.decode(enc.encode(reconstructed))
        bkd.assert_allclose(projected, reconstructed, rtol=1e-10)

    def test_square_orthogonal_exact_round_trip(self, bkd):
        theta = 0.7
        c, s = np.cos(theta), np.sin(theta)
        P = bkd.array([[c, s], [-s, c]])
        enc = LinearEncoder(P, bkd)
        states = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        decoded = enc.decode(enc.encode(states))
        bkd.assert_allclose(decoded, states, rtol=1e-12)

    def test_encode_jacobian(self, bkd):
        P = bkd.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        enc = LinearEncoder(P, bkd)
        bkd.assert_allclose(enc.encode_jacobian(), P)

    def test_invalid_projection_matrix(self, bkd):
        import pytest

        with pytest.raises(ValueError, match="2D"):
            LinearEncoder(bkd.array([1.0, 2.0, 3.0]), bkd)
