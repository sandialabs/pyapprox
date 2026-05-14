import numpy as np
import pytest

from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.encoders.pca import (
    PCAFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)


class TestIdentityFunctionEncoder:
    def test_ncodes_equals_ngrid(self, bkd) -> None:
        enc = IdentityFunctionEncoder(10, bkd)
        assert enc.ncodes() == 10
        assert enc.ngrid() == 10

    def test_encode_decode_roundtrip(self, bkd) -> None:
        np.random.seed(42)
        ngrid, N = 8, 5
        enc = IdentityFunctionEncoder(ngrid, bkd)
        f = bkd.array(np.random.randn(ngrid, N))
        bkd.assert_allclose(enc.decode(enc.encode(f)), f)

    def test_decode_std_equals_decode(self, bkd) -> None:
        np.random.seed(42)
        ngrid, N = 8, 5
        enc = IdentityFunctionEncoder(ngrid, bkd)
        codes = bkd.array(np.random.randn(ngrid, N))
        bkd.assert_allclose(enc.decode_std(codes), enc.decode(codes))

    def test_protocol_compliance(self, bkd) -> None:
        enc = IdentityFunctionEncoder(5, bkd)
        assert isinstance(enc, FunctionEncoderProtocol)


class TestPCAFunctionEncoder:
    def _make_low_rank_data(self, bkd, ngrid=20, N=15, rank=5):
        np.random.seed(42)
        A = np.random.randn(ngrid, rank)
        B = np.random.randn(rank, N)
        data_np = A @ B + 0.1 * np.random.randn(ngrid, N)
        return bkd.array(data_np)

    def test_fit_from_data_ncodes(self, bkd) -> None:
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=3)
        assert enc.ncodes() == 3
        assert enc.ngrid() == 20

    def test_fit_from_data_variance_fraction(self, bkd) -> None:
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(
            data, bkd, variance_fraction=0.99
        )
        assert 1 <= enc.ncodes() <= 15

    def test_fit_from_data_requires_exactly_one(self, bkd) -> None:
        data = self._make_low_rank_data(bkd)
        with pytest.raises(ValueError, match="Exactly one"):
            PCAFunctionEncoder.fit_from_data(data, bkd)
        with pytest.raises(ValueError, match="Exactly one"):
            PCAFunctionEncoder.fit_from_data(
                data, bkd, ncodes=3, variance_fraction=0.9
            )

    def test_encode_decode_shapes(self, bkd) -> None:
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=5)
        codes = enc.encode(data)
        assert codes.shape == (5, 15)
        recovered = enc.decode(codes)
        assert recovered.shape == (20, 15)

    def test_roundtrip_full_rank(self, bkd) -> None:
        """With ncodes == min(ngrid, N), roundtrip is near-exact."""
        np.random.seed(42)
        ngrid, N = 10, 8
        data = bkd.array(np.random.randn(ngrid, N))
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=N)
        recovered = enc.decode(enc.encode(data))
        bkd.assert_allclose(recovered, data, atol=1e-10)

    def test_roundtrip_truncated_reduces_error(self, bkd) -> None:
        """More codes gives better reconstruction."""
        data = self._make_low_rank_data(bkd, ngrid=20, N=15, rank=5)
        enc3 = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=3)
        enc5 = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=5)
        err3 = float(bkd.to_numpy(
            bkd.sum((enc3.decode(enc3.encode(data)) - data) ** 2)
        ))
        err5 = float(bkd.to_numpy(
            bkd.sum((enc5.decode(enc5.encode(data)) - data) ** 2)
        ))
        assert err5 < err3

    def test_decode_std_zero_codes(self, bkd) -> None:
        """decode_std of zero codes should be zero."""
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=3)
        codes = bkd.array(np.zeros((3, 1)))
        decoded_std = enc.decode_std(codes)
        bkd.assert_allclose(decoded_std, bkd.zeros((20, 1)), atol=1e-14)

    def test_decode_std_nonnegative(self, bkd) -> None:
        """decode_std output must be non-negative for non-negative input."""
        np.random.seed(42)
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=5)
        std_codes = bkd.array(np.abs(np.random.randn(5, 10)))
        result = enc.decode_std(std_codes)
        assert bkd.all_bool(result >= 0)

    def test_decode_std_variance_propagation(self, bkd) -> None:
        """decode_std matches sqrt(P^2 @ sigma_c^2) computed explicitly."""
        np.random.seed(42)
        ngrid, ncodes = 10, 3
        Q, _ = np.linalg.qr(np.random.randn(ngrid, ncodes))
        basis = bkd.array(Q)
        mean = bkd.array(np.zeros((ngrid, 1)))
        enc = PCAFunctionEncoder(basis, mean, bkd)

        std_codes_np = np.abs(np.random.randn(ncodes, 5))
        std_codes = bkd.array(std_codes_np)

        expected_np = np.sqrt((Q ** 2) @ (std_codes_np ** 2))
        result = enc.decode_std(std_codes)
        bkd.assert_allclose(result, bkd.array(expected_np), atol=1e-12)

    def test_constructor_with_explicit_basis(self, bkd) -> None:
        """Can construct from pre-computed basis and mean."""
        np.random.seed(42)
        ngrid, ncodes = 10, 3
        Q, _ = np.linalg.qr(np.random.randn(ngrid, ncodes))
        basis = bkd.array(Q)
        mean = bkd.array(np.random.randn(ngrid, 1))
        enc = PCAFunctionEncoder(basis, mean, bkd)
        assert enc.ncodes() == ncodes
        assert enc.ngrid() == ngrid

    def test_protocol_compliance(self, bkd) -> None:
        data = self._make_low_rank_data(bkd)
        enc = PCAFunctionEncoder.fit_from_data(data, bkd, ncodes=3)
        assert isinstance(enc, FunctionEncoderProtocol)
