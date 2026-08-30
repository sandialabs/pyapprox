import numpy as np
from pyapprox.surrogates.kerneloperator.encoders.identity import (
    IdentityFunctionEncoder,
)
from pyapprox.surrogates.kerneloperator.protocols import (
    FunctionEncoderProtocol,
)


class TestIdentityFunctionEncoder:
    def test_latent_dim_equals_full_dim(self, bkd) -> None:
        enc = IdentityFunctionEncoder(10, bkd)
        assert enc.latent_dim() == 10
        assert enc.full_dim() == 10

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
