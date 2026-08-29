"""Tests for operator-learning field encoders."""

from typing import Any, List

import numpy as np
import pytest
from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    DiagonalMassMatrix,
    IdentityMassMatrix,
)
from pyapprox.surrogates.operatorlearning import (
    FieldEncoderProtocol,
    GramProjectionEncoder,
    IdentityFieldEncoder,
    ProductFieldEncoder,
    orthonormalize_basis,
)
from pyapprox.util.backends.protocols import Backend


def _fourier_basis(bkd: Backend, ngrid: int, ncodes: int) -> Any:
    """Orthonormal sine basis under the trapezoidal-style uniform rule.

    Columns are sqrt(2) * sin(j pi x) at interior points, which are
    orthonormal with respect to the uniform weight 1 / ngrid.
    """
    x = np.linspace(0.0, 1.0, ngrid + 2)[1:-1]
    modes = np.arange(1, ncodes + 1)
    basis = np.sqrt(2.0) * np.sin(np.outer(x, modes) * np.pi)
    return bkd.asarray(basis)


def _uniform_mass(bkd: Backend, ngrid: int) -> DiagonalMassMatrix:
    """Diagonal mass for the uniform rule that makes _fourier_basis orthonormal."""
    return DiagonalMassMatrix(bkd.full((ngrid,), 1.0 / (ngrid + 1)), bkd)


class TestIdentityFieldEncoder:
    def test_roundtrip(self, bkd: Backend) -> None:
        encoder = IdentityFieldEncoder(4, bkd)
        f = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        bkd.assert_allclose(encoder.decode(encoder.encode(f)), f)

    def test_dimensions(self, bkd: Backend) -> None:
        encoder = IdentityFieldEncoder(5, bkd)
        assert encoder.ncodes() == 5
        assert encoder.ngrid() == 5

    def test_is_isometry_default(self, bkd: Backend) -> None:
        assert IdentityFieldEncoder(3, bkd).is_isometry()

    def test_is_isometry_can_be_declared_false(self, bkd: Backend) -> None:
        assert not IdentityFieldEncoder(3, bkd, is_isometry=False).is_isometry()

    def test_satisfies_protocol(self, bkd: Backend) -> None:
        assert isinstance(IdentityFieldEncoder(3, bkd), FieldEncoderProtocol)


class TestGramProjectionEncoder:
    """T1: encode recovers known coefficients; round-trip; isometry."""

    def test_recovers_known_coefficients(self, bkd: Backend) -> None:
        """f = sum_j c_j psi_j encodes back to c."""
        ngrid, ncodes = 64, 5
        basis = _fourier_basis(bkd, ngrid, ncodes)
        encoder = GramProjectionEncoder(basis, _uniform_mass(bkd, ngrid), bkd)
        coefs = bkd.asarray([[1.0], [-2.0], [0.5], [3.0], [-0.25]])
        f = encoder.decode(coefs)
        bkd.assert_allclose(encoder.encode(f), coefs, atol=1e-13)

    def test_roundtrip_in_span(self, bkd: Backend) -> None:
        ngrid, ncodes = 64, 4
        basis = _fourier_basis(bkd, ngrid, ncodes)
        encoder = GramProjectionEncoder(basis, _uniform_mass(bkd, ngrid), bkd)
        coefs = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        f = encoder.decode(coefs)
        bkd.assert_allclose(encoder.encode(f), coefs, atol=1e-13)

    def test_isometry_preserves_norm(self, bkd: Backend) -> None:
        """||f||_Y == ||encode(f)||_2 for an orthonormal basis."""
        ngrid, ncodes = 64, 5
        basis = _fourier_basis(bkd, ngrid, ncodes)
        mass = _uniform_mass(bkd, ngrid)
        encoder = GramProjectionEncoder(basis, mass, bkd)
        coefs = bkd.asarray([[1.0], [-2.0], [0.5], [3.0], [-0.25]])
        f = encoder.decode(coefs)
        y_norm_sq = bkd.sum(f * mass.apply(f))
        code_norm_sq = bkd.sum(encoder.encode(f) ** 2)
        bkd.assert_allclose(
            bkd.asarray([y_norm_sq]), bkd.asarray([code_norm_sq]), rtol=1e-12
        )

    def test_gram_is_identity_for_orthonormal_basis(self, bkd: Backend) -> None:
        ngrid, ncodes = 64, 4
        basis = _fourier_basis(bkd, ngrid, ncodes)
        encoder = GramProjectionEncoder(basis, _uniform_mass(bkd, ngrid), bkd)
        bkd.assert_allclose(encoder.gram(), bkd.eye(ncodes), atol=1e-13)
        assert encoder.is_isometry()

    def test_non_orthonormal_basis_reports_not_isometry(self, bkd: Backend) -> None:
        """A merely independent basis must not claim to be an isometry."""
        basis = bkd.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        encoder = GramProjectionEncoder(
            basis, IdentityMassMatrix(3, bkd), bkd
        )
        assert not encoder.is_isometry()

    def test_wrong_inner_product_detected(self, bkd: Backend) -> None:
        """A basis orthonormal under one mass is not under another.

        Guards the lumping error: with a non-diagonal mass matrix, a
        diagonal approximation would wrongly report an isometry.
        """
        ngrid, ncodes = 32, 3
        basis = _fourier_basis(bkd, ngrid, ncodes)
        assert GramProjectionEncoder(
            basis, _uniform_mass(bkd, ngrid), bkd
        ).is_isometry()
        assert not GramProjectionEncoder(
            basis, IdentityMassMatrix(ngrid, bkd), bkd
        ).is_isometry()

    def test_nondiagonal_mass_is_used_exactly(self, bkd: Backend) -> None:
        """Encoding uses the full mass matrix, not its diagonal."""
        dense = bkd.asarray(
            [[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]]
        )
        basis = bkd.asarray([[1.0], [0.0], [0.0]])
        encoder = GramProjectionEncoder(
            basis, ConstantDenseMassMatrix(dense, bkd), bkd
        )
        f = bkd.asarray([[0.0], [1.0], [0.0]])
        # psi^T M f picks out M[0, 1] = 0.5; a lumped mass would give 0.
        bkd.assert_allclose(encoder.encode(f), bkd.asarray([[0.5]]))

    def test_rejects_1d_basis(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            GramProjectionEncoder(
                bkd.asarray([1.0, 2.0]), IdentityMassMatrix(2, bkd), bkd
            )

    def test_rejects_non_mass_matrix(self, bkd: Backend) -> None:
        with pytest.raises(TypeError, match="MassMatrixProtocol"):
            GramProjectionEncoder(
                bkd.asarray([[1.0], [0.0]]), "not_a_mass", bkd
            )

    def test_rejects_wrong_encode_shape(self, bkd: Backend) -> None:
        encoder = GramProjectionEncoder(
            bkd.asarray([[1.0], [0.0]]), IdentityMassMatrix(2, bkd), bkd
        )
        with pytest.raises(ValueError, match="leading dimension"):
            encoder.encode(bkd.asarray([[1.0], [2.0], [3.0]]))

    def test_rejects_wrong_decode_shape(self, bkd: Backend) -> None:
        encoder = GramProjectionEncoder(
            bkd.asarray([[1.0], [0.0]]), IdentityMassMatrix(2, bkd), bkd
        )
        with pytest.raises(ValueError, match="leading dimension"):
            encoder.decode(bkd.asarray([[1.0], [2.0]]))

    def test_satisfies_protocol(self, bkd: Backend) -> None:
        encoder = GramProjectionEncoder(
            bkd.asarray([[1.0], [0.0]]), IdentityMassMatrix(2, bkd), bkd
        )
        assert isinstance(encoder, FieldEncoderProtocol)


class TestOrthonormalizeBasis:
    def test_makes_gram_identity(self, bkd: Backend) -> None:
        basis = bkd.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mass = IdentityMassMatrix(3, bkd)
        ortho = orthonormalize_basis(basis, mass, bkd)
        encoder = GramProjectionEncoder(ortho, mass, bkd)
        bkd.assert_allclose(encoder.gram(), bkd.eye(2), atol=1e-13)
        assert encoder.is_isometry()

    def test_preserves_span(self, bkd: Backend) -> None:
        """Orthonormalizing changes the basis but not the space it spans."""
        basis = bkd.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mass = IdentityMassMatrix(3, bkd)
        ortho = orthonormalize_basis(basis, mass, bkd)
        # Any field in the span round-trips exactly through the new basis.
        f = bkd.dot(basis, bkd.asarray([[2.0], [-1.0]]))
        encoder = GramProjectionEncoder(ortho, mass, bkd)
        bkd.assert_allclose(encoder.decode(encoder.encode(f)), f, atol=1e-13)

    def test_nondiagonal_mass(self, bkd: Backend) -> None:
        dense = bkd.asarray(
            [[2.0, 0.5, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 2.0]]
        )
        mass = ConstantDenseMassMatrix(dense, bkd)
        basis = bkd.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        ortho = orthonormalize_basis(basis, mass, bkd)
        encoder = GramProjectionEncoder(ortho, mass, bkd)
        bkd.assert_allclose(encoder.gram(), bkd.eye(2), atol=1e-13)


class TestProductFieldEncoder:
    def _parts(self, bkd: Backend) -> List[FieldEncoderProtocol]:
        return [IdentityFieldEncoder(2, bkd), IdentityFieldEncoder(3, bkd)]

    def test_dimensions_sum(self, bkd: Backend) -> None:
        product = ProductFieldEncoder(self._parts(bkd), bkd)
        assert product.ncodes() == 5
        assert product.ngrid() == 5
        assert product.nfields() == 2

    def test_roundtrip(self, bkd: Backend) -> None:
        product = ProductFieldEncoder(self._parts(bkd), bkd)
        f = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        bkd.assert_allclose(product.decode(product.encode(f)), f)

    def test_product_of_isometries_is_isometry(self, bkd: Backend) -> None:
        assert ProductFieldEncoder(self._parts(bkd), bkd).is_isometry()

    def test_not_isometry_if_any_part_is_not(self, bkd: Backend) -> None:
        parts = [
            IdentityFieldEncoder(2, bkd),
            IdentityFieldEncoder(3, bkd, is_isometry=False),
        ]
        assert not ProductFieldEncoder(parts, bkd).is_isometry()

    def test_norm_is_sum_of_field_norms(self, bkd: Backend) -> None:
        """||c||^2 == sum_j ||f_j||^2 for unit scalings."""
        product = ProductFieldEncoder(self._parts(bkd), bkd)
        f = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        codes = product.encode(f)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(codes**2)]),
            bkd.asarray([bkd.sum(f**2)]),
        )

    def test_scalings_roundtrip(self, bkd: Backend) -> None:
        """Scalings change the induced norm but decode still inverts encode."""
        product = ProductFieldEncoder(
            self._parts(bkd), bkd, scalings=bkd.asarray([4.0, 9.0])
        )
        f = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        bkd.assert_allclose(product.decode(product.encode(f)), f)

    def test_scalings_weight_the_norm(self, bkd: Backend) -> None:
        """||c||^2 == sum_j alpha_j ||f_j||^2."""
        product = ProductFieldEncoder(
            self._parts(bkd), bkd, scalings=bkd.asarray([4.0, 9.0])
        )
        f = bkd.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        codes = product.encode(f)
        expected = 4.0 * (1.0 + 4.0) + 9.0 * (9.0 + 16.0 + 25.0)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(codes**2)]), bkd.asarray([expected])
        )

    def test_single_field_matches_bare_encoder(self, bkd: Backend) -> None:
        """Wrapping one encoder changes nothing."""
        bare = IdentityFieldEncoder(3, bkd)
        product = ProductFieldEncoder([bare], bkd)
        f = bkd.asarray([[1.0], [2.0], [3.0]])
        bkd.assert_allclose(product.encode(f), bare.encode(f))

    def test_composes_heterogeneous_encoders(self, bkd: Backend) -> None:
        """A product may mix encoder types."""
        ngrid, ncodes = 32, 3
        gram = GramProjectionEncoder(
            _fourier_basis(bkd, ngrid, ncodes), _uniform_mass(bkd, ngrid), bkd
        )
        product = ProductFieldEncoder([gram, IdentityFieldEncoder(2, bkd)], bkd)
        assert product.ncodes() == ncodes + 2
        assert product.ngrid() == ngrid + 2
        assert product.is_isometry()

    def test_rejects_empty(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            ProductFieldEncoder([], bkd)

    def test_rejects_non_encoder(self, bkd: Backend) -> None:
        with pytest.raises(TypeError, match="FieldEncoderProtocol"):
            ProductFieldEncoder(["not_an_encoder"], bkd)

    def test_rejects_wrong_scaling_shape(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="wrong shape"):
            ProductFieldEncoder(
                self._parts(bkd), bkd, scalings=bkd.asarray([1.0])
            )

    def test_rejects_nonpositive_scaling(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ProductFieldEncoder(
                self._parts(bkd), bkd, scalings=bkd.asarray([1.0, 0.0])
            )

    def test_satisfies_protocol(self, bkd: Backend) -> None:
        product = ProductFieldEncoder(self._parts(bkd), bkd)
        assert isinstance(product, FieldEncoderProtocol)
