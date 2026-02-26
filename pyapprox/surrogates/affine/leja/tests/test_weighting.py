"""Tests for Leja weighting strategies."""

from scipy import stats

from pyapprox.probability import ScipyContinuousMarginal


class TestChristoffelWeighting:
    """Tests for ChristoffelWeighting."""

    def test_weights_shape(self, bkd) -> None:
        """Test that weights have correct shape."""
        from pyapprox.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        assert weights.shape == (3, 1)

    def test_weights_positive(self, bkd) -> None:
        """Test that weights are positive."""
        from pyapprox.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        assert bkd.all_bool(weights > 0)

    def test_jacobian_shape(self, bkd) -> None:
        """Test that Jacobian has correct shape."""
        from pyapprox.surrogates.affine.leja import ChristoffelWeighting

        weighting = ChristoffelWeighting(bkd)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        basis_jac = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        jac = weighting.jacobian(samples, basis_values, basis_jac)

        assert jac.shape == (3, 1)


class TestPDFWeighting:
    """Tests for PDFWeighting."""

    def test_weights_shape(self, bkd) -> None:
        """Test that PDF weights have correct shape."""
        from pyapprox.surrogates.affine.leja import PDFWeighting

        # Use typing wrapper for scipy distribution
        rv = ScipyContinuousMarginal(stats.uniform(-1, 2), bkd)

        # PDFWeighting expects a callable that returns backend arrays
        # ScipyContinuousMarginal uses __call__ for PDF (FunctionProtocol)
        # Input shape: (1, nsamples), output shape: (1, nsamples)
        def pdf_callable(samples):
            return rv(bkd.reshape(samples, (1, -1)))[0, :]

        weighting = PDFWeighting(bkd, pdf_callable)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        assert weights.shape == (3, 1)

    def test_weights_match_pdf(self, bkd) -> None:
        """Test that weights match the PDF values."""
        from pyapprox.surrogates.affine.leja import PDFWeighting

        # Use typing wrapper for scipy distribution
        rv = ScipyContinuousMarginal(stats.norm(0, 1), bkd)

        # PDFWeighting expects a callable that returns backend arrays
        # ScipyContinuousMarginal uses __call__ for PDF (FunctionProtocol)
        # Input shape: (1, nsamples), output shape: (1, nsamples)
        def pdf_callable(samples):
            return rv(bkd.reshape(samples, (1, -1)))[0, :]

        weighting = PDFWeighting(bkd, pdf_callable)
        samples = bkd.asarray([[0.0, 0.5, 1.0]])
        basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
        weights = weighting(samples, basis_values)

        # Get expected PDF values using the typed distribution
        # rv() expects (1, nsamples) and returns (1, nsamples)
        expected = rv(samples)[0, :]
        bkd.assert_allclose(weights[:, 0], expected, rtol=1e-10)
