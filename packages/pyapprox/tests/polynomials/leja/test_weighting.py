"""Tests for Leja weighting strategies."""

import pytest
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

    def test_accepts_a_marginal_pdf_directly(self, bkd) -> None:
        """A marginal's own pdf is a valid argument, unwrapped.

        ``marginal.pdf`` requires (1, nsamples) and rejects 1D input,
        so passing it is what pins the calling convention. A wrapper
        that reshapes would accept either and prove nothing.
        """
        from pyapprox.probability import UniformMarginal
        from pyapprox.surrogates.affine.leja import PDFWeighting

        marginal = UniformMarginal(0.0, 1.0, bkd)
        weighting = PDFWeighting(bkd, marginal.pdf)
        samples = bkd.asarray([[0.1, 0.5, 0.9]])
        basis_values = bkd.asarray([[1.0], [1.0], [1.0]])

        weights = weighting(samples, basis_values)
        assert weights.shape == (3, 1)
        # Uniform on [0, 1] has density 1 everywhere inside it.
        bkd.assert_allclose(
            weights[:, 0], bkd.asarray([1.0, 1.0, 1.0]), rtol=1e-12
        )

    def test_rejects_samples_that_are_not_a_single_row(self, bkd) -> None:
        """Univariate weighting: anything but (1, nsamples) is a bug.

        Passing 1D samples reaches the pdf as an array it cannot
        interpret, and the error it raises names the pdf rather than
        the caller, so the shape is checked here instead.
        """
        from pyapprox.probability import UniformMarginal
        from pyapprox.surrogates.affine.leja import PDFWeighting

        weighting = PDFWeighting(
            bkd, UniformMarginal(0.0, 1.0, bkd).pdf
        )
        basis_values = bkd.asarray([[1.0], [1.0]])
        for bad in (
            bkd.asarray([0.1, 0.5]),
            bkd.asarray([[0.1, 0.5], [0.2, 0.6]]),
        ):
            with pytest.raises(ValueError, match=r"\(1, nsamples\)"):
                weighting(bad, basis_values)

    def test_rejects_a_pdf_returning_the_wrong_count(self, bkd) -> None:
        """One weight per sample, or the sequence is silently wrong."""
        from pyapprox.surrogates.affine.leja import PDFWeighting

        weighting = PDFWeighting(bkd, lambda s: bkd.asarray([1.0]))
        with pytest.raises(ValueError, match="one value per sample"):
            weighting(
                bkd.asarray([[0.1, 0.5, 0.9]]), bkd.asarray([[1.0]])
            )
