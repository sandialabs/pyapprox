"""Tests for PrecomputedKLE.

The class exists so a computed basis can outlive the objects that
built it, so the tests that matter are the ones showing a
PrecomputedKLE assembled from another KLE's arrays evaluates
identically to it. Exact equality is the right assertion there: the
same arrays go through the same arithmetic, so any difference is a
defect rather than rounding.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.surrogates.kle.nystrom_kle import create_nystrom_kle
from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.surrogates.kle.protocols import KLEProtocol


def _coords(bkd, npoints=60):
    return bkd.array(np.linspace(0.0, 1.0, npoints)[None, :])


def _kernel(bkd, lenscale=0.3):
    return ExponentialKernel(bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd)


def _coef(bkd, nterms, nsamples=4, seed=0):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.standard_normal((nterms, nsamples)))


def _from_kle(kle, bkd, mean_field=None, **kwargs):
    """Rebuild a KLE as a PrecomputedKLE from its own arrays."""
    if mean_field is None:
        mean_field = bkd.zeros((kle.eigenvectors().shape[0],))
    return PrecomputedKLE(
        kle.eigenvalues(),
        kle.eigenvectors(),
        mean_field,
        bkd=bkd,
        **kwargs,
    )


class TestReproducesItsSource:
    """A rebuilt basis must evaluate as the one it came from."""

    def test_matches_mesh_kle(self, bkd) -> None:
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(coords, kernel, nterms=6, bkd=bkd)
        rebuilt = _from_kle(source, bkd)
        coef = _coef(bkd, 6)
        bkd.assert_allclose(rebuilt(coef), source(coef), rtol=0.0, atol=0.0)

    def test_matches_mesh_kle_with_mean_and_sigma(self, bkd) -> None:
        """sigma and a non-zero mean must be carried, not assumed away."""
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(
            coords, kernel, sigma=2.5, mean_field=3.0, nterms=5, bkd=bkd
        )
        rebuilt = PrecomputedKLE(
            source.eigenvalues(),
            source.eigenvectors(),
            source.mean_field(),
            sigma=2.5,
            bkd=bkd,
        )
        coef = _coef(bkd, 5)
        bkd.assert_allclose(rebuilt(coef), source(coef), rtol=0.0, atol=0.0)

    def test_matches_mesh_kle_with_use_log(self, bkd) -> None:
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(
            coords, kernel, nterms=5, use_log=True, mean_field=0.5, bkd=bkd
        )
        rebuilt = PrecomputedKLE(
            source.eigenvalues(),
            source.eigenvectors(),
            source.mean_field(),
            use_log=True,
            bkd=bkd,
        )
        coef = _coef(bkd, 5)
        bkd.assert_allclose(rebuilt(coef), source(coef), rtol=0.0, atol=0.0)

    def test_matches_nystrom_at_extended_points(self, bkd) -> None:
        """The case the whole design rests on.

        A Nystrom basis extended to points outside the collocation set
        is just a taller array, so a PrecomputedKLE built from it must
        reproduce ``evaluate_at`` there. If this fails, storing an
        extended basis is not a valid thing to do and the class needs
        the kernel after all.
        """
        coords, kernel = _coords(bkd, 100), _kernel(bkd)
        source = create_nystrom_kle(
            kernel, coords, 6, bkd, nlandmarks=40, sigma=1.5
        )
        # Midpoints: in neither the landmark nor the collocation set.
        grid = np.linspace(0.0, 1.0, 100)
        query = bkd.array(((grid[:-1] + grid[1:]) / 2)[None, :])
        rebuilt = PrecomputedKLE(
            source.eigenvalues(),
            source.eigenvectors_at(query),
            bkd.zeros((query.shape[1],)),
            sigma=1.5,
            bkd=bkd,
        )
        coef = _coef(bkd, 6)
        bkd.assert_allclose(
            rebuilt(coef),
            source.evaluate_at(query, coef),
            rtol=0.0,
            atol=0.0,
        )

    def test_row_count_need_not_match_the_source(self, bkd) -> None:
        """A stored basis is defined by its own rows, not its origin.

        The Nystrom case above extends 40 landmarks to 99 query points,
        so npoints and the source's collocation count genuinely differ.
        Pinning it separately keeps the property from being read as an
        accident of that test's fixture.
        """
        coords, kernel = _coords(bkd, 100), _kernel(bkd)
        source = create_nystrom_kle(kernel, coords, 6, bkd, nlandmarks=40)
        query = bkd.array(np.linspace(0.05, 0.95, 17)[None, :])
        rebuilt = PrecomputedKLE(
            source.eigenvalues(),
            source.eigenvectors_at(query),
            bkd.zeros((17,)),
            bkd=bkd,
        )
        assert rebuilt.npoints() == 17
        assert rebuilt.nterms() == 6
        assert rebuilt(_coef(bkd, 6)).shape == (17, 4)


class TestProtocol:
    """It must be usable anywhere a KLE is consumed."""

    def test_satisfies_kle_protocol(self, bkd) -> None:
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(coords, kernel, nterms=4, bkd=bkd)
        assert isinstance(_from_kle(source, bkd), KLEProtocol)

    def test_accessors_round_trip(self, bkd) -> None:
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(coords, kernel, nterms=4, bkd=bkd)
        rebuilt = _from_kle(source, bkd, sigma=2.0, use_log=True)
        bkd.assert_allclose(rebuilt.eigenvalues(), source.eigenvalues())
        bkd.assert_allclose(rebuilt.eigenvectors(), source.eigenvectors())
        assert rebuilt.nterms() == 4
        assert rebuilt.sigma() == 2.0
        assert rebuilt.use_log() is True
        assert rebuilt.bkd() is bkd

    def test_weighted_eigenvectors_fold_in_sigma(self, bkd) -> None:
        """weighted = unweighted * sqrt(lambda) * sigma.

        Pinned because the save/load layer stores the *unweighted*
        form, and would silently double-apply the scaling if these two
        were ever confused.
        """
        coords, kernel = _coords(bkd), _kernel(bkd)
        source = MeshKLE(coords, kernel, nterms=4, bkd=bkd)
        rebuilt = _from_kle(source, bkd, sigma=3.0)
        expected = (
            source.eigenvectors() * bkd.sqrt(source.eigenvalues()) * 3.0
        )
        bkd.assert_allclose(rebuilt.weighted_eigenvectors(), expected)


class TestValidation:
    """Inconsistent arrays are refused where the mistake was made.

    Each of these would otherwise surface as a broadcasting error
    inside a matmul at evaluation time, reporting shapes that have
    already been combined rather than the array that was wrong.
    """

    def _valid(self, bkd, npoints=10, nterms=3):
        return (
            bkd.array(np.linspace(1.0, 0.1, nterms)),
            bkd.array(np.random.RandomState(0).standard_normal(
                (npoints, nterms))),
            bkd.zeros((npoints,)),
        )

    def test_accepts_consistent_arrays(self, bkd) -> None:
        """The negative cases below mean nothing without this."""
        vals, vecs, mean = self._valid(bkd)
        assert PrecomputedKLE(vals, vecs, mean, bkd=bkd).nterms() == 3

    def test_rejects_column_count_mismatch(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="columns"):
            PrecomputedKLE(vals[:2], vecs, mean, bkd=bkd)

    def test_rejects_row_count_mismatch(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="same points"):
            PrecomputedKLE(vals, vecs, mean[:5], bkd=bkd)

    def test_rejects_1d_eigenvectors(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="must be 2D"):
            PrecomputedKLE(vals[:1], vecs[:, 0], mean, bkd=bkd)

    def test_rejects_2d_eigenvalues(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="must be 1D"):
            PrecomputedKLE(vals[:, None], vecs, mean, bkd=bkd)

    def test_rejects_2d_mean_field(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="mean_field must be 1D"):
            PrecomputedKLE(vals, vecs, mean[:, None], bkd=bkd)

    def test_rejects_negative_eigenvalues(self, bkd) -> None:
        """The expansion takes sqrt of each eigenvalue.

        An unchecked negative becomes NaN and reaches every field
        evaluation, so it is refused at construction rather than
        discovered in output.
        """
        vals, vecs, mean = self._valid(bkd)
        bad = bkd.array(np.array([1.0, 0.5, -1e-3]))
        with pytest.raises(ValueError, match="non-negative"):
            PrecomputedKLE(bad, vecs, mean, bkd=bkd)

    def test_requires_a_backend(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        with pytest.raises(ValueError, match="bkd"):
            PrecomputedKLE(vals, vecs, mean)

    def test_rejects_wrong_coef_shape(self, bkd) -> None:
        vals, vecs, mean = self._valid(bkd)
        kle = PrecomputedKLE(vals, vecs, mean, bkd=bkd)
        with pytest.raises(ValueError, match="ndim"):
            kle(bkd.array(np.zeros(3)))
        with pytest.raises(ValueError, match="nterms"):
            kle(bkd.array(np.zeros((5, 2))))
