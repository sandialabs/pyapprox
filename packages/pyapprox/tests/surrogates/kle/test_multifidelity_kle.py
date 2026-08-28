"""Tests for extending one Nystrom basis to several meshes.

The property under test is that a single eigensolve can be sampled on
point sets it was not solved on, and that every sample expands the *same*
field: one coefficient vector, one realization, several resolutions. That
is what lets a multifidelity estimator attribute differences between
fidelities to discretization rather than to bases that disagree.

Meshes here are bare coordinate arrays. Nothing in the module under test
knows what a mesh is, so nothing in these tests builds one; the
integration tier covers real meshes.

Equality against ``evaluate_at`` is asserted exactly: both paths run the
same arrays through the same arithmetic, so any difference is a defect
rather than rounding.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.multifidelity import (
    nystrom_kle_on_mesh,
    nystrom_kles_on_meshes,
)
from pyapprox.surrogates.kle.nystrom_kle import create_nystrom_kle
from pyapprox.surrogates.kle.precomputed_kle import PrecomputedKLE
from pyapprox.surrogates.kle.protocols import KLEProtocol


def _kernel(bkd, lenscale=0.3):
    return ExponentialKernel(bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd)


def _coords(bkd, npoints, lo=0.0, hi=1.0):
    return bkd.array(np.linspace(lo, hi, npoints)[None, :])


def _coef(bkd, nterms, nsamples=3, seed=0):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.standard_normal((nterms, nsamples)))


def _nystrom(bkd, nterms=5, npoints=60, **kwargs):
    return create_nystrom_kle(
        _kernel(bkd), _coords(bkd, npoints), nterms, bkd,
        nlandmarks=25, **kwargs,
    )


class TestSingleMesh:
    """One extension, checked against the basis it came from."""

    def test_result_satisfies_kle_protocol(self, bkd) -> None:
        kle = nystrom_kle_on_mesh(_nystrom(bkd), _coords(bkd, 17))
        assert isinstance(kle, KLEProtocol)
        assert isinstance(kle, PrecomputedKLE)

    def test_matches_evaluate_at(self, bkd) -> None:
        """The whole point: a plain KLE that computes what Nystrom does."""
        nystrom = _nystrom(bkd)
        coords = _coords(bkd, 17)
        coef = _coef(bkd, nystrom.nterms())
        bkd.assert_allclose(
            nystrom_kle_on_mesh(nystrom, coords)(coef),
            nystrom.evaluate_at(coords, coef),
            rtol=0.0,
            atol=0.0,
        )

    def test_carries_the_eigenvalues_over(self, bkd) -> None:
        nystrom = _nystrom(bkd)
        kle = nystrom_kle_on_mesh(nystrom, _coords(bkd, 17))
        bkd.assert_allclose(kle.eigenvalues(), nystrom.eigenvalues())
        assert kle.nterms() == nystrom.nterms()

    def test_basis_has_a_row_per_point(self, bkd) -> None:
        kle = nystrom_kle_on_mesh(_nystrom(bkd, nterms=5), _coords(bkd, 17))
        assert kle.eigenvectors().shape == (17, 5)

    def test_at_the_landmarks_reproduces_the_landmark_basis(
        self, bkd
    ) -> None:
        """Extension to where the basis already lives changes nothing."""
        nystrom = _nystrom(bkd)
        kle = nystrom_kle_on_mesh(nystrom, nystrom.landmark_coords())
        bkd.assert_allclose(
            kle.eigenvectors(), nystrom.eigenvectors(), rtol=1e-12
        )

    def test_sigma_is_carried_over(self, bkd) -> None:
        nystrom = _nystrom(bkd, sigma=2.5)
        coords = _coords(bkd, 17)
        coef = _coef(bkd, nystrom.nterms())
        bkd.assert_allclose(
            nystrom_kle_on_mesh(nystrom, coords)(coef),
            nystrom.evaluate_at(coords, coef),
            rtol=0.0,
            atol=0.0,
        )

    def test_use_log_is_carried_over(self, bkd) -> None:
        nystrom = _nystrom(bkd, use_log=True, mean_field=0.5)
        coords = _coords(bkd, 17)
        coef = _coef(bkd, nystrom.nterms())
        values = nystrom_kle_on_mesh(nystrom, coords)(coef)
        assert bool(bkd.all_bool(values > 0.0))
        bkd.assert_allclose(
            values, nystrom.evaluate_at(coords, coef), rtol=0.0, atol=0.0
        )

    def test_mean_field_is_carried_over(self, bkd) -> None:
        nystrom = _nystrom(bkd, mean_field=3.0)
        kle = nystrom_kle_on_mesh(nystrom, _coords(bkd, 17))
        bkd.assert_allclose(
            kle.mean_field(), bkd.full((17,), 3.0), rtol=1e-12
        )


class TestSeveralMeshes:
    """The multifidelity property: one coefficient vector, one field."""

    def test_returns_one_kle_per_mesh_in_order(self, bkd) -> None:
        meshes = [_coords(bkd, n) for n in (9, 17, 33)]
        kles = nystrom_kles_on_meshes(_nystrom(bkd), meshes)
        assert len(kles) == 3
        assert [kle.eigenvectors().shape[0] for kle in kles] == [9, 17, 33]

    def test_every_mesh_matches_evaluate_at(self, bkd) -> None:
        nystrom = _nystrom(bkd)
        meshes = [_coords(bkd, n) for n in (9, 17, 33)]
        coef = _coef(bkd, nystrom.nterms())
        for kle, coords in zip(nystrom_kles_on_meshes(nystrom, meshes),
                               meshes):
            bkd.assert_allclose(
                kle(coef), nystrom.evaluate_at(coords, coef),
                rtol=0.0, atol=0.0,
            )

    def test_meshes_agree_where_they_share_a_point(self, bkd) -> None:
        """One field sampled twice, not two fields that resemble each other.

        The endpoints are the only coordinates these two point sets have
        in common; the field there must be identical, since it is the
        same eigenfunction evaluated at the same x.
        """
        nystrom = _nystrom(bkd)
        coarse, fine = _coords(bkd, 9), _coords(bkd, 33)
        coef = _coef(bkd, nystrom.nterms())
        kle_coarse, kle_fine = nystrom_kles_on_meshes(
            nystrom, [coarse, fine]
        )
        values_coarse, values_fine = kle_coarse(coef), kle_fine(coef)
        for icoarse, ifine in ((0, 0), (-1, -1)):
            bkd.assert_allclose(
                values_coarse[icoarse], values_fine[ifine], rtol=1e-12
            )

    def test_shares_the_spectrum_across_meshes(self, bkd) -> None:
        nystrom = _nystrom(bkd)
        kles = nystrom_kles_on_meshes(
            nystrom, [_coords(bkd, n) for n in (9, 33)]
        )
        bkd.assert_allclose(kles[0].eigenvalues(), kles[1].eigenvalues())

    def test_accepts_no_meshes(self, bkd) -> None:
        assert nystrom_kles_on_meshes(_nystrom(bkd), []) == []


class TestRejects:
    """Mistakes that would otherwise produce plausible wrong numbers."""

    def test_rejects_non_nystrom(self, bkd) -> None:
        frozen = nystrom_kle_on_mesh(_nystrom(bkd), _coords(bkd, 17))
        with pytest.raises(TypeError, match="must be a NystromKLE"):
            nystrom_kle_on_mesh(frozen, _coords(bkd, 9))

    def test_rejects_1d_coords(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be 2D"):
            nystrom_kle_on_mesh(
                _nystrom(bkd), bkd.linspace(0.0, 1.0, 9)
            )

    def test_rejects_wrong_spatial_dimension(self, bkd) -> None:
        coords2d = bkd.array(np.zeros((2, 9)))
        with pytest.raises(ValueError, match="spatial dimension"):
            nystrom_kle_on_mesh(_nystrom(bkd), coords2d)
