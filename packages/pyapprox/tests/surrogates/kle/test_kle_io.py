"""Tests for saving and reloading a KLE basis.

The point of the format is that a basis outlives the objects that
built it, so the assertions are about what a reloaded KLE *computes*
rather than about the bytes. Realization equality is asserted exactly:
the same arrays go through the same arithmetic, so any difference is a
defect rather than rounding.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.io import (
    SCHEMA_VERSION,
    load_kle,
    save_kle,
)
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


class TestRoundTrip:
    """A reloaded basis must compute what the saved one did."""

    def test_mesh_kle(self, bkd, tmp_path) -> None:
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=6, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        coef = _coef(bkd, 6)
        bkd.assert_allclose(
            load_kle(path, bkd)(coef), source(coef), rtol=0.0, atol=0.0
        )

    def test_mesh_kle_with_mean(self, bkd, tmp_path) -> None:
        source = MeshKLE(
            _coords(bkd), _kernel(bkd), mean_field=3.0, nterms=5, bkd=bkd
        )
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        coef = _coef(bkd, 5)
        bkd.assert_allclose(
            load_kle(path, bkd)(coef), source(coef), rtol=0.0, atol=0.0
        )

    def test_nystrom_kle_at_its_landmarks(self, bkd, tmp_path) -> None:
        """A Nystrom KLE stores the basis where its own accessors are.

        ``eigenvectors()`` and ``mean_field()`` both report at the
        landmarks, so a reload reproduces ``__call__`` rather than
        ``evaluate_at`` somewhere else.
        """
        source = create_nystrom_kle(
            _kernel(bkd), _coords(bkd, 100), 6, bkd, nlandmarks=40
        )
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        coef = _coef(bkd, 6)
        bkd.assert_allclose(
            load_kle(path, bkd)(coef), source(coef), rtol=0.0, atol=0.0
        )

    def test_nystrom_basis_extended_before_saving(
        self, bkd, tmp_path
    ) -> None:
        """The intended workflow for storing a basis somewhere new.

        A loaded KLE cannot extend itself, so extension happens first
        and the result is what gets stored. This is the path a caller
        takes to persist a basis at quadrature points.
        """
        source = create_nystrom_kle(
            _kernel(bkd), _coords(bkd, 100), 6, bkd, nlandmarks=40
        )
        grid = np.linspace(0.0, 1.0, 100)
        query = bkd.array(((grid[:-1] + grid[1:]) / 2)[None, :])
        extended = PrecomputedKLE(
            source.eigenvalues(),
            source.eigenvectors_at(query),
            source.mean_field_at(query),
            bkd=bkd,
        )
        path = tmp_path / "kle.npz"
        save_kle(path, extended)
        coef = _coef(bkd, 6)
        bkd.assert_allclose(
            load_kle(path, bkd)(coef),
            source.evaluate_at(query, coef),
            rtol=0.0,
            atol=0.0,
        )

    def test_precomputed_kle_is_idempotent(self, bkd, tmp_path) -> None:
        """Saving a loaded basis again must change nothing."""
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=5, bkd=bkd)
        first, second = tmp_path / "a.npz", tmp_path / "b.npz"
        save_kle(first, source)
        once = load_kle(first, bkd)
        save_kle(second, once)
        twice = load_kle(second, bkd)
        coef = _coef(bkd, 5)
        bkd.assert_allclose(twice(coef), once(coef), rtol=0.0, atol=0.0)
        bkd.assert_allclose(
            twice.eigenvectors(), once.eigenvectors(), rtol=0.0, atol=0.0
        )

    def test_arrays_survive_unchanged(self, bkd, tmp_path) -> None:
        """The basis itself round-trips bit for bit.

        Stronger than realization equality, and the reason the
        unweighted basis is the stored form: writing and reading is a
        copy, with no arithmetic to lose a bit in.
        """
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=6, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        loaded = load_kle(path, bkd)
        bkd.assert_allclose(
            loaded.eigenvalues(), source.eigenvalues(), rtol=0.0, atol=0.0
        )
        bkd.assert_allclose(
            loaded.eigenvectors(), source.eigenvectors(), rtol=0.0, atol=0.0
        )
        bkd.assert_allclose(
            loaded.mean_field(), source.mean_field(), rtol=0.0, atol=0.0
        )


class TestScalars:
    """sigma and use_log are stated by the caller, not inferred."""

    def test_sigma_is_recorded(self, bkd, tmp_path) -> None:
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=5, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source, sigma=2.5)
        assert load_kle(path, bkd).sigma() == 2.5

    def test_sigma_scales_realizations(self, bkd, tmp_path) -> None:
        """The stored scalar must reach the arithmetic.

        Recording sigma and then not applying it would leave
        ``sigma()`` correct while every realization was wrong, which is
        how a defaulted sigma went unnoticed before it was measured.
        """
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=5, bkd=bkd)
        plain, scaled = tmp_path / "a.npz", tmp_path / "b.npz"
        save_kle(plain, source)
        save_kle(scaled, source, sigma=3.0)
        coef = _coef(bkd, 5)
        # Not exact: the scaled basis folds sigma in before the matmul
        # while the reference multiplies after it, so the two differ by
        # the order of a floating-point multiply.
        bkd.assert_allclose(
            load_kle(scaled, bkd)(coef),
            load_kle(plain, bkd)(coef) * 3.0,
            rtol=1e-13,
            atol=1e-15,
        )

    def test_use_log_is_recorded_and_applied(self, bkd, tmp_path) -> None:
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=5, bkd=bkd)
        plain, logged = tmp_path / "a.npz", tmp_path / "b.npz"
        save_kle(plain, source)
        save_kle(logged, source, use_log=True)
        loaded = load_kle(logged, bkd)
        assert loaded.use_log() is True
        coef = _coef(bkd, 5)
        bkd.assert_allclose(
            loaded(coef),
            bkd.exp(load_kle(plain, bkd)(coef)),
            rtol=1e-14,
            atol=0.0,
        )

    def test_defaults_are_the_neutral_values(self, bkd, tmp_path) -> None:
        """Omitting them must mean "no scaling, no exponential"."""
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=4, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        loaded = load_kle(path, bkd)
        assert loaded.sigma() == 1.0
        assert loaded.use_log() is False


class TestLoadedObject:
    """What comes back is a usable KLE."""

    def test_satisfies_kle_protocol(self, bkd, tmp_path) -> None:
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=4, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        assert isinstance(load_kle(path, bkd), KLEProtocol)

    def test_loads_under_either_backend(self, bkd, tmp_path) -> None:
        """The archive is numpy, so it is backend-neutral.

        Saved under whichever backend the fixture supplies and loaded
        under both, so the cross-backend direction is covered whichever
        way round the parametrization runs.
        """
        from pyapprox.util.backends.numpy import NumpyBkd

        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=5, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        native = load_kle(path, bkd)
        numpy_bkd = NumpyBkd()
        as_numpy = load_kle(path, numpy_bkd)
        coef = np.random.RandomState(0).standard_normal((5, 3))
        np.testing.assert_allclose(
            bkd.to_numpy(native(bkd.array(coef))),
            numpy_bkd.to_numpy(as_numpy(numpy_bkd.array(coef))),
            rtol=1e-14,
        )


class TestRejects:
    """A malformed or foreign archive fails where it is read."""

    def test_rejects_non_kle(self, bkd, tmp_path) -> None:
        with pytest.raises(TypeError, match="KLEProtocol"):
            save_kle(tmp_path / "kle.npz", object())

    def test_rejects_unknown_schema_version(self, bkd, tmp_path) -> None:
        """A future format must not be read as though it were this one.

        Without the check the arrays would be handed to PrecomputedKLE
        under whatever meaning this version assumes, which is how a
        format change turns into wrong numbers instead of an error.
        """
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=4, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        with np.load(path) as archive:
            entries = dict(archive)
        entries["schema_version"] = SCHEMA_VERSION + 1
        np.savez(path, **entries)
        with pytest.raises(ValueError, match="schema version"):
            load_kle(path, bkd)

    def test_rejects_missing_entry(self, bkd, tmp_path) -> None:
        source = MeshKLE(_coords(bkd), _kernel(bkd), nterms=4, bkd=bkd)
        path = tmp_path / "kle.npz"
        save_kle(path, source)
        with np.load(path) as archive:
            entries = dict(archive)
        del entries["mean_field"]
        np.savez(path, **entries)
        with pytest.raises(ValueError, match="mean_field"):
            load_kle(path, bkd)

    def test_rejects_foreign_archive(self, bkd, tmp_path) -> None:
        """An npz written by something else is not a KLE."""
        path = tmp_path / "other.npz"
        np.savez(path, something=np.zeros(3))
        with pytest.raises(ValueError, match="schema_version"):
            load_kle(path, bkd)
