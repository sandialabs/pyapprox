"""A random field solved once and reused across real 2D meshes.

Crosses ``pyapprox.surrogates.kle`` and ``pyapprox.pde.galerkin.mesh``, so
it lives in the integration tier: the unit tests for this capability use
bare coordinate arrays, because the library deliberately knows nothing
about meshes. What is only checkable here is that the workflow survives
contact with an actual mesh type -- solve on the fine mesh, persist,
reload later, and extend to meshes that did not exist at solve time.

**The meshes are chosen NOT to be nested.** On ``[0, 1]^2`` the node sets
of ``nx = ny = 2, 4, 8`` are exact subsets of one another, so a coarse
evaluation would only ever ask the extension for points it already holds
and the off-mesh path -- the entire reason Nystrom is used here -- would
go untested. Against an 8x8 mesh, 5x5 and 7x7 share only the four domain
corners, leaving 89% and 94% of their nodes genuinely new.

Non-nested does not mean approximate. Nystrom evaluates the eigenfunction
directly as ``phi(x) = C(x, S) T``, which is exact at any point, so the
assertions below are still exact equality; the coarse meshes simply
exercise arithmetic the nested ones would skip.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle import (
    create_nystrom_kle,
    load_nystrom_kle,
    nystrom_kles_on_meshes,
    save_nystrom_kle,
)
from pyapprox.surrogates.kle.protocols import KLEProtocol
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from pyapprox.pde.galerkin.mesh.structured import (  # noqa: E402
    StructuredMesh2D,
)

# 8 is the high-fidelity mesh; 5 and 7 are non-nested against it (see the
# module docstring). 2 and 4 would be exact subsets and are avoided.
FINE, COARSE = 8, (5, 7)


def _nodes(bkd, n):
    """Return the ``(2, nnodes)`` nodal coordinates of an n-by-n square."""
    mesh = StructuredMesh2D(
        nx=n, ny=n, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
    )
    return mesh.nodes()


def _kernel(bkd, lenscale=0.4):
    return ExponentialKernel(bkd.full((2,), lenscale), (0.01, 100.0), 2, bkd)


def _coef(bkd, nterms, nsamples=3, seed=0):
    rng = np.random.RandomState(seed)
    return bkd.array(rng.standard_normal((nterms, nsamples)))


def _shared_columns(bkd, left, right, tol=1e-12):
    """Index pairs where two coordinate arrays hold the same point."""
    a = np.asarray(bkd.to_numpy(left))
    b = np.asarray(bkd.to_numpy(right))
    pairs = []
    for i in range(a.shape[1]):
        hits = np.nonzero(np.abs(b - a[:, i : i + 1]).max(axis=0) < tol)[0]
        pairs.extend((i, int(j)) for j in hits)
    return pairs


class TestNonNestedMeshes:
    """The mesh choice itself, asserted rather than assumed."""

    def test_coarse_meshes_are_not_nested_in_the_fine_one(self, bkd) -> None:
        """Guards the premise: nesting here would void the tests below."""
        fine = _nodes(bkd, FINE)
        for n in COARSE:
            coarse = _nodes(bkd, n)
            shared = len(_shared_columns(bkd, coarse, fine))
            npts = int(coarse.shape[1])
            assert shared < npts, (
                f"{n}x{n} is nested in {FINE}x{FINE}; the extension would "
                "never be evaluated off the solve mesh"
            )
            assert shared / npts < 0.2

    def test_nested_sizes_would_have_been_nested(self, bkd) -> None:
        """Documents the trap: 4x4 *is* a subset of 8x8."""
        fine, nested = _nodes(bkd, FINE), _nodes(bkd, 4)
        assert len(_shared_columns(bkd, nested, fine)) == int(
            nested.shape[1]
        )


class TestSolveOnceReuseEverywhere:
    """One eigensolve on the fine mesh, sampled on all three."""

    def test_every_mesh_reproduces_the_source_basis(self, bkd) -> None:
        nystrom = create_nystrom_kle(
            _kernel(bkd), _nodes(bkd, FINE), 6, bkd, nlandmarks=30
        )
        meshes = [_nodes(bkd, n) for n in (FINE,) + COARSE]
        coef = _coef(bkd, nystrom.nterms())
        for kle, coords in zip(nystrom_kles_on_meshes(nystrom, meshes),
                               meshes):
            assert isinstance(kle, KLEProtocol)
            bkd.assert_allclose(
                kle(coef),
                nystrom.evaluate_at(coords, coef),
                rtol=0.0,
                atol=0.0,
            )

    def test_meshes_agree_at_the_corners_they_share(self, bkd) -> None:
        """One field sampled three ways, not three similar fields."""
        nystrom = create_nystrom_kle(
            _kernel(bkd), _nodes(bkd, FINE), 6, bkd, nlandmarks=30
        )
        fine_coords = _nodes(bkd, FINE)
        coef = _coef(bkd, nystrom.nterms())
        fine_values = nystrom_kles_on_meshes(nystrom, [fine_coords])[0](coef)
        for n in COARSE:
            coords = _nodes(bkd, n)
            values = nystrom_kles_on_meshes(nystrom, [coords])[0](coef)
            shared = _shared_columns(bkd, coords, fine_coords)
            assert shared, f"{n}x{n} shares no node with the fine mesh"
            for icoarse, ifine in shared:
                bkd.assert_allclose(
                    values[icoarse], fine_values[ifine], rtol=1e-12
                )


class TestSolveNowBindLater:
    """Persist the basis, then extend to meshes it never saw."""

    def test_reloaded_basis_extends_to_new_meshes(self, bkd, tmp_path):
        nystrom = create_nystrom_kle(
            _kernel(bkd), _nodes(bkd, FINE), 6, bkd, nlandmarks=30
        )
        path = tmp_path / "field.npz"
        save_nystrom_kle(path, nystrom)

        # Everything below stands in for a later session: the meshes are
        # built after the save, from an object that never met them.
        loaded = load_nystrom_kle(path, _kernel(bkd), bkd)
        meshes = [_nodes(bkd, n) for n in COARSE]
        coef = _coef(bkd, loaded.nterms())
        for kle, coords in zip(nystrom_kles_on_meshes(loaded, meshes),
                               meshes):
            bkd.assert_allclose(
                kle(coef),
                nystrom.evaluate_at(coords, coef),
                rtol=0.0,
                atol=0.0,
            )

    def test_reload_matches_the_presave_object_on_every_mesh(
        self, bkd, tmp_path
    ) -> None:
        nystrom = create_nystrom_kle(
            _kernel(bkd), _nodes(bkd, FINE), 6, bkd, nlandmarks=30
        )
        path = tmp_path / "field.npz"
        save_nystrom_kle(path, nystrom)
        loaded = load_nystrom_kle(path, _kernel(bkd), bkd)
        meshes = [_nodes(bkd, n) for n in (FINE,) + COARSE]
        coef = _coef(bkd, nystrom.nterms())
        for before, after in zip(
            nystrom_kles_on_meshes(nystrom, meshes),
            nystrom_kles_on_meshes(loaded, meshes),
        ):
            bkd.assert_allclose(
                after(coef), before(coef), rtol=0.0, atol=0.0
            )

    def test_a_mesh_can_be_added_one_at_a_time(self, bkd, tmp_path) -> None:
        """Adaptive use: a fidelity appears after the study has started."""
        nystrom = create_nystrom_kle(
            _kernel(bkd), _nodes(bkd, FINE), 6, bkd, nlandmarks=30
        )
        path = tmp_path / "field.npz"
        save_nystrom_kle(path, nystrom)
        loaded = load_nystrom_kle(path, _kernel(bkd), bkd)
        coef = _coef(bkd, loaded.nterms())

        first = nystrom_kles_on_meshes(loaded, [_nodes(bkd, COARSE[0])])
        # ... time passes, the estimator asks for another fidelity ...
        second = nystrom_kles_on_meshes(loaded, [_nodes(bkd, COARSE[1])])
        both = nystrom_kles_on_meshes(
            loaded, [_nodes(bkd, n) for n in COARSE]
        )
        for incremental, batched in zip(first + second, both):
            bkd.assert_allclose(
                incremental(coef), batched(coef), rtol=0.0, atol=0.0
            )
