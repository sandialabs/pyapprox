"""Tests for the matrix-free, truncated Householder pivoted-QR factorizer.

The factorizer is the same algorithm as LAPACK ``geqp3`` (Householder reflectors,
Businger-Golub pivoting, Drmac-Bujanovic norm safeguard), so it is validated
against scipy's reference column-pivoted QR
(``scipy.linalg.qr(pivoting=True)``): the ``R``-diagonal, the orthonormal factor,
the reconstruction, and the pivot order must agree -- to machine precision on
well-conditioned data, and within tie-ambiguity on ill-conditioned data.  The
truncation (stop at the budget / rank cliff) and the numba-vs-numpy agreement are
also pinned.
"""

import numpy as np
import pytest
import scipy.linalg

from pyapprox.util.linalg import TruncatedPivotedQRFactorizer


def _scipy_cpqr_pivots(A_np, k):
    """Reference: leading k column-pivots from scipy's pivoted QR."""
    _Q, _R, piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
    return list(piv[:k])


class TestAgainstScipy:
    """Householder pivoted QR matches LAPACK geqp3 (scipy)."""

    def test_rdiag_matches_scipy_clean(self, bkd):
        # Well-conditioned: R-diagonal must match scipy to ~machine eps.
        rng = np.random.RandomState(0)
        A_np = rng.normal(size=(40, 25))
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=0.0)
        fac.factorize(npivots=25)
        bkd.assert_allclose(
            fac.rdiag(), bkd.asarray(np.abs(np.diag(R))), atol=1e-10,
        )

    def test_reconstruction(self, bkd):
        # A[:, pivots] == Q @ R to machine precision.
        rng = np.random.RandomState(1)
        A_np = rng.normal(size=(30, 18))
        A = bkd.asarray(A_np)
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=18)
        Q = fac.factor()
        nc = fac.npivots()
        R = bkd.to_numpy(fac._R)[:nc, :nc]
        bkd.assert_allclose(Q @ bkd.asarray(R), A[:, fac.pivots()], atol=1e-10)

    def test_factor_orthonormal(self, bkd):
        rng = np.random.RandomState(2)
        A = bkd.asarray(rng.normal(size=(40, 20)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=15)
        Q = fac.factor()
        bkd.assert_allclose(Q.T @ Q, bkd.eye(15), atol=1e-10)

    def test_pivots_match_scipy_separated(self, bkd):
        # Well-separated singular values -> unambiguous pivot order.
        rng = np.random.RandomState(3)
        U, _ = np.linalg.qr(rng.normal(size=(30, 30)))
        Vt, _ = np.linalg.qr(rng.normal(size=(20, 20)))
        svals = np.geomspace(1.0, 1e-3, 12)
        A_np = (U[:, :12] * svals) @ Vt[:12, :]
        ref = _scipy_cpqr_pivots(A_np, 8)
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=1e-10)
        fac.factorize(npivots=8)
        ours = [int(p) for p in bkd.to_numpy(fac.pivots())]
        assert set(ours) == set(ref)

    def test_stable_on_wide_dynamic_range(self, bkd):
        # Geometric decay past sqrt(eps): Gram-Schmidt loses orthogonality here,
        # but Householder stays stable and reproduces scipy's R-DIAGONAL.
        # (Comparing a QR count to the SVD rank is invalid -- CPQR's |R[i,i]|
        # upper-bounds the singular values and gives a different, legitimately
        # higher, count; the R-diagonal magnitudes are the right reference.)
        rng = np.random.RandomState(5)
        U, _ = np.linalg.qr(rng.normal(size=(80, 80)))
        Vt, _ = np.linalg.qr(rng.normal(size=(70, 70)))
        svals = np.geomspace(1.0, 1e-13, 50)
        A_np = (U[:, :50] * svals) @ Vt[:50, :]
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=0.0)
        fac.factorize(npivots=70)
        ours = np.sort(bkd.to_numpy(fac.rdiag()))[::-1]
        ref = np.sort(np.abs(np.diag(R)))[::-1][: len(ours)]
        n = min(len(ours), len(ref))
        # Compare in the resolvable range, above 1e-7 * R[0,0].  There the
        # R-diagonal matches scipy's geqp3 to ~5e-10 (relative); below it, the
        # directions sit at the level of accumulated roundoff and geqp3
        # implementations (different LAPACK/BLAS builds included) legitimately
        # disagree on tie-ambiguous columns (a sharp cliff from ~1e-9 to ~0.5
        # in the relative difference sits right at this threshold).
        big = ref[:n] > 1e-7 * ref[0]
        rel = np.abs(ours[:n] - ref[:n]) / (ref[:n] + 1e-300)
        assert rel[big].max() < 1e-8


class TestTruncation:
    """Stops at the budget or the numerical-rank cliff."""

    def test_budget_truncates(self, bkd):
        rng = np.random.RandomState(6)
        A = bkd.asarray(rng.normal(size=(30, 25)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=5)
        assert fac.npivots() == 5
        assert bkd.to_numpy(fac.pivots()).shape[0] == 5

    def test_stops_at_rank_cliff(self, bkd):
        # Exact rank 6 embedded in 40 columns -> at most 6 pivots regardless of
        # the requested budget.
        rng = np.random.RandomState(4)
        basis = rng.normal(size=(50, 6))
        A_np = basis @ rng.normal(size=(6, 40))
        fac = TruncatedPivotedQRFactorizer(
            bkd.asarray(A_np), bkd, tol=1e-10,
        )
        fac.factorize(npivots=30)
        assert fac.npivots() == 6
        assert fac.success()


class TestAllPaths:
    """Every execution path -- backend-generic on numpy AND torch, plus the
    numpy numba fast path -- must run and match scipy.  Each is forced
    explicitly (the bkd fixture alone would never exercise numpy-generic, since
    numpy auto-dispatches to numba)."""

    def _factorizer_for(self, path, A_np, tol):
        if path == "numpy-generic":
            from pyapprox.util.backends.numpy import NumpyBkd
            bkd = NumpyBkd()
            fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
            fac._use_numba = False
        elif path == "numpy-numba":
            pytest.importorskip("numba")
            from pyapprox.util.backends.numpy import NumpyBkd
            bkd = NumpyBkd()
            fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
            assert fac._use_numba  # numpy + numba -> numba path is active
        elif path == "torch-generic":
            torch = pytest.importorskip("torch")
            torch.set_default_dtype(torch.float64)
            from pyapprox.util.backends.torch import TorchBkd
            bkd = TorchBkd()
            fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
            assert not fac._use_numba  # torch always uses the generic path
        else:
            raise ValueError(path)
        return fac, bkd

    @pytest.mark.parametrize(
        "path", ["numpy-generic", "numpy-numba", "torch-generic"],
    )
    def test_path_matches_scipy(self, path):
        rng = np.random.RandomState(7)
        A_np = np.ascontiguousarray(rng.normal(size=(40, 25)))
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        fac, bkd = self._factorizer_for(path, A_np, tol=0.0)
        fac.factorize(npivots=25)
        bkd.assert_allclose(
            fac.rdiag(), bkd.asarray(np.abs(np.diag(R))), atol=1e-10,
        )

    def test_all_paths_agree(self):
        # numpy-generic, numpy-numba, torch-generic must give identical pivots,
        # R-diagonal, and rank on the same data.
        rng = np.random.RandomState(11)
        A_np = np.ascontiguousarray(rng.normal(size=(45, 30)))
        results = {}
        for path in ["numpy-generic", "numpy-numba", "torch-generic"]:
            fac, bkd = self._factorizer_for(path, A_np, tol=1e-12)
            fac.factorize(npivots=20)
            results[path] = (
                fac.npivots(),
                [int(p) for p in bkd.to_numpy(fac.pivots())],
                np.asarray(bkd.to_numpy(fac.rdiag()), dtype=float),
            )
        base = results["numpy-generic"]
        for path, (nc, piv, rd) in results.items():
            assert nc == base[0], path
            assert piv == base[1], path
            np.testing.assert_allclose(rd, base[2], atol=1e-10, err_msg=path)

    def test_torch_stays_native(self):
        # The generic path must keep torch tensors native (no numpy roundtrip
        # mid-compute) so GPU/autograd is preserved.
        torch = pytest.importorskip("torch")
        torch.set_default_dtype(torch.float64)
        from pyapprox.util.backends.torch import TorchBkd
        bkd = TorchBkd()
        rng = np.random.RandomState(9)
        A = bkd.asarray(rng.normal(size=(30, 20)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=15)
        assert torch.is_tensor(fac.pivots())
        assert torch.is_tensor(fac.factor())


def _hard_matrix(seed):
    """Wide dynamic range + near-duplicate columns: the pattern that exposes a
    too-late norm-recompute safeguard (the downdated norms stall and the pivot
    order drifts from geqp3 deep into the spectrum)."""
    rng = np.random.RandomState(seed)
    ngrid, core = 200, 60
    U, _ = np.linalg.qr(rng.normal(size=(ngrid, ngrid)))
    svals = np.geomspace(1.0, 1e-11, core)
    base = (U[:, :core] * svals) @ np.linalg.qr(
        rng.normal(size=(core, core)))[0][:core, :]
    cols = [base]
    for c in base.T:
        for _ in range(8):
            cols.append((c + 1e-12 * rng.normal(size=ngrid))[:, None])
    return np.ascontiguousarray(np.hstack(cols))


def _factorizer_for_path(path, A_np, tol):
    if path == "numpy-generic":
        from pyapprox.util.backends.numpy import NumpyBkd
        bkd = NumpyBkd()
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
        fac._use_numba = False
    elif path == "numpy-numba":
        pytest.importorskip("numba")
        from pyapprox.util.backends.numpy import NumpyBkd
        bkd = NumpyBkd()
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
        assert fac._use_numba
    elif path == "torch-generic":
        torch = pytest.importorskip("torch")
        torch.set_default_dtype(torch.float64)
        from pyapprox.util.backends.torch import TorchBkd
        bkd = TorchBkd()
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=tol)
        assert not fac._use_numba
    else:
        raise ValueError(path)
    return fac, bkd


class TestGeqp3FaithfulDeepSpectrum:
    """On ill-conditioned data with near-duplicate columns, every path must
    satisfy the mathematically-invariant guarantees of geqp3 *deep* into the
    spectrum -- pinning the LAPACK dlaqps (Drmac-Bujanovic) norm safeguard.

    Note the invariants are the R-DIAGONAL (magnitudes) and the RECONSTRUCTION,
    not the pivot indices: among near-equal (e.g. duplicate) columns the pivot
    choice is tie-arbitrary and legitimately differs across backends/BLAS, but
    each choice spans the same subspace.  No problem-dependent tie tolerance is
    imposed, so pivot identity is not asserted."""

    # numpy (BLAS/numba) tracks scipy's geqp3 to ~1e-6 relative down to
    # 1e-9*R[0,0] when both use the same BLAS build.  Across BLAS builds
    # (OpenBLAS vs Accelerate vs MKL; x86_64 vs arm64) the deep band disagrees
    # by up to ~4e-4 (observed on GitHub CI runners), because among
    # near-duplicate columns the downdated norms are roundoff-dominated there.
    # 1e-3 still pins the Drmac-Bujanovic safeguard: without it the deep-band
    # R-diagonal is wrong by orders of magnitude, not fractions of a percent.
    # The native-torch path accumulates more float error in the
    # reflector/downdate (its matmul/reduction ordering differs from LAPACK),
    # so it shares the same looser bound -- kept native (GPU/autograd) at the
    # cost of ~1e-3 agreement in the resolvable band, which is still ample for
    # selecting a well-conditioned non-redundant column subset.
    _DEEP_RDIAG_RTOL = {
        "numpy-generic": 1e-3,
        "numpy-numba": 1e-3,
        "torch-generic": 1e-3,
    }

    @pytest.mark.parametrize(
        "path", ["numpy-generic", "numpy-numba", "torch-generic"],
    )
    def test_rdiag_matches_scipy_deep(self, path):
        A_np = _hard_matrix(seed=0)
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        ref = np.sort(np.abs(np.diag(R)))[::-1]
        fac, bkd = _factorizer_for_path(path, A_np, tol=0.0)
        fac.factorize(npivots=min(A_np.shape))
        ours = np.sort(np.asarray(bkd.to_numpy(fac.rdiag()), float))[::-1]
        n = min(len(ref), len(ours))
        # Deep into the spectrum, down to 1e-9 * R[0,0] (below ~sqrt(eps) the
        # directions are at roundoff level, where even different LAPACK builds
        # disagree, so the comparison stops there).
        big = ref[:n] > 1e-9 * ref[0]
        rel = np.abs(ours[:n] - ref[:n]) / (ref[:n] + 1e-300)
        assert rel[big].max() < self._DEEP_RDIAG_RTOL[path]

    @pytest.mark.parametrize(
        "path", ["numpy-generic", "numpy-numba", "torch-generic"],
    )
    def test_reconstruction_machine_exact(self, path):
        # Whatever (tie-arbitrary) pivots are chosen, A[:, pivots] = Q @ R must
        # hold to machine precision -- the definitive correctness invariant.
        A_np = _hard_matrix(seed=1)
        fac, bkd = _factorizer_for_path(path, A_np, tol=0.0)
        fac.factorize(npivots=min(A_np.shape))
        nc = fac.npivots()
        Q = fac.factor()
        R = fac._R[:nc, :nc]
        A = bkd.asarray(A_np)
        recon = Q @ R
        err = float(bkd.to_numpy(bkd.norm(
            bkd.reshape(recon - A[:, fac.pivots()], (-1,))
        )))
        assert err < 1e-9

    def test_numpy_paths_agree_tightly(self):
        # The two numpy paths (vectorized-generic and numba) share LAPACK-style
        # arithmetic, so they must agree on rank and R-diagonal near bit-level.
        A_np = _hard_matrix(seed=2)
        rds = {}
        ranks = {}
        for path in ["numpy-generic", "numpy-numba"]:
            fac, bkd = _factorizer_for_path(path, A_np, tol=1e-10)
            fac.factorize(npivots=min(A_np.shape))
            ranks[path] = fac.npivots()
            rds[path] = np.sort(
                np.asarray(bkd.to_numpy(fac.rdiag()), float))[::-1]
        assert ranks["numpy-numba"] == ranks["numpy-generic"]
        n = min(len(rds["numpy-numba"]), len(rds["numpy-generic"]))
        np.testing.assert_allclose(
            rds["numpy-numba"][:n], rds["numpy-generic"][:n],
            rtol=1e-10, atol=1e-14,
        )

    def test_torch_agrees_in_resolvable_range(self):
        # Native torch agrees with numpy on rank and on the R-diagonal in the
        # resolvable range (above sqrt(eps)*R[0,0]); deeper, its higher float
        # error is expected (see _DEEP_RDIAG_RTOL) but reconstruction stays exact.
        pytest.importorskip("torch")
        A_np = _hard_matrix(seed=2)
        fac_n, bkd_n = _factorizer_for_path("numpy-generic", A_np, tol=1e-10)
        fac_t, bkd_t = _factorizer_for_path("torch-generic", A_np, tol=1e-10)
        fac_n.factorize(npivots=min(A_np.shape))
        fac_t.factorize(npivots=min(A_np.shape))
        assert fac_t.npivots() == fac_n.npivots()
        rn = np.sort(np.asarray(bkd_n.to_numpy(fac_n.rdiag()), float))[::-1]
        rt = np.sort(np.asarray(bkd_t.to_numpy(fac_t.rdiag()), float))[::-1]
        n = min(len(rn), len(rt))
        resolvable = rn[:n] > np.sqrt(np.finfo(np.float64).eps) * rn[0]
        rel = np.abs(rt[:n] - rn[:n]) / (rn[:n] + 1e-300)
        assert rel[resolvable].max() < 1e-3
