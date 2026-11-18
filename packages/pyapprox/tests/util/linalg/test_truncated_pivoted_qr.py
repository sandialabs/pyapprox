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

from dataclasses import dataclass

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg import TruncatedPivotedQRFactorizer

_PATHS = ["numpy-generic", "numpy-numba", "torch-generic"]


def _scipy_cpqr_pivots(A_np: NDArray[np.float64], k: int) -> list[int]:
    """Reference: leading k column-pivots from scipy's pivoted QR."""
    _Q, _R, piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
    return [int(p) for p in piv[:k]]


@dataclass(frozen=True)
class _PathResult:
    """What a factorization produced, converted to numpy for comparison."""

    npivots: int
    pivots: list[int]
    rdiag: NDArray[np.float64]
    recon_err: float

    def sorted_rdiag(self) -> NDArray[np.float64]:
        return np.sort(self.rdiag)[::-1]


def _run_factorizer(
    fac: TruncatedPivotedQRFactorizer[Array], A: Array, npivots: int,
) -> _PathResult:
    """Factorize ``A`` (the matrix ``fac`` was built on) and record rank,
    pivots, R-diagonal and the norm of ``A[:, pivots] - Q @ R``."""
    bkd = fac.bkd()
    fac.factorize(npivots=npivots)
    nc = fac.npivots()
    assert fac._R is not None
    recon = fac.factor() @ fac._R[:nc, :nc]
    resid = bkd.reshape(recon - A[:, fac.pivots()], (-1,))
    return _PathResult(
        npivots=nc,
        pivots=[int(p) for p in bkd.to_numpy(fac.pivots())],
        rdiag=np.asarray(bkd.to_numpy(fac.rdiag()), dtype=np.float64),
        recon_err=float(bkd.to_numpy(bkd.norm(resid))),
    )


def _result_for_path(
    path: str, A_np: NDArray[np.float64], tol: float, npivots: int,
) -> _PathResult:
    """Run one execution path, forced explicitly: the ``bkd`` fixture alone
    never exercises numpy-generic, since numpy dispatches to numba when it
    is installed."""
    if path in ("numpy-generic", "numpy-numba"):
        if path == "numpy-numba":
            pytest.importorskip("numba")
        bkd_np = NumpyBkd()
        A_bkd_np = bkd_np.asarray(A_np)
        fac_np = TruncatedPivotedQRFactorizer(A_bkd_np, bkd_np, tol=tol)
        if path == "numpy-generic":
            fac_np._use_numba = False
        else:
            assert fac_np._use_numba  # numpy + numba -> numba path is active
        return _run_factorizer(fac_np, A_bkd_np, npivots)
    if path == "torch-generic":
        pytest.importorskip("torch")
        import torch

        from pyapprox.util.backends.torch import TorchBkd

        torch.set_default_dtype(torch.float64)
        bkd_t = TorchBkd()
        A_t = bkd_t.asarray(A_np)
        fac_t = TruncatedPivotedQRFactorizer(A_t, bkd_t, tol=tol)
        assert not fac_t._use_numba  # torch always uses the generic path
        return _run_factorizer(fac_t, A_t, npivots)
    raise ValueError(path)


class TestAgainstScipy:
    """Householder pivoted QR matches LAPACK geqp3 (scipy)."""

    def test_rdiag_matches_scipy_clean(self, bkd: Backend[Array]) -> None:
        # Well-conditioned: R-diagonal must match scipy to ~machine eps.
        rng = np.random.RandomState(0)
        A_np = rng.normal(size=(40, 25))
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        fac = TruncatedPivotedQRFactorizer(bkd.asarray(A_np), bkd, tol=0.0)
        fac.factorize(npivots=25)
        bkd.assert_allclose(
            fac.rdiag(), bkd.asarray(np.abs(np.diag(R))), atol=1e-10,
        )

    def test_reconstruction(self, bkd: Backend[Array]) -> None:
        # A[:, pivots] == Q @ R to machine precision.
        rng = np.random.RandomState(1)
        A_np = rng.normal(size=(30, 18))
        A = bkd.asarray(A_np)
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=18)
        Q = fac.factor()
        nc = fac.npivots()
        assert fac._R is not None
        R = fac._R[:nc, :nc]
        bkd.assert_allclose(Q @ R, A[:, fac.pivots()], atol=1e-10)

    def test_factor_orthonormal(self, bkd: Backend[Array]) -> None:
        rng = np.random.RandomState(2)
        A = bkd.asarray(rng.normal(size=(40, 20)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=15)
        Q = fac.factor()
        bkd.assert_allclose(Q.T @ Q, bkd.eye(15), atol=1e-10)

    def test_pivots_match_scipy_separated(self, bkd: Backend[Array]) -> None:
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

    def test_stable_on_wide_dynamic_range(self, bkd: Backend[Array]) -> None:
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

    def test_budget_truncates(self, bkd: Backend[Array]) -> None:
        rng = np.random.RandomState(6)
        A = bkd.asarray(rng.normal(size=(30, 25)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=5)
        assert fac.npivots() == 5
        assert bkd.to_numpy(fac.pivots()).shape[0] == 5

    def test_stops_at_rank_cliff(self, bkd: Backend[Array]) -> None:
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
    explicitly by ``_result_for_path``."""

    @pytest.mark.parametrize("path", _PATHS)
    def test_path_matches_scipy(self, path: str) -> None:
        rng = np.random.RandomState(7)
        A_np = np.ascontiguousarray(rng.normal(size=(40, 25)))
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        res = _result_for_path(path, A_np, tol=0.0, npivots=25)
        NumpyBkd().assert_allclose(
            res.rdiag, np.abs(np.diag(R)), atol=1e-10,
        )

    def test_all_paths_agree(self) -> None:
        # numpy-generic, numpy-numba, torch-generic must give identical pivots,
        # R-diagonal, and rank on the same data.
        rng = np.random.RandomState(11)
        A_np = np.ascontiguousarray(rng.normal(size=(45, 30)))
        results = {
            path: _result_for_path(path, A_np, tol=1e-12, npivots=20)
            for path in _PATHS
        }
        base = results["numpy-generic"]
        for path, res in results.items():
            assert res.npivots == base.npivots, path
            assert res.pivots == base.pivots, path
            NumpyBkd().assert_allclose(
                res.rdiag, base.rdiag, atol=1e-10, err_msg=path,
            )

    def test_torch_stays_native(self) -> None:
        # The generic path must keep torch tensors native (no numpy roundtrip
        # mid-compute) so GPU/autograd is preserved.
        pytest.importorskip("torch")
        import torch

        from pyapprox.util.backends.torch import TorchBkd

        torch.set_default_dtype(torch.float64)
        bkd = TorchBkd()
        rng = np.random.RandomState(9)
        A = bkd.asarray(rng.normal(size=(30, 20)))
        fac = TruncatedPivotedQRFactorizer(A, bkd, tol=0.0)
        fac.factorize(npivots=15)
        assert torch.is_tensor(fac.pivots())
        assert torch.is_tensor(fac.factor())


def _hard_matrix(seed: int) -> NDArray[np.float64]:
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

    @pytest.mark.parametrize("path", _PATHS)
    def test_rdiag_matches_scipy_deep(self, path: str) -> None:
        A_np = _hard_matrix(seed=0)
        _Q, R, _piv = scipy.linalg.qr(A_np, pivoting=True, mode="economic")
        ref = np.sort(np.abs(np.diag(R)))[::-1]
        ours = _result_for_path(
            path, A_np, tol=0.0, npivots=min(A_np.shape),
        ).sorted_rdiag()
        n = min(len(ref), len(ours))
        # Deep into the spectrum, down to 1e-9 * R[0,0] (below ~sqrt(eps) the
        # directions are at roundoff level, where even different LAPACK builds
        # disagree, so the comparison stops there).
        big = ref[:n] > 1e-9 * ref[0]
        rel = np.abs(ours[:n] - ref[:n]) / (ref[:n] + 1e-300)
        assert rel[big].max() < self._DEEP_RDIAG_RTOL[path]

    @pytest.mark.parametrize("path", _PATHS)
    def test_reconstruction_machine_exact(self, path: str) -> None:
        # Whatever (tie-arbitrary) pivots are chosen, A[:, pivots] = Q @ R must
        # hold to machine precision -- the definitive correctness invariant.
        A_np = _hard_matrix(seed=1)
        res = _result_for_path(path, A_np, tol=0.0, npivots=min(A_np.shape))
        assert res.recon_err < 1e-9

    def test_numpy_paths_agree_tightly(self) -> None:
        # The two numpy paths (vectorized-generic and numba) must agree on rank
        # and on the R-diagonal to roundoff relative to R[0,0]. A per-entry
        # relative tolerance would be wrong: the diagonal spans about 1e10, and
        # its smallest entries carry an absolute error of order eps * R[0,0]
        # that differs between the two paths (measured: up to 2.5% relative on
        # the smallest entry, 7e-12 relative to R[0,0], numba 0.68.1 against
        # numpy 2.5.4).
        A_np = _hard_matrix(seed=2)
        npiv = min(A_np.shape)
        gen = _result_for_path("numpy-generic", A_np, tol=1e-10, npivots=npiv)
        nb = _result_for_path("numpy-numba", A_np, tol=1e-10, npivots=npiv)
        assert nb.npivots == gen.npivots
        n = min(len(nb.rdiag), len(gen.rdiag))
        ref = gen.sorted_rdiag()[:n]
        NumpyBkd().assert_allclose(
            nb.sorted_rdiag()[:n], ref, rtol=0.0, atol=1e-10 * float(ref[0]),
        )

    def test_torch_agrees_in_resolvable_range(self) -> None:
        # Native torch agrees with numpy on rank and on the R-diagonal in the
        # resolvable range (above sqrt(eps)*R[0,0]); deeper, its higher float
        # error is expected (see _DEEP_RDIAG_RTOL) but reconstruction stays exact.
        pytest.importorskip("torch")
        A_np = _hard_matrix(seed=2)
        npiv = min(A_np.shape)
        res_n = _result_for_path("numpy-generic", A_np, tol=1e-10, npivots=npiv)
        res_t = _result_for_path("torch-generic", A_np, tol=1e-10, npivots=npiv)
        assert res_t.npivots == res_n.npivots
        rn = res_n.sorted_rdiag()
        rt = res_t.sorted_rdiag()
        n = min(len(rn), len(rt))
        resolvable = rn[:n] > np.sqrt(np.finfo(np.float64).eps) * rn[0]
        rel = np.abs(rt[:n] - rn[:n]) / (rn[:n] + 1e-300)
        assert rel[resolvable].max() < 1e-3
