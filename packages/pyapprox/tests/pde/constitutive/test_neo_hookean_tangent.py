"""Tests for the NeoHookean 3D stress and tangent modulus.

Verifies:
1. compute_stress_3d matches sympy_stress_3d (numeric vs symbolic)
2. compute_tangent_3d matches sympy differentiation of the symbolic PK1
3. 3D reduces exactly to 2D for plane deformations (F33=1, no out-of-
   plane coupling) — the check that catches F^{-1} vs F^{-T} mix-ups
4. Major symmetry A_iJkL = A_kLiJ (A is the Hessian of an energy)
5. Closed form at F = I
"""

import numpy as np
import sympy as sp
from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)

_LAMDA, _MU = 2.0, 3.0


def _random_f_batch(nbatch, seed=0):
    """Random 3x3 deformation gradients near identity with det > 0."""
    rng = np.random.RandomState(seed)
    fs = np.eye(3)[..., None] + 0.15 * rng.randn(3, 3, nbatch)
    dets = np.linalg.det(np.moveaxis(fs, -1, 0))
    assert np.all(dets > 0.1)
    return fs


def _as_nested(fs, bkd):
    """Convert (3, 3, nbatch) numpy array to the nested-tuple convention."""
    return tuple(
        tuple(bkd.asarray(fs[i, j]) for j in range(3)) for i in range(3)
    )


class TestNeoHookean3DStress:

    def test_stress_matches_sympy(self, bkd):
        model = NeoHookeanStress(_LAMDA, _MU)
        f_symbs = sp.symbols("f11 f12 f13 f21 f22 f23 f31 f32 f33")
        F_sym = tuple(tuple(f_symbs[3 * i + j] for j in range(3)) for i in range(3))
        P_sym = model.sympy_stress_3d(F_sym)

        fs = _random_f_batch(5)
        F = _as_nested(fs, bkd)
        P = model.compute_stress_3d(F, bkd)
        args = [fs[i, j] for i in range(3) for j in range(3)]
        for i in range(3):
            for j in range(3):
                expected = sp.lambdify(f_symbs, P_sym[i][j], "numpy")(*args)
                bkd.assert_allclose(P[i][j], bkd.asarray(expected), rtol=1e-12)

    def test_stress_3d_reduces_to_2d(self, bkd):
        """Plane deformation: in-plane 3D PK1 equals the 2D PK1.

        With F33=1 and zero out-of-plane coupling, J and the in-plane
        block of F^{-T} coincide with their 2D counterparts. This is the
        regression test for using F^{-1} in place of F^{-T} (they agree
        for symmetric F, so random non-symmetric F is essential).
        """
        model = NeoHookeanStress(_LAMDA, _MU)
        rng = np.random.RandomState(3)
        f2 = np.eye(2)[..., None] + 0.15 * rng.randn(2, 2, 5)

        F11, F12 = bkd.asarray(f2[0, 0]), bkd.asarray(f2[0, 1])
        F21, F22 = bkd.asarray(f2[1, 0]), bkd.asarray(f2[1, 1])
        P2 = model.compute_stress_2d(F11, F12, F21, F22, bkd)

        zero = bkd.zeros((5,))
        one = bkd.ones((5,))
        F3 = (
            (F11, F12, zero),
            (F21, F22, zero),
            (zero, zero, one),
        )
        P3 = model.compute_stress_3d(F3, bkd)

        bkd.assert_allclose(P3[0][0], P2[0], rtol=1e-12)
        bkd.assert_allclose(P3[0][1], P2[1], rtol=1e-12)
        bkd.assert_allclose(P3[1][0], P2[2], rtol=1e-12)
        bkd.assert_allclose(P3[1][1], P2[3], rtol=1e-12)
        # out-of-plane shear stresses vanish for plane deformations
        bkd.assert_allclose(P3[0][2], zero, atol=1e-14)
        bkd.assert_allclose(P3[2][0], zero, atol=1e-14)


class TestNeoHookean3DTangent:

    def test_tangent_matches_sympy(self, bkd):
        """All 81 components match differentiation of the symbolic PK1."""
        model = NeoHookeanStress(_LAMDA, _MU)
        f_symbs = sp.symbols("f11 f12 f13 f21 f22 f23 f31 f32 f33")
        F_sym = tuple(tuple(f_symbs[3 * i + j] for j in range(3)) for i in range(3))
        P_sym = model.sympy_stress_3d(F_sym)

        fs = _random_f_batch(5, seed=1)
        F = _as_nested(fs, bkd)
        A = model.compute_tangent_3d(F, bkd)
        args = [fs[i, j] for i in range(3) for j in range(3)]
        for i in range(3):
            for J in range(3):
                for k in range(3):
                    for L in range(3):
                        dP = sp.diff(P_sym[i][J], f_symbs[3 * k + L])
                        expected = sp.lambdify(f_symbs, dP, "numpy")(*args)
                        bkd.assert_allclose(
                            A[i, J, k, L],
                            bkd.asarray(np.broadcast_to(expected, (5,)).copy()),
                            rtol=1e-10,
                            atol=1e-13,
                        )

    def test_tangent_3d_reduces_to_2d(self, bkd):
        """In-plane 3D tangent entries equal the 2D dict values."""
        model = NeoHookeanStress(_LAMDA, _MU)
        rng = np.random.RandomState(7)
        f2 = np.eye(2)[..., None] + 0.15 * rng.randn(2, 2, 5)

        F11, F12 = bkd.asarray(f2[0, 0]), bkd.asarray(f2[0, 1])
        F21, F22 = bkd.asarray(f2[1, 0]), bkd.asarray(f2[1, 1])
        A2 = model.compute_tangent_2d(F11, F12, F21, F22, bkd)

        zero = bkd.zeros((5,))
        one = bkd.ones((5,))
        F3 = (
            (F11, F12, zero),
            (F21, F22, zero),
            (zero, zero, one),
        )
        A3 = model.compute_tangent_3d(F3, bkd)

        for i in range(2):
            for J in range(2):
                for k in range(2):
                    for L in range(2):
                        key = f"A_{i + 1}{J + 1}{k + 1}{L + 1}"
                        bkd.assert_allclose(A3[i, J, k, L], A2[key], rtol=1e-12)

    def test_tangent_major_symmetry(self, bkd):
        """A_iJkL = A_kLiJ: the tangent is the Hessian of an energy."""
        model = NeoHookeanStress(_LAMDA, _MU)
        fs = _random_f_batch(5, seed=2)
        A = model.compute_tangent_3d(_as_nested(fs, bkd), bkd)
        for i in range(3):
            for J in range(3):
                for k in range(3):
                    for L in range(3):
                        bkd.assert_allclose(
                            A[i, J, k, L], A[k, L, i, J], rtol=1e-12
                        )

    def test_tangent_at_identity(self, bkd):
        """At F = I: A_iJkL = mu*(d_ik*d_JL + d_Jk*d_Li) + lam*d_Ji*d_Lk."""
        model = NeoHookeanStress(_LAMDA, _MU)
        one = bkd.ones((2,))
        zero = bkd.zeros((2,))
        F = (
            (one, zero, zero),
            (zero, one, zero),
            (zero, zero, one),
        )
        A = model.compute_tangent_3d(F, bkd)
        for i in range(3):
            for J in range(3):
                for k in range(3):
                    for L in range(3):
                        expected = (
                            _MU * ((i == k) * (J == L) + (J == k) * (L == i))
                            + _LAMDA * (J == i) * (L == k)
                        )
                        bkd.assert_allclose(
                            A[i, J, k, L],
                            bkd.full((2,), float(expected)),
                            atol=1e-14,
                        )
