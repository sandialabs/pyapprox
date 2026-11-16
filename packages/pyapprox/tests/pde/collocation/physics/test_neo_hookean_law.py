"""The neo-Hookean law in the nested form collocation hyperelasticity
reads: protocol conformance, agreement with the stress model it adapts,
and its tangent and Lame sensitivities against central differences of
its own stress."""

from typing import List

import numpy as np
import pytest

from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticLawProtocol,
    LameSensitiveLawProtocol,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.util.backends.protocols import Array, Backend

_NPTS = 5
_EPS = 1e-6


def _deformation_gradient(bkd: Backend[Array], ndim: int) -> List[List[Array]]:
    rng = np.random.default_rng(3 + ndim)
    return [
        [
            bkd.asarray((1.0 if i == J else 0.0) + 0.1 * rng.standard_normal(_NPTS))
            for J in range(ndim)
        ]
        for i in range(ndim)
    ]


def _material(bkd: Backend[Array]) -> tuple[Array, Array]:
    rng = np.random.default_rng(11)
    mu = bkd.asarray(1.0 + rng.uniform(size=_NPTS))
    lamda = bkd.asarray(0.5 + rng.uniform(size=_NPTS))
    return mu, lamda


def _shifted(F: List[List[Array]], k: int, L: int, step: float) -> List[List[Array]]:
    return [
        [entry + step if (i, J) == (k, L) else entry for J, entry in enumerate(row)]
        for i, row in enumerate(F)
    ]


class TestNeoHookeanLaw:
    def test_satisfies_physics_protocols(self, bkd: Backend[Array]) -> None:
        law = NeoHookeanLaw(bkd)
        assert isinstance(law, HyperelasticLawProtocol)
        assert isinstance(law, LameSensitiveLawProtocol)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_stress_matches_stress_model(self, bkd: Backend[Array], ndim: int) -> None:
        """Per-node mu and lamda give the stress model's per-node stress."""
        F = _deformation_gradient(bkd, ndim)
        mu, lamda = _material(bkd)
        P = NeoHookeanLaw(bkd).stress(F, mu, lamda)
        model = NeoHookeanStress(lamda, mu)
        if ndim == 1:
            expected = [[model.compute_stress_1d(F[0][0], bkd)]]
        elif ndim == 2:
            P11, P12, P21, P22 = model.compute_stress_2d(
                F[0][0], F[0][1], F[1][0], F[1][1], bkd
            )
            expected = [[P11, P12], [P21, P22]]
        else:
            P3 = model.compute_stress_3d(tuple(tuple(row) for row in F), bkd)
            expected = [list(row) for row in P3]
        for i in range(ndim):
            for J in range(ndim):
                bkd.assert_allclose(P[i][J], expected[i][J], rtol=1e-14)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_no_stress_undeformed(self, bkd: Backend[Array], ndim: int) -> None:
        identity = [
            [bkd.full((_NPTS,), 1.0 if i == J else 0.0) for J in range(ndim)]
            for i in range(ndim)
        ]
        mu, lamda = _material(bkd)
        for row in NeoHookeanLaw(bkd).stress(identity, mu, lamda):
            for entry in row:
                bkd.assert_allclose(entry, bkd.zeros((_NPTS,)), atol=1e-14)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_tangent_is_stress_derivative(self, bkd: Backend[Array], ndim: int) -> None:
        """A[i][J][k][L] equals the central difference of P[i][J] in
        F[k][L], node by node."""
        law = NeoHookeanLaw(bkd)
        F = _deformation_gradient(bkd, ndim)
        mu, lamda = _material(bkd)
        A = law.tangent(F, mu, lamda)
        for k in range(ndim):
            for L in range(ndim):
                plus = law.stress(_shifted(F, k, L, _EPS), mu, lamda)
                minus = law.stress(_shifted(F, k, L, -_EPS), mu, lamda)
                for i in range(ndim):
                    for J in range(ndim):
                        fd = (plus[i][J] - minus[i][J]) / (2.0 * _EPS)
                        bkd.assert_allclose(A[i][J][k][L], fd, rtol=1e-7, atol=1e-8)

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_lame_sensitivities_are_stress_derivatives(
        self, bkd: Backend[Array], ndim: int
    ) -> None:
        law = NeoHookeanLaw(bkd)
        F = _deformation_gradient(bkd, ndim)
        mu, lamda = _material(bkd)
        dmu = law.stress_mu_sensitivity(F, mu, lamda)
        dlamda = law.stress_lamda_sensitivity(F, mu, lamda)
        fd_mu = (
            law.stress(F, mu + _EPS, lamda),
            law.stress(F, mu - _EPS, lamda),
        )
        fd_lamda = (
            law.stress(F, mu, lamda + _EPS),
            law.stress(F, mu, lamda - _EPS),
        )
        for i in range(ndim):
            for J in range(ndim):
                bkd.assert_allclose(
                    dmu[i][J],
                    (fd_mu[0][i][J] - fd_mu[1][i][J]) / (2.0 * _EPS),
                    rtol=1e-7,
                    atol=1e-8,
                )
                bkd.assert_allclose(
                    dlamda[i][J],
                    (fd_lamda[0][i][J] - fd_lamda[1][i][J]) / (2.0 * _EPS),
                    rtol=1e-7,
                    atol=1e-8,
                )

    def test_lame_sensitivities_refused_in_3d(self, numpy_bkd: Backend[Array]) -> None:
        law = NeoHookeanLaw(numpy_bkd)
        F = _deformation_gradient(numpy_bkd, 3)
        mu, lamda = _material(numpy_bkd)
        with pytest.raises(NotImplementedError, match="1D and 2D"):
            law.stress_mu_sensitivity(F, mu, lamda)
        with pytest.raises(NotImplementedError, match="1D and 2D"):
            law.stress_lamda_sensitivity(F, mu, lamda)
