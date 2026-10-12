"""Tests for the advection formulations.

Checked through identities, not restated formulas: the two strong terms
differ by ``u div(v)``; the conservative weak integrand is the
non-conservative one after integration by parts; one definition works on
sympy expressions and on arrays.
"""

import pickle
from typing import Any

import numpy as np
import pytest
import sympy as sp

from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.util.backends.numpy import NumpyBkd

_X, _Y = sp.symbols("x y")


def _derivative(expr: Any, dim: int) -> Any:
    return sp.diff(expr, (_X, _Y)[dim])


class TestAdvectionForms:
    def test_strong_terms_differ_by_u_div_v(self) -> None:
        velocity = [_X * _Y + 1, sp.sin(_X) - _Y**2]
        u = sp.exp(_X) * sp.cos(2 * _Y)
        conservative = ConservativeAdvection().strong_term(
            velocity, u, _derivative
        )
        nonconservative = NonConservativeAdvection().strong_term(
            velocity, u, _derivative
        )
        divergence = sum(_derivative(v, ii) for ii, v in enumerate(velocity))
        assert sp.simplify(conservative - nonconservative - u * divergence) == 0

    def test_weak_integrands_match_by_parts(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Integrated over [0, 1]^2 by tensor Gauss quadrature, for a
        divergence-free v and a test function vanishing on the boundary,
        -(u, v.grad w) equals (v.grad u, w)."""
        bkd = numpy_bkd
        pts, wts = np.polynomial.legendre.leggauss(20)
        pts, wts = 0.5 * (pts + 1.0), 0.5 * wts
        xx, yy = np.meshgrid(pts, pts, indexing="ij")
        weights = np.outer(wts, wts)
        velocity = [-(yy - 0.5), xx - 0.5]
        u = np.exp(xx) * np.cos(yy)
        grad_u = [u, -np.exp(xx) * np.sin(yy)]
        # Asymmetric, so the symmetric rotation does not cancel the term.
        w = xx**2 * (1 - xx) * yy * (1 - yy) ** 2
        grad_w = [
            (2 * xx - 3 * xx**2) * yy * (1 - yy) ** 2,
            xx**2 * (1 - xx) * (1 - yy) * (1 - 3 * yy),
        ]
        integrals = [
            np.sum(
                weights
                * form.weak_integrand(velocity, u, grad_u, w, grad_w)
            )
            for form in (NonConservativeAdvection(), ConservativeAdvection())
        ]
        bkd.assert_allclose(
            bkd.asarray([integrals[1]]), bkd.asarray([integrals[0]]),
            rtol=1e-12,
        )
        # Not vacuous: the integral (about 7e-4) is far above round-off.
        assert abs(integrals[0]) > 1e-5

    def test_flux(self, numpy_bkd: NumpyBkd) -> None:
        """Only the conservative form integrates a flux by parts."""
        velocity = [np.array([1.0, 2.0]), np.array([-1.0, 0.5])]
        u = np.array([3.0, 4.0])
        assert NonConservativeAdvection().flux(velocity, u) is None
        flux = ConservativeAdvection().flux(velocity, u)
        assert flux is not None
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.stack(flux)),
            numpy_bkd.asarray(np.stack([velocity[0] * u, velocity[1] * u])),
        )

    @pytest.mark.parametrize(
        "form", [NonConservativeAdvection(), ConservativeAdvection()]
    )
    def test_satisfies_protocol_and_pickles(
        self, form: AdvectionFormProtocol
    ) -> None:
        assert isinstance(form, AdvectionFormProtocol)
        restored = pickle.loads(pickle.dumps(form))
        assert type(restored) is type(form)
