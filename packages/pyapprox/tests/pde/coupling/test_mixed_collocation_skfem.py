r"""A collocation subdomain beside an skfem one, coupled through a
polynomial skeleton, against a manufactured solution. NumPy only: skfem
is.

The unit square is cut at x = 1/2 (harness: ``tests/_helpers/
coupling_mixed.py``): collocation on the left (18 Chebyshev nodes per
direction, exact flux testing), Q2 skfem on the right (n x n cells,
Neumann data on its top and bottom), skeleton 14 CGL nodes on the
interface, its two ends held at the exact solution. The skfem ports read
the skeleton by L2 projection onto their Q2 trace; the collocation ports
by interpolation. Smooth solution :math:`e^x \sin 3x \cos 2y`.

- Convergence: :math:`D \Delta u - \nabla\cdot(v u) + r u + f = 0` with
  constant (D, v, r) = (1, (1, 0.5), -1) and with variable
  :math:`D = 1 + xy/2`, no advection or reaction. For n = 4, 8, 16 both
  sides' max nodal errors fall at the Q2 rate, fitted rate at least 2.5
  (observed 2.6-2.9 for the skfem side, constant coefficients: 1.6e-3,
  2.2e-4, 3.6e-5); the skfem side's total interface flux converges to the
  exact :math:`\int D\, \partial u^*/\partial n` at rate at least 3; the
  two sides' fluxes cancel against every free skeleton function (1e-9).
- Wrong-normal control: the skfem flux gathered with the inward normal.
  The error stays above 1 at every level.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
import sympy as sp

from tests._helpers.coupling_collocation import SMOOTH, Coefficients
from tests._helpers.coupling_mixed import MixedStrip

_NCOL, _NSKEL = 18, 14
_LEVELS = (4, 8, 16)
_CASES = {
    "adr": Coefficients(1.0, (1.0, 0.5), -1.0),
    "variable_diffusion": Coefficients("1 + x*y/2", None, None),
}


def _rate(errors: list[float]) -> float:
    """The least-squares rate in h over the levels, h halving each."""
    slope = np.polyfit(np.log(1.0 / np.asarray(_LEVELS)), np.log(errors), 1)[0]
    return float(slope)


def _exact_flux(coefs: Coefficients) -> float:
    r"""The skfem side's outward Neumann data integrated over the
    interface: :math:`-\int_0^1 D u_x(1/2, y)\, dy`, its normal being -x."""
    x, y = sp.symbols("x y")
    u = sp.sympify(SMOOTH, locals={"x": x, "y": y})
    diffusion = sp.sympify(str(coefs.diffusion), locals={"x": x, "y": y})
    integrand = sp.lambdify(y, (-diffusion * sp.diff(u, x)).subs(x, 0.5), "numpy")
    t, w = np.polynomial.legendre.leggauss(30)
    return float(0.5 * np.sum(w * integrand(0.5 * (t + 1.0))))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_converges_at_the_galerkin_rate(case: str) -> None:
    coefs = _CASES[case]
    exact_flux = _exact_flux(coefs)
    collocation, galerkin, flux = [], [], []
    for n in _LEVELS:
        strip = MixedStrip(coefs, SMOOTH, _NCOL, n, _NSKEL)
        lam = strip.solve_skeleton()
        ec, eg = strip.errors(lam)
        collocation.append(ec)
        galerkin.append(eg)
        flux.append(abs(strip.galerkin_total_flux(lam) - exact_flux))
        left, right = strip.gathered(lam)
        free = [d for d in range(lam.shape[0]) if d not in strip.pinned()]
        assert np.abs(left[free] + right[free]).max() < 1e-9
    assert _rate(galerkin) > 2.5, galerkin
    assert _rate(collocation) > 2.5, collocation
    assert _rate(flux) > 3.0, flux


def test_a_reversed_galerkin_normal_stalls() -> None:
    coefs = _CASES["adr"]
    for n in _LEVELS:
        strip = MixedStrip(
            coefs, SMOOTH, _NCOL, n, _NSKEL, reverse_galerkin_normal=True
        )
        assert min(strip.errors(strip.solve_skeleton())) > 1.0
