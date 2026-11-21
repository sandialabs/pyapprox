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
  :math:`D = 1 + xy/2`, no advection or reaction. For n = 4, 8, 16 the
  skfem side's L2 error converges at the Q2 rate, fitted rate within 0.1
  of 3 (constant coefficients: 5.8e-4, 7.5e-5, 9.4e-6); the collocation
  side's max nodal error, set by the interface data, at least 2.5; the
  skfem side's total interface flux converges to the exact
  :math:`\int D\, \partial u^*/\partial n` at rate at least 3; the two
  sides' fluxes cancel against every free skeleton function (1e-9).
- Wrong-normal control: the skfem flux gathered with the inward normal.
  The error stays above 1 at every level.
- Dirichlet ends: the skfem top and bottom Dirichlet too, so its
  interface ends are essential rows and its ports read the skeleton by
  the mortar projection. A quadratic (collocation 12, Q2 3 x 3, skeleton
  8) is reproduced (1e-10; observed 9e-13); the ADR case converges as
  above (skfem L2 6.0e-4, 7.5e-5, 9.4e-6, rate 3.00; collocation max
  1.0e-3, 9.8e-5, 4.1e-6). The skfem max nodal error is not asserted: it
  sits at the interface node next to a Dirichlet end and converges there
  at about h^2.5 (n = 16 to 32: 1.1e-5 to 1.9e-6), while away from the
  ends it converges faster than h^3. Control: plain projection onto the port nodes' own
  functions misses the quadratic and stalls on the smooth case (errors
  above 0.05 at every level; observed 0.11-0.21).
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
# Reproduced exactly by Q2, the skeleton and collocation alike.
_QUADRATIC = "1 + x - 2*y + x*y + y**2"
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
        collocation.append(strip.errors(lam)[0])
        galerkin.append(strip.galerkin_l2_error(lam))
        flux.append(abs(strip.galerkin_total_flux(lam) - exact_flux))
        left, right = strip.gathered(lam)
        free = [d for d in range(lam.shape[0]) if d not in strip.pinned()]
        assert np.abs(left[free] + right[free]).max() < 1e-9
    assert abs(_rate(galerkin) - 3.0) < 0.1, galerkin
    assert _rate(collocation) > 2.5, collocation
    assert _rate(flux) > 3.0, flux


def test_a_reversed_galerkin_normal_stalls() -> None:
    coefs = _CASES["adr"]
    for n in _LEVELS:
        strip = MixedStrip(
            coefs, SMOOTH, _NCOL, n, _NSKEL, reverse_galerkin_normal=True
        )
        assert min(strip.errors(strip.solve_skeleton())) > 1.0


def test_mortar_ends_reproduce_a_quadratic() -> None:
    strip = MixedStrip(_CASES["adr"], _QUADRATIC, 12, 3, 8, dirichlet_ends=True)
    assert max(strip.errors(strip.solve_skeleton())) < 1e-10


def test_mortar_ends_converge_at_the_galerkin_rate() -> None:
    collocation, galerkin = [], []
    for n in _LEVELS:
        strip = MixedStrip(_CASES["adr"], SMOOTH, _NCOL, n, _NSKEL, dirichlet_ends=True)
        lam = strip.solve_skeleton()
        collocation.append(strip.errors(lam)[0])
        galerkin.append(strip.galerkin_l2_error(lam))
        left, right = strip.gathered(lam)
        free = [d for d in range(lam.shape[0]) if d not in strip.pinned()]
        assert np.abs(left[free] + right[free]).max() < 1e-9
    assert abs(_rate(galerkin) - 3.0) < 0.1, galerkin
    assert _rate(collocation) > 2.5, collocation


def test_plain_projection_with_dirichlet_ends_stalls() -> None:
    quadratic = MixedStrip(
        _CASES["adr"], _QUADRATIC, 12, 3, 8, dirichlet_ends=True, mortar=False
    )
    assert max(quadratic.errors(quadratic.solve_skeleton())) > 0.05
    for n in _LEVELS:
        strip = MixedStrip(
            _CASES["adr"],
            SMOOTH,
            _NCOL,
            n,
            _NSKEL,
            dirichlet_ends=True,
            mortar=False,
        )
        assert max(strip.errors(strip.solve_skeleton())) > 0.05
