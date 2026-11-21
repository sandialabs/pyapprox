r"""The solid-mechanics strong forms, by checks that need no derivation.

- The small-strain stress is symmetric, and in 1D is
  :math:`(\lambda + 2\mu) u_x`.
- The neo-Hookean stress vanishes in the undeformed state (:math:`F = I`)
  and, linearized there, is the small-strain stress, in 1, 2 and 3
  dimensions with :math:`\lambda = 2`, :math:`\mu = 1 + x`; in 1D it is
  :math:`\mu F + (\lambda \ln F - \mu)/F`.
- The equilibrium rows of a 1D bar with :math:`u = x^2`, :math:`\lambda =
  \mu = 1` give the forcing :math:`-6`; the rows' fluxes are :math:`-S`.
- Refusals: a dimension outside 1-3, a non-square stress, a Lamé value
  with an undeclared name.
"""

import pytest
import sympy as sp

from pyapprox.pde.expressions.symbols import derivative, standard
from pyapprox.pde.verification.conditions import Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.solid_mechanics import (
    DISPLACEMENT_FIELDS,
    displacement_gradient,
    equilibrium_terms,
    linear_stress,
    neo_hookean_stress,
)

_X = sp.Symbol("x")


def _zero(stress: sp.Matrix, ndim: int) -> sp.Matrix:
    """``stress`` with every displacement-gradient jet set to zero."""
    jets = displacement_gradient(ndim)
    return stress.xreplace({jets[i, j]: 0 for i in range(ndim) for j in range(ndim)})


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_linear_stress_is_symmetric(ndim: int) -> None:
    stress = linear_stress(ndim, "2", "1 + x")
    assert sp.simplify(stress - stress.T) == sp.zeros(ndim, ndim)


def test_linear_stress_in_1d() -> None:
    u_x = derivative("u", "x", standard(1))
    assert sp.simplify(linear_stress(1, "2", "1 + x")[0, 0] - (4 + 2 * _X) * u_x) == 0


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_neo_hookean_vanishes_undeformed(ndim: int) -> None:
    stress = neo_hookean_stress(ndim, "2", "1 + x")
    assert sp.simplify(_zero(stress, ndim)) == sp.zeros(ndim, ndim)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_neo_hookean_linearizes_to_the_small_strain_stress(ndim: int) -> None:
    """d/de P(e G) at e = 0 equals sigma(G) for a fixed gradient G."""
    jets = displacement_gradient(ndim)
    eps = sp.Symbol("eps")
    entries = [sp.Rational(k + 1, 7) * (-1) ** k for k in range(ndim * ndim)]
    scaled = {
        jets[i, j]: eps * entries[i * ndim + j]
        for i in range(ndim)
        for j in range(ndim)
    }
    fixed = {
        jets[i, j]: entries[i * ndim + j] for i in range(ndim) for j in range(ndim)
    }
    tangent = sp.diff(neo_hookean_stress(ndim, "2", "1 + x").xreplace(scaled), eps)
    expected = linear_stress(ndim, "2", "1 + x").xreplace(fixed)
    assert sp.simplify(tangent.subs(eps, 0) - expected) == sp.zeros(ndim, ndim)


def test_neo_hookean_in_1d() -> None:
    deformation = 1 + derivative("u", "x", standard(1))
    expected = deformation + (2 * sp.log(deformation) - 1) / deformation
    assert sp.simplify(neo_hookean_stress(1, 2, 1)[0, 0] - expected) == 0


def test_equilibrium_rows_and_forcing() -> None:
    stress = linear_stress(2, "1", "1")
    terms = equilibrium_terms(stress)
    assert [term.row() for term in terms] == list(DISPLACEMENT_FIELDS[:2])
    bar = ManufacturedProblem(
        {"u": "x**2"},
        equilibrium_terms(linear_stress(1, "1", "1")),
        Box([(0.0, 1.0)]),
        {"left": Dirichlet(), "right": Dirichlet()},
    )
    assert sp.simplify(bar.forcing("u") + 6) == 0
    names = ManufacturedProblem(
        {"u": "x", "v": "y"},
        terms,
        Box([(0.0, 1.0)] * 2),
        {side: Dirichlet() for side in Box([(0.0, 1.0)] * 2).boundaries()},
    ).names()
    for i, term in enumerate(terms):
        flux = term.flux(names)
        assert flux is not None
        assert all(sp.simplify(flux[j] + stress[i, j]) == 0 for j in range(2))


def test_refusals() -> None:
    with pytest.raises(ValueError, match="1 to 3"):
        displacement_gradient(4)
    with pytest.raises(ValueError, match="1 to 3"):
        linear_stress(0, 1, 1)
    with pytest.raises(ValueError, match="square"):
        equilibrium_terms(sp.zeros(2, 3))
    with pytest.raises(ValueError, match="1 to 3"):
        equilibrium_terms(sp.zeros(4, 4))
    with pytest.raises(ValueError):
        neo_hookean_stress(2, "q", 1)
