r"""Strong forms of solid mechanics, as terms of a verification problem.

Each stress law is written here from its formula and imports nothing from
the constitutive models it is used to check, so a problem built from it
is an independent statement of the physics. The fields are the
displacement components :math:`u, v, w` in the coordinates :math:`x, y,
z`; row :math:`i` of the equilibrium :math:`-\nabla\cdot S = f` is the
divergence term :math:`\nabla\cdot(-S_{i\cdot})`, so a problem's natural
data are :math:`-S n`.

- :func:`linear_stress`: :math:`\sigma = \lambda\,\mathrm{tr}(\varepsilon)
  I + 2\mu\varepsilon`, :math:`\varepsilon = (\nabla u + \nabla u^T)/2`
  (plane strain in 2D).
- :func:`neo_hookean_stress`: the compressible neo-Hookean first Piola
  stress :math:`P = \mu F + (\lambda \ln J - \mu) F^{-T}`, :math:`F = I +
  \nabla u`, :math:`J = \det F`.
- :func:`equilibrium_terms`: the rows of :math:`-\nabla\cdot S`.

Lamé values are expressions in the coordinates, as text (read without
being run), sympy or numbers.
"""

from typing import List, Tuple

import sympy as sp

from pyapprox.pde.expressions.symbols import gradient, pde_names, standard
from pyapprox.pde.verification.terms import Divergence, TermProtocol
from pyapprox.util.expressions.parsing import Value, parse

DISPLACEMENT_FIELDS: Tuple[str, ...] = ("u", "v", "w")


def _check_ndim(ndim: int) -> None:
    if not 1 <= ndim <= 3:
        raise ValueError(f"a displacement has 1 to 3 components, got {ndim}")


def _lame(ndim: int, lam: Value, mu: Value) -> Tuple[sp.Expr, sp.Expr]:
    """``lam`` and ``mu`` parsed over the coordinates of ``ndim``."""
    names = pde_names(standard(ndim))
    return parse(lam, names), parse(mu, names)


def displacement_gradient(ndim: int) -> sp.Matrix:
    r""":math:`\nabla u` as jet symbols: entry (i, j) is the derivative of
    field i along coordinate j.

    Raises
    ------
    ValueError
        If ``ndim`` is not 1, 2 or 3.
    """
    _check_ndim(ndim)
    coords = standard(ndim)
    rows = [gradient(field, coords) for field in DISPLACEMENT_FIELDS[:ndim]]
    return sp.Matrix(rows)


def linear_stress(ndim: int, lam: Value, mu: Value) -> sp.Matrix:
    r"""The small-strain stress :math:`\sigma` with Lamé values ``lam``
    and ``mu``.

    Raises
    ------
    ValueError
        If ``ndim`` is not 1, 2 or 3, or a Lamé value uses anything but
        numbers, the coordinates and known functions.
    """
    grad = displacement_gradient(ndim)
    lam_value, mu_value = _lame(ndim, lam, mu)
    strain = (grad + grad.T) / 2
    stress: sp.Matrix = (
        lam_value * strain.trace() * sp.eye(ndim) + 2 * mu_value * strain
    )
    return stress


def neo_hookean_stress(ndim: int, lam: Value, mu: Value) -> sp.Matrix:
    r"""The neo-Hookean first Piola stress :math:`P` with Lamé values
    ``lam`` and ``mu``, using :math:`F^{-T} = \mathrm{adj}(F)^T / J`.

    Raises
    ------
    ValueError
        If ``ndim`` is not 1, 2 or 3, or a Lamé value uses anything but
        numbers, the coordinates and known functions.
    """
    deformation = sp.eye(ndim) + displacement_gradient(ndim)
    jacobian = deformation.det()
    lam_value, mu_value = _lame(ndim, lam, mu)
    stress: sp.Matrix = (
        mu_value * deformation
        + (lam_value * sp.log(jacobian) - mu_value)
        * deformation.adjugate().T
        / jacobian
    )
    return stress


def equilibrium_terms(stress: sp.Matrix) -> List[TermProtocol]:
    r"""The rows of :math:`-\nabla\cdot S`: one :class:`Divergence` per
    displacement component, field ``i``'s flux :math:`-S_{i\cdot}`.

    Raises
    ------
    ValueError
        If ``stress`` is not square with 1 to 3 rows.
    """
    nrows, ncols = stress.shape
    if nrows != ncols:
        raise ValueError(f"stress must be square, got shape {stress.shape}")
    _check_ndim(nrows)
    return [
        Divergence(DISPLACEMENT_FIELDS[i], [-stress[i, j] for j in range(nrows)])
        for i in range(nrows)
    ]
