r"""The strong and weak forms a pointwise physics states, for reading.

A physics gives the weak-form integrands :math:`f_0, f_1`; the PDE it
solves is the strong form :math:`f_0 - \nabla \cdot f_1 = 0`, with natural
boundary flux :math:`f_1 \cdot n`. :func:`strong_residual` recovers the
strong form by total differentiation through the jets (``u_x`` gives
``u_xx`` and ``u_xy``), and :func:`describe` prints all three per field,
so a user can check the ``f0``/``f1`` split against the PDE they meant:
a term placed in ``f1`` is differentiated in the strong form, and its
normal component is what a Neumann condition prescribes.
:func:`strong_form_difference` makes that check exact: it compares the
derived strong form with one the user writes out.

Only sympy is used here.
"""

from typing import List, Sequence

import sympy as sp

from pyapprox.pde.expressions.symbols import derivative, jets, normal, pde_names
from pyapprox.pde.pointwise.derivatives import RealDifferentiation
from pyapprox.pde.pointwise.spec import PointwisePhysics
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse


def total_derivative(
    physics: PointwisePhysics, expression: sp.Expr, direction: str
) -> sp.Expr:
    """The derivative of ``expression`` along coordinate ``direction``,
    with every field and aux-field jet differentiated by the chain rule.
    Variables are real and derivatives branch-wise, as in
    :class:`~pyapprox.pde.pointwise.derivatives.RealDifferentiation`.

    Raises
    ------
    ValueError
        If ``direction`` is not a coordinate of the physics.
    """
    coords = physics.coords()
    if direction not in coords:
        raise ValueError(f"direction must be one of {list(coords)}, got {direction!r}")
    diff = RealDifferentiation(physics.names().variables())
    result = diff(expression, sp.Symbol(direction))
    named = list(physics.field_names()) + [data.name for data in physics.aux()]
    for name in named:
        # The integrands use values and gradients only.
        for symbol, directions in jets(name, coords, 1).items():
            if symbol in expression.free_symbols:
                result += diff(expression, symbol) * derivative(
                    name, directions + direction, coords
                )
    return result


def strong_residual(physics: PointwisePhysics, field: str) -> sp.Expr:
    """``f0 - ∇·f1`` of ``field``: the strong form is this equal to zero.

    Raises
    ------
    KeyError
        If ``field`` is not a field.
    """
    divergence = sum(
        (
            total_derivative(physics, component, direction)
            for component, direction in zip(physics.f1(field), physics.coords())
        ),
        sp.Integer(0),
    )
    result: sp.Expr = physics.f0(field) - divergence
    return result


def strong_form_difference(
    physics: PointwisePhysics, field: str, expected: Value
) -> sp.Expr:
    """``f0 - ∇·f1`` of ``field`` minus the strong residual the user
    meant, simplified: ``0`` when the ``f0``/``f1`` split states that PDE.

    ``expected`` is the left side of ``expected = 0``, as text or sympy,
    and may use second jets (``u_xx``) besides everything the integrands
    may use. Its sign counts: ``u_xx + u_yy + f`` is the negative of
    Poisson's ``-f - u_xx - u_yy`` and differs from it. Agreement fixes
    the PDE but not the natural boundary condition; fluxes differing by a
    divergence-free field give the same strong form, so check
    :func:`boundary_flux` too.

    Raises
    ------
    KeyError
        If ``field`` is not a field.
    ValueError
        If ``expected`` fails to parse.
    """
    declared = physics.names()
    named = list(physics.field_names()) + [data.name for data in physics.aux()]
    geometry = pde_names(
        physics.coords(),
        named,
        {s.name: v for s, v in declared.constants().items()},
        declared.functions(),
        jet_order=2,
    )
    names = Names(
        [s.name for s in geometry.variables()] + list(physics.params()),
        {s.name: v for s, v in geometry.constants().items()},
        geometry.functions(),
    )
    difference: sp.Expr = sp.simplify(
        strong_residual(physics, field) - parse(expected, names)
    )
    return difference


def boundary_flux(physics: PointwisePhysics, field: str) -> sp.Expr:
    """``f1·n`` of ``field``, the natural boundary flux.

    Raises
    ------
    KeyError
        If ``field`` is not a field.
    """
    result: sp.Expr = sum(
        (c * n for c, n in zip(physics.f1(field), normal(physics.coords()))),
        sp.Integer(0),
    )
    return result


def _weak(physics: PointwisePhysics, field: str) -> str:
    test = f"v_{field}"
    terms: List[str] = []
    f0 = physics.f0(field)
    if f0 != 0:
        terms.append(f"{test}*({f0})")
    for component, direction in zip(physics.f1(field), physics.coords()):
        if component != 0:
            terms.append(f"d{test}/d{direction}*({component})")
    return " + ".join(terms) if terms else "0"


def describe(physics: PointwisePhysics, fields: Sequence[str] = ()) -> str:
    """The strong form, weak form and natural boundary flux of each field
    (all of them by default), as text.

    Jets are written as in the specification: ``u_xy`` is the second
    derivative of ``u`` in ``x`` and ``y``. Expressions are expanded, so
    terms appear in sympy's order, not the specification's.

    Raises
    ------
    KeyError
        If a requested name is not a field.
    """
    names = list(fields) or list(physics.field_names())
    lines: List[str] = []
    for name in names:
        lines.append(f"field {name}:")
        lines.append(f"  strong:  {sp.expand(strong_residual(physics, name))} = 0")
        lines.append(f"  weak:    integral of {_weak(physics, name)} dx")
        lines.append(f"  flux:    f1.n = {boundary_flux(physics, name)}")
    return "\n".join(lines)


def _latex(physics: PointwisePhysics, expression: sp.Expr) -> str:
    # Normals as n_x; jets already print as u_{xy}.
    coords = physics.coords()
    names = {n: f"n_{{{c}}}" for n, c in zip(normal(coords), coords)}
    text: str = sp.latex(expression, symbol_names=names)
    return text


def _weak_latex(physics: PointwisePhysics, field: str) -> str:
    test = f"v_{{{field}}}"
    terms: List[str] = []
    f0 = physics.f0(field)
    if f0 != 0:
        terms.append(rf"{test} \left({_latex(physics, f0)}\right)")
    for component, direction in zip(physics.f1(field), physics.coords()):
        if component != 0:
            terms.append(
                rf"\partial_{{{direction}}} {test} "
                rf"\left({_latex(physics, component)}\right)"
            )
    body = " + ".join(terms) if terms else "0"
    return rf"\int_\Omega {body} \, dx"


def describe_latex(physics: PointwisePhysics, fields: Sequence[str] = ()) -> str:
    r""":func:`describe` as LaTeX: one ``aligned`` block of the strong
    form, weak form and natural boundary flux of each field.

    The text renders in Jupyter (``IPython.display.Math``), Quarto and
    Sphinx math, or in a LaTeX document inside a display environment.

    Raises
    ------
    KeyError
        If a requested name is not a field.
    """
    names = list(fields) or list(physics.field_names())
    rows: List[str] = []
    for name in names:
        strong = _latex(physics, sp.expand(strong_residual(physics, name)))
        flux = _latex(physics, boundary_flux(physics, name))
        rows.append(rf"&\text{{field }} {name}: \\")
        rows.append(rf"&\quad\text{{strong:}} && {strong} = 0 \\")
        rows.append(rf"&\quad\text{{weak:}} && {_weak_latex(physics, name)} \\")
        rows.append(rf"&\quad\text{{flux:}} && f_1 \cdot n = {flux} \\")
    return "\\begin{aligned}\n" + "\n".join(rows) + "\n\\end{aligned}"
