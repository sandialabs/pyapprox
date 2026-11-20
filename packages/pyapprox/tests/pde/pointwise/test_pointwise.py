r"""Pointwise physics: specification, symbolic derivatives, generated sets
and the backend kernel.

Weak residuals :math:`R(u; v) = \int v f_0 + \nabla v \cdot f_1\, dx`,
i.e. strong forms :math:`-\nabla \cdot f_1 + f_0 = 0`, in 2D:

- Poisson :math:`-\Delta u = f` (``f`` an aux field): :math:`f_0 = -f`,
  :math:`f_1 = \nabla u`.
- Helmholtz :math:`-\Delta u - k^2 u = f`: :math:`f_0 = -k_2 u - f`,
  :math:`f_1 = \nabla u`, with the parameter ``k2`` standing for
  :math:`k^2`.
- A nonlinear coupled pair ``u, w`` with parameters ``alpha, beta``, aux
  field ``f`` used by value and gradient, and dependence on ``x`` and
  ``t``:
  :math:`f_{0,u} = \alpha u^2 w - f + \sin(x) u_x w`,
  :math:`f_{1,u} = ((1 + \alpha u^2) u_x + \beta w, (1 + \alpha u^2) u_y)`,
  :math:`f_{0,w} = e^{\beta w} - u f_y`,
  :math:`f_{1,w} = (w_x, \cos(t) u w_y)`.
- A one-way coupled pair: :math:`f_{0,u} = u w`,
  :math:`f_{1,u} = \nabla u`, :math:`f_{1,w} = \nabla w`, so the
  ``(w, u)`` block is zero.

Symbolic: Poisson's ``g0, g1, g2`` and Helmholtz's ``g1, g2`` are absent,
the ``g3`` of both the identity, Helmholtz's ``g0 = -k2`` and its
``k2``-derivative ``f0 = -u`` with ``f1`` absent; the one-way pair
reports ``(w, u)`` zero.

Generated sets (no backend): the documented entry order; zero terms have
no entry; constants and ``pi`` become floats; a subexpression shared by
the nonlinear pair's outputs is computed once, and substituting the
replacements back recovers the derivatives exactly; ``_cse`` names are
refused.

Forms (no backend): Poisson's strong residual is ``-f - u_xx - u_yy`` and
its flux ``nx u_x + ny u_y``; for the nonlinear pair, with concrete
functions substituted for every jet, the strong residual equals sympy's
direct :math:`f_0 - \nabla \cdot f_1` of the composed expressions
(simplifies to zero); the text and LaTeX descriptions of Poisson match
their expected strings.

Kernel, on both backends at five random points: Helmholtz values against
the formulas (1e-12); every Jacobian block and parameter derivative of
every spec against central differences of the kernel's residual (step
1e-6; 1e-7 relative, 1e-8 absolute); ``DerivativeChecker`` on the
nonlinear pair's state and parameter Jacobians at each point (error
ratio at most 1e-6); on torch the same Jacobians against
``torch.func.jacrev`` of the residual (1e-10). The
backend's dtype, a pickle round trip, and refusals of bad specs and
inputs.
"""

import pickle
from typing import Callable, Dict, Generic, List, Optional, Union

import numpy as np
import pytest
import sympy as sp

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.expressions.symbols import jets, standard
from pyapprox.pde.pointwise.derivatives import PointwiseDerivatives
from pyapprox.pde.pointwise.forms import (
    boundary_flux,
    describe,
    describe_latex,
    strong_form_difference,
    strong_residual,
    total_derivative,
)
from pyapprox.pde.pointwise.generated import generate
from pyapprox.pde.pointwise.kernel import (
    BackendPointwiseKernel,
    PointwiseInputs,
    PointwiseJacobianBlock,
    PointwiseResidual,
    PointwiseTerms,
)
from pyapprox.pde.pointwise.spec import AuxField, Field, PointwisePhysics
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.expressions.names import CustomFunction

_XY = standard(2)
_NQ = 5


def _poisson() -> PointwisePhysics:
    return PointwisePhysics(
        _XY,
        [Field("u")],
        f0={"u": "-f"},
        f1={"u": ("u_x", "u_y")},
        aux=[AuxField("f")],
    )


def _helmholtz() -> PointwisePhysics:
    return PointwisePhysics(
        _XY,
        [Field("u")],
        f0={"u": "-k2*u - f"},
        f1={"u": ("u_x", "u_y")},
        params=["k2"],
        aux=[AuxField("f")],
    )


def _nonlinear_pair() -> PointwisePhysics:
    return PointwisePhysics(
        _XY,
        [Field("u"), Field("w")],
        f0={"u": "alpha*u**2*w - f + sin(x)*u_x*w", "w": "exp(beta*w) - u*f_y"},
        f1={
            "u": ("(1 + alpha*u**2)*u_x + beta*w", "(1 + alpha*u**2)*u_y"),
            "w": ("w_x", "cos(t)*u*w_y"),
        },
        params=["alpha", "beta"],
        aux=[AuxField("f")],
    )


def _one_way_pair() -> PointwisePhysics:
    return PointwisePhysics(
        _XY,
        [Field("u"), Field("w")],
        f0={"u": "u*w"},
        f1={"u": ("u_x", "u_y"), "w": ("w_x", "w_y")},
    )


_SPECS: Dict[str, Callable[[], PointwisePhysics]] = {
    "poisson": _poisson,
    "helmholtz": _helmholtz,
    "nonlinear_pair": _nonlinear_pair,
    "one_way_pair": _one_way_pair,
}


def _identity(bkd: Backend[Array], value: Array) -> Array:
    return value


def _inputs(physics: PointwisePhysics, bkd: Backend[Array]) -> PointwiseInputs[Array]:
    rng = np.random.default_rng(1)
    ndim = physics.ndim()
    named = list(physics.field_names()) + [data.name for data in physics.aux()]
    return PointwiseInputs(
        coords=bkd.asarray(rng.uniform(0.0, 1.0, (ndim, _NQ))),
        values={n: bkd.asarray(rng.uniform(-1.0, 1.0, (_NQ,))) for n in named},
        gradients={n: bkd.asarray(rng.uniform(-1.0, 1.0, (ndim, _NQ))) for n in named},
        params=bkd.asarray(rng.uniform(0.5, 1.5, (physics.nparams(),))),
        t=0.3,
    )


def _replaced(
    inputs: PointwiseInputs[Array],
    values: Optional[Dict[str, Array]] = None,
    gradients: Optional[Dict[str, Array]] = None,
    params: Optional[Array] = None,
) -> PointwiseInputs[Array]:
    """``inputs`` with some values, gradients or the parameters replaced."""
    return PointwiseInputs(
        inputs.coords,
        {**inputs.values, **(values or {})},
        {**inputs.gradients, **(gradients or {})},
        inputs.params if params is None else params,
        inputs.t,
    )


def _stacked(residual: PointwiseResidual[Array], bkd: Backend[Array]) -> Array:
    """``(f0, f1)`` as rows, shape ``(1 + ndim, nq)``."""
    return bkd.concatenate([residual.f0[None, :], residual.f1], axis=0)


def _dense_block(
    block: Optional[PointwiseJacobianBlock[Array]],
    ndim: int,
    npts: int,
    bkd: Backend[Array],
) -> Array:
    """The pointwise Jacobian of ``(f0, f1)`` by ``(u, ∇u)``, absent terms
    as zeros, shape ``(1 + ndim, 1 + ndim, npts)``."""
    g0 = bkd.zeros((npts,))
    g1 = g2 = bkd.zeros((ndim, npts))
    g3 = bkd.zeros((ndim, ndim, npts))
    if block is not None:
        g0 = g0 if block.g0 is None else block.g0
        g1 = g1 if block.g1 is None else block.g1
        g2 = g2 if block.g2 is None else block.g2
        g3 = g3 if block.g3 is None else block.g3
    top = bkd.concatenate([g0[None, None, :], g1[None, :, :]], axis=1)
    bottom = bkd.concatenate([g2[:, None, :], g3], axis=1)
    return bkd.concatenate([top, bottom], axis=0)


def _dense_terms(
    terms: Optional[PointwiseTerms[Array]],
    ndim: int,
    npts: int,
    bkd: Backend[Array],
) -> Array:
    """``(∂f0, ∂f1)`` by one parameter, absent terms as zeros, shape
    ``(1 + ndim, npts)``."""
    f0, f1 = bkd.zeros((npts,)), bkd.zeros((ndim, npts))
    if terms is not None:
        f0 = f0 if terms.f0 is None else terms.f0
        f1 = f1 if terms.f1 is None else terms.f1
    return bkd.concatenate([f0[None, :], f1], axis=0)


def _shifted(
    inputs: PointwiseInputs[Array],
    bkd: Backend[Array],
    field: str,
    column: int,
    step: float,
) -> PointwiseInputs[Array]:
    """``inputs`` with ``field``'s value (``column = 0``) or gradient
    component ``column - 1`` shifted by ``step`` at every point."""
    if column == 0:
        return _replaced(inputs, values={field: inputs.values[field] + step})
    grad = inputs.gradients[field]
    ndim = grad.shape[0]
    unit = bkd.asarray([[1.0 if ii == column - 1 else 0.0] for ii in range(ndim)])
    return _replaced(inputs, gradients={field: grad + step * unit})


class TestSpec:
    def test_integrands(self) -> None:
        physics = _helmholtz()
        u, k2, f = sp.symbols("u k2 f")
        assert physics.f0("u") == -k2 * u - f
        assert physics.f1("u") == sp.symbols("u_x u_y")
        assert physics.field_names() == ("u",)
        assert physics.params() == ("k2",)
        # A field without integrands has zero ones.
        assert _one_way_pair().f0("w") == 0

    def test_constants_stay_symbols(self) -> None:
        physics = PointwisePhysics(
            _XY, [Field("u")], {"u": "c*u"}, {}, constants={"c": 2.0}
        )
        assert physics.f0("u") == sp.Symbol("c") * sp.Symbol("u")
        assert physics.f1("u") == (0, 0)

    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            ({"f0": {"w": "u"}}, ValueError, "not a field"),
            ({"f1": {"u": ("u_x",)}}, ValueError, "2 components"),
            ({"f1": {"u": "u_x"}}, ValueError, "2 components"),
            ({"f0": {"u": "nx*u"}}, ValueError, "outward normal"),
            ({"f0": {"u": "u_xx"}}, ValueError, "unknown name 'u_xx'"),
            ({"f0": {"u": "g"}}, ValueError, "unknown name 'g'"),
            ({"params": ["u_x"]}, ValueError, "more than once"),
            ({"params": ["t"]}, ValueError, "more than once"),
            ({"aux": [AuxField("u")]}, ValueError, "more than once"),
            ({"fields": []}, ValueError, "at least one field"),
            ({"coords": ()}, ValueError, "at least one coordinate"),
            ({"fields": ["u"]}, TypeError, "must be Field"),
            ({"aux": ["f"]}, TypeError, "must be AuxField"),
        ],
    )
    def test_refusals(self, kwargs: Dict[str, object], error: type, match: str) -> None:
        arguments: Dict[str, object] = {
            "coords": _XY,
            "fields": [Field("u")],
            "f0": {},
            "f1": {},
        }
        arguments.update(kwargs)
        with pytest.raises(error, match=match):
            PointwisePhysics(**arguments)  # type: ignore[arg-type]

    def test_field_refusals(self) -> None:
        with pytest.raises(NotImplementedError, match="scalar"):
            Field("u", shape=(2,))
        with pytest.raises(ValueError, match="field name"):
            Field("u_1")
        with pytest.raises(ValueError, match="field name"):
            AuxField("t")


class TestSymbolic:
    def test_poisson(self) -> None:
        derivatives = PointwiseDerivatives(_poisson())
        block = derivatives.jacobian_block("u", "u")
        assert block is not None
        assert (block.g0, block.g1, block.g2) == (None, None, None)
        assert block.g3 == ((1, 0), (0, 1))
        assert derivatives.zero_jacobian_blocks() == ()

    def test_helmholtz(self) -> None:
        derivatives = PointwiseDerivatives(_helmholtz())
        block = derivatives.jacobian_block("u", "u")
        assert block is not None
        assert block.g0 == -sp.Symbol("k2")
        assert (block.g1, block.g2) == (None, None)
        assert block.g3 == ((1, 0), (0, 1))
        terms = derivatives.parameter_derivative("u", "k2")
        assert terms is not None
        assert terms.f0 == -sp.Symbol("u") and terms.f1 is None

    def test_zero_block_reported(self) -> None:
        derivatives = PointwiseDerivatives(_one_way_pair())
        assert derivatives.zero_jacobian_blocks() == (("w", "u"),)
        assert derivatives.jacobian_block("w", "u") is None
        coupling = derivatives.jacobian_block("u", "w")
        assert coupling is not None and coupling.g0 == sp.Symbol("u")
        assert (coupling.g1, coupling.g2, coupling.g3) == (None, None, None)

    def test_parameter_without_effect(self) -> None:
        physics = PointwisePhysics(
            _XY, [Field("u")], {"u": "a*u"}, {}, params=["a", "b"]
        )
        assert PointwiseDerivatives(physics).parameter_derivative("u", "b") is None

    def test_undifferentiable_function(self) -> None:
        unknown = CustomFunction(sp.Function("g"), _identity)
        physics = PointwisePhysics(
            _XY, [Field("u")], {"u": "g(u)"}, {}, functions={"g": unknown}
        )
        with pytest.raises(ValueError, match="cannot differentiate"):
            PointwiseDerivatives(physics)

    def test_refuses_non_physics(self) -> None:
        with pytest.raises(TypeError, match="PointwisePhysics"):
            PointwiseDerivatives("u")  # type: ignore[arg-type]


class TestGenerated:
    def test_layout(self) -> None:
        sets = generate(PointwiseDerivatives(_one_way_pair()))
        assert [e.key for e in sets.residual.entries] == [
            ("f0", "u"),
            ("f1", "u"),
            ("f0", "w"),
            ("f1", "w"),
        ]
        assert [(e.start, e.shape) for e in sets.residual.entries] == [
            (0, ()),
            (1, (2,)),
            (3, ()),
            (4, (2,)),
        ]
        assert sets.residual.noutputs() == 6
        # Zero terms and the zero (w, u) block have no entry.
        assert [e.key for e in sets.jacobian.entries] == [
            ("g0", "u", "u"),
            ("g3", "u", "u"),
            ("g0", "u", "w"),
            ("g3", "w", "w"),
        ]
        assert not sets.jacobian.has(("g0", "w", "u"))
        assert sets.jacobian.entry(("g3", "w", "w")).start == 6
        with pytest.raises(KeyError):
            sets.jacobian.entry(("g1", "u", "u"))
        assert sets.parameters.noutputs() == 0

    def test_variables_in_name_order(self) -> None:
        sets = generate(PointwiseDerivatives(_helmholtz()))
        assert sets.residual.variables == sp.symbols("u u_x u_y f k2")
        assert sets.jacobian.variables == (sp.Symbol("k2"),)

    def test_constants_become_numbers(self) -> None:
        physics = PointwisePhysics(
            _XY, [Field("u")], {"u": "c*pi*u"}, {}, constants={"c": 2.0}
        )
        (output,) = generate(PointwiseDerivatives(physics)).jacobian.outputs
        assert output.is_Float
        assert abs(float(output) - 2.0 * np.pi) < 1e-14

    def test_common_subexpressions(self) -> None:
        physics = _nonlinear_pair()
        derivatives = PointwiseDerivatives(physics)
        sets = generate(derivatives)
        jacobian = sets.jacobian
        # alpha*u**2 appears in g2 and g3 of (u, u); it is computed once.
        assert len(jacobian.replacements) > 0
        assert all(s.name.startswith("_cse") for s, _ in jacobian.replacements)
        block = derivatives.jacobian_block("u", "u")
        assert block is not None and block.g3 is not None
        entry = jacobian.entry(("g3", "u", "u"))
        expanded = jacobian.expanded()[entry.start : entry.start + entry.size()]
        expected = [e for row in block.g3 for e in row]
        assert all(sp.simplify(a - b) == 0 for a, b in zip(expanded, expected))

    def test_refusals(self) -> None:
        physics = PointwisePhysics(
            _XY, [Field("u")], {"u": "_cse0*u"}, {}, params=["_cse0"]
        )
        with pytest.raises(ValueError, match="reserved"):
            generate(PointwiseDerivatives(physics))
        with pytest.raises(TypeError, match="PointwiseDerivatives"):
            generate(physics)  # type: ignore[arg-type]

    def test_imports_no_backend(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import pyapprox.pde.pointwise.generated\n"
            "heavy = [m for m in ('pyapprox.util.backends.numpy', 'torch')\n"
            "         if m in sys.modules]\n"
            "print(heavy)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"


class TestForms:
    def test_poisson(self) -> None:
        physics = _poisson()
        f, u_xx, u_yy = sp.symbols("f u_xx u_yy")
        assert strong_residual(physics, "u") == -f - u_xx - u_yy
        nx, ny, u_x, u_y = sp.symbols("nx ny u_x u_y")
        assert boundary_flux(physics, "u") == nx * u_x + ny * u_y

    def test_matches_direct_differentiation(self) -> None:
        physics = _nonlinear_pair()
        x, y, t = sp.symbols("x y t")
        concrete = {
            "u": sp.sin(x) * sp.cos(2 * y) + t,
            "w": sp.exp(x * y),
            "f": x**2 + y**3,
        }
        # Every jet up to second order, replaced by the concrete function's
        # derivative.
        substitution = {
            symbol: (
                sp.diff(function, *[sp.Symbol(d) for d in directions])
                if directions
                else function
            )
            for name, function in concrete.items()
            for symbol, directions in jets(name, _XY, 2).items()
        }
        for field in physics.field_names():
            composed_f1 = [c.xreplace(substitution) for c in physics.f1(field)]
            direct = physics.f0(field).xreplace(substitution) - (
                sp.diff(composed_f1[0], x) + sp.diff(composed_f1[1], y)
            )
            generated = strong_residual(physics, field).xreplace(substitution)
            assert sp.simplify(generated - direct) == 0

    def test_describe(self) -> None:
        assert describe(_poisson()) == (
            "field u:\n"
            "  strong:  -f - u_xx - u_yy = 0\n"
            "  weak:    integral of v_u*(-f) + dv_u/dx*(u_x) + dv_u/dy*(u_y) dx\n"
            "  flux:    f1.n = nx*u_x + ny*u_y"
        )
        latex = describe_latex(_poisson())
        assert latex.startswith("\\begin{aligned}")
        assert r"- f - u_{xx} - u_{yy} = 0" in latex
        assert r"\partial_{x} v_{u} \left(u_{x}\right)" in latex
        assert r"f_1 \cdot n = n_{x} u_{x} + n_{y} u_{y}" in latex

    def test_strong_form_check(self) -> None:
        # Nonlinear diffusion -div((1 + alpha u^2) grad u) = f, written out.
        physics = PointwisePhysics(
            _XY,
            [Field("u")],
            {"u": "-f"},
            {"u": ("(1 + alpha*u**2)*u_x", "(1 + alpha*u**2)*u_y")},
            params=["alpha"],
            aux=[AuxField("f")],
        )
        meant = "-(1 + alpha*u**2)*(u_xx + u_yy) - 2*alpha*u*(u_x**2 + u_y**2) - f"
        assert strong_form_difference(physics, "u", meant) == 0
        # Wrong-model controls: a dropped chain-rule term, a flipped sign.
        dropped = "-(1 + alpha*u**2)*(u_xx + u_yy) - f"
        assert strong_form_difference(physics, "u", dropped) != 0
        flipped = "(1 + alpha*u**2)*(u_xx + u_yy) + 2*alpha*u*(u_x**2 + u_y**2) + f"
        assert strong_form_difference(physics, "u", flipped) != 0
        # A flux with the wrong sign fails against the intended Poisson.
        wrong = PointwisePhysics(
            _XY,
            [Field("u")],
            {"u": "-f"},
            {"u": ("-u_x", "-u_y")},
            aux=[AuxField("f")],
        )
        assert strong_form_difference(wrong, "u", "-u_xx - u_yy - f") != 0
        assert strong_form_difference(_poisson(), "u", "-u_xx - u_yy - f") == 0

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="unknown name 'u_xxx'"):
            strong_form_difference(_poisson(), "u", "u_xxx")
        with pytest.raises(ValueError, match="direction"):
            total_derivative(_poisson(), sp.Symbol("u"), "z")
        with pytest.raises(KeyError):
            strong_residual(_poisson(), "w")


class TestKernel:
    def test_helmholtz_values(self, bkd: Backend[Array]) -> None:
        physics = _helmholtz()
        kernel = BackendPointwiseKernel(physics, bkd)
        inputs = _inputs(physics, bkd)
        assert inputs.params is not None
        k2, u = inputs.params[0], inputs.values["u"]
        residual = kernel.residual(inputs)["u"]
        bkd.assert_allclose(residual.f0, -k2 * u - inputs.values["f"], rtol=1e-12)
        bkd.assert_allclose(residual.f1, inputs.gradients["u"], rtol=1e-12)
        block = kernel.jacobian(inputs)[("u", "u")]
        assert block is not None and block.g0 is not None and block.g3 is not None
        assert block.g1 is None and block.g2 is None
        bkd.assert_allclose(block.g0, bkd.zeros((_NQ,)) - k2, rtol=1e-12)
        assert block.g3.shape == (2, 2, _NQ)
        bkd.assert_allclose(block.g3[:, :, 2], bkd.eye(2), rtol=1e-12)
        terms = kernel.parameter_derivatives(inputs)[("u", "k2")]
        assert terms is not None and terms.f0 is not None and terms.f1 is None
        bkd.assert_allclose(terms.f0, -u, rtol=1e-12)

    @pytest.mark.parametrize("spec", sorted(_SPECS))
    def test_central_differences(self, bkd: Backend[Array], spec: str) -> None:
        physics = _SPECS[spec]()
        kernel = BackendPointwiseKernel(physics, bkd)
        inputs = _inputs(physics, bkd)
        ndim, step, fields = physics.ndim(), 1e-6, physics.field_names()
        blocks = kernel.jacobian(inputs)
        for trial in fields:
            for column in range(1 + ndim):
                plus = kernel.residual(_shifted(inputs, bkd, trial, column, step))
                minus = kernel.residual(_shifted(inputs, bkd, trial, column, -step))
                for test in fields:
                    fd = (_stacked(plus[test], bkd) - _stacked(minus[test], bkd)) / (
                        2 * step
                    )
                    dense = _dense_block(blocks[(test, trial)], ndim, _NQ, bkd)
                    bkd.assert_allclose(dense[:, column, :], fd, rtol=1e-7, atol=1e-8)
        params = inputs.params
        assert params is not None
        terms = kernel.parameter_derivatives(inputs)
        for kk, param in enumerate(physics.params()):
            unit = bkd.asarray(
                [1.0 if jj == kk else 0.0 for jj in range(physics.nparams())]
            )
            plus = kernel.residual(_replaced(inputs, params=params + step * unit))
            minus = kernel.residual(_replaced(inputs, params=params - step * unit))
            for test in fields:
                fd = (_stacked(plus[test], bkd) - _stacked(minus[test], bkd)) / (
                    2 * step
                )
                dense = _dense_terms(terms[(test, param)], ndim, _NQ, bkd)
                bkd.assert_allclose(dense, fd, rtol=1e-7, atol=1e-8)

    def test_derivative_checker(self, bkd: Backend[Array]) -> None:
        physics = _nonlinear_pair()
        kernel = BackendPointwiseKernel(physics, bkd)
        inputs = _inputs(physics, bkd)
        for point in range(_NQ):
            single = _at_point(inputs, point)
            for test in physics.field_names():
                for trial in physics.field_names():
                    state = _StateFunction(kernel, single, test, trial)
                    _assert_checks(state.as_function(), state.sample())
                params = _ParamFunction(kernel, single, test)
                _assert_checks(params.as_function(), params.sample())

    def test_torch_autograd(self, torch_bkd: TorchBkd) -> None:
        import torch

        bkd = torch_bkd
        physics = _nonlinear_pair()
        kernel = BackendPointwiseKernel(physics, bkd)
        inputs = _at_point(_inputs(physics, bkd), 3)
        for test in physics.field_names():
            functions: List[
                Union[_StateFunction[torch.Tensor], _ParamFunction[torch.Tensor]]
            ] = [
                _StateFunction(kernel, inputs, test, trial)
                for trial in physics.field_names()
            ]
            functions.append(_ParamFunction(kernel, inputs, test))
            for function in functions:
                sample = function.sample()
                expected: torch.Tensor = torch.func.jacrev(function.flat_value)(
                    sample[:, 0]
                )
                bkd.assert_allclose(function.jacobian(sample), expected, rtol=1e-10)

    def test_dtype_follows_the_backend(self, bkd: Backend[Array]) -> None:
        physics = _nonlinear_pair()
        kernel = BackendPointwiseKernel(physics, bkd)
        inputs = _inputs(physics, bkd)
        dtype = bkd.zeros((1,)).dtype
        assert kernel.residual(inputs)["w"].f1.dtype == dtype
        block = kernel.jacobian(inputs)[("u", "u")]
        assert block is not None and block.g3 is not None
        assert block.g3.dtype == dtype

    def test_pickles(self, bkd: Backend[Array]) -> None:
        physics = _helmholtz()
        kernel = BackendPointwiseKernel(physics, bkd)
        copy = pickle.loads(pickle.dumps(kernel))
        inputs = _inputs(physics, bkd)
        bkd.assert_allclose(
            copy.residual(inputs)["u"].f0, kernel.residual(inputs)["u"].f0
        )

    def test_unused_inputs_not_needed(self, numpy_bkd: Backend[Array]) -> None:
        kernel = BackendPointwiseKernel(_poisson(), numpy_bkd)
        # Poisson's Jacobian is constant: no fields, aux data or parameters.
        block = kernel.jacobian(PointwiseInputs(numpy_bkd.zeros((2, 3))))[("u", "u")]
        assert block is not None and block.g3 is not None
        assert block.g3.shape == (2, 2, 3)
        assert (
            kernel.parameter_derivatives(PointwiseInputs(numpy_bkd.zeros((2, 3)))) == {}
        )

    def test_unsupported_backend_function(self, numpy_bkd: Backend[Array]) -> None:
        physics = PointwisePhysics(_XY, [Field("u")], {"u": "tan(u)"}, {})
        with pytest.raises(ValueError, match=r"no backend method for \['tan'\]"):
            BackendPointwiseKernel(physics, numpy_bkd)

    def test_input_refusals(self, numpy_bkd: Backend[Array]) -> None:
        bkd = numpy_bkd
        physics = _helmholtz()
        kernel = BackendPointwiseKernel(physics, bkd)
        good = _inputs(physics, bkd)
        with pytest.raises(ValueError, match="coords must have shape"):
            kernel.residual(PointwiseInputs(bkd.zeros((3, _NQ)), good.values))
        with pytest.raises(ValueError, match="value of 'f'"):
            kernel.residual(
                PointwiseInputs(
                    good.coords, {"u": good.values["u"]}, good.gradients, good.params
                )
            )
        with pytest.raises(ValueError, match="gradient of 'u'"):
            kernel.residual(_replaced(good, gradients={"u": bkd.zeros((2, 2))}))
        with pytest.raises(ValueError, match="parameter k2"):
            kernel.residual(PointwiseInputs(good.coords, good.values, good.gradients))
        with pytest.raises(ValueError, match="params must have shape"):
            kernel.residual(_replaced(good, params=bkd.zeros((2,))))


def _at_point(inputs: PointwiseInputs[Array], point: int) -> PointwiseInputs[Array]:
    """``inputs`` restricted to one quadrature point."""
    window = slice(point, point + 1)
    return PointwiseInputs(
        inputs.coords[:, window],
        {n: v[window] for n, v in inputs.values.items()},
        {n: g[:, window] for n, g in inputs.gradients.items()},
        inputs.params,
        inputs.t,
    )


def _assert_checks(
    function: FunctionWithJacobianFromCallable[Array], sample: Array
) -> None:
    checker = DerivativeChecker(function)
    # Central differences: entries linear in the input leave forward
    # differences only about six decades deep, too shallow for the ratio.
    errors = checker.check_derivatives(sample, central=True)
    assert checker.bkd().to_float(checker.error_ratio(errors[0])) <= 1e-6


class _StateFunction(Generic[Array]):
    """At one point, ``(u_b, ∇u_b) -> (f0_a, f1_a)`` with every other input
    fixed, and its generated Jacobian."""

    def __init__(
        self,
        kernel: BackendPointwiseKernel[Array],
        inputs: PointwiseInputs[Array],
        test: str,
        trial: str,
    ) -> None:
        self._kernel, self._inputs = kernel, inputs
        self._test, self._trial = test, trial

    def sample(self) -> Array:
        """The trial field's value and gradient, shape ``(1 + ndim, 1)``."""
        return self._kernel.bkd().concatenate(
            [
                self._inputs.values[self._trial][:, None],
                self._inputs.gradients[self._trial],
            ],
            axis=0,
        )

    def _at(self, sample: Array) -> PointwiseInputs[Array]:
        return _replaced(
            self._inputs,
            values={self._trial: sample[0]},
            gradients={self._trial: sample[1:]},
        )

    def value(self, sample: Array) -> Array:
        residual = self._kernel.residual(self._at(sample))[self._test]
        return _stacked(residual, self._kernel.bkd())

    def flat_value(self, column: Array) -> Array:
        return self.value(column[:, None])[:, 0]

    def jacobian(self, sample: Array) -> Array:
        block = self._kernel.jacobian(self._at(sample))[(self._test, self._trial)]
        ndim = self._kernel.physics().ndim()
        return _dense_block(block, ndim, 1, self._kernel.bkd())[:, :, 0]

    def as_function(self) -> FunctionWithJacobianFromCallable[Array]:
        n = 1 + self._kernel.physics().ndim()
        return FunctionWithJacobianFromCallable(
            n, n, self.value, self.jacobian, self._kernel.bkd()
        )


class _ParamFunction(Generic[Array]):
    """At one point, ``p -> (f0_a, f1_a)`` and its generated Jacobian."""

    def __init__(
        self,
        kernel: BackendPointwiseKernel[Array],
        inputs: PointwiseInputs[Array],
        test: str,
    ) -> None:
        self._kernel, self._inputs, self._test = kernel, inputs, test

    def sample(self) -> Array:
        """The parameters, shape ``(nparams, 1)``."""
        params = self._inputs.params
        if params is None:
            raise ValueError("these inputs have no parameters")
        return params[:, None]

    def value(self, sample: Array) -> Array:
        inputs = _replaced(self._inputs, params=sample[:, 0])
        return _stacked(self._kernel.residual(inputs)[self._test], self._kernel.bkd())

    def flat_value(self, column: Array) -> Array:
        return self.value(column[:, None])[:, 0]

    def jacobian(self, sample: Array) -> Array:
        bkd, physics = self._kernel.bkd(), self._kernel.physics()
        terms = self._kernel.parameter_derivatives(
            _replaced(self._inputs, params=sample[:, 0])
        )
        columns = [
            _dense_terms(terms[(self._test, param)], physics.ndim(), 1, bkd)
            for param in physics.params()
        ]
        return bkd.concatenate(columns, axis=1)

    def as_function(self) -> FunctionWithJacobianFromCallable[Array]:
        physics = self._kernel.physics()
        return FunctionWithJacobianFromCallable(
            1 + physics.ndim(),
            physics.nparams(),
            self.value,
            self.jacobian,
            self._kernel.bkd(),
        )
