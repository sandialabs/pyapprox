r"""Stokes beside Darcy with Beavers-Joseph-Saffman, coupled through the
interface pressure (``tests/_helpers/coupling_stokes_darcy.py``). NumPy
only: skfem is.

Stokes Taylor-Hood Q2-Q1 on [0, 1/2] x [0, 1] as a port-loaded response
(traction -lambda n in, normal velocity out, through a normal-projection
transfer), primal Darcy Q2 on [1/2, 1] x [0, 1] pinned to lambda
(declared as receiving the interface pressure and returning the normal
velocity); 8 skeleton nodes; 4, 8, 16 cells per direction on each side.

- With BJS: the Stokes velocity converges at 3, its pressure at 2 and
  the Darcy pressure at 3 (last pair of levels, within 0.1); the
  interface masses cancel against every skeleton function (1e-12).
- Controls, which stall (rate below 0.5): BJS dropped, and the Stokes
  normal flipped; both still balance the masses, so the balance alone
  proves nothing.
- The declared kinds give one unknown per skeleton DOF; DeclaredKinds
  refuses a response outside its protocol and leaves everything but the
  kinds to the wrapped response.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import math

import numpy as np

from pyapprox.pde.coupling.responses import DeclaredKinds, Flux, Trace
from tests._helpers.coupling_stokes_darcy import (
    InterfacePressure,
    NormalVelocity,
    StokesDarcyStrip,
)

_LEVELS = (4, 8, 16)
_NSKEL = 8


def _errors(bjs: bool = True, stokes_sign: float = -1.0) -> list[dict[str, float]]:
    out = []
    for ncells in _LEVELS:
        strip = StokesDarcyStrip(ncells, _NSKEL, bjs, stokes_sign)
        out.append(strip.errors(strip.solve_skeleton()))
    return out


def _rate(errors: list[dict[str, float]], name: str) -> float:
    return math.log2(errors[-2][name] / errors[-1][name])


def test_bjs_converges_at_the_optimal_rates() -> None:
    errors = _errors()
    for name, expected in (("velocity", 3.0), ("pressure", 2.0), ("darcy", 3.0)):
        assert abs(_rate(errors, name) - expected) < 0.1, (name, errors)


@pytest.mark.parametrize(
    "bjs,stokes_sign", [(False, -1.0), (True, 1.0)], ids=["no_bjs", "flipped"]
)
def test_controls_stall(bjs: bool, stokes_sign: float) -> None:
    errors = _errors(bjs, stokes_sign)
    assert _rate(errors, "velocity") < 0.5, errors
    assert errors[-1]["velocity"] > 1e-2, errors


@pytest.mark.parametrize("bjs", [True, False])
def test_interface_masses_cancel(bjs: bool) -> None:
    strip = StokesDarcyStrip(8, _NSKEL, bjs=bjs)
    stokes, darcy = strip.gathered(strip.solve_skeleton())
    assert float(np.abs(stokes).max()) > 1e-2
    np.testing.assert_allclose(stokes + darcy, 0.0, atol=1e-12)


def test_declared_kinds_give_one_unknown_per_dof() -> None:
    strip = StokesDarcyStrip(4, _NSKEL)
    layout = strip.layout()
    assert layout.nunknowns() == strip.exact_pressure().shape[0]
    assert layout.coupling == ()


def test_declared_kinds_wrap_only_the_kinds() -> None:
    strip = StokesDarcyStrip(4, _NSKEL)
    inner = strip.darcy_pinned()
    wrapped = DeclaredKinds(inner, InterfacePressure(), NormalVelocity())
    assert (inner.receives(), inner.returns()) == (Trace(), Flux())
    assert (wrapped.receives(), wrapped.returns()) == (
        InterfacePressure(),
        NormalVelocity(),
    )
    assert wrapped.wrapped() is inner
    assert (wrapped.nport(), wrapped.nparams(), wrapped.nqoi()) == (
        inner.nport(),
        inner.nparams(),
        inner.nqoi(),
    )
    assert wrapped.bkd() is inner.bkd()
    with pytest.raises(TypeError):
        DeclaredKinds(object(), Trace(), Flux())  # type: ignore[arg-type]
