"""Importing the numerical constitutive and collocation physics packages
must not load sympy: only manufactured solutions need it."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "pyapprox.pde.constitutive",
        "pyapprox.pde.constitutive.coefficient_functions",
        "pyapprox.pde.collocation.physics",
    ],
)
def test_import_does_not_load_sympy(module: str) -> None:
    code = f"import sys\nimport {module}\nprint('sympy' in sys.modules)\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_symbolic_stress_still_works() -> None:
    """The sympy expressions import sympy where they are built."""
    import sympy as sp

    from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress

    F = sp.Symbol("F", positive=True)
    P = NeoHookeanStress(lamda=2.0, mu=1.0).sympy_stress_1d(F)
    assert sp.simplify(P.subs(F, 1)) == 0
