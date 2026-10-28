"""Each ODE benchmark residual's derivatives bundle, by finite differences.

At a random state, adjoint and parameter (seed 7; states in [0.2, 1.5],
parameters in [0.2, 1.0], t = 0.3), every term the bundle declares
passes the bundle checker: forward-difference errors below 1e-6, dot
tests below 1e-12, stated zeros zero.
"""

from typing import Callable, Union

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.functions.ode.chemical_reaction import (
    ChemicalReactionResidual,
)
from pyapprox_benchmarks.functions.ode.coupled_springs import CoupledSpringsResidual
from pyapprox_benchmarks.functions.ode.hastings_ecology import (
    HastingsEcologyResidual,
)
from pyapprox_benchmarks.functions.ode.linear_ode import (
    LinearODEResidual,
    QuadraticODEResidual,
)
from pyapprox_benchmarks.functions.ode.lotka_volterra import LotkaVolterraResidual
from pyapprox_benchmarks.functions.ode.time_modulated_quadratic import (
    TimeModulatedQuadraticODE,
)

_Residual = Union[
    LinearODEResidual[Array],
    QuadraticODEResidual[Array],
    TimeModulatedQuadraticODE[Array],
    ChemicalReactionResidual[Array],
    CoupledSpringsResidual[Array],
    LotkaVolterraResidual[Array],
    HastingsEcologyResidual[Array],
]
_A3 = [[-1.0, 0.2, 0.0], [0.1, -2.0, 0.3], [0.0, 0.4, -1.5]]


def _factories() -> dict[str, Callable[[Backend[Array]], _Residual[Array]]]:
    return {
        "linear": lambda bkd: LinearODEResidual(
            bkd.asarray(_A3), bkd.asarray([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]]), bkd
        ),
        "quadratic": lambda bkd: QuadraticODEResidual(bkd.asarray(_A3), bkd),
        "time-modulated": lambda bkd: TimeModulatedQuadraticODE(bkd.asarray(_A3), bkd),
        "chemical": lambda bkd: ChemicalReactionResidual(bkd),
        "springs": lambda bkd: CoupledSpringsResidual(bkd),
        "lotka-volterra": lambda bkd: LotkaVolterraResidual(3, bkd),
        "hastings": lambda bkd: HastingsEcologyResidual(bkd),
    }


@pytest.mark.parametrize("name", list(_factories()))
def test_bundle_passes_the_checker(bkd: Backend[Array], name: str) -> None:
    residual = _factories()[name](bkd)
    nparams = residual.nparams()
    rng = np.random.default_rng(7)
    param = bkd.asarray(rng.uniform(0.2, 1.0, nparams))
    nstates = residual.mass_matrix().as_matrix().shape[0]
    state = bkd.asarray(rng.uniform(0.2, 1.5, nstates))
    adj = bkd.asarray(rng.normal(size=nstates))
    report = ODEResidualBundleChecker(residual).check(state, 0.3, adj, param)
    assert "param" in report.fd_errors
    assert report.failures() == []
