"""Each ODE benchmark residual's derivatives bundle against its methods.

The residuals offer their derivatives both as the operator bundle from
``derivatives()`` and as the matrix and vector methods the tiered
steppers read. At a random state and parameter, every populated operator
of the bundle must reproduce the corresponding method, applied and
transposed, and the terms a bundle states as zero must be declared zero.
"""

from typing import Callable, Union

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives import DerivativeOperator
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import LinearizationPoint
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


def _column(bkd: Backend[Array], vec: Array) -> Array:
    return bkd.reshape(vec, (vec.shape[0], 1))


def _applied(bkd: Backend[Array], op: DerivativeOperator[Array], vec: Array) -> Array:
    return op.apply(_column(bkd, vec))[:, 0]


def _transposed(
    bkd: Backend[Array], op: DerivativeOperator[Array], vec: Array
) -> Array:
    return op.apply_transpose(_column(bkd, vec))[:, 0]


@pytest.mark.parametrize("name", list(_factories()))
def test_bundle_reproduces_the_residuals_methods(
    bkd: Backend[Array], name: str
) -> None:
    residual = _factories()[name](bkd)
    nparams = residual.nparams()
    rng = np.random.default_rng(7)
    residual.set_param(bkd.asarray(rng.uniform(0.2, 1.0, nparams)))
    nstates = residual.mass_matrix().as_matrix().shape[0]
    state = bkd.asarray(rng.uniform(0.2, 1.5, nstates))
    adj = bkd.asarray(rng.normal(size=nstates))
    v = bkd.asarray(rng.normal(size=nparams))
    w = bkd.asarray(rng.normal(size=nstates))
    time = 0.3
    bundle = residual.derivatives()
    assert bundle.jacobian is not None and bundle.param is not None
    assert bundle.initial_state is not None
    jacobian = bundle.jacobian(state, time, LinearizationPoint(0))
    matrix = residual.jacobian(state, time)
    param_matrix = residual.param_jacobian(state, time)
    initial = residual.initial_param_jacobian()
    pairs = [
        (_applied(bkd, jacobian, w), bkd.dot(matrix, w)),
        (_transposed(bkd, jacobian, w), bkd.dot(matrix.T, w)),
        (_applied(bkd, bundle.param(state, time), v), bkd.dot(param_matrix, v)),
        (_transposed(bkd, bundle.param(state, time), w), bkd.dot(param_matrix.T, w)),
        (_applied(bkd, bundle.initial_state(), v), bkd.dot(initial, v)),
        (_transposed(bkd, bundle.initial_state(), w), bkd.dot(initial.T, w)),
    ]
    if bundle.state_state is not None:
        assert bundle.state_param is not None and bundle.param_param is not None
        assert isinstance(
            residual,
            (
                LinearODEResidual,
                QuadraticODEResidual,
                TimeModulatedQuadraticODE,
                LotkaVolterraResidual,
            ),
        )
        mixed = bundle.state_param(state, time, adj)
        pairs += [
            (
                _applied(bkd, bundle.state_state(state, time, adj), w),
                residual.state_state_hvp(state, time, adj, w),
            ),
            (_applied(bkd, mixed, v), residual.state_param_hvp(state, time, adj, v)),
            (
                _transposed(bkd, mixed, w),
                residual.param_state_hvp(state, time, adj, w),
            ),
            (
                _applied(bkd, bundle.param_param(state, time, adj), v),
                residual.param_param_hvp(state, time, adj, v),
            ),
        ]
    for actual, expected in pairs:
        bkd.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
