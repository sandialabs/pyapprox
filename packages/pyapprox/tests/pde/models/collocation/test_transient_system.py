r"""The transient collocation system through the shared transient map.

1D ADR on 12 Chebyshev nodes, Dirichlet 1 on the left and gradient Robin
(alpha 2, beta 0.5, g 0.3) on the right, the diffusion parameterized as
0.3 + p_0 + p_1 m(x); the functional is the value at the middle node at
t = 4e-3 (8 steps of 5e-4). Every case runs BE, CN and Heun under three
injected boundary treatments: the step-end pin, the pin at every block,
and the step-end pin with the essential rows differentiated
(``CoupledRows``).

- ``ImplicitMapFunction`` gradient and HVP pass central differences
  (best error below 1e-6, the error ratio below 1e-6), and the HVP is
  symmetric to 1e-12;
- the adjoint gradient equals the tangent Jacobian row of the final state
  to 1e-10;
- ``initial_incompatibility`` reports zero for compatible data and the
  mismatch otherwise, and ``initial_state`` sets the essential entries;
- refusals: a treatment without essential rows, essential data depending
  on the parameters, a non-system.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ArrayParameters,
    ImplicitMapFunction,
    TangentColumns,
)
from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.ode.differentiated_rows import (
    CoupledRows,
    RowRatesTreatmentProtocol,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.pinned_rows import AtEveryBlock, AtStepEnd, PinPlacementProtocol
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    heun,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    ParameterVectorSetterProtocol,
    TransientDiscretizationProtocol,
    TransientImplicitMap,
    TransientPoint,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
    MappedBoundaryData,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.field_maps.basis_expansion import BasisExpansion
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.transient_system import (
    CollocationTransientSystem,
    collocation_transient_system,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_N1D = 12
_OBSERVED = _N1D // 2
_SAMPLE = [[0.2], [-0.1]]
_DIRECTION = [[0.7], [-1.1]]

_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "backward_euler": backward_euler,
    "crank_nicolson": crank_nicolson,
    "heun": heun,
}
_TREATMENTS: dict[
    str, Tuple[Callable[[], PinPlacementProtocol], Optional[RowRatesTreatmentProtocol]]
] = {
    "step_end": (AtStepEnd, None),
    "every_block": (AtEveryBlock, None),
    "differentiated": (AtStepEnd, CoupledRows()),
}


def _mesh(bkd: Backend[Array]) -> TransformedMesh1D[Array]:
    return TransformedMesh1D(_N1D, bkd)


def _physics(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        ChebyshevBasis1D(_mesh(bkd), bkd),
        bkd,
        diffusion=0.3,
        velocity=[bkd.full((_N1D,), 0.5)],
        reaction=-0.4,
    )


def _robin(bkd: Backend[Array]) -> List[ClosureRowsProtocol[Array]]:
    mesh = _mesh(bkd)
    right = mesh.boundary_indices(1)
    normal = GradientNormalOperator(
        bkd,
        right,
        mesh.boundary_normals(1),
        [ChebyshevBasis1D(mesh, bkd).derivative_matrix(1, 0)],
    )
    return [
        RobinRows(
            bkd,
            StateLayout.uniform(_N1D, 1),
            [RobinPart(right, normal, 2.0, 0.5, 0.3)],
        )
    ]


def _ends(bkd: Backend[Array]) -> DirichletRows[Array]:
    return DirichletRows(
        bkd,
        StateLayout.uniform(_N1D, 1),
        [DirichletPart(_mesh(bkd).boundary_indices(0), 1.0)],
    )


def _parameters(
    bkd: Backend[Array], physics: AdvectionDiffusionReaction[Array]
) -> ParameterState[Array]:
    rng = np.random.default_rng(5)
    modes = [bkd.ones((_N1D,)), bkd.asarray(rng.uniform(-0.3, 0.3, _N1D))]
    return ParameterState(
        create_diffusion_parameterization(physics, bkd, BasisExpansion(bkd, 0.3, modes))
    )


def _transient(
    bkd: Backend[Array], treatment: str
) -> CollocationTransientSystem[Array]:
    physics = _physics(bkd)
    placement, rows = _TREATMENTS[treatment]
    return collocation_transient_system(
        CollocationSystem(physics, _ends(bkd), _robin(bkd)),
        placement(),
        rows,
        _parameters(bkd, physics),
    )


def _init(bkd: Backend[Array]) -> Array:
    nodes = _mesh(bkd).points()[0]
    return bkd.reshape(1.0 + 0.5 * (nodes + 1.0) * bkd.cos(nodes), (-1, 1))


def _function(
    bkd: Backend[Array],
    transient: CollocationTransientSystem[Array],
    method: str,
    entries: List[int],
    strategy: Optional[AdjointRows | TangentColumns] = None,
) -> ImplicitMapFunction[TransientPoint[Array, Array], Array, Array]:
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    functional = FixedTrajectoryFunctional(
        TrajectoryFunctional.from_terms(
            bkd,
            len(entries),
            state_terms=(EndpointTerm(StateEntries(bkd, _N1D, entries)),),
        )
    )
    implicit = TransientImplicitMap(
        transient,
        transient,
        RungeKutta(_TABLEAUX[method](), None if method == "heun" else newton),
        RecomputeAlways(transient.solved.solver),
        FixedStep(5e-4, (0.0, 4e-3)),
        _init(bkd),
        functional,
    )
    if strategy is None:
        return ImplicitMapFunction(implicit, ArrayParameters(bkd))
    return ImplicitMapFunction(implicit, ArrayParameters(bkd), strategy)


_CASES = [(method, treatment) for method in _TABLEAUX for treatment in _TREATMENTS]


@pytest.mark.parametrize("method, treatment", _CASES)
class TestDerivatives:
    def test_gradient_and_hvp_match_central_differences(
        self, numpy_bkd: Backend[Array], method: str, treatment: str
    ) -> None:
        bkd = numpy_bkd
        model = _function(bkd, _transient(bkd, treatment), method, [_OBSERVED])
        sample, direction = bkd.asarray(_SAMPLE), bkd.asarray(_DIRECTION)
        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(
            sample, direction=direction, relative=True, central=True
        )
        for error in errors:
            assert bkd.to_float(bkd.min(error)) <= 1e-6
            assert bkd.to_float(checker.error_ratio(error)) <= 1e-6
        hvp = model.derivatives().hvp
        assert hvp is not None
        other = bkd.asarray([[-0.4], [0.9]])
        bkd.assert_allclose(
            bkd.sum(hvp(sample, direction) * other),
            bkd.sum(hvp(sample, other) * direction),
            rtol=1e-12,
        )

    def test_adjoint_equals_tangent(
        self, bkd: Backend[Array], method: str, treatment: str
    ) -> None:
        transient = _transient(bkd, treatment)
        scalar = _function(bkd, transient, method, [_OBSERVED], AdjointRows())
        final = _function(bkd, transient, method, list(range(_N1D)), TangentColumns())
        sample = bkd.asarray(_SAMPLE)
        adjoint, tangent = scalar.derivatives().jacobian, final.derivatives().jacobian
        assert adjoint is not None and tangent is not None
        bkd.assert_allclose(
            adjoint(sample)[0], tangent(sample)[_OBSERVED], rtol=1e-10, atol=1e-14
        )


class TestTheRecord:
    def test_protocols(self, bkd: Backend[Array]) -> None:
        transient = _transient(bkd, "step_end")
        assert isinstance(transient, TransientDiscretizationProtocol)
        assert isinstance(transient, ParameterVectorSetterProtocol)
        assert transient.nparams() == 2

    def test_initial_state_and_incompatibility(self, bkd: Backend[Array]) -> None:
        """The compatible start: zero essential mismatch, and the Robin
        residual it has. A start off by 0.25 on the Dirichlet node reports
        0.25, and ``initial_state`` sets that node to 1."""
        transient = _transient(bkd, "step_end")
        init = _init(bkd)
        mismatch, residual = transient.initial_incompatibility(init, 0.0)
        assert mismatch == 0.0
        robin = _robin(bkd)[0].residual(init[:, 0], 0.0)
        assert residual == pytest.approx(bkd.to_float(bkd.max(bkd.abs(robin))))
        left = bkd.to_int(_mesh(bkd).boundary_indices(0)[0])
        shift = bkd.zeros((_N1D, 1))
        shift = bkd.index_update(shift, (left, 0), 0.25)
        mismatch, _ = transient.initial_incompatibility(init + shift, 0.0)
        assert mismatch == pytest.approx(0.25)
        bkd.assert_allclose(transient.initial_state(init + shift, 0.0), init)


class TestRefusals:
    def test_treatment_without_essential_rows(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        periodic: List[ClosureRowsProtocol[Array]] = [
            PeriodicRows(
                bkd,
                StateLayout.uniform(_N1D, 1),
                mesh.reference_points(),
                [
                    PeriodicPart(
                        mesh.boundary_indices(0),
                        mesh.boundary_indices(1),
                        ChebyshevBasis1D(mesh, bkd).derivative_matrix(1, 0),
                    )
                ],
            )
        ]
        system: CollocationSystem[Array] = CollocationSystem(
            _physics(bkd), None, periodic
        )
        with pytest.raises(ValueError, match="no essential rows"):
            collocation_transient_system(system, AtStepEnd(), CoupledRows())

    def test_parameter_dependent_essential_data(self, bkd: Backend[Array]) -> None:
        mesh = _mesh(bkd)
        left = mesh.boundary_indices(0)
        data = MappedBoundaryData(
            BasisExpansion(bkd, 1.0, [bkd.ones((left.shape[0],))]),
            bkd.asarray([0.5]),
            bkd,
        )
        physics = _physics(bkd)
        ends = DirichletRows(
            bkd, StateLayout.uniform(_N1D, 1), [DirichletPart(left, data)]
        )
        system: CollocationSystem[Array] = CollocationSystem(physics, ends)
        with pytest.raises(ValueError, match="independent of the parameters"):
            collocation_transient_system(
                system, AtStepEnd(), None, _parameters(bkd, physics)
            )

    def test_not_a_system(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="CollocationSystem"):
            collocation_transient_system(
                _physics(bkd),  # type: ignore[arg-type]
                AtStepEnd(),
                None,
            )
