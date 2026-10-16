"""The element-wise contract on a producer with no assembly library.

``tests/_helpers/mock_elementwise.py`` is 1D P1 Galerkin for
-(kappa u')' + c u' [+ u^2] = f in closed form, backend-generic, so the
contract of ``pyapprox.pde.elementwise`` and its generic helpers run on
NumPy and Torch without skfem.

Checks:
- 1D P1 is nodally exact when the solution lies in the trial space or
  is the quadratic of a constant load, so the interior residual rows
  vanish (atol 1e-12) for three manufactured cases on 8 cells: u = 1
  with u^2 = f = 1; u = x with c = f = 0.7; u = x(1 - x)/2 with
  kappa = f = 1. Wrong-model control: f + 1 leaves exactly h in each
  interior row.
- Two halves of a random partition (seed 1) scatter to the assembled
  residual (rtol 1e-12) at a random state (seed 0, entries in
  [0.5, 1.5]); dropping one element breaks it.
- Element Jacobian actions and their transposes satisfy the per-element
  dot identity, while advection (c = 0.7) makes the two actions differ
  by more than 1e-3; ``element_jacobian_blocks`` applied to a direction
  equals the action.
- DerivativeChecker (central, error ratio <= 1e-6) on the assembled
  state Jacobian with reaction and advection, and on the modal
  parameter Jacobian.
- The homogeneity check accepts the mock; malformed indices, states and
  directions raise.
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.protocols.spatial_operator import SpatialOperatorProtocol
from pyapprox.pde.elementwise import (
    ElementSet,
    ElementwiseOperatorProtocol,
    RestrictedElementOperatorProtocol,
    check_restricted_operator,
    element_jacobian_blocks,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.mock_elementwise import (
    MockP1Operator,
    MockP1Restricted,
    ModalCoefficientParameterization,
)

_NCELLS = 8


def _elements(indices: list[int], bkd: Backend[Array]) -> ElementSet[Array]:
    return ElementSet(bkd.asarray(indices, dtype=bkd.int64_dtype()), bkd)


def _state(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(0).uniform(0.5, 1.5, _NCELLS + 1))


def _local(restricted: MockP1Restricted[Array], state: Array) -> Array:
    return state[restricted.global_dofs()[restricted.element_state_dofs()]]


def _scatter(restricted: MockP1Restricted[Array], state: Array) -> Array:
    """Sum the element residuals into global rows, independently of the
    operator's own assembly."""
    bkd = restricted.bkd()
    rows = bkd.to_numpy(restricted.global_dofs()[restricted.element_row_dofs()])
    scatter = np.zeros((_NCELLS + 1, rows.size))
    scatter[rows.reshape(-1), np.arange(rows.size)] = 1.0
    residuals = restricted.element_residuals(_local(restricted, state), 0.0)
    return bkd.dot(bkd.asarray(scatter), bkd.reshape(residuals, (-1,)))


def _manufactured(
    case: str, bkd: Backend[Array]
) -> Tuple[MockP1Operator[Array], Array, Array]:
    """An operator, its exact nodal solution and the forcing it solves."""
    nodes = np.linspace(0.0, 1.0, _NCELLS + 1)
    if case == "reaction":
        operator = MockP1Operator(_NCELLS, True, bkd)
        solution, forcing = np.ones_like(nodes), np.ones_like(nodes)
    elif case == "advection":
        operator = MockP1Operator(_NCELLS, False, bkd, velocity=0.7)
        solution, forcing = nodes.copy(), np.full_like(nodes, 0.7)
    else:
        operator = MockP1Operator(_NCELLS, False, bkd)
        solution, forcing = 0.5 * nodes * (1.0 - nodes), np.ones_like(nodes)
    return operator, bkd.asarray(solution), bkd.asarray(forcing)


class TestContract:
    def test_satisfies_the_protocols(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd)
        assert isinstance(operator, ElementwiseOperatorProtocol)
        assert isinstance(operator, SpatialOperatorProtocol)
        restricted = operator.restrict(_elements([0, 3, 5], bkd))
        assert isinstance(restricted, RestrictedElementOperatorProtocol)
        check_restricted_operator(restricted)

    @pytest.mark.parametrize("case", ["reaction", "advection", "diffusion"])
    def test_interior_rows_vanish_at_exact_solutions(
        self, bkd: Backend[Array], case: str
    ) -> None:
        operator, solution, forcing = _manufactured(case, bkd)
        operator.set_coefficients(bkd.ones((_NCELLS,)), forcing)
        residual = operator.spatial_residual(solution, 0.0)
        bkd.assert_allclose(residual[1:-1], bkd.zeros((_NCELLS - 1,)), atol=1e-12)
        # Wrong model: one more unit of load is h per interior row.
        operator.set_coefficients(bkd.ones((_NCELLS,)), forcing + 1.0)
        shifted = operator.spatial_residual(solution, 0.0)
        bkd.assert_allclose(
            shifted[1:-1], bkd.full((_NCELLS - 1,), 1.0 / _NCELLS), rtol=1e-12
        )

    def test_partition_sums_to_the_assembly(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd, velocity=0.7)
        state = _state(bkd)
        order = np.random.default_rng(1).permutation(_NCELLS)
        total = bkd.zeros((_NCELLS + 1,))
        for half in (order[: _NCELLS // 2], order[_NCELLS // 2 :]):
            restricted = operator.restrict(_elements(sorted(half.tolist()), bkd))
            total = total + _scatter(restricted, state)
        assembled = operator.spatial_residual(state, 0.0)
        bkd.assert_allclose(total, assembled, rtol=1e-12)
        dropped = _scatter(
            operator.restrict(_elements(list(range(1, _NCELLS)), bkd)), state
        )
        assert float(bkd.max(bkd.abs(dropped - assembled))) > 1e-3


class TestActions:
    def test_transpose_is_the_adjoint_of_the_action(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd, velocity=0.7)
        restricted = operator.restrict(_elements([1, 2, 6], bkd))
        local = _local(restricted, _state(bkd))
        rng = np.random.default_rng(2)
        directions = bkd.asarray(rng.normal(size=(3, 2, 2)))
        adjoints = bkd.asarray(rng.normal(size=(3, 2, 2)))
        forward = restricted.element_jacobian_actions(local, 0.0, directions)
        backward = restricted.element_jacobian_transpose_actions(local, 0.0, adjoints)
        bkd.assert_allclose(
            bkd.einsum("eik,eik->ek", forward, adjoints),
            bkd.einsum("eik,eik->ek", directions, backward),
            rtol=1e-12,
        )
        as_forward = restricted.element_jacobian_actions(local, 0.0, adjoints)
        assert float(bkd.max(bkd.abs(as_forward - backward))) > 1e-3

    def test_blocks_reproduce_the_action(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd, velocity=0.7)
        restricted = operator.restrict(_elements([0, 4], bkd))
        local = _local(restricted, _state(bkd))
        directions = bkd.asarray(np.random.default_rng(3).normal(size=(2, 2, 3)))
        blocks = element_jacobian_blocks(restricted, local, 0.0)
        bkd.assert_allclose(
            bkd.einsum("eij,ejk->eik", blocks, directions),
            restricted.element_jacobian_actions(local, 0.0, directions),
            rtol=1e-12,
        )

    def test_rejects_malformed_inputs(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, False, bkd)
        with pytest.raises(ValueError, match="element indices"):
            operator.restrict(_elements([0, _NCELLS], bkd))
        restricted = operator.restrict(_elements([0, 1], bkd))
        local = _local(restricted, _state(bkd))
        with pytest.raises(ValueError, match="local_states"):
            restricted.element_residuals(local[:1], 0.0)
        with pytest.raises(ValueError, match="3D"):
            restricted.element_jacobian_actions(local, 0.0, local)
        with pytest.raises(ValueError, match="leading shape"):
            restricted.element_jacobian_transpose_actions(
                local, 0.0, bkd.zeros((2, 3, 1))
            )


class TestDerivatives:
    def test_state_jacobian(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd, velocity=0.7)
        nstates = operator.nstates()

        def fun(sample: Array) -> Array:
            return operator.spatial_residual(sample[:, 0], 0.0)[:, None]

        def jac(sample: Array) -> Array:
            return operator.spatial_jacobian(sample[:, 0], 0.0)

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=nstates, nvars=nstates, fun=fun, jacobian=jac, bkd=bkd
            )
        )
        errors = checker.check_derivatives(
            _state(bkd)[:, None], verbosity=0, central=True
        )[0]
        assert float(checker.error_ratio(errors)) <= 1e-6

    def test_param_jacobian(self, bkd: Backend[Array]) -> None:
        operator = MockP1Operator(_NCELLS, True, bkd, velocity=0.7)
        parameterization = ModalCoefficientParameterization(operator, bkd)
        param_jacobian = parameterization.param_derivatives().param_jacobian
        assert param_jacobian is not None
        state = _state(bkd)

        def fun(sample: Array) -> Array:
            parameterization.apply(sample[:, 0])
            return operator.spatial_residual(state, 0.0)[:, None]

        def jac(sample: Array) -> Array:
            parameterization.apply(sample[:, 0])
            return param_jacobian(state, 0.0, sample[:, 0])

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=operator.nstates(), nvars=6, fun=fun, jacobian=jac, bkd=bkd
            )
        )
        params = bkd.asarray(np.random.default_rng(4).uniform(-1.0, 1.0, (6, 1)))
        errors = checker.check_derivatives(params, verbosity=0, central=True)[0]
        assert float(checker.error_ratio(errors)) <= 1e-6
