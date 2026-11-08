r"""The time-free residual derivatives bundle and its Jacobian levels.

A 2-state residual :math:`R(u; p)` with parameter space of size 3 and the
Jacobian :math:`[[1, 2], [3, 4]]`:

- ``ResidualJacobian.general`` passes the state and the label to its
  function; ``constant`` passes only the label and declares :math:`R`
  affine; holding no function, two, or a non-callable is refused.
- The named constructors fill the fields they name; ``STATED_ZERO``
  becomes a zero operator between the right spaces and is recognized as
  a stated zero.
- Invariants: ``param`` needs ``jacobian`` and comes with
  ``param_space``; the three second-order fields come together and need
  ``param``; a constant Jacobian needs the stated-zero ``state_state``;
  non-callable fields and a Jacobian of the wrong type are refused.
"""

from typing import Generic, Optional

import pytest

from pyapprox.implicitfunction.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ResidualDerivatives,
    ResidualJacobian,
    ResidualSecondDerivativeFn,
    is_stated_zero,
)
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, DenseFormat
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    VectorSpaceProtocol,
)

_POINT = LinearizationPoint(0, ("test",))


class _Parts(Generic[Array]):
    """A Jacobian, a parameter Jacobian and a second derivative of a 2-state,
    3-parameter residual."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        self.states, self.params = backend_spaces(bkd, 2, 3)
        self.matrix = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.seen: list[Array] = []

    def general(
        self, state: Array, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        self.seen.append(state)
        return LinearizedOperator(
            BackendDenseOperator(self.matrix, self.states.descriptor(), self.bkd),
            point,
        )

    def constant(
        self, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return self.general(self.bkd.zeros((2,)), point)

    def param(self, state: Array) -> DerivativeOperator[Array]:
        return BackendDenseOperator(
            self.bkd.ones((2, 3)),
            self.params.descriptor(),
            self.bkd,
            range=self.states.descriptor(),
        )

    def second(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return BackendDenseOperator(
            self.bkd.eye(2), self.states.descriptor(), self.bkd
        )

    def cross(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return self.param(state)

    def param_param(self, state: Array, adj: Array) -> DerivativeOperator[Array]:
        return BackendDenseOperator(
            self.bkd.eye(3), self.params.descriptor(), self.bkd
        )


def _zero_maps(
    field: Optional[ResidualSecondDerivativeFn[Array]],
    domain: VectorSpaceProtocol[Array],
    bkd: Backend[Array],
) -> Array:
    """The field's operator applied to a ones column of ``domain``."""
    assert field is not None
    op = field(bkd.zeros((2,)), bkd.zeros((2,)))
    return op.apply(bkd.ones((domain.descriptor().size, 1)))


class TestTheJacobian:
    def test_a_general_jacobian_sees_the_state(self, bkd: Backend[Array]) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        jacobian = ResidualJacobian.general(parts.general)
        state = bkd.asarray([0.5, -1.0])
        snapshot = jacobian(state, _POINT)
        assert snapshot.linearization() == _POINT
        bkd.assert_allclose(parts.seen[-1], state)
        assert not jacobian.affine()
        assert repr(jacobian) == "ResidualJacobian.general"

    def test_a_constant_jacobian_does_not(self, bkd: Backend[Array]) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        jacobian = ResidualJacobian.constant(parts.constant)
        snapshot = jacobian(bkd.asarray([0.5, -1.0]), _POINT)
        bkd.assert_allclose(parts.seen[-1], bkd.zeros((2,)))
        bkd.assert_allclose(
            snapshot.apply(bkd.asarray([[1.0], [1.0]])),
            bkd.asarray([[3.0], [7.0]]),
        )
        assert jacobian.affine()
        assert repr(jacobian) == "ResidualJacobian.constant"

    def test_it_holds_exactly_one_function(self, bkd: Backend[Array]) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        with pytest.raises(ValueError, match="exactly one"):
            ResidualJacobian()
        with pytest.raises(ValueError, match="exactly one"):
            ResidualJacobian(general=parts.general, constant=parts.constant)
        with pytest.raises(TypeError, match="holds a function"):
            ResidualJacobian.general(3.0)  # type: ignore[arg-type]


class TestNamedConstructors:
    def test_state_only(self, bkd: Backend[Array]) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        bundle = ResidualDerivatives.state_only(ResidualJacobian.general(parts.general))
        assert bundle.jacobian is not None
        assert bundle.param is None and bundle.state_state is None
        assert ResidualDerivatives.none().jacobian is None

    def test_first_order(self, bkd: Backend[Array]) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        bundle = ResidualDerivatives.first_order(
            parts.params, ResidualJacobian.general(parts.general), parts.param
        )
        assert bundle.param == parts.param and bundle.param_space is parts.params
        assert bundle.state_state is None

    def test_second_order_states_zeros(self, bkd: Backend[Array]) -> None:
        """A constant Jacobian with the stated-zero state curvature, and a
        stated-zero cross term: both zero operators of the right shapes,
        recognized as stated."""
        parts: _Parts[Array] = _Parts(bkd)
        bundle = ResidualDerivatives.second_order(
            parts.states,
            parts.params,
            ResidualJacobian.constant(parts.constant),
            parts.param,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=parts.param_param,
        )
        assert is_stated_zero(bundle.state_state)
        assert is_stated_zero(bundle.state_param)
        assert not is_stated_zero(bundle.param_param)
        bkd.assert_allclose(
            _zero_maps(bundle.state_state, parts.states, bkd), bkd.zeros((2, 1))
        )
        bkd.assert_allclose(
            _zero_maps(bundle.state_param, parts.params, bkd), bkd.zeros((2, 1))
        )


class TestInvariants:
    def test_param_needs_the_jacobian_and_its_space(
        self, bkd: Backend[Array]
    ) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        jacobian = ResidualJacobian.general(parts.general)
        with pytest.raises(ValueError, match="requires jacobian"):
            ResidualDerivatives(param=parts.param, param_space=parts.params)
        with pytest.raises(ValueError, match="param_space"):
            ResidualDerivatives(jacobian=jacobian, param=parts.param)
        with pytest.raises(ValueError, match="param_space"):
            ResidualDerivatives(jacobian=jacobian, param_space=parts.params)

    def test_second_order_comes_together_after_first(
        self, bkd: Backend[Array]
    ) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        jacobian = ResidualJacobian.general(parts.general)
        with pytest.raises(ValueError, match="come together"):
            ResidualDerivatives(
                jacobian=jacobian,
                param=parts.param,
                param_space=parts.params,
                state_state=parts.second,
            )
        with pytest.raises(ValueError, match="requires param"):
            ResidualDerivatives(
                jacobian=jacobian,
                state_state=parts.second,
                state_param=parts.cross,
                param_param=parts.param_param,
            )

    def test_an_affine_residual_states_zero_curvature(
        self, bkd: Backend[Array]
    ) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        with pytest.raises(ValueError, match="STATED_ZERO"):
            ResidualDerivatives.second_order(
                parts.states,
                parts.params,
                ResidualJacobian.constant(parts.constant),
                parts.param,
                state_state=parts.second,
                state_param=parts.cross,
                param_param=parts.param_param,
            )

    def test_fields_are_callables_of_the_right_type(
        self, bkd: Backend[Array]
    ) -> None:
        parts: _Parts[Array] = _Parts(bkd)
        with pytest.raises(TypeError, match="ResidualJacobian"):
            ResidualDerivatives(jacobian=parts.general)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="callable"):
            ResidualDerivatives(
                jacobian=ResidualJacobian.general(parts.general),
                param=bkd.ones((2, 3)),  # type: ignore[arg-type]
                param_space=parts.params,
            )
