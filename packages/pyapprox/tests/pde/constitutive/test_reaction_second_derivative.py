"""The reaction second derivative is an accessor, never a bound method."""

import pickle
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    LinearReaction,
    ReactionFunctionProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd

_COORDS = np.zeros((1, 3))
_STATE = np.array([0.5, -1.0, 2.0])


def _cube(x: NDArray[Any], u: NDArray[Any]) -> NDArray[Any]:
    return u**3


def _cube_derivative(x: NDArray[Any], u: NDArray[Any]) -> NDArray[Any]:
    return 3.0 * u**2


def _cube_second_derivative(
    x: NDArray[Any], u: NDArray[Any]
) -> NDArray[Any]:
    return 6.0 * u


class TestReactionSecondDerivative:
    def test_callable_reaction_without_it_is_none(self) -> None:
        reaction = CallableReaction(_cube, _cube_derivative)
        assert isinstance(reaction, ReactionFunctionProtocol)
        assert reaction.second_derivative_function() is None

    def test_callable_reaction_with_it(self, numpy_bkd: NumpyBkd) -> None:
        reaction = CallableReaction(
            _cube, _cube_derivative, _cube_second_derivative
        )
        second = reaction.second_derivative_function()
        assert second is not None
        numpy_bkd.assert_allclose(second(_COORDS, _STATE), 6.0 * _STATE)

    def test_same_class_both_ways(self) -> None:
        """Capability is a value, not a method attached per instance."""
        with_it = CallableReaction(
            _cube, _cube_derivative, _cube_second_derivative
        )
        without = CallableReaction(_cube, _cube_derivative)
        assert not hasattr(with_it, "second_derivative")
        assert not hasattr(without, "second_derivative")

    def test_linear_reaction_is_exact_zero(self, numpy_bkd: NumpyBkd) -> None:
        second = LinearReaction(2.0).second_derivative_function()
        assert second is not None
        numpy_bkd.assert_allclose(second(_COORDS, _STATE), np.zeros(3))

    def test_pickle_round_trip(self, numpy_bkd: NumpyBkd) -> None:
        reaction = CallableReaction(
            _cube, _cube_derivative, _cube_second_derivative
        )
        restored = pickle.loads(pickle.dumps(reaction))
        second = restored.second_derivative_function()
        assert second is not None
        numpy_bkd.assert_allclose(second(_COORDS, _STATE), 6.0 * _STATE)
