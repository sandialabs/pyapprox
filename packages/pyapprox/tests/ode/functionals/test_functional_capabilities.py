"""Each functional's derivative bundle lists exactly what it provides.

The capability a functional declares in ``derivatives()`` is what
consumers rely on, so each is checked against what the functional
implements; a wrong declaration would either hide a capability or
promise one that raises.
"""

import dataclasses
from typing import Any, Callable, Dict, Literal, Set

import numpy as np
import pytest

from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.functionals.derivatives import (
    TransientFunctionalDerivatives,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.functionals.tikhonov import TikhonovAugmentedFunctional
from pyapprox.ode.functionals.time_indexed_observation import (
    TimeIndexedObservationFunctional,
)
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.functionals.weighted_endpoint import (
    WeightedEndpointFunctional,
)
from pyapprox.ode.time_quadrature import (
    midpoint_quadrature,
    trapezoidal_quadrature,
)
from pyapprox.optimization.implicitfunction.functionals.mean_squared_error import (  # noqa: E501
    MSEFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (  # noqa: E501
    SubsetOfStatesAdjointFunctional,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd

_NSTATES, _NPARAMS = 3, 2
_JAC = {"state_jacobian", "param_jacobian"}
_HVPS = {"state_state_hvp", "state_param_hvp", "param_state_hvp", "param_param_hvp"}


class _ValueOnlyFunctional:
    """A transient functional with no derivatives: the sum of the final
    state."""

    def __init__(self, bkd: NumpyBkd) -> None:
        self._bkd = bkd

    def bkd(self) -> NumpyBkd:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return _NSTATES

    def nparams(self) -> int:
        return _NPARAMS

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["transient"]:
        return "transient"

    def derivatives(self) -> TransientFunctionalDerivatives[Any]:
        return TransientFunctionalDerivatives.none()

    def set_time_quadrature(self, quadrature: Any) -> None:
        pass

    def __call__(self, sol: Any, param: Any) -> Any:
        return self._bkd.reshape(self._bkd.sum(sol[:, -1]), (1, 1))


def _with_quadrature(functional: Any, bkd: NumpyBkd) -> Any:
    """Inject a time-diagonal rule, which the functional's capability
    depends on."""
    functional.set_time_quadrature(
        trapezoidal_quadrature(bkd.linspace(0.0, 1.0, 4), bkd)
    )
    return functional


def _present(functional: Any) -> Set[str]:
    bundle = functional.derivatives()
    return {
        field.name
        for field in dataclasses.fields(bundle)
        if getattr(bundle, field.name) is not None
    }


_FACTORIES: Dict[str, Callable[[NumpyBkd], Any]] = {
    "endpoint": lambda bkd: EndpointFunctional(0, _NSTATES, _NPARAMS, bkd),
    "all_states_endpoint": lambda bkd: AllStatesEndpointFunctional(
        _NSTATES, _NPARAMS, bkd
    ),
    "weighted_endpoint": lambda bkd: WeightedEndpointFunctional(
        bkd.ones((_NSTATES, 1)), _NPARAMS, bkd
    ),
    "transient_mse": lambda bkd: TransientMSEFunctional(
        _NSTATES, _NPARAMS, [(0, bkd.asarray(np.array([1, 2])))], bkd
    ),
    "time_integrated_l2": lambda bkd: _with_quadrature(
        TimeIntegratedWeightedL2Functional(bkd.eye(_NSTATES), _NPARAMS, bkd),
        bkd,
    ),
    "time_indexed_observation": lambda bkd: TimeIndexedObservationFunctional(
        bkd.asarray(np.array([[0], [1]]), dtype=bkd.int64_dtype()),
        bkd.ones((2, 1)),
        [1, 2],
        _NSTATES,
        _NPARAMS,
        bkd,
    ),
    "mse": lambda bkd: MSEFunctional(_NSTATES, _NPARAMS, bkd),
    "subset_of_states": lambda bkd: SubsetOfStatesAdjointFunctional(
        _NSTATES, _NPARAMS, bkd.arange(_NSTATES), bkd
    ),
    "weighted_sum": lambda bkd: WeightedSumFunctional(
        bkd.ones((_NSTATES, 1)), _NPARAMS, bkd
    ),
}

_EXPECTED: Dict[str, Set[str]] = {
    "endpoint": _JAC | _HVPS,
    "all_states_endpoint": _JAC
    | _HVPS
    | {"apply_state_jacobian", "row_functional"},
    "weighted_endpoint": _JAC | _HVPS,
    "transient_mse": _JAC | _HVPS,
    "time_integrated_l2": _JAC | _HVPS,
    "time_indexed_observation": _JAC
    | {"apply_state_jacobian", "row_functional"},
    "mse": _JAC | _HVPS,
    "subset_of_states": _JAC,
    "weighted_sum": _JAC | _HVPS,
}
_STEADY = {"mse", "subset_of_states", "weighted_sum"}


class TestDeclaredCapabilities:
    @pytest.mark.parametrize("name", sorted(_FACTORIES))
    def test_bundle_lists_exactly_the_capabilities(
        self, numpy_bkd: NumpyBkd, name: str
    ) -> None:
        assert _present(_FACTORIES[name](numpy_bkd)) == _EXPECTED[name]

    @pytest.mark.parametrize("name", sorted(_FACTORIES))
    def test_declared_fields_are_the_functional_methods(
        self, numpy_bkd: NumpyBkd, name: str
    ) -> None:
        """A present field is the functional's own method, so declaring a
        capability cannot drift from implementing it."""
        functional = _FACTORIES[name](numpy_bkd)
        bundle = functional.derivatives()
        for field_name in _present(functional):
            assert getattr(bundle, field_name) == getattr(
                functional, field_name
            )

    @pytest.mark.parametrize(
        "name", sorted(n for n in _FACTORIES if n not in _STEADY)
    )
    def test_every_transient_functional_accepts_the_quadrature(
        self, numpy_bkd: NumpyBkd, name: str
    ) -> None:
        functional = _FACTORIES[name](numpy_bkd)
        functional.set_time_quadrature(
            trapezoidal_quadrature(numpy_bkd.linspace(0.0, 1.0, 4), numpy_bkd)
        )


class TestQuadratureDependentCapability:
    """A functional that integrates over time has per-step HVPs only
    under a rule whose samples are the stored columns."""

    def test_capability_needs_a_quadrature(self, numpy_bkd: NumpyBkd) -> None:
        functional = TimeIntegratedWeightedL2Functional(
            numpy_bkd.eye(_NSTATES), _NPARAMS, numpy_bkd
        )
        with pytest.raises(RuntimeError, match="time quadrature not set"):
            functional.derivatives()

    def test_midpoint_rule_drops_the_hvps(self, numpy_bkd: NumpyBkd) -> None:
        functional = TimeIntegratedWeightedL2Functional(
            numpy_bkd.eye(_NSTATES), _NPARAMS, numpy_bkd
        )
        functional.set_time_quadrature(
            midpoint_quadrature(numpy_bkd.linspace(0.0, 1.0, 4), numpy_bkd)
        )
        assert _present(functional) == _JAC
        wrapped = TikhonovAugmentedFunctional(functional, 0.5, numpy_bkd)
        assert _present(wrapped) == _JAC

    def test_endpoint_keeps_hvps_under_midpoint_rule(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """An endpoint functional ignores the quadrature."""
        functional = EndpointFunctional(0, _NSTATES, _NPARAMS, numpy_bkd)
        functional.set_time_quadrature(
            midpoint_quadrature(numpy_bkd.linspace(0.0, 1.0, 4), numpy_bkd)
        )
        assert _present(functional) == _JAC | _HVPS


class TestTikhonovMirrorsInner:
    def test_hvp_inner_gives_hvp_fields(self, numpy_bkd: NumpyBkd) -> None:
        inner = WeightedEndpointFunctional(
            numpy_bkd.ones((_NSTATES, 1)), _NPARAMS, numpy_bkd
        )
        wrapped = TikhonovAugmentedFunctional(inner, 0.5, numpy_bkd)
        assert _present(wrapped) == _JAC | _HVPS

    def test_value_only_inner_gives_no_derivatives(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """A value-only inner functional makes a value-only wrapper."""
        wrapped = TikhonovAugmentedFunctional(
            _ValueOnlyFunctional(numpy_bkd), 0.5, numpy_bkd
        )
        assert _present(wrapped) == set()

    def test_action_inner_gives_action_only(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The state-Jacobian action is the inner one; the rows are not
        offered, since they would omit the parameter cost."""
        inner = TimeIndexedObservationFunctional(
            numpy_bkd.asarray(np.array([[0]]), dtype=numpy_bkd.int64_dtype()),
            numpy_bkd.ones((1, 1)),
            [2],
            _NSTATES,
            _NPARAMS,
            numpy_bkd,
        )
        wrapped = TikhonovAugmentedFunctional(inner, 0.5, numpy_bkd)
        assert _present(wrapped) == _JAC | {"apply_state_jacobian"}
        assert (
            wrapped.derivatives().apply_state_jacobian
            == inner.apply_state_jacobian
        )

    def test_forwards_the_quadrature(self, numpy_bkd: NumpyBkd) -> None:
        inner = TimeIntegratedWeightedL2Functional(
            numpy_bkd.eye(_NSTATES), _NPARAMS, numpy_bkd
        )
        wrapped = TikhonovAugmentedFunctional(inner, 0.5, numpy_bkd)
        sol = numpy_bkd.ones((_NSTATES, 4))
        param = numpy_bkd.zeros((_NPARAMS, 1))
        wrapped.set_time_quadrature(
            trapezoidal_quadrature(numpy_bkd.linspace(0.0, 1.0, 4), numpy_bkd)
        )
        # The inner functional received the quadrature (it cannot
        # evaluate without one), and at p = 0 the cost adds nothing.
        numpy_bkd.assert_allclose(
            wrapped(sol, param), inner(sol, param), rtol=1e-14
        )
