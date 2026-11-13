r"""A steady functional as a function of the parameters.

The pinned quadratic of ``tests._helpers.implicitfunction_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`) with :math:`Q = g + r`
(:math:`g = \tfrac12 u^\top D u + p_1 u_1`, :math:`r = \tfrac{3}{20}
\|p\|^2`), solved by Newton (``atol`` 1e-13) from zero with dense LU:

- Values equal a separate ``SteadySystem`` solve and ``SteadyDerivatives``
  value at three samples to 1e-12.
- A scalar functional gives the Jacobian and the HVP, both passing central
  finite differences (minimum relative error below 1e-6) at
  :math:`p = (0.7, -0.4)`.
- The functional stacked twice (no more entries than parameters: one
  adjoint per entry) and three times (more: one tangent per parameter)
  gives every row equal to the scalar gradient to 1e-12, and no HVP.
- ``AdjointRows`` and ``TangentColumns`` passed explicitly give the
  default's Jacobian to 1e-12 for one, two and three entries.
- No derivatives when the residual has no parameter Jacobian or a term
  has none; the values still work.
- Each sample clears the setup strategy once.
- Refusals: ``init_state`` of the wrong shape, a non-setter, a
  non-functional, a non-strategy, parameters the residual does not read,
  a solve that does not converge.
"""

from typing import Generic, List, Optional, Tuple

import pytest

from pyapprox.implicitfunction.setups import (
    KeepAll,
    OperatorBuilderProtocol,
    SteadyRequest,
)
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_function import (
    AdjointRows,
    SteadyImplicitFunction,
    SteadyJacobianProtocol,
    TangentColumns,
)
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import SolverHandleProtocol
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import (
    QUADRATIC_PARAM,
    pinned_quadratic,
    quadratic_functional,
)

_Format = DenseFormat


class _Ignored(Generic[Array]):
    """A setter that moves nothing the residual reads."""

    def nparams(self) -> int:
        return 2

    def set_param(self, param: Array) -> None:
        pass


class _CountingClears(Generic[Array]):
    """``KeepAll`` that counts its clears."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._inner: KeepAll[Array, _Format[Array]] = KeepAll(BackendDenseLU(bkd))
        self.clears = 0

    def handle(
        self,
        request: SteadyRequest[Array, _Format[Array]],
        builder: OperatorBuilderProtocol[
            SteadyRequest[Array, _Format[Array]], Array, _Format[Array]
        ],
    ) -> SolverHandleProtocol[Array]:
        return self._inner.handle(request, builder)

    def clear(self) -> None:
        self.clears += 1
        self._inner.clear()


def _function(
    bkd: Backend[Array],
    residual_order: int = 2,
    functional_order: int = 2,
    nqoi: int = 1,
    maxiters: int = 30,
    jacobian: Optional[SteadyJacobianProtocol] = None,
) -> Tuple[SteadyImplicitFunction[Array, _Format[Array]], _CountingClears[Array]]:
    f, pinned = pinned_quadratic(bkd, residual_order)
    one = quadratic_functional(f, functional_order)
    functional = one if nqoi == 1 else StateFunctional.stack(bkd, [one] * nqoi)
    setups = _CountingClears(bkd)
    function: SteadyImplicitFunction[Array, _Format[Array]] = SteadyImplicitFunction(
        pinned,
        f.states,
        f,
        setups,
        Newton(atol=1e-13, rtol=0.0, maxiters=maxiters),
        functional,
        bkd.zeros((3, 1)),
        bkd,
        jacobian,
    )
    return function, setups


def _param(bkd: Backend[Array]) -> Array:
    return bkd.reshape(bkd.asarray(QUADRATIC_PARAM), (2, 1))


def _samples(bkd: Backend[Array]) -> Array:
    p = _param(bkd)
    return bkd.hstack([p, 0.5 * p, p + 0.1])


def _min_errors(
    bkd: Backend[Array], function: SteadyImplicitFunction[Array, _Format[Array]]
) -> List[float]:
    errors = DerivativeChecker(function).check_derivatives(
        _param(bkd), relative=True, central=True
    )
    return [bkd.to_float(bkd.min(e)) for e in errors]


class TestValues:
    def test_satisfies_function_protocol(self, bkd: Backend[Array]) -> None:
        function, _ = _function(bkd)
        assert isinstance(function, FunctionProtocol)
        assert function.nvars() == 2 and function.nqoi() == 1

    def test_values_equal_a_separate_solve(self, bkd: Backend[Array]) -> None:
        function, _ = _function(bkd)
        samples = _samples(bkd)
        f, pinned = pinned_quadratic(bkd)
        functional = quadratic_functional(f)
        setups: KeepAll[Array, _Format[Array]] = KeepAll(BackendDenseLU(bkd))
        system = SteadySystem(pinned, f.states, setups)
        derivs = SteadyDerivatives(pinned, f.states, functional)
        expected = []
        for ii in range(3):
            f.set_param(samples[:, ii])
            setups.clear()
            newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
            point = system.solve(newton, bkd.zeros((3, 1)))
            expected.append(derivs.value(point))
        bkd.assert_allclose(function(samples), bkd.hstack(expected), rtol=1e-12)


class TestDerivatives:
    def test_scalar_gives_jacobian_and_hvp(self, bkd: Backend[Array]) -> None:
        function, _ = _function(bkd)
        derivs = function.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is not None
        errors = _min_errors(bkd, function)
        assert len(errors) == 2
        assert max(errors) < 1e-6

    @pytest.mark.parametrize("nqoi", [2, 3], ids=["adjoint", "tangent"])
    def test_vector_rows_equal_the_scalar_gradient(
        self, bkd: Backend[Array], nqoi: int
    ) -> None:
        scalar, _ = _function(bkd)
        vector, _ = _function(bkd, nqoi=nqoi)
        derivs = vector.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is None
        scalar_jacobian = scalar.derivatives().jacobian
        assert scalar_jacobian is not None
        p = _param(bkd)
        gradient = scalar_jacobian(p)
        bkd.assert_allclose(
            derivs.jacobian(p), bkd.vstack([gradient] * nqoi), rtol=1e-12
        )

    @pytest.mark.parametrize("nqoi", [1, 2, 3])
    @pytest.mark.parametrize(
        "strategy", [AdjointRows(), TangentColumns()], ids=["adjoint", "tangent"]
    )
    def test_either_strategy_overrides_the_default(
        self, bkd: Backend[Array], nqoi: int, strategy: SteadyJacobianProtocol
    ) -> None:
        default, _ = _function(bkd, nqoi=nqoi)
        chosen, _ = _function(bkd, nqoi=nqoi, jacobian=strategy)
        expected, actual = default.derivatives().jacobian, chosen.derivatives().jacobian
        assert expected is not None and actual is not None
        p = _param(bkd)
        bkd.assert_allclose(actual(p), expected(p), rtol=1e-12)
        assert repr(strategy) in repr(chosen)

    @pytest.mark.parametrize("orders", [(0, 2), (2, 0)], ids=["residual", "functional"])
    def test_none_without_first_derivatives(
        self, bkd: Backend[Array], orders: Tuple[int, int]
    ) -> None:
        function, _ = _function(
            bkd, residual_order=orders[0], functional_order=orders[1]
        )
        assert function.derivatives().jacobian is None
        assert function(_param(bkd)).shape == (1, 1)

    def test_first_order_gives_no_hvp(self, bkd: Backend[Array]) -> None:
        function, _ = _function(bkd, residual_order=1)
        derivs = function.derivatives()
        assert derivs.jacobian is not None and derivs.hvp is None


class TestSetups:
    def test_each_sample_clears_the_strategy(self, bkd: Backend[Array]) -> None:
        function, setups = _function(bkd)
        function(_samples(bkd))
        assert setups.clears == 3
        jacobian = function.derivatives().jacobian
        assert jacobian is not None
        jacobian(_param(bkd))
        assert setups.clears == 4


class TestRefusals:
    def test_wrong_arguments(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        functional = quadratic_functional(f)
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        setups: KeepAll[Array, _Format[Array]] = KeepAll(BackendDenseLU(bkd))
        zeros = bkd.zeros((3, 1))
        with pytest.raises(ValueError, match="init_state"):
            SteadyImplicitFunction(
                pinned,
                f.states,
                f,
                setups,
                newton,
                functional,
                bkd.zeros((3,)),
                bkd,
            )
        with pytest.raises(TypeError, match="ParameterSetterProtocol"):
            SteadyImplicitFunction(
                pinned,
                f.states,
                3.0,  # type: ignore[arg-type]
                setups,
                newton,
                functional,
                zeros,
                bkd,
            )
        with pytest.raises(TypeError, match="StateFunctional"):
            SteadyImplicitFunction(
                pinned,
                f.states,
                f,
                setups,
                newton,
                3.0,  # type: ignore[arg-type]
                zeros,
                bkd,
            )

    def test_parameters_the_residual_does_not_read(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        setups: KeepAll[Array, _Format[Array]] = KeepAll(BackendDenseLU(bkd))
        function: SteadyImplicitFunction[Array, _Format[Array]] = (
            SteadyImplicitFunction(
                pinned,
                f.states,
                _Ignored(),
                setups,
                Newton(atol=1e-13, rtol=0.0, maxiters=30),
                quadratic_functional(f),
                bkd.zeros((3, 1)),
                bkd,
            )
        )
        with pytest.raises(ValueError, match="does not read these parameters"):
            function(_param(bkd))

    def test_a_solve_that_does_not_converge(self, bkd: Backend[Array]) -> None:
        function, _ = _function(bkd, maxiters=1)
        with pytest.raises(RuntimeError, match="did not converge"):
            function(_param(bkd))

    def test_a_jacobian_that_is_not_a_strategy(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="SteadyJacobianProtocol"):
            _function(bkd, jacobian=3.0)  # type: ignore[arg-type]
