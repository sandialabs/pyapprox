r"""Stack-generic suites on array stacks: a steady functional as a
function of the parameters.

``SteadyImplicitFunction`` takes states as backend arrays, so these
suites run on stacks whose vectors are arrays (``TypedStackCase[A, F,
A]``: CSR, dense NumPy, dense Torch), served by an ``array_case``
fixture. A replicated parameter space would make it generic in the
stack's vector type, and these suites would then take ``case``.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned to :math:`(p_0^2 + 1, p_0 p_1)`) with :math:`Q = g + r`
(:math:`g = \tfrac12 u^\top D u + p_1 u_1`, :math:`r = \tfrac{3}{20}
\|p\|^2`), solved by Newton (``atol`` 1e-13) from zero with the stack's
direct solver:

- Values equal a separate ``SteadySystem`` solve and ``SteadyDerivatives``
  value at three samples to 1e-12.
- A scalar functional gives the Jacobian, the HVP and the weighted HVP,
  passing central finite differences (minimum relative error below
  1e-6) at :math:`p = (0.7, -0.4)`.
- The functional stacked twice (no more entries than parameters: one
  adjoint per entry) and three times (more: one tangent per parameter)
  gives every row equal to the scalar gradient to 1e-12, and a weighted
  HVP equal to the weights' sum times the scalar HVP to 1e-10 that
  passes central finite differences of the weighted gradient.
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

from pyapprox.implicitfunction.implicit_map import (
    AdjointRows,
    ImplicitJacobianProtocol,
    TangentColumns,
)
from pyapprox.implicitfunction.setups import (
    KeepAll,
    OperatorBuilderProtocol,
    SteadyRequest,
)
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_function import SteadyImplicitFunction
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.linalg.operators import LinearSolverProtocol, SolverHandleProtocol
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    pinned_quadratic,
    quadratic_functional,
)
from tests._helpers.linalg.stack_case import A, F, TypedStackCase


class _Ignored(Generic[A]):
    """A setter that moves nothing the residual reads."""

    def nparams(self) -> int:
        return 2

    def set_param(self, param: A) -> None:
        pass


class _CountingClears(Generic[A, F]):
    """``KeepAll`` that counts its clears."""

    def __init__(self, solver: LinearSolverProtocol[A, F]) -> None:
        self._inner: KeepAll[A, F] = KeepAll(solver)
        self.clears = 0

    def handle(
        self,
        request: SteadyRequest[A, F],
        builder: OperatorBuilderProtocol[SteadyRequest[A, F], A, F],
    ) -> SolverHandleProtocol[A]:
        return self._inner.handle(request, builder)

    def clear(self) -> None:
        self.clears += 1
        self._inner.clear()


def _function(
    case: TypedStackCase[A, F, A],
    residual_order: int = 2,
    functional_order: int = 2,
    nqoi: int = 1,
    maxiters: int = 30,
    jacobian: Optional[ImplicitJacobianProtocol] = None,
) -> Tuple[SteadyImplicitFunction[A, F], _CountingClears[A, F]]:
    f, pinned = pinned_quadratic(case, residual_order)
    one = quadratic_functional(f, functional_order)
    functional = one if nqoi == 1 else StateFunctional.stack(case.bkd, [one] * nqoi)
    setups: _CountingClears[A, F] = _CountingClears(case.direct_solver())
    function: SteadyImplicitFunction[A, F] = SteadyImplicitFunction(
        pinned,
        f.states,
        f,
        setups,
        Newton(atol=1e-13, rtol=0.0, maxiters=maxiters),
        functional,
        f.states.zeros(1),
        case.bkd,
        jacobian,
    )
    return function, setups


def _param(case: TypedStackCase[A, F, A]) -> A:
    return case.bkd.reshape(case.bkd.asarray(QUADRATIC_PARAM), (2, 1))


def _samples(case: TypedStackCase[A, F, A]) -> A:
    p = _param(case)
    return case.bkd.hstack([p, 0.5 * p, p + 0.1])


def _min_errors(
    case: TypedStackCase[A, F, A], function: SteadyImplicitFunction[A, F]
) -> List[float]:
    errors = DerivativeChecker(function).check_derivatives(
        _param(case), relative=True, central=True
    )
    return [case.bkd.to_float(case.bkd.min(e)) for e in errors]


class ValuesSuite:
    def test_satisfies_function_protocol(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        function, _ = _function(array_case)
        assert isinstance(function, FunctionProtocol)
        assert function.nvars() == 2 and function.nqoi() == 1

    def test_values_equal_a_separate_solve(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        case, bkd = array_case, array_case.bkd
        function, _ = _function(case)
        samples = _samples(case)
        f, pinned = pinned_quadratic(case)
        functional = quadratic_functional(f)
        setups: KeepAll[A, F] = KeepAll(case.direct_solver())
        system = SteadySystem(pinned, f.states, setups)
        derivs = SteadyDerivatives(pinned, f.states, functional)
        expected = []
        for ii in range(3):
            f.set_param(samples[:, ii])
            setups.clear()
            newton: Newton[A, A] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
            point = system.solve(newton, f.states.zeros(1))
            expected.append(derivs.value(point))
        bkd.assert_allclose(function(samples), bkd.hstack(expected), rtol=1e-12)


class DerivativesSuite:
    def test_scalar_gives_jacobian_and_hvp(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        function, _ = _function(array_case)
        derivs = function.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is not None and derivs.whvp is not None
        errors = _min_errors(array_case, function)
        assert len(errors) == 2
        assert max(errors) < 1e-6

    @pytest.mark.parametrize("nqoi", [2, 3], ids=["adjoint", "tangent"])
    def test_vector_rows_equal_the_scalar_gradient(
        self, array_case: TypedStackCase[A, F, A], nqoi: int
    ) -> None:
        scalar, _ = _function(array_case)
        vector, _ = _function(array_case, nqoi=nqoi)
        derivs = vector.derivatives()
        assert derivs.jacobian is not None
        scalar_jacobian = scalar.derivatives().jacobian
        assert scalar_jacobian is not None
        p = _param(array_case)
        gradient = scalar_jacobian(p)
        array_case.bkd.assert_allclose(
            derivs.jacobian(p), array_case.bkd.vstack([gradient] * nqoi), rtol=1e-12
        )

    @pytest.mark.parametrize("nqoi", [2, 3])
    def test_vector_whvp_is_the_weighted_scalar_hvp(
        self, array_case: TypedStackCase[A, F, A], nqoi: int
    ) -> None:
        """Every entry is the scalar functional, so the weighted HVP is
        the scalar HVP times the weights' sum; and it passes central
        finite differences of the weighted gradient."""
        bkd = array_case.bkd
        scalar, _ = _function(array_case)
        vector, _ = _function(array_case, nqoi=nqoi)
        scalar_hvp = scalar.derivatives().hvp
        whvp = vector.derivatives().whvp
        # The plain HVP is defined for one entry only.
        assert vector.derivatives().hvp is None
        assert scalar_hvp is not None and whvp is not None
        p = _param(array_case)
        direction = bkd.reshape(bkd.asarray([0.3, -1.1]), (2, 1))
        weights = bkd.reshape(bkd.asarray([0.7, -0.4, 1.3][:nqoi]), (nqoi, 1))
        bkd.assert_allclose(
            whvp(p, direction, weights),
            bkd.sum(weights) * scalar_hvp(p, direction),
            rtol=1e-10,
        )
        errors = DerivativeChecker(vector).check_derivatives(
            p, relative=True, central=True, weights=weights
        )
        assert bkd.to_float(bkd.min(errors[1])) < 1e-6

    @pytest.mark.parametrize("nqoi", [1, 2, 3])
    @pytest.mark.parametrize(
        "strategy", [AdjointRows(), TangentColumns()], ids=["adjoint", "tangent"]
    )
    def test_either_strategy_overrides_the_default(
        self,
        array_case: TypedStackCase[A, F, A],
        nqoi: int,
        strategy: ImplicitJacobianProtocol,
    ) -> None:
        default, _ = _function(array_case, nqoi=nqoi)
        chosen, _ = _function(array_case, nqoi=nqoi, jacobian=strategy)
        expected, actual = default.derivatives().jacobian, chosen.derivatives().jacobian
        assert expected is not None and actual is not None
        p = _param(array_case)
        array_case.bkd.assert_allclose(actual(p), expected(p), rtol=1e-12)
        assert repr(strategy) in repr(chosen)

    @pytest.mark.parametrize("orders", [(0, 2), (2, 0)], ids=["residual", "functional"])
    def test_none_without_first_derivatives(
        self, array_case: TypedStackCase[A, F, A], orders: Tuple[int, int]
    ) -> None:
        function, _ = _function(
            array_case, residual_order=orders[0], functional_order=orders[1]
        )
        assert function.derivatives().jacobian is None
        assert function(_param(array_case)).shape == (1, 1)

    def test_first_order_gives_no_hvp(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        function, _ = _function(array_case, residual_order=1)
        derivs = function.derivatives()
        assert derivs.jacobian is not None
        assert derivs.hvp is None and derivs.whvp is None


class SetupsSuite:
    def test_each_sample_clears_the_strategy(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        function, setups = _function(array_case)
        function(_samples(array_case))
        assert setups.clears == 3
        jacobian = function.derivatives().jacobian
        assert jacobian is not None
        jacobian(_param(array_case))
        assert setups.clears == 4


class RefusalsSuite:
    def test_wrong_arguments(self, array_case: TypedStackCase[A, F, A]) -> None:
        case, bkd = array_case, array_case.bkd
        f, pinned = pinned_quadratic(case)
        functional = quadratic_functional(f)
        newton: Newton[A, A] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        setups: KeepAll[A, F] = KeepAll(case.direct_solver())
        zeros = f.states.zeros(1)
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

    def test_parameters_the_residual_does_not_read(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        case = array_case
        f, pinned = pinned_quadratic(case)
        setups: KeepAll[A, F] = KeepAll(case.direct_solver())
        function: SteadyImplicitFunction[A, F] = SteadyImplicitFunction(
            pinned,
            f.states,
            _Ignored(),
            setups,
            Newton(atol=1e-13, rtol=0.0, maxiters=30),
            quadratic_functional(f),
            f.states.zeros(1),
            case.bkd,
        )
        with pytest.raises(ValueError, match="does not read these parameters"):
            function(_param(case))

    def test_a_solve_that_does_not_converge(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        function, _ = _function(array_case, maxiters=1)
        with pytest.raises(RuntimeError, match="did not converge"):
            function(_param(array_case))

    def test_a_jacobian_that_is_not_a_strategy(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        with pytest.raises(TypeError, match="ImplicitJacobianProtocol"):
            _function(array_case, jacobian=3.0)  # type: ignore[arg-type]
