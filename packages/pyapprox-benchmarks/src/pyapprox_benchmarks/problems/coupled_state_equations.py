"""Forward UQ problem built on a nonlinear coupled state equation.

The quantity of interest is defined implicitly: each parameter sample
requires a Newton solve of

.. math::
    a u_1^2 + u_2^2 - 1 = 0, \\qquad u_1^2 - b u_2^2 - 1 = 0,

after which a scalar functional reads the solved state. The problem
exists to exercise machinery that propagates uncertainty through an
implicit solve, and it carries adjoint gradients and second-order
adjoint Hessian-vector products for consumers that use derivative
information (gradient-enhanced surrogates, local expansions, active
subspaces). The functional is a constructor argument, so the same
state equation and prior support quantities of interest with quite
different geometry.
"""

from typing import Generic, Optional

from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianAndHVPProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.weighted_sum import (
    WeightedSumFunctional,
)
from pyapprox.optimization.implicitfunction.operator.as_function import (
    ImplicitFunctionOfParameters,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_hvp import (
    AdjointOperatorWithJacobianAndHVP,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.functions.algebraic.wildeys_nonlinear_state_equation import (
    NonLinearCoupledStateEquations,
)
from pyapprox_benchmarks.problems.forward_uq import ForwardUQProblem


class CoupledStateEquationsForwardUQ(Generic[Array]):
    """Forward UQ over the nonlinear coupled state equations.

    The parameters :math:`(a, b)` are independent uniforms. The support
    keeps :math:`a` bounded away from 1, where the real solution branch
    degenerates: the closed form :math:`u_2^2 = (1-a)/(1+ab)` requires
    :math:`a < 1`, and Newton's convergence deteriorates as that bound
    is approached. :math:`b` is unconstrained by feasibility.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    functional : ParameterizedFunctionalWithJacobianAndHVPProtocol, optional
        Scalar functional of ``(state, param)`` defining the quantity of
        interest. Defaults to reading the second state component. Supply
        one to change what the problem measures without changing the
        state equation, the prior, or the adjoint machinery: a
        least-squares misfit turns the same solve into a
        parameter-estimation objective, and reading the first state
        instead is a ``WeightedSumFunctional`` with the weights
        transposed. Must declare ``nqoi() == 1`` and the four
        second-order terms.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        functional: Optional[
            ParameterizedFunctionalWithJacobianAndHVPProtocol[Array]
        ] = None,
    ) -> None:
        state_eq: NonLinearCoupledStateEquations[Array] = (
            NonLinearCoupledStateEquations(bkd)
        )
        nstates = state_eq.nstates()
        if functional is None:
            # Reading one state component as a weighted sum keeps the
            # functional scalar AND second-order capable; the
            # subset-of-states functional is vector valued for forward
            # sensitivities and declares no hvp.
            weights = bkd.asarray([[0.0]] * (nstates - 1) + [[1.0]])
            functional = WeightedSumFunctional(weights, state_eq.nparams(), bkd)
        elif not isinstance(
            functional, ParameterizedFunctionalWithJacobianAndHVPProtocol
        ):
            raise TypeError(
                "functional must satisfy "
                "ParameterizedFunctionalWithJacobianAndHVPProtocol, got "
                f"{type(functional).__name__}"
            )
        operator = AdjointOperatorWithJacobianAndHVP(state_eq, functional)
        # Start Newton inside the solution branch for the whole support.
        init_state = bkd.full((nstates, 1), 0.75)
        function = ImplicitFunctionOfParameters(operator, init_state)
        prior = IndependentJoint(
            [UniformMarginal(0.1, 0.9, bkd), UniformMarginal(0.2, 2.0, bkd)],
            bkd,
        )
        self._problem: ForwardUQProblem[
            ImplicitFunctionOfParameters[Array], Array
        ] = ForwardUQProblem(
            "coupled_state_equations",
            function,
            prior,
            description=(
                "Uncertainty propagation through a nonlinear coupled state "
                "equation solved by Newton's method. The quantity of "
                "interest is a scalar functional of the solved state; "
                "adjoint gradients and second-order adjoint Hessian-vector "
                "products are available."
            ),
        )

    def problem(
        self,
    ) -> ForwardUQProblem[ImplicitFunctionOfParameters[Array], Array]:
        """Return the forward UQ problem."""
        return self._problem
