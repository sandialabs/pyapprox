"""Expose an implicitly constrained QoI as a plain function of parameters.

The adjoint operators evaluate one parameter sample at a time and take the
initial state used to start the state solve. Consumers that speak
:class:`FunctionProtocol` — surrogates, sensitivity analysis, forward UQ
drivers, optimizers — want a batch map ``(nvars, nsamples) ->
(nqoi, nsamples)`` whose ``Derivatives`` bundle carries whatever the
operator supports. This module bridges the two.
"""

from typing import Generic, Optional

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
)
from pyapprox.optimization.implicitfunction.operator.adjoint_operator import (
    AdjointOperator,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    OperatorHVPFn,
)
from pyapprox.util.backends.protocols import Array, Backend


class ImplicitFunctionOfParameters(Generic[Array]):
    """A scalar implicitly constrained QoI as a function of its parameters.

    Wraps an adjoint operator so that ``f(samples)`` solves the state
    equation once per column and returns the functional value, and the
    ``Derivatives`` bundle exposes the adjoint gradient (and, when the
    operator supports it, the second-order adjoint Hessian-vector
    product).

    Parameters
    ----------
    operator : AdjointOperator
        Operator pairing a parameterized state equation with a scalar
        functional. The bundle declares ``hvp`` as well as ``jacobian``
        when the operator's own bundle carries one.
    init_state : Array
        Initial state used to start every state solve.
        Shape: ``(nstates, 1)``.
    """

    def __init__(
        self,
        operator: AdjointOperator[Array],
        init_state: Array,
    ) -> None:
        state_eq = operator.state_equation()
        validate_sample(state_eq.nstates(), init_state)
        if operator.functional().nqoi() != 1:
            raise ValueError(
                "operator functional must have nqoi == 1, got "
                f"{operator.functional().nqoi()}"
            )
        self._operator = operator
        self._init_state = init_state
        self._bkd = operator.bkd()
        # Capability is decided once, at construction, from the
        # operator's bundle.
        operator_hvp = operator.derivatives().hvp
        self._operator_hvp: Optional[OperatorHVPFn[Array]] = operator_hvp
        if operator_hvp is not None:
            self._derivs: Derivatives[Array] = Derivatives.second_order(
                jacobian=self.jacobian, hvp=self._hvp
            )
        else:
            self._derivs = Derivatives.first_order(jacobian=self.jacobian)

    def bkd(self) -> Backend[Array]:
        """Return the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of parameters."""
        return self._operator.nparams()

    def nqoi(self) -> int:
        """Return the number of quantities of interest (always 1)."""
        return 1

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle."""
        return self._derivs

    def __call__(self, samples: Array) -> Array:
        """Evaluate the QoI at each parameter sample.

        Parameters
        ----------
        samples : Array
            Shape: ``(nvars, nsamples)``.

        Returns
        -------
        Array
            Shape: ``(1, nsamples)``.
        """
        validate_samples(self.nvars(), samples)
        values = [
            self._operator(self._init_state, samples[:, ii : ii + 1])
            for ii in range(samples.shape[1])
        ]
        return self._bkd.reshape(
            self._bkd.hstack([self._bkd.flatten(value) for value in values]),
            (1, samples.shape[1]),
        )

    def jacobian(self, sample: Array) -> Array:
        """Adjoint gradient at one sample. Shape: ``(1, nvars)``."""
        validate_sample(self.nvars(), sample)
        return self._operator.jacobian(self._init_state, sample)

    def _hvp(self, sample: Array, vec: Array) -> Array:
        """Second-order adjoint Hessian-vector product.

        Parameters
        ----------
        sample : Array
            Shape: ``(nvars, 1)``.
        vec : Array
            Shape: ``(nvars, 1)``.

        Returns
        -------
        Array
            Shape: ``(nvars, 1)``.
        """
        operator_hvp = self._operator_hvp
        if operator_hvp is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        return operator_hvp(self._init_state, sample, vec)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"nqoi={self.nqoi()}, "
            f"operator={type(self._operator).__name__})"
        )
