"""Conformance tests for the Krylov solvers on every stack.

Conjugate gradients on SPD operators, GMRES on nonsymmetric ones, both on
stored and matrix-free operators; and the true-residual test that turns a
solve conjugate gradients cannot do (a strongly nonsymmetric operator)
into ``converged=False``.
"""

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    GMRES,
    ConjugateGradient,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import NumpyVectorSpace, ScipyCSROperator
from tests._helpers.linalg.operator_suites.krylov import (
    ConjugateGradientSuite,
    GMRESSuite,
    SharedSuite,
)

_U = SpaceDescriptor("u", 6)


class TestConjugateGradient(ConjugateGradientSuite):
    pass


class TestGMRES(GMRESSuite):
    pass


class TestShared(SharedSuite):
    pass


class TestSettings:
    def test_refuses_bad_settings(self) -> None:
        bkd = NumpyBkd()
        space = NumpyVectorSpace(_U, np.float64)
        with pytest.raises(ValueError, match="tol"):
            ConjugateGradient(space, bkd, tol=0.0)
        with pytest.raises(ValueError, match="maxiter"):
            GMRES(space, bkd, maxiter=0)
        with pytest.raises(ValueError, match="restart"):
            GMRES(space, bkd, restart=0)
        with pytest.raises(TypeError, match="InnerProductSpaceProtocol"):
            GMRES(bkd, bkd)  # type: ignore[arg-type]

    def test_refuses_complex_scalars(self) -> None:
        bkd = NumpyBkd()
        complex_space = SpaceDescriptor("z", 2, dtype="complex128")
        space = NumpyVectorSpace(complex_space, np.complex128)
        op = ScipyCSROperator.identity(complex_space, np.complex128)
        with pytest.raises(TypeError, match="real scalars"):
            ConjugateGradient(space, bkd).setup(op)
        with pytest.raises(TypeError, match="real scalars"):
            GMRES(space, bkd).setup(op)
