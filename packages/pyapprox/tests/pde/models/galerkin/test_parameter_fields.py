"""The parameter Jacobian operator ``residual_derivatives`` builds."""

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.pde.models.galerkin.parameter_fields import _ParamJacobianOperator
from pyapprox.pde.parameterizations.derivatives import DenseParamJacobian
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import MatrixFree

_NSTATES, _NPARAMS = 5, 3


def _operator(
    bkd: Backend[Array],
) -> tuple[_ParamJacobianOperator[Array], Array]:
    matrix = bkd.asarray(
        np.random.default_rng(3).normal(0.0, 1.0, (_NSTATES, _NPARAMS))
    )
    states, params = backend_spaces(bkd, _NSTATES, _NPARAMS)
    op = _ParamJacobianOperator(
        DenseParamJacobian(matrix), params.descriptor(), states.descriptor()
    )
    return op, matrix


class TestParamJacobianOperator:
    def test_spaces_and_format(self, bkd: Backend[Array]) -> None:
        op = _operator(bkd)[0]
        assert op.domain().size == _NPARAMS
        assert op.range().size == _NSTATES
        assert isinstance(op.format(), MatrixFree)
        assert not op.properties().symmetric

    def test_actions_apply_the_jacobian(self, bkd: Backend[Array]) -> None:
        op, matrix = _operator(bkd)
        vecs = bkd.asarray(np.arange(2.0 * _NPARAMS).reshape(_NPARAMS, 2))
        wvecs = bkd.asarray(np.arange(3.0 * _NSTATES).reshape(_NSTATES, 3))
        bkd.assert_allclose(op.apply(vecs), matrix @ vecs, rtol=1e-14)
        bkd.assert_allclose(op.apply_transpose(wvecs), matrix.T @ wvecs, rtol=1e-14)

    def test_refuses_blocks_off_its_spaces(self, bkd: Backend[Array]) -> None:
        op = _operator(bkd)[0]
        with pytest.raises(ValueError, match="shape"):
            op.apply(bkd.zeros((_NSTATES, 1)))
        with pytest.raises(ValueError, match="shape"):
            op.apply_transpose(bkd.zeros((_NPARAMS,)))
