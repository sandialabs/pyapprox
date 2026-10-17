"""Conformance tests for the stored operators of the SciPy and backend stacks.

Every protocol test runs on ScipyCSROperator (NumPy arrays) and on
BackendDenseOperator under NumPy and Torch.
"""

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator
from tests._helpers.operator_stacks import StackCase as _Case
from tests._helpers.operator_stacks import random_matrix as _random
from tests._helpers.operator_stacks import spd_matrix as _spd

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)


class TestStoredOperatorConformance:
    def test_satisfies_protocols(self, case: _Case) -> None:
        op = case.build(_random(3, 5, 0), _U, _P)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op, DiagonalCapableProtocol)
        assert isinstance(op.format(), case.format_type)
        assert op.domain() == _U and op.range() == _P
        assert op.properties() == OperatorProperties()

    def test_apply_and_transpose_match_the_matrix(self, case: _Case) -> None:
        """Rectangular A: x in the domain (5), y in the range (3)."""
        bkd = case.bkd
        matrix = _random(3, 5, 1)
        op = case.build(matrix, _U, _P)
        x, y = case.vec(5, 2, 2), case.vec(3, 4, 3)
        dense = bkd.asarray(matrix)
        bkd.assert_allclose(op.apply(x), bkd.dot(dense, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(y), bkd.dot(dense.T, y), rtol=1e-12)

    def test_dot_identity(self, case: _Case) -> None:
        r"""<A x, y> = <x, A^T y>, the check that replaces independent
        transpose implementations."""
        bkd = case.bkd
        op = case.build(_random(3, 5, 4), _U, _P)
        x, y = case.vec(5, 1, 5), case.vec(3, 1, 6)
        lhs = bkd.sum(op.apply(x) * y)
        rhs = bkd.sum(x * op.apply_transpose(y))
        bkd.assert_allclose(bkd.reshape(lhs, (1,)), bkd.reshape(rhs, (1,)), rtol=1e-12)

    def test_apply_with_out(self, case: _Case) -> None:
        bkd = case.bkd
        op = case.build(_random(3, 5, 7), _U, _P)
        x = case.vec(5, 2, 8)
        out = bkd.zeros((3, 2))
        result = op.apply(x, out=out)
        bkd.assert_allclose(result, op.apply(x), rtol=1e-12)
        assert (result is out) == case.in_place

    def test_apply_never_aliases_its_input(self, case: _Case) -> None:
        op = case.build(_spd(5, 9), _U)
        x = case.vec(5, 1, 10)
        with pytest.raises(ValueError, match="alias"):
            op.apply(x, out=x)
        with pytest.raises(ValueError, match="alias"):
            op.apply_transpose(x, out=x)

    def test_rejects_wrong_input_shape(self, case: _Case) -> None:
        op = case.build(_random(3, 5, 11), _U, _P)
        with pytest.raises(ValueError, match="ncols"):
            op.apply(case.vec(3, 1, 12))
        with pytest.raises(ValueError, match="ncols"):
            op.apply(case.bkd.zeros((5,)))
        with pytest.raises(ValueError, match="ncols"):
            op.apply_transpose(case.vec(5, 1, 13))

    def test_rejects_matrix_shape_disagreeing_with_layouts(self, case: _Case) -> None:
        with pytest.raises(ValueError, match="range.size, domain.size"):
            case.build(_random(5, 3, 14), _U, _P)

    def test_diagonal_matches_the_matrix(self, case: _Case) -> None:
        bkd = case.bkd
        matrix = _random(5, 5, 15)
        diagonal = case.build(matrix, _U).diagonal()
        assert tuple(diagonal.shape) == (5, 1)
        # np.diag returns a read-only view, which torch warns about.
        bkd.assert_allclose(diagonal, bkd.asarray(np.diag(matrix).copy()[:, None]))
        with pytest.raises(ValueError, match="diagonal"):
            case.build(_random(3, 5, 16), _U, _P).diagonal()

    def test_identity_in_the_partner_format(self, case: _Case) -> None:
        """The identity is stored in its stack's format, so it combines
        with that stack's operators without format erasure."""
        bkd = case.bkd
        ident = case.identity(_U)
        assert isinstance(ident.format(), case.format_type)
        assert ident.properties() == OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )
        x = case.vec(5, 3, 17)
        bkd.assert_allclose(ident.apply(x), x)
        bkd.assert_allclose(ident.diagonal(), bkd.ones((5, 1)))

    def test_from_diagonal(self, case: _Case) -> None:
        bkd = case.bkd
        d = bkd.asarray(np.arange(1.0, 6.0)[:, None])
        op = case.from_diagonal(d, _U)
        assert isinstance(op.format(), case.format_type)
        props = op.properties()
        assert props.diagonal and props.symmetric and props.spd
        assert not props.identity
        x = case.vec(5, 2, 18)
        bkd.assert_allclose(op.apply(x), d * x, rtol=1e-12)
        bkd.assert_allclose(op.diagonal(), d)
        negative = case.from_diagonal(-d, _U)
        assert not negative.properties().spd
        with pytest.raises(ValueError, match="single vector"):
            case.from_diagonal(bkd.ones((5, 2)), _U)

    def test_verifies_declared_properties(self, case: _Case) -> None:
        symmetric = OperatorProperties(symmetric=True)
        case.build(_spd(5, 19), _U, properties=symmetric)
        with pytest.raises(ValueError, match="declared symmetric"):
            case.build(_random(5, 5, 20), _U, properties=symmetric)
        diagonal = OperatorProperties(symmetric=True, diagonal=True)
        with pytest.raises(ValueError, match="declared diagonal"):
            case.build(_spd(5, 21), _U, properties=diagonal)
        identity = OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )
        with pytest.raises(ValueError, match="declared identity"):
            case.build(2.0 * np.eye(5), _U, properties=identity)

    def test_symmetric_requires_one_space(self, case: _Case) -> None:
        """Equal sizes but different spaces: not a symmetric operator."""
        with pytest.raises(ValueError, match="symmetric operator"):
            case.build(
                np.eye(5),
                _U,
                SpaceDescriptor("v", 5),
                OperatorProperties(symmetric=True),
            )


class TestOperatorProperties:
    def test_implications_are_enforced(self) -> None:
        with pytest.raises(ValueError, match="spd=True requires symmetric"):
            OperatorProperties(spd=True)
        with pytest.raises(ValueError, match="diagonal=True requires symmetric"):
            OperatorProperties(diagonal=True)
        with pytest.raises(ValueError, match="identity=True requires diagonal"):
            OperatorProperties(symmetric=True, spd=True, identity=True)
        with pytest.raises(ValueError, match="identity=True requires spd"):
            OperatorProperties(symmetric=True, diagonal=True, identity=True)


class TestScipyCSROperator:
    def test_rejects_dense_matrix(self) -> None:
        with pytest.raises(TypeError, match="scipy sparse"):
            ScipyCSROperator(np.eye(5), _U)

    def test_rejects_dtype_disagreeing_with_layout(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            ScipyCSROperator(csr_matrix(np.eye(5, dtype=np.float32)), _U)

    def test_format_exposes_the_stored_matrix(self) -> None:
        matrix = csr_matrix(_random(3, 5, 22))
        op = ScipyCSROperator(matrix, _U, _P)
        NumpyBkd().assert_allclose(op.format().matrix().toarray(), matrix.toarray())


class TestBackendDenseOperator:
    def test_rejects_layout_dtype_other_than_the_backends(
        self, bkd: Backend[Any]
    ) -> None:
        with pytest.raises(ValueError, match="configured dtype"):
            BackendDenseOperator(
                bkd.asarray(np.eye(5)), SpaceDescriptor("u", 5, dtype="float32"), bkd
            )

    def test_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        """d(y^T A x)/dx = A^T y through autograd."""
        import torch

        matrix = _random(3, 5, 23)
        op = BackendDenseOperator(torch_bkd.asarray(matrix), _U, torch_bkd, _P)
        rng = np.random.default_rng(24)
        x = torch.tensor(rng.normal(size=(5, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(3, 1)))
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        torch_bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)
