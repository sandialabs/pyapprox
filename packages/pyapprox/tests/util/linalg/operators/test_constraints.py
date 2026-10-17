"""Conformance tests for row sets and row-constrained operators.

Row replacement is checked against ``apply_dirichlet_rows``, the row
edit the PDE code uses today, so the decorator is a drop-in for it.
"""

from typing import Any

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.pde.sparse_utils import apply_dirichlet_rows
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
)
from pyapprox.util.linalg.operators import (
    OperatorProperties,
    RowReplacedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
    SpaceMismatchError,
    Transposed,
    rows_replaced_properties,
    rows_zeroed_properties,
)
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipyCSROperator
from tests._helpers.operator_stacks import StackCase, random_matrix, spd_matrix

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)
_ROWS = [3, 0, 3]  # unsorted, with a duplicate: the set is {0, 3}
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)


def _keep(n: int, rows: list[int]) -> np.ndarray[Any, np.dtype[np.float64]]:
    mask = np.ones((n, 1))
    mask[rows] = 0.0
    return mask


def _dot_identity(case: StackCase, op: Any, seed: int) -> None:
    bkd = case.bkd
    x = case.vec(op.domain().size, 1, seed)
    y = case.vec(op.range().size, 1, seed + 1)
    lhs = bkd.reshape(bkd.sum(op.apply(x) * y), (1,))
    rhs = bkd.reshape(bkd.sum(x * op.apply_transpose(y)), (1,))
    bkd.assert_allclose(lhs, rhs, rtol=1e-12)


class TestRowSet:
    def test_satisfies_protocol_and_normalizes(self, case: StackCase) -> None:
        rows = case.rows(_U, _ROWS)
        assert isinstance(rows, RowSetProtocol)
        assert rows.count() == 2
        assert rows.descriptor() == _U

    def test_operations(self, case: StackCase) -> None:
        bkd = case.bkd
        rows = case.rows(_U, _ROWS)
        keep = case.dense(_keep(5, [0, 3]))
        x, y = case.vec(5, 2, 0), case.vec(5, 2, 1)
        bkd.assert_allclose(rows.zero_rows(x), keep * x)
        bkd.assert_allclose(rows.replace_rows(y, x), keep * y + (1 - keep) * x)
        bkd.assert_allclose(rows.add_rows(y, x), y + (1 - keep) * x)

    def test_out_and_aliasing(self, case: StackCase) -> None:
        bkd = case.bkd
        rows = case.rows(_U, _ROWS)
        keep = case.dense(_keep(5, [0, 3]))
        x, y = case.vec(5, 1, 2), case.vec(5, 1, 3)
        expected = keep * y + (1 - keep) * x
        result = rows.replace_rows(y, x, out=y)
        bkd.assert_allclose(result, expected)
        assert (result is y) == case.in_place
        with pytest.raises(ValueError, match="alias"):
            rows.replace_rows(y, x, out=x)
        with pytest.raises(ValueError, match="alias"):
            rows.add_rows(y, x, out=x)

    def test_difference(self, case: StackCase) -> None:
        bkd = case.bkd
        rows = case.rows(_U, [0, 1, 3]).difference(case.rows(_U, [1, 4]))
        assert rows.count() == 2
        x = case.vec(5, 1, 4)
        bkd.assert_allclose(rows.zero_rows(x), case.dense(_keep(5, [0, 3])) * x)
        with pytest.raises(SpaceMismatchError):
            rows.difference(case.rows(SpaceDescriptor("v", 5), [0]))

    def test_rejects_out_of_range_rows(self, case: StackCase) -> None:
        with pytest.raises(ValueError, match=r"\[0, 5\)"):
            case.rows(_U, [0, 5])
        with pytest.raises(ValueError, match=r"\[0, 5\)"):
            case.rows(_U, [-1])


class TestRowReplacedOperator:
    def test_matches_apply_dirichlet_rows(self, case: StackCase) -> None:
        """Action, transpose and stored format all equal P A + (I - P)."""
        bkd = case.bkd
        a = random_matrix(5, 5, 5)
        reference = case.dense(apply_dirichlet_rows(a, np.array([0, 3])))
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, _ROWS))
        x = case.vec(5, 2, 6)
        bkd.assert_allclose(op.apply(x), bkd.dot(reference, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(reference.T, x), rtol=1e-12)
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), reference)
        _dot_identity(case, op, 7)

    def test_apply_with_out(self, case: StackCase) -> None:
        bkd = case.bkd
        a = random_matrix(5, 5, 8)
        reference = case.dense(apply_dirichlet_rows(a, np.array([0, 3])))
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, _ROWS))
        x = case.vec(5, 1, 9)
        out = bkd.zeros((5, 1))
        result = op.apply(x, out=out)
        bkd.assert_allclose(result, bkd.dot(reference, x), rtol=1e-12)
        assert (result is out) == case.in_place
        result = op.apply_transpose(x, out=out)
        bkd.assert_allclose(result, bkd.dot(reference.T, x), rtol=1e-12)

    def test_clears_symmetry_unless_diagonal(self, case: StackCase) -> None:
        rows = case.rows(_U, _ROWS)
        spd = RowReplacedOperator(
            case.build(spd_matrix(5, 10), _U, properties=_SPD), rows
        )
        assert spd.properties() == OperatorProperties()
        d = case.dense(np.arange(1.0, 6.0)[:, None])
        diagonal = RowReplacedOperator(case.from_diagonal(d, _U), rows)
        assert diagonal.properties() == _DIAG_SPD

    def test_empty_set_is_the_operator(self, case: StackCase) -> None:
        bkd = case.bkd
        a = spd_matrix(5, 11)
        stored = case.build(a, _U, properties=_SPD)
        op = RowReplacedOperator(stored, case.rows(_U, []))
        assert op.properties() == _SPD
        x = case.vec(5, 2, 12)
        bkd.assert_allclose(op.apply(x), stored.apply(x))
        bkd.assert_allclose(op.apply_transpose(x), stored.apply_transpose(x))
        bkd.assert_allclose(case.format_dense(op.format()), case.dense(a))

    def test_rejects_rectangular_and_foreign_rows(self, case: StackCase) -> None:
        with pytest.raises(SpaceMismatchError, match="row-replaced"):
            RowReplacedOperator(
                case.build(random_matrix(3, 5, 13), _U, _P), case.rows(_P, [0])
            )
        with pytest.raises(SpaceMismatchError, match="row set"):
            RowReplacedOperator(
                case.identity(_U), case.rows(SpaceDescriptor("v", 5), [0])
            )


class TestRowsZeroedOperator:
    def test_matches_masked_products(self, case: StackCase) -> None:
        """P A on a 3x5 operator; its transpose zeros A^T's columns."""
        bkd = case.bkd
        a = random_matrix(3, 5, 14)
        keep = _keep(3, [1])
        op = RowsZeroedOperator(case.build(a, _U, _P), case.rows(_P, [1]))
        reference = case.dense(keep * a)
        x, y = case.vec(5, 2, 15), case.vec(3, 2, 16)
        bkd.assert_allclose(op.apply(x), bkd.dot(reference, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(y), bkd.dot(reference.T, y), rtol=1e-12)
        bkd.assert_allclose(case.format_dense(op.format()), reference)
        transposed = Transposed(op)
        bkd.assert_allclose(
            case.format_dense(transposed.format()), case.dense(a.T * keep.T)
        )
        _dot_identity(case, op, 17)

    def test_clears_spd(self, case: StackCase) -> None:
        rows = case.rows(_U, _ROWS)
        spd = RowsZeroedOperator(
            case.build(spd_matrix(5, 18), _U, properties=_SPD), rows
        )
        assert spd.properties() == OperatorProperties()
        d = case.dense(np.arange(1.0, 6.0)[:, None])
        diagonal = RowsZeroedOperator(case.from_diagonal(d, _U), rows)
        assert diagonal.properties() == OperatorProperties(
            symmetric=True, diagonal=True
        )

    def test_rejects_rows_of_the_domain(self, case: StackCase) -> None:
        with pytest.raises(SpaceMismatchError, match="row set"):
            RowsZeroedOperator(
                case.build(random_matrix(3, 5, 19), _U, _P), case.rows(_U, [0])
            )


class TestPropertyRules:
    @pytest.mark.parametrize(
        "props, nrows, replaced, zeroed",
        [
            (_SPD, 0, _SPD, _SPD),
            (_SPD, 2, OperatorProperties(), OperatorProperties()),
            (
                _DIAG_SPD,
                2,
                _DIAG_SPD,
                OperatorProperties(symmetric=True, diagonal=True),
            ),
            (
                OperatorProperties(
                    symmetric=True, spd=True, diagonal=True, identity=True
                ),
                1,
                OperatorProperties(
                    symmetric=True, spd=True, diagonal=True, identity=True
                ),
                OperatorProperties(symmetric=True, diagonal=True),
            ),
        ],
    )
    def test_table(
        self,
        props: OperatorProperties,
        nrows: int,
        replaced: OperatorProperties,
        zeroed: OperatorProperties,
    ) -> None:
        assert rows_replaced_properties(props, nrows) == replaced
        assert rows_zeroed_properties(props, nrows) == zeroed


class TestScipyRowEdits:
    def test_matches_sparse_apply_dirichlet_rows_with_missing_diagonal(self) -> None:
        """A replaced row whose diagonal is absent from the pattern still
        gets a unit diagonal, as in apply_dirichlet_rows."""
        a = random_matrix(5, 5, 20)
        a[3, 3] = 0.0
        matrix = csr_matrix(a)
        op = RowReplacedOperator(ScipyCSROperator(matrix, _U), NumpyRowSet(_U, _ROWS))
        reference = apply_dirichlet_rows(matrix, np.array([0, 3]))
        NumpyBkd().assert_allclose(op.format().matrix().toarray(), reference.toarray())

    def test_rejects_another_stacks_row_set(self, numpy_bkd: Backend[Any]) -> None:
        op = RowReplacedOperator(
            ScipyCSROperator.identity(_U), BackendRowSet(_U, [0], numpy_bkd)
        )
        with pytest.raises(TypeError, match="NumpyRowSet"):
            op.format()

    def test_rejects_overlapping_out(self) -> None:
        rows = NumpyRowSet(_U, _ROWS)
        block = np.ones((5, 2))
        with pytest.raises(ValueError, match="alias"):
            rows.replace_rows(np.zeros((5, 1)), block[:, :1], out=block[:, :1])


class TestBackendGraph:
    def test_row_replaced_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        r"""d(y^T R x)/dx = R^T y through autograd, R = P A + (I - P)."""
        import torch

        bkd = torch_bkd
        a = random_matrix(5, 5, 21)
        op = RowReplacedOperator(
            BackendDenseOperator(bkd.asarray(a), _U, bkd), BackendRowSet(_U, _ROWS, bkd)
        )
        rng = np.random.default_rng(22)
        x = torch.tensor(rng.normal(size=(5, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(5, 1)))
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)
