"""Conformance tests for row sets and row-constrained operators.

Row replacement is checked against ``apply_dirichlet_rows``, the row
edit the PDE code uses today, so the decorator is a drop-in for it.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.pde.sparse_utils import apply_dirichlet_rows
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendRowSet,
)
from pyapprox.util.linalg.operators import (
    OperatorProperties,
    RowReplacedOperator,
    SpaceDescriptor,
    rows_replaced_properties,
    rows_zeroed_properties,
    symmetric_eliminated_properties,
)
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipyCSROperator
from tests._helpers.linalg.operator_suites.constraints import (
    RowReplacedOperatorSuite,
    RowSetSuite,
    RowsZeroedOperatorSuite,
    SymmetricEliminatedOperatorSuite,
)
from tests._helpers.linalg.stack_case import (
    random_matrix,
)

_U = SpaceDescriptor("u", 5)
_ROWS = [3, 0, 3]  # unsorted, with a duplicate: the set is {0, 3}
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)


class TestRowSet(RowSetSuite):
    pass


class TestRowReplacedOperator(RowReplacedOperatorSuite):
    pass


class TestRowsZeroedOperator(RowsZeroedOperatorSuite):
    pass


class TestSymmetricEliminatedOperator(SymmetricEliminatedOperatorSuite):
    pass


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

    @pytest.mark.parametrize(
        "props, nrows, eliminated",
        [
            (_SPD, 0, _SPD),
            (_SPD, 2, _SPD),
            (_DIAG_SPD, 2, _DIAG_SPD),
            (OperatorProperties(symmetric=True), 2, OperatorProperties(symmetric=True)),
            (OperatorProperties(), 2, OperatorProperties()),
            # The unit diagonal on the set is not zero.
            (OperatorProperties(zero=True), 2, OperatorProperties()),
            (OperatorProperties(zero=True), 0, OperatorProperties(zero=True)),
        ],
    )
    def test_symmetric_elimination_keeps_structure(
        self, props: OperatorProperties, nrows: int, eliminated: OperatorProperties
    ) -> None:
        assert symmetric_eliminated_properties(props, nrows) == eliminated


class TestScipyRowEdits:
    def test_matches_sparse_apply_dirichlet_rows_with_missing_diagonal(self) -> None:
        """A replaced row whose diagonal is absent from the pattern still
        gets a unit diagonal, as in apply_dirichlet_rows."""
        a = random_matrix(5, 5, 20)
        a[3, 3] = 0.0
        matrix = csr_matrix(a)
        op = RowReplacedOperator(
            ScipyCSROperator(matrix, _U, scalar=np.float64), NumpyRowSet(_U, _ROWS)
        )
        reference = apply_dirichlet_rows(matrix, np.array([0, 3]))
        NumpyBkd().assert_allclose(op.format().matrix().toarray(), reference.toarray())

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_replaced_rows_match_the_dense_replacement(self, seed: int) -> None:
        """Random sparse patterns (empty rows, rows without a diagonal) and
        random row sets, the first and last rows included: the CSR built
        directly equals replacing the dense rows, and stays canonical."""
        rng = np.random.default_rng(seed)
        n = 30
        dense = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.15)
        dense[5, :] = 0.0
        dense[np.arange(0, n, 4), np.arange(0, n, 4)] = 0.0
        rows = sorted({0, n - 1, 5, *rng.choice(n, size=8, replace=False).tolist()})
        u = SpaceDescriptor("u", n, "float64")
        op = RowReplacedOperator(
            ScipyCSROperator(csr_matrix(dense), u, scalar=np.float64),
            NumpyRowSet(u, rows),
        )
        expected = dense.copy()
        expected[rows, :] = 0.0
        expected[rows, rows] = 1.0
        matrix = op.format().matrix()
        assert matrix.has_canonical_format
        NumpyBkd().assert_allclose(matrix.toarray(), expected, rtol=0.0, atol=0.0)

    def test_rejects_another_stacks_row_set(self, numpy_bkd: Backend[Array]) -> None:
        op = RowReplacedOperator(  # type: ignore[misc]
            ScipyCSROperator.identity(_U, np.float64), BackendRowSet(_U, [0], numpy_bkd)
        )
        with pytest.raises(TypeError, match="NumpyRowSet"):
            op.format()

    def test_rejects_overlapping_out(self) -> None:
        rows = NumpyRowSet(_U, _ROWS)
        block = np.ones((5, 2))
        with pytest.raises(ValueError, match="alias"):
            rows.replace_rows(np.zeros((5, 1)), block[:, :1], out=block[:, :1])
