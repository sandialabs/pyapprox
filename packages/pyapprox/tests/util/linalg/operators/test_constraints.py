"""Conformance tests for row sets and row-constrained operators.

Row replacement is checked against ``apply_dirichlet_rows``, the row
edit the PDE code uses today, so the decorator is a drop-in for it.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
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
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
    SpaceMismatchError,
    SymmetricEliminatedOperator,
    Transposed,
    rows_replaced_properties,
    rows_zeroed_properties,
    symmetric_eliminated_properties,
)
from pyapprox.util.linalg.scipy_stack import NumpyRowSet, ScipyCSROperator
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)
_ROWS = [3, 0, 3]  # unsorted, with a duplicate: the set is {0, 3}
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)


def _keep(n: int, rows: list[int]) -> NDArray[np.float64]:
    mask = np.ones((n, 1))
    mask[rows] = 0.0
    return mask


class TestRowSet:
    def test_satisfies_protocol_and_normalizes(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        rows = case.rows(_U, _ROWS)
        assert isinstance(rows, RowSetProtocol)
        assert rows.count() == 2
        assert rows.descriptor() == _U

    def test_operations(self, case: TypedStackCase[V, F, A]) -> None:
        rows = case.rows(_U, _ROWS)
        keep = case.dense(_keep(5, [0, 3]))
        x, y = case.vec(_U, 2, 0), case.vec(_U, 2, 1)
        xa, ya = case.array(x), case.array(y)
        case.assert_vector(rows.zero_rows(x), keep * xa)
        case.assert_vector(rows.replace_rows(y, x), keep * ya + (1 - keep) * xa)
        case.assert_vector(rows.add_rows(y, x), ya + (1 - keep) * xa)

    def test_out_and_aliasing(self, case: TypedStackCase[V, F, A]) -> None:
        rows = case.rows(_U, _ROWS)
        keep = case.dense(_keep(5, [0, 3]))
        x, y = case.vec(_U, 1, 2), case.vec(_U, 1, 3)
        expected = keep * case.array(y) + (1 - keep) * case.array(x)
        result = rows.replace_rows(y, x, out=y)
        case.assert_vector(result, expected)
        assert (result is y) == case.in_place
        with pytest.raises(ValueError, match="alias"):
            rows.replace_rows(y, x, out=x)
        with pytest.raises(ValueError, match="alias"):
            rows.add_rows(y, x, out=x)

    def test_difference(self, case: TypedStackCase[V, F, A]) -> None:
        rows = case.rows(_U, [0, 1, 3]).difference(case.rows(_U, [1, 4]))
        assert rows.count() == 2
        x = case.vec(_U, 1, 4)
        case.assert_vector(
            rows.zero_rows(x), case.dense(_keep(5, [0, 3])) * case.array(x)
        )
        with pytest.raises(SpaceMismatchError):
            rows.difference(case.rows(SpaceDescriptor("v", 5), [0]))

    def test_rejects_out_of_range_rows(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(ValueError, match=r"\[0, 5\)"):
            case.rows(_U, [0, 5])
        with pytest.raises(ValueError, match=r"\[0, 5\)"):
            case.rows(_U, [-1])


class TestRowReplacedOperator:
    def test_matches_apply_dirichlet_rows(self, case: TypedStackCase[V, F, A]) -> None:
        """Action, transpose and stored format all equal P A + (I - P)."""
        bkd = case.bkd
        a = random_matrix(5, 5, 5)
        replaced: NDArray[np.float64] = np.asarray(
            apply_dirichlet_rows(a, np.array([0, 3])), dtype=np.float64
        )
        reference = case.dense(replaced)
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, _ROWS))
        x = case.vec(_U, 2, 6)
        case.assert_vector(op.apply(x), case.times(reference, x), rtol=1e-12)
        case.assert_vector(
            op.apply_transpose(x), case.times(reference.T, x), rtol=1e-12
        )
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), reference)
        case.assert_dot_identity(op, 7)

    def test_apply_with_out(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = random_matrix(5, 5, 8)
        replaced: NDArray[np.float64] = np.asarray(
            apply_dirichlet_rows(a, np.array([0, 3])), dtype=np.float64
        )
        reference = case.dense(replaced)
        op = RowReplacedOperator(case.build(a, _U), case.rows(_U, _ROWS))
        x = case.vec(_U, 1, 9)
        out = case.vector(bkd.zeros((5, 1)), _U)
        result = op.apply(x, out=out)
        case.assert_vector(result, case.times(reference, x), rtol=1e-12)
        assert (result is out) == case.in_place
        result = op.apply_transpose(x, out=out)
        case.assert_vector(result, case.times(reference.T, x), rtol=1e-12)

    def test_clears_symmetry_unless_diagonal(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        rows = case.rows(_U, _ROWS)
        spd = RowReplacedOperator(
            case.build(spd_matrix(5, 10), _U, properties=_SPD), rows
        )
        assert spd.properties() == OperatorProperties()
        entries: NDArray[np.float64] = np.arange(1.0, 6.0, dtype=np.float64)[:, None]
        d = case.vector(case.dense(entries), _U)
        diagonal = RowReplacedOperator(case.from_diagonal(d, _U), rows)
        assert diagonal.properties() == _DIAG_SPD

    def test_empty_set_is_the_operator(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = spd_matrix(5, 11)
        stored = case.build(a, _U, properties=_SPD)
        op = RowReplacedOperator(stored, case.rows(_U, []))
        assert op.properties() == _SPD
        x = case.vec(_U, 2, 12)
        case.assert_vector(op.apply(x), case.array(stored.apply(x)))
        case.assert_vector(op.apply_transpose(x), case.array(stored.apply_transpose(x)))
        bkd.assert_allclose(case.format_dense(op.format()), case.dense(a))

    def test_rejects_rectangular_and_foreign_rows(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        with pytest.raises(SpaceMismatchError, match="row-replaced"):
            RowReplacedOperator(
                case.build(random_matrix(3, 5, 13), _U, _P), case.rows(_P, [0])
            )
        with pytest.raises(SpaceMismatchError, match="row set"):
            RowReplacedOperator(
                case.identity(_U), case.rows(SpaceDescriptor("v", 5), [0])
            )


class TestRowsZeroedOperator:
    def test_matches_masked_products(self, case: TypedStackCase[V, F, A]) -> None:
        """P A on a 3x5 operator; its transpose zeros A^T's columns."""
        bkd = case.bkd
        a = random_matrix(3, 5, 14)
        keep = _keep(3, [1])
        op = RowsZeroedOperator(case.build(a, _U, _P), case.rows(_P, [1]))
        reference = case.dense(keep * a)
        x, y = case.vec(_U, 2, 15), case.vec(_P, 2, 16)
        case.assert_vector(op.apply(x), case.times(reference, x), rtol=1e-12)
        case.assert_vector(
            op.apply_transpose(y), case.times(reference.T, y), rtol=1e-12
        )
        bkd.assert_allclose(case.format_dense(op.format()), reference)
        transposed = Transposed(op)
        bkd.assert_allclose(
            case.format_dense(transposed.format()), case.dense(a.T * keep.T)
        )
        case.assert_dot_identity(op, 17)

    def test_clears_spd(self, case: TypedStackCase[V, F, A]) -> None:
        rows = case.rows(_U, _ROWS)
        spd = RowsZeroedOperator(
            case.build(spd_matrix(5, 18), _U, properties=_SPD), rows
        )
        assert spd.properties() == OperatorProperties()
        entries: NDArray[np.float64] = np.arange(1.0, 6.0, dtype=np.float64)[:, None]
        d = case.vector(case.dense(entries), _U)
        diagonal = RowsZeroedOperator(case.from_diagonal(d, _U), rows)
        assert diagonal.properties() == OperatorProperties(
            symmetric=True, diagonal=True
        )

    def test_rejects_rows_of_the_domain(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(SpaceMismatchError, match="row set"):
            RowsZeroedOperator(
                case.build(random_matrix(3, 5, 19), _U, _P), case.rows(_U, [0])
            )


def _eliminated(a: NDArray[np.float64], rows: list[int]) -> NDArray[np.float64]:
    """P A P + (I - P), from masks."""
    keep = _keep(a.shape[0], rows)
    result: NDArray[np.float64] = keep * a * keep.T + np.diag(1.0 - keep[:, 0])
    return result


class TestSymmetricEliminatedOperator:
    def test_matches_the_dense_elimination(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """Action, transpose and stored format all equal P A P + (I - P),
        on an unsymmetric A so a transpose slip shows."""
        bkd = case.bkd
        a = random_matrix(5, 5, 21)
        reference = case.dense(_eliminated(a, [0, 3]))
        op = SymmetricEliminatedOperator(case.build(a, _U), case.rows(_U, _ROWS))
        x = case.vec(_U, 2, 22)
        case.assert_vector(op.apply(x), case.times(reference, x), rtol=1e-12)
        case.assert_vector(
            op.apply_transpose(x), case.times(reference.T, x), rtol=1e-12
        )
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), reference)
        case.assert_dot_identity(op, 23)

    def test_keeps_spd(self, case: TypedStackCase[V, F, A]) -> None:
        """An SPD operand stays SPD, so its stored form factors by
        Cholesky, where row replacement alone would clear the property."""
        bkd = case.bkd
        stored = case.build(spd_matrix(5, 24), _U, properties=_SPD)
        op = SymmetricEliminatedOperator(stored, case.rows(_U, _ROWS))
        assert op.properties() == _SPD
        np.linalg.cholesky(bkd.to_numpy(case.format_dense(op.format())))

    def test_empty_set_is_the_operator(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = random_matrix(5, 5, 25)
        stored = case.build(a, _U)
        op = SymmetricEliminatedOperator(stored, case.rows(_U, []))
        x = case.vec(_U, 2, 26)
        case.assert_vector(op.apply(x), case.array(stored.apply(x)))
        bkd.assert_allclose(case.format_dense(op.format()), case.dense(a))

    def test_rejects_rectangular_and_foreign_rows(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        with pytest.raises(SpaceMismatchError, match="symmetric elimination"):
            SymmetricEliminatedOperator(
                case.build(random_matrix(3, 5, 27), _U, _P), case.rows(_P, [0])
            )
        with pytest.raises(SpaceMismatchError, match="row set"):
            SymmetricEliminatedOperator(
                case.identity(_U), case.rows(SpaceDescriptor("v", 5), [0])
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
