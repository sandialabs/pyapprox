"""Unit tests for DirichletConstraintSet and the BC role protocols."""

from typing import Any, Sequence

import numpy as np
import pytest
from pyapprox.pde.boundary import (
    ConstraintSetProtocol,
    DirichletConstraintSet,
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from scipy.sparse import csr_matrix, issparse


class _EssentialBC:
    """Minimal essential BC with optionally time-varying values."""

    def __init__(
        self,
        dofs: Sequence[int],
        vals: Sequence[float],
        bkd: Backend[Any],
        time_scale: float = 0.0,
    ) -> None:
        self._bkd = bkd
        self._dofs = bkd.asarray(
            np.asarray(dofs, dtype=np.int64), dtype=bkd.int64_dtype()
        )
        self._vals = bkd.asarray(np.asarray(vals, dtype=np.float64))
        self._time_scale = time_scale

    def bkd(self) -> Backend[Any]:
        return self._bkd

    def constrained_dofs(self) -> Any:
        return self._dofs

    def constrained_values(self, time: float) -> Any:
        return self._vals * (1.0 + self._time_scale * time)


def _example_set(
    bkd: Backend[Any], time_scale: float = 0.0
) -> DirichletConstraintSet[Any]:
    """Constraint set on 6 DOFs constraining [0, 3] and [5]."""
    bc0 = _EssentialBC([0, 3], [1.0, 2.0], bkd, time_scale)
    bc1 = _EssentialBC([5], [7.0], bkd, time_scale)
    return DirichletConstraintSet([bc0, bc1], nstates=6, bkd=bkd)


class TestProtocols:
    def test_roles_are_disjoint(self, numpy_bkd: NumpyBkd) -> None:
        bc = _EssentialBC([0], [1.0], numpy_bkd)
        assert isinstance(bc, EssentialBCProtocol)
        assert not isinstance(bc, WeakFormBCProtocol)

    def test_constraint_set_satisfies_protocol(self, numpy_bkd: NumpyBkd) -> None:
        cs = _example_set(numpy_bkd)
        assert isinstance(cs, ConstraintSetProtocol)


class TestConstruction:
    def test_accessors(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        assert cs.ndofs() == 3
        assert cs.nstates() == 6
        assert [int(d) for d in cs.dofs()] == [0, 3, 5]

    def test_values_time_dependence(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd, time_scale=1.0)
        bkd.assert_allclose(cs.values(0.0), bkd.asarray([1.0, 2.0, 7.0]))
        bkd.assert_allclose(cs.values(1.0), bkd.asarray([2.0, 4.0, 14.0]))

    def test_shared_dofs_last_bc_wins(self, bkd: Backend[Any]) -> None:
        """Corner DOFs shared by adjacent boundaries take the last value."""
        bc0 = _EssentialBC([0, 3], [1.0, 2.0], bkd)
        bc1 = _EssentialBC([3, 5], [30.0, 7.0], bkd)
        cs = DirichletConstraintSet([bc0, bc1], nstates=6, bkd=bkd)
        assert cs.ndofs() == 3
        assert [int(d) for d in cs.dofs()] == [0, 3, 5]
        bkd.assert_allclose(cs.values(0.0), bkd.asarray([1.0, 30.0, 7.0]))

    def test_triple_shared_dof_last_bc_wins(self, bkd: Backend[Any]) -> None:
        """A 3D-corner-style DOF shared by three face BCs."""
        bc0 = _EssentialBC([0, 1], [1.0, 2.0], bkd)
        bc1 = _EssentialBC([0, 2], [10.0, 3.0], bkd)
        bc2 = _EssentialBC([0, 4], [100.0, 5.0], bkd)
        cs = DirichletConstraintSet([bc0, bc1, bc2], nstates=6, bkd=bkd)
        assert [int(d) for d in cs.dofs()] == [0, 1, 2, 4]
        bkd.assert_allclose(
            cs.values(0.0), bkd.asarray([100.0, 2.0, 3.0, 5.0])
        )

    def test_repeated_dof_within_one_bc_raises(
        self, bkd: Backend[Any]
    ) -> None:
        bc = _EssentialBC([0, 3, 0], [1.0, 2.0, 3.0], bkd)
        with pytest.raises(ValueError, match="more than once"):
            DirichletConstraintSet([bc], nstates=6, bkd=bkd)

    def test_out_of_range_dofs_raise(self, bkd: Backend[Any]) -> None:
        bc = _EssentialBC([0, 6], [1.0, 2.0], bkd)
        with pytest.raises(ValueError, match="must lie in"):
            DirichletConstraintSet([bc], nstates=6, bkd=bkd)

    def test_non_essential_bc_raises(self, bkd: Backend[Any]) -> None:
        not_a_bc: Any = object()
        with pytest.raises(TypeError, match="EssentialBCProtocol"):
            DirichletConstraintSet([not_a_bc], nstates=6, bkd=bkd)

    def test_nonpositive_nstates_raises(self, bkd: Backend[Any]) -> None:
        with pytest.raises(ValueError, match="nstates"):
            DirichletConstraintSet([], nstates=0, bkd=bkd)


class TestApply:
    def test_apply_to_residual(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        residual = bkd.asarray(np.full(6, 9.0))
        state = bkd.asarray(np.arange(6.0))
        out = cs.apply_to_residual(residual, state, 0.0)
        bkd.assert_allclose(
            out, bkd.asarray([-1.0, 9.0, 9.0, 1.0, 9.0, -2.0])
        )
        # input untouched
        bkd.assert_allclose(residual, bkd.asarray(np.full(6, 9.0)))

    def test_apply_to_jacobian_dense(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        jacobian = bkd.asarray(np.full((6, 6), 2.0))
        out = cs.apply_to_jacobian(jacobian)
        expected = np.full((6, 6), 2.0)
        expected[[0, 3, 5], :] = np.eye(6)[[0, 3, 5]]
        bkd.assert_allclose(out, bkd.asarray(expected))

    def test_apply_to_jacobian_sparse(self, numpy_bkd: NumpyBkd) -> None:
        cs = _example_set(numpy_bkd)
        jacobian = csr_matrix(np.full((6, 6), 2.0))
        out = cs.apply_to_jacobian(jacobian)
        assert issparse(out)
        expected = np.full((6, 6), 2.0)
        expected[[0, 3, 5], :] = np.eye(6)[[0, 3, 5]]
        numpy_bkd.assert_allclose(out.toarray(), expected)

    def test_apply_to_mass_cached(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        mass = bkd.asarray(np.full((6, 6), 3.0))
        out1 = cs.apply_to_mass(mass)
        out2 = cs.apply_to_mass(mass)
        assert out1 is out2
        expected = np.full((6, 6), 3.0)
        expected[[0, 3, 5], :] = np.eye(6)[[0, 3, 5]]
        bkd.assert_allclose(out1, bkd.asarray(expected))
        # a different matrix object invalidates the cache
        other = bkd.asarray(np.full((6, 6), 4.0))
        out3 = cs.apply_to_mass(other)
        assert out3 is not out1
        expected_other = np.full((6, 6), 4.0)
        expected_other[[0, 3, 5], :] = np.eye(6)[[0, 3, 5]]
        bkd.assert_allclose(out3, bkd.asarray(expected_other))

    def test_apply_to_mass_sparse_cached(self, numpy_bkd: NumpyBkd) -> None:
        cs = _example_set(numpy_bkd)
        mass = csr_matrix(np.full((6, 6), 3.0))
        out1 = cs.apply_to_mass(mass)
        assert cs.apply_to_mass(mass) is out1
        assert issparse(out1)


class TestZeroing:
    def test_zero_rows_dense(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        matrix = bkd.asarray(np.full((6, 4), 2.0))
        out = cs.zero_rows(matrix)
        expected = np.full((6, 4), 2.0)
        expected[[0, 3, 5], :] = 0.0
        bkd.assert_allclose(out, bkd.asarray(expected))

    def test_zero_rows_sparse(self, numpy_bkd: NumpyBkd) -> None:
        cs = _example_set(numpy_bkd)
        matrix = csr_matrix(np.full((6, 4), 2.0))
        out = cs.zero_rows(matrix)
        assert issparse(out)
        expected = np.full((6, 4), 2.0)
        expected[[0, 3, 5], :] = 0.0
        numpy_bkd.assert_allclose(out.toarray(), expected)

    def test_zero_cols_dense(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        matrix = bkd.asarray(np.full((4, 6), 2.0))
        out = cs.zero_cols(matrix)
        expected = np.full((4, 6), 2.0)
        expected[:, [0, 3, 5]] = 0.0
        bkd.assert_allclose(out, bkd.asarray(expected))

    def test_zero_cols_sparse(self, numpy_bkd: NumpyBkd) -> None:
        cs = _example_set(numpy_bkd)
        matrix = csr_matrix(np.full((4, 6), 2.0))
        out = cs.zero_cols(matrix)
        assert issparse(out)
        expected = np.full((4, 6), 2.0)
        expected[:, [0, 3, 5]] = 0.0
        numpy_bkd.assert_allclose(out.toarray(), expected)

    def test_zero_entries(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        vec = bkd.asarray(np.full(6, 2.0))
        out = cs.zero_entries(vec)
        expected = np.full(6, 2.0)
        expected[[0, 3, 5]] = 0.0
        bkd.assert_allclose(out, bkd.asarray(expected))


class TestInjectAndClassification:
    def test_inject(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd, time_scale=1.0)
        state = bkd.asarray(np.zeros(6))
        out = cs.inject(state, 1.0)
        bkd.assert_allclose(
            out, bkd.asarray([2.0, 0.0, 0.0, 4.0, 0.0, 14.0])
        )
        bkd.assert_allclose(state, bkd.asarray(np.zeros(6)))

    def test_classification(self, bkd: Backend[Any]) -> None:
        cs = _example_set(bkd)
        cls = cs.classification()
        assert cls.essential == [0, 3, 5]
        assert cls.row_replaced == [0, 3, 5]
        assert cs.classification() is cls


class TestEmptySetNoOp:
    """Every method must be an exact passthrough for an empty set."""

    def test_accessors(self, bkd: Backend[Any]) -> None:
        cs = DirichletConstraintSet([], nstates=6, bkd=bkd)
        assert cs.ndofs() == 0
        assert len(cs.dofs()) == 0
        assert len(cs.values(0.0)) == 0

    def test_passthrough_identity(self, bkd: Backend[Any]) -> None:
        cs = DirichletConstraintSet([], nstates=6, bkd=bkd)
        vec = bkd.asarray(np.arange(6.0))
        mat = bkd.asarray(np.full((6, 6), 2.0))
        assert cs.apply_to_residual(vec, vec, 0.0) is vec
        assert cs.apply_to_jacobian(mat) is mat
        assert cs.apply_to_mass(mat) is mat
        assert cs.zero_rows(mat) is mat
        assert cs.zero_cols(mat) is mat
        assert cs.zero_entries(vec) is vec
        assert cs.inject(vec, 0.0) is vec

    def test_passthrough_sparse(self, numpy_bkd: NumpyBkd) -> None:
        cs = DirichletConstraintSet([], nstates=6, bkd=numpy_bkd)
        mat = csr_matrix(np.eye(6))
        assert cs.apply_to_jacobian(mat) is mat
        assert cs.apply_to_mass(mat) is mat
        assert cs.zero_rows(mat) is mat
        assert cs.zero_cols(mat) is mat

    def test_classification_empty(self, bkd: Backend[Any]) -> None:
        cs = DirichletConstraintSet([], nstates=6, bkd=bkd)
        cls = cs.classification()
        assert cls.essential == []
        assert cls.row_replaced == []
