from typing import Union

import numpy as np
import pytest

from pyapprox.util.backends.protocols import Array, Backend


# Example function using the backend
def foo(x: Array, backend: Backend[Array]) -> Union[Array]:
    """
    Example function that computes the dot product of an identity matrix
    and the input array using the specified backend.

    Args:
        x (Array): Input array.
        backend (Backend[Array]): Backend for array operations.

    Returns:
        Union[Array, float]: Result of the dot product.
    """
    identity = backend.eye(
        x.shape[0],
        x.shape[1] if x.ndim > 1 else None,
        dtype=x.dtype,
    )
    return identity @ x


# Base test class
class TestBackend:

    def test_2d_array(self, bkd: Backend[Array]) -> None:
        """
        Test the foo function with a 2D array.
        """
        array_2d = bkd.array([[1, 2], [3, 4]], dtype=bkd.double_dtype())
        result = foo(array_2d, bkd)
        expected = (
            array_2d  # Identity matrix dot product should return the original array
        )
        bkd.assert_allclose(result, expected)

    def test_1d_array(self, bkd: Backend[Array]) -> None:
        """
        Test the foo function with a 1D array.
        """
        array_1d = bkd.array([1, 2], dtype=bkd.double_dtype())
        result = foo(array_1d, bkd)
        expected = (
            array_1d  # Identity matrix dot product should return the original array
        )
        bkd.assert_allclose(result, expected)


# TODO:
# complete tests of all functions in backend protocol:
# use __test__ = False pattern typing/interface/functions/fromcallable/tests/ and
# load_tests to avoid running base class.


class TestTorchBkdDtype:
    """Test that TorchBkd respects device and dtype configuration."""

    @pytest.mark.parametrize(
        "dtype_name",
        ["float32", "float64"],
    )
    def test_tensor_creation_dtype(self, dtype_name: str) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        dtype = getattr(torch, dtype_name)
        bkd = TorchBkd(dtype=dtype)

        assert bkd.array([[1.0, 2.0]]).dtype == dtype
        assert bkd.zeros((2, 3)).dtype == dtype
        assert bkd.ones((2, 3)).dtype == dtype
        assert bkd.eye(3).dtype == dtype
        assert bkd.full((2,), 5.0).dtype == dtype
        assert bkd.empty((2,)).dtype == dtype
        assert bkd.linspace(0.0, 1.0, 5).dtype == dtype
        assert bkd.logspace(0.0, 1.0, 5).dtype == dtype

    def test_default_dtype_matches_init(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd32 = TorchBkd(dtype=torch.float32)
        assert bkd32.default_dtype() == torch.float32

        bkd64 = TorchBkd(dtype=torch.float64)
        assert bkd64.default_dtype() == torch.float64

    def test_default_constructor_uses_torch_default(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        torch.set_default_dtype(torch.float64)
        bkd = TorchBkd()
        assert bkd.default_dtype() == torch.float64
        assert bkd.zeros((2,)).dtype == torch.float64

    def test_explicit_dtype_overrides_default(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd(dtype=torch.float32)
        # Explicit dtype arg to method should override backend default
        result = bkd.zeros((2,), dtype=torch.float64)
        assert result.dtype == torch.float64

    def test_arange_device(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        result = bkd.arange(5)
        assert result.device == torch.device("cpu")

    def test_asarray_device(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        result = bkd.asarray([1.0, 2.0])
        assert result.device == torch.device("cpu")

    def test_tril_indices_device(self) -> None:
        torch = pytest.importorskip("torch")
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()
        row_idx, col_idx = bkd.tril_indices(3)
        assert row_idx.device == torch.device("cpu")
        assert col_idx.device == torch.device("cpu")


class TestTorchBkdMPS:
    """Test TorchBkd with MPS device (skipped if unavailable)."""

    def test_mps_tensor_creation(self, torch_mps_bkd: Backend[Array]) -> None:
        torch = pytest.importorskip("torch")
        result = torch_mps_bkd.zeros((2, 3))
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "mps"
        assert result.dtype.is_floating_point

    def test_mps_to_numpy(self, torch_mps_bkd: Backend[Array]) -> None:
        tensor = torch_mps_bkd.ones((3,))
        arr = torch_mps_bkd.to_numpy(tensor)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, np.ones(3), rtol=1e-6)


class TestDtypePredicates:
    """is_floating_dtype / is_integer_dtype agree across backends.

    Estimator templates guard their continuous-relaxation math with
    is_floating_dtype, so the predicate must reject every non-float dtype
    identically on every backend.
    """

    def test_float_is_floating(self, bkd: Backend[Array]) -> None:
        assert bkd.is_floating_dtype(bkd.array([1.0, 2.0]))

    def test_float_is_not_integer(self, bkd: Backend[Array]) -> None:
        assert not bkd.is_integer_dtype(bkd.array([1.0, 2.0]))

    def test_int_is_not_floating(self, bkd: Backend[Array]) -> None:
        int_array = bkd.asarray(bkd.array([1.0, 2.0]), dtype=bkd.int64_dtype())
        assert not bkd.is_floating_dtype(int_array)

    def test_int_is_integer(self, bkd: Backend[Array]) -> None:
        int_array = bkd.asarray(bkd.array([1.0, 2.0]), dtype=bkd.int64_dtype())
        assert bkd.is_integer_dtype(int_array)

    def test_bool_is_not_floating(self, bkd: Backend[Array]) -> None:
        """Bool must not pass the float guard on either backend.

        numpy and torch disagree on whether bool counts as an integer
        dtype, so only the positive float check is reliable here.
        """
        assert not bkd.is_floating_dtype(bkd.array([True, False]))

    def test_complex_is_not_floating(self, bkd: Backend[Array]) -> None:
        complex_array = bkd.asarray(
            bkd.array([1.0, 2.0]), dtype=bkd.complex_dtype()
        )
        assert not bkd.is_floating_dtype(complex_array)


class TestCdistShapeConsistency:
    """Distances must not depend on how many points are passed at once.

    torch.cdist switches to a matrix-multiply identity above 25 rows,
    which is accurate enough in isolation but makes the result depend on
    the batch size: the same pair of points gets different distances
    according to how many points accompany them. Blockwise kernel
    evaluation assembles a matrix from pieces and uses it in place of
    the whole, so that inconsistency is a correctness bug rather than a
    tolerance question.
    """

    def test_block_matches_corresponding_rows(self, bkd: Backend[Array]) -> None:
        """A slice of the inputs must give a slice of the output.

        The block is deliberately below torch's 25-row threshold while
        the full call is above it, so the two take different internal
        paths. Without the compute_mode override this differs by ~2e-08
        in float64.
        """
        import numpy as np

        np.random.seed(0)
        pts = bkd.array(np.random.uniform(0.0, 1.0, (37, 2)))
        # scaling by a lengthscale is what pushes the values into the
        # range where the identity's cancellation shows up
        scaled = pts / bkd.full((2,), 0.4)
        full = bkd.cdist(scaled, scaled)
        block = bkd.cdist(scaled[7:14], scaled)
        bkd.assert_allclose(block, full[7:14], rtol=1e-14)

    def test_every_block_matches(self, bkd: Backend[Array]) -> None:
        """Not just one lucky offset."""
        import numpy as np

        np.random.seed(0)
        pts = bkd.array(np.random.uniform(0.0, 1.0, (37, 2)))
        scaled = pts / bkd.full((2,), 0.4)
        full = bkd.cdist(scaled, scaled)
        for start in range(0, 37, 7):
            stop = min(start + 7, 37)
            bkd.assert_allclose(
                bkd.cdist(scaled[start:stop], scaled),
                full[start:stop],
                rtol=1e-14,
            )

    def test_symmetric_and_zero_diagonal(self, bkd: Backend[Array]) -> None:
        """Properties the mm identity can violate through cancellation.

        Squaring and subtracting can leave a small negative under the
        square root on the diagonal, where the true distance is exactly
        zero.
        """
        import numpy as np

        np.random.seed(0)
        pts = bkd.array(np.random.uniform(0.0, 1.0, (37, 2)))
        dists = bkd.cdist(pts, pts)
        bkd.assert_allclose(dists, dists.T, rtol=1e-14)
        bkd.assert_allclose(
            bkd.diag(dists), bkd.full((37,), 0.0), atol=1e-15, rtol=0.0
        )


class TestConj:
    def test_conjugates_complex_entries(self, bkd: Backend[Array]) -> None:
        x = bkd.asarray([[1.0 + 2.0j], [-3.0j]], dtype=bkd.complex_dtype())
        expected = bkd.asarray([[1.0 - 2.0j], [3.0j]], dtype=bkd.complex_dtype())
        bkd.assert_allclose(bkd.conj(x), expected)

    def test_leaves_real_entries_unchanged(self, bkd: Backend[Array]) -> None:
        x = bkd.asarray([[1.5], [-2.0]])
        bkd.assert_allclose(bkd.conj(x), x)


class TestLuSolveTranspose:
    """``lu_solve(transpose=True)`` solves with the plain transpose
    :math:`A^T` on every backend, complex matrices included."""

    @pytest.mark.parametrize("complex_entries", [False, True], ids=["real", "complex"])
    def test_solves_with_the_transpose(
        self, bkd: Backend[Array], complex_entries: bool
    ) -> None:
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 4)) + 4.0 * np.eye(4)
        b = rng.normal(size=(4, 2))
        dtype = bkd.double_dtype()
        if complex_entries:
            a = a + 1j * rng.normal(size=(4, 4))
            b = b + 1j * rng.normal(size=(4, 2))
            dtype = bkd.complex_dtype()
        lu, pivots = bkd.lu_factor(bkd.asarray(a, dtype=dtype))
        x = bkd.lu_solve(lu, pivots, bkd.asarray(b, dtype=dtype), transpose=True)
        bkd.assert_allclose(
            bkd.dot(bkd.asarray(a.T, dtype=dtype), x),
            bkd.asarray(b, dtype=dtype),
            rtol=1e-12,
        )


class TestStackedLu:
    """A ``(k, n, n)`` stack factors each matrix on its own: solving a
    ``(k, n, ncols)`` right-hand side with it, plain or transposed, gives
    each matrix's own solution, real and complex, blocks that need row
    swaps included (a zero leading entry)."""

    @pytest.mark.parametrize("complex_entries", [False, True], ids=["real", "complex"])
    @pytest.mark.parametrize("transpose", [False, True], ids=["plain", "transposed"])
    def test_each_matrix_is_solved_on_its_own(
        self, bkd: Backend[Array], complex_entries: bool, transpose: bool
    ) -> None:
        rng = np.random.default_rng(5)
        stack = rng.normal(size=(6, 3, 3)) + 2.0 * np.eye(3)
        rhs = rng.normal(size=(6, 3, 2))
        # Zero leading entries: the first column's pivot is not row 0.
        stack[1, 0, 0] = 0.0
        stack[4, :2, 0] = 0.0
        dtype = bkd.double_dtype()
        if complex_entries:
            stack = stack + 1j * rng.normal(size=(6, 3, 3))
            rhs = rhs + 1j * rng.normal(size=(6, 3, 2))
            dtype = bkd.complex_dtype()
        lu, pivots = bkd.lu_factor(bkd.asarray(stack, dtype=dtype))
        x = bkd.lu_solve(lu, pivots, bkd.asarray(rhs, dtype=dtype), transpose=transpose)
        for i in range(6):
            matrix = stack[i].T if transpose else stack[i]
            bkd.assert_allclose(
                bkd.dot(bkd.asarray(matrix, dtype=dtype), x[i]),
                bkd.asarray(rhs[i], dtype=dtype),
                rtol=1e-12,
                atol=1e-13,
            )
