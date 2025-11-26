from typing import Any, Optional, Union, Sequence, List, Tuple, overload

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backend import Backend


# Implement the PyTorch backend
class TorchBkd(Backend[torch.Tensor]):  # Specify torch.Tensor type
    @staticmethod
    def dot(Amat: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
        # for typing consistancy make sure dot always returns an array
        return torch.matmul(Amat, Bmat)

    @staticmethod
    def eye(
        nrows: int, ncols: Optional[int] = None, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.eye(
            nrows, ncols if ncols is not None else nrows, dtype=dtype
        )

    @staticmethod
    def inv(matrix: torch.Tensor) -> torch.Tensor:
        result = torch.linalg.inv(matrix)
        assert isinstance(
            result, torch.Tensor
        ), "Expected torch.linalg.inv to return a torch.Tensor"
        return result

    @staticmethod
    def array(
        array: Union[Sequence[Any], torch.Tensor, float, int],
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        return torch.tensor(array, dtype=dtype)

    @staticmethod
    def asarray(
        array: Union[Sequence[Any], torch.Tensor, float, int],
        dtype: Optional[Any] = None,
    ) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype)

    @staticmethod
    def assert_allclose(
        actual: torch.Tensor,
        desired: torch.Tensor,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: Optional[str] = None,
    ) -> None:
        torch.testing.assert_close(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=err_msg,
        )

    @staticmethod
    def double_dtype() -> Any:
        return torch.double

    @staticmethod
    def complex_dtype() -> Any:
        return torch.cdouble

    @staticmethod
    def int64_dtype() -> Any:
        return torch.int64

    @staticmethod
    def stack(
        arrays: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        axis: int = 0,
    ) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    @staticmethod
    def linspace(start: float, stop: float, num: int) -> torch.Tensor:
        return torch.linspace(start, stop, num)

    @staticmethod
    def logspace(start: float, stop: float, num: int) -> torch.Tensor:
        return torch.logspace(start, stop, num)

    @staticmethod
    def to_numpy(array: torch.Tensor) -> NDArray[Any]:
        return array.numpy()

    @staticmethod
    def flatten(array: torch.Tensor) -> torch.Tensor:
        return array.flatten()

    @staticmethod
    def meshgrid(
        *arrays: torch.Tensor, indexing: str = "xy"
    ) -> Tuple[torch.Tensor, ...]:
        return torch.meshgrid(*arrays, indexing=indexing)

    @staticmethod
    def reshape(array: torch.Tensor, newshape: Sequence[int]) -> torch.Tensor:
        return torch.reshape(array, newshape)

    @staticmethod
    def sum(
        array: torch.Tensor,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims=False,
    ) -> torch.Tensor:
        return torch.sum(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def sin(array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    @staticmethod
    def cos(array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    @staticmethod
    def full(
        shape: Tuple[int, ...], fill_value: float, dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def zeros(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(
        shape: Tuple[int, ...], dtype: Optional[Any] = None
    ) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype)

    @staticmethod
    def arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.arange(*args, **kwargs)

    @staticmethod
    def prod(
        array: torch.Tensor,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> torch.Tensor:
        if axis is None:
            return torch.prod(array, keepdim=keepdims)
        return torch.prod(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def any(
        array: torch.Tensor, axis: Optional[int] = None, keepdims: bool = False
    ) -> torch.Tensor:
        return torch.any(array, dim=axis, keepdim=keepdims)

    @staticmethod
    def log(array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    @staticmethod
    def exp(array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    @staticmethod
    def copy(array: torch.Tensor) -> torch.Tensor:
        return array.clone()

    @staticmethod
    def erf(array: torch.Tensor) -> torch.Tensor:
        return torch.erf(array)

    @staticmethod
    def erfinv(array: torch.Tensor) -> torch.Tensor:
        return torch.erfinv(array)
