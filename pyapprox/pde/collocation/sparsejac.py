from abc import ABC, abstractmethod
from typing import Union

from pyapprox.util.backends.template import Array, BackendMixin


class SparseJacobian(ABC):
    # needed for ndarray * SparseJacobian so numpy does not mulitply
    # SparseJacobian by each entry of ndarray separately
    __array_priority__ = 1
    
    def __init__(self, bkd: BackendMixin, shape: tuple, sparse_jac: Array):
        self._shape = shape
        self._bkd = bkd
        self.set_sparse_jacobian(sparse_jac)

    def __repr__(self):
        return "{0}(shape={1})".format(self.__class__.__name__, self._shape)

    @abstractmethod
    def copy(self) -> "SparseJacobian":
        raise NotImplementedError

    @abstractmethod
    def get_jacobian(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def set_sparse_jacobian(self, jac: Array):
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: "SparseJacobian"):
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: "SparseJacobian"):
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other: Union[Array, float]):
        raise NotImplementedError

    def __rmul__(self, other: Union[Array, float]):
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(self, other: Union[Array, float]):
        raise NotImplementedError

    @abstractmethod
    def rdot(self, other: Array):
        raise NotImplementedError

    def _check_other(self, other: Union[Array, float]):
        if not isinstance(other, (self._bkd.array_type(), float)):
            raise NotImplementedError(f"cannot appply jacobian and {other}")
        if (
                not isinstance(other, float) and other.shape != self._shape[:1]
                and not other.ndim == 0
        ):
            raise RuntimeError("Other Array has the wrong shape")


class DenseJac(SparseJacobian):
    def copy(self) -> "DenseJac":
        return DenseJac(
            self._bkd, self._shape, self._bkd.copy(self._sparse_jac)
        )

    def set_sparse_jacobian(self, jac: Array):
        if not isinstance(jac, self._bkd.array_type()):
            raise ValueError("Sparse jacobian must be an array")
        if jac.shape != self._shape:
            raise ValueError("jac has the wrong shape")
        self._sparse_jac = jac

    def get_jacobian(self) -> Array:
        return self._sparse_jac

    def __neg__(self):
        return DenseJac(self._bkd, self._shape, -self._sparse_jac)

    def __mul__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float) or other.ndim == 0:
            return DenseJac(self._bkd, self._shape, self._sparse_jac * other)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac * other[..., None]
        )

    def __truediv__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float) or other.ndim == 0:
            return DenseJac(self._bkd, self._shape, self._sparse_jac / other)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac / other[..., None]
        )

    def rdot(self, other: Array):
        # Compute A @ self.get_jacobian() in sparse format
        if not isinstance(other, self._bkd.array_type()):
            raise NotImplementedError(f"cannot dot product {0} with jacobian")
        if other.shape[1] != self._shape[0]:
            raise NotImplementedError("Array has the wrong shape")
        return DenseJac(self._bkd, self._shape, other @ self._sparse_jac)

    def __add__(self, other: "SparseJacobian"):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot add {other}")
        if isinstance(other, ZeroJac):
            return other.__add__(self)
        if isinstance(other, DiagJac):
            stride = self._shape[0]
            ninput_funs = self._shape[1] // self._shape[0]
            dense_jac = self._bkd.copy(self._sparse_jac)
            for ii in range(ninput_funs):
                dense_jac[
                    :, ii * stride : (ii + 1) * stride
                ] += self._bkd.diag(other._sparse_jac[:, ii])
            return DenseJac(self._bkd, self._shape, dense_jac)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac + other._sparse_jac
        )

    def __sub__(self, other: "SparseJacobian"):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot subtract {other}")
        if isinstance(other, ZeroJac):
            return other.__rsub__(self)
        if isinstance(other, DiagJac):
            stride = self._shape[0]
            ninput_funs = self._shape[1] // self._shape[0]
            dense_jac = self._bkd.copy(self._sparse_jac)
            for ii in range(ninput_funs):
                dense_jac[
                    :, ii * stride : (ii + 1) * stride
                ] -= self._bkd.diag(other._sparse_jac[:, ii])
            return DenseJac(self._bkd, self._shape, dense_jac)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac - other._sparse_jac
        )

    def __rsub__(self, other: "SparseJacobian"):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot subtract {other}")
        if isinstance(other, ZeroJac):
            return other.__rsub__(self)
        if isinstance(other, DiagJac):
            stride = self._shape[0]
            ninput_funs = self._shape[1] // self._shape[0]
            dense_jac = -self._sparse_jac
            for ii in range(ninput_funs):
                dense_jac[
                    :, ii * stride : (ii + 1) * stride
                ] += self._bkd.diag(other._sparse_jac[:, ii])
            return DenseJac(self._bkd, self._shape, dense_jac)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac - other._sparse_jac
        )


class DiagJac(SparseJacobian):
    def copy(self) -> "DiagJac":
        return DiagJac(
            self._bkd, self._shape, self._bkd.copy(self._sparse_jac)
        )

    def set_sparse_jacobian(self, jac: Array):
        if not isinstance(jac, self._bkd.array_type()):
            raise ValueError("Sparse jacobian must be an array")
        ninput_funs = self._shape[1] // self._shape[0]
        if jac.shape != (self._shape[0], ninput_funs):
            raise ValueError("jac has the wrong shape")
        self._sparse_jac = jac

    def get_jacobian(self) -> Array:
        return self._bkd.hstack(
            [
                self._bkd.diag(self._sparse_jac[:, ii])
                for ii in range(self._sparse_jac.shape[1])
            ]
        )

    def __neg__(self):
        return DiagJac(self._bkd, self._shape, -self._sparse_jac)

    def __mul__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float) or other.ndim == 0:
            return DiagJac(self._bkd, self._shape, self._sparse_jac * other)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac * other[:, None]
        )

    def __truediv__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float) or other.ndim == 0:
            return DiagJac(self._bkd, self._shape, self._sparse_jac / other)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac / other[:, None]
        )

    def rdot(self, other: Array):
        # Compute A @ self.get_jacobian() in sparse format
        if not isinstance(other, self._bkd.array_type()):
            raise NotImplementedError(f"cannot dot product {0} with jacobian")
        if other.shape[1] != self._shape[0]:
            raise NotImplementedError("Array has the wrong shape")
        dense_jac = self._bkd.empty(self._shape)
        # exploit fact dot product of derivative matrix is with
        # diag matrix
        stride = self._shape[0]
        ninput_funs = self._shape[1] // self._shape[0]
        for ii in range(ninput_funs):
            dense_jac[:, ii * stride : (ii + 1) * stride] = (
                other * self._sparse_jac[:, ii]
            )
        return DenseJac(self._bkd, self._shape, dense_jac)

    def __add__(self, other: "SparseJacobian"):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot add {other}")
        if isinstance(other, ZeroJac):
            return other.__add__(self)
        if isinstance(other, DenseJac):
            return other.__add__(self)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac + other._sparse_jac
        )

    def __sub__(self, other: "SparseJacobian"):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot subtract {other}")
        if isinstance(other, ZeroJac):
            return other.__rsub__(self)
        if isinstance(other, DenseJac):
            return other.__rsub__(self)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac - other._sparse_jac
        )


class ZeroJac(SparseJacobian):
    def __init__(self, bkd: BackendMixin, shape: tuple):
        super().__init__(bkd, shape, None)

    def copy(self) -> "ZeroJac":
        return ZeroJac(self._bkd, self._shape)

    def new_type(self, other: SparseJacobian):
        if isinstance(other, ZeroJac):
            return ZeroJac()
        return other.jac_type()

    def set_sparse_jacobian(self, jac: Array):
        if jac is not None:
            raise ValueError("Sparse jacobian must be None")

    def get_jacobian(self) -> Array:
        return self._bkd.zeros(self._shape)

    def __mul__(self, other: Union[Array, float]):
        self._check_other(other)
        return ZeroJac(self._bkd, self._shape)

    def __truediv__(self, other: Union[Array, float]):
        self._check_other(other)
        return ZeroJac(self._bkd, self._shape)

    def __add__(self, other: SparseJacobian):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot add {other}")
        if isinstance(other, ZeroJac):
            return ZeroJac(self._bkd, self._shape)
        return other.__class__(
            self._bkd, other._shape, self._bkd.copy(other._sparse_jac)
        )

    def __sub__(self, other: SparseJacobian):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot subtract {other}")
        if isinstance(other, ZeroJac):
            return ZeroJac(self._bkd, self._shape)
        return other.__class__(self._bkd, other._shape, -other._sparse_jac)

    def __rsub__(self, other: SparseJacobian):
        if not isinstance(other, SparseJacobian):
            raise ValueError(f"Cannot subtract {other}")
        return other.__class__(
            self._bkd, other._shape, self._bkd.copy(other._sparse_jac)
        )

    def rdot(self, other: Array):
        # Compute A @ self.get_jacobian() in sparse format
        if not isinstance(other, self._bkd.array_type()):
            raise NotImplementedError(f"cannot dot product {0} with jacobian")
        if other.shape[1] != self._shape[0]:
            raise NotImplementedError("Array has the wrong shape")
        return ZeroJac(self._bkd, self._shape)

    def __neg__(self):
        return ZeroJac(self._bkd, self._shape)
