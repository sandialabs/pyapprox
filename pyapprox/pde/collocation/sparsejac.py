from abc import ABC, abstractmethod
from typing import Union

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin


class SparseJacobian(ABC):
    def __init__(self, bkd: LinAlgMixin, shape: tuple, sparse_jac: Array):
        self._shape = shape
        self._bkd = bkd
        self.set_sparse_jacobian(sparse_jac)

    def __repr__(self):
        return "{0}(shape={1})".format(self.__class__.__name__, self._shape)

    @abstractmethod
    def new_type(self, other: "SparseJacobian"):
        raise NotImplementedError

    @abstractmethod
    def get_jacobian(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def set_sparse_jacobian(self, jac: Array):
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: "SparseJacobian"):
        raise NotImplementedError

    # def __radd__(self, other: "SparseJacobian"):
    #     return self.__add__(other)

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
        if not isinstance(other, float) and other.shape != self._shape[:1]:
            raise RuntimeError("Other Array has the wrong shape")


class DenseJac(SparseJacobian):
    def new_type(self, other: SparseJacobian):
        return DenseJac()

    def set_sparse_jacobian(self, jac: Array):
        if not isinstance(jac, self._bkd.array_type()):
            raise ValueError("Sparse jacobian must be an array")
        if jac.shape != self._shape:
            raise ValueError("jac has the wrong shape")
        self._sparse_jac = jac

    def get_jacobian(self) -> Array:
        return self._sparse_jac

    def __mul__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float):
            return DenseJac(self._bkd, self._shape, self._sparse_jac * other)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac * other[..., None]
        )

    def __truediv__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float):
            return DenseJac(self._bkd, self._shape, self._sparse_jac / other)
        return DenseJac(
            self._bkd, self._shape, self._sparse_jac / other[..., None]
        )

    def rdot(self, other: Array):
        # Compute A @ self.get_jacobian() in sparse format
        if not isinstance(other, self._bkd.array_type()):
            raise NotImplementedError(f"cannot dot product {0} with jacobian")
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
                dense_jac[:, ii*stride:(ii+1)*stride] += self._bkd.diag(other._sparse_jac[:, ii])
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
                dense_jac[:, ii*stride:(ii+1)*stride] -= self._bkd.diag(other._sparse_jac[:, ii])
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
                dense_jac[:, ii*stride:(ii+1)*stride] += self._bkd.diag(other._sparse_jac[:, ii])
            return DenseJac(self._bkd, self._shape, dense_jac)
        return DenseJac(
             self._bkd, self._shape, self._sparse_jac - other._sparse_jac
        )


class DiagJac(SparseJacobian):
    def new_type(self, other: SparseJacobian):
        if isinstance(other, DenseJac):
            return DenseJac
        return DiagJac()

    def set_sparse_jacobian(self, jac: Array):
        if not isinstance(jac, self._bkd.array_type()):
            raise ValueError("Sparse jacobian must be an array")
        ninput_funs = self._shape[1] / self._shape[0]
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

    def __mul__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float):
            return DiagJac(self._bkd, self._shape, self._sparse_jac * other)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac * other[:, None]
        )

    def __truediv__(self, other: Union[Array, float]):
        self._check_other(other)
        if isinstance(other, float):
            return DiagJac(self._bkd, self._shape, self._sparse_jac / other)
        return DiagJac(
            self._bkd, self._shape, self._sparse_jac / other[:, None]
        )

    def rdot(self, other: Array):
        # Compute A @ self.get_jacobian() in sparse format
        if not isinstance(other, self._bkd.array_type()):
            raise NotImplementedError(f"cannot dot product {0} with jacobian")
        dense_jac = self._bkd.empty(self._shape)
        # exploit fact dot product of derivative matrix is with
        # diag matrix
        stride = self.basis.mesh.nmesh_pts()
        for ii in range(self.ninput_funs()):
            dense_jac[:, ii*stride:(ii+1)*stride] = (
                other * self._bkd.diag(
                    self._sparse_jac[:, ii*stride:(ii+1)*stride]
                )
            )
        return DiagJac(self._bkd, self._shape, dense_jac)

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
    def __init__(self, bkd: LinAlgMixin, shape: tuple):
        super().__init__(bkd, shape, None)

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
        return other.__class__(
            self._bkd, other._shape, -other._sparse_jac
        )

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
        return ZeroJac(self._bkd, self._shape)



import numpy as np
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
def test_sparse_jacobian():
    bkd = NumpyLinAlgMixin
    nrows, ninput_funs = 3, 2
    shape = (nrows, nrows * ninput_funs)
    dense_jac_array = bkd.asarray(np.random.normal(0, 1, shape))
    dense_jac = DenseJac(bkd, shape, dense_jac_array)
    diag_jac_array = bkd.asarray(np.random.normal(0, 1, (nrows, ninput_funs)))
    diag_jac = DiagJac(bkd, shape, diag_jac_array)
    zero_jac = ZeroJac(bkd, shape)
    vec = bkd.arange(1, nrows+1)

    # operations on zero jacobian
    assert isinstance(zero_jac * 2., ZeroJac)
    assert isinstance(zero_jac * vec, ZeroJac)
    assert isinstance(zero_jac - 2. * zero_jac, ZeroJac)
    assert isinstance(zero_jac + zero_jac, ZeroJac)
    assert isinstance(zero_jac / 2., ZeroJac)
    assert isinstance(zero_jac / vec, ZeroJac)

    # operations on diagonal jacobian
    assert isinstance(diag_jac * 2., DiagJac)
    assert bkd.allclose(
        (diag_jac * 2.).get_jacobian(),
        bkd.hstack(
            [bkd.diag(2 * diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac * vec, DiagJac)
    assert bkd.allclose(
        (diag_jac * vec).get_jacobian(),
        bkd.hstack(
            [bkd.diag(vec * diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac - 2. * diag_jac, DiagJac)
    assert bkd.allclose(
        (diag_jac - 2. * diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(-diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac + diag_jac, DiagJac)
    assert bkd.allclose(
        (diag_jac + diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(2*diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac / 2., DiagJac)
    assert bkd.allclose(
        (diag_jac / 2.).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii] / 2) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac / vec, DiagJac)
    assert bkd.allclose(
        (diag_jac / vec).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii] / vec) for ii in range(2)]
        )
    )

    # operations on dense jacobian
    assert isinstance(dense_jac * 2., DenseJac)
    assert bkd.allclose(
        (dense_jac * 2.).get_jacobian(),
        2. * dense_jac_array
    )
    assert isinstance(dense_jac * vec, DenseJac)
    assert bkd.allclose(
        (dense_jac * vec).get_jacobian(),
        dense_jac_array * vec[:, None]
    )
    assert isinstance(dense_jac - 2. * dense_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac - 2. * dense_jac).get_jacobian(),
        -dense_jac_array
    )
    assert isinstance(dense_jac + dense_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac + dense_jac).get_jacobian(),
        2. * dense_jac_array
    )
    assert isinstance(dense_jac / 2., DenseJac)
    assert bkd.allclose(
        (dense_jac / 2.).get_jacobian(),
        dense_jac_array / 2.
    )
    assert isinstance(dense_jac / vec, DenseJac)
    assert bkd.allclose(
        (dense_jac / vec).get_jacobian(),
        dense_jac_array / vec[:, None]
    )

    # Combining zero and diagonal jacobians
    assert isinstance(diag_jac - 2. * zero_jac, DiagJac)
    assert bkd.allclose(
        (diag_jac - 2. * zero_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(diag_jac + zero_jac, DiagJac)
    assert bkd.allclose(
        (diag_jac + zero_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(zero_jac - 2. * diag_jac, DiagJac)
    assert bkd.allclose(
        (zero_jac - 2. * diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(-2. * diag_jac_array[:, ii]) for ii in range(2)]
        )
    )
    assert isinstance(zero_jac + diag_jac, DiagJac)
    assert bkd.allclose(
        (zero_jac + diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        )
    )

    # Combining zero and dense jacobians
    assert isinstance(dense_jac - 2. * zero_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac - 2. * zero_jac).get_jacobian(),
        dense_jac_array
    )
    assert isinstance(dense_jac + zero_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac + zero_jac).get_jacobian(),
        dense_jac_array
    )
    assert isinstance(zero_jac - 2. * dense_jac, DenseJac)
    assert bkd.allclose(
        (zero_jac - 2. * dense_jac).get_jacobian(),
        -2 * dense_jac_array
    )
    assert isinstance(zero_jac + dense_jac, DenseJac)
    assert bkd.allclose(
        (zero_jac + dense_jac).get_jacobian(),
        dense_jac_array
    )

    # Combining diag and dense jacobians
    assert isinstance(diag_jac - 2. * dense_jac, DenseJac)
    assert bkd.allclose(
        (diag_jac - 2. * dense_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        ) - 2 * dense_jac_array
    )
    assert isinstance(diag_jac + dense_jac, DenseJac)
    assert bkd.allclose(
        (diag_jac + dense_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        ) + dense_jac_array
    )
    assert isinstance(dense_jac - 2. * diag_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac - 2. * diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(-2. * diag_jac_array[:, ii]) for ii in range(2)]
        ) + dense_jac_array
    )
    assert isinstance(dense_jac + diag_jac, DenseJac)
    assert bkd.allclose(
        (dense_jac + diag_jac).get_jacobian(),
        bkd.hstack(
            [bkd.diag(diag_jac_array[:, ii]) for ii in range(2)]
        ) + dense_jac_array
    )
    


test_sparse_jacobian()
