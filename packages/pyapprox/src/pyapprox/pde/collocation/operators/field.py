"""Field classes for spectral collocation methods.

Provides unified field representation on collocation meshes with Jacobian
tracking for automatic differentiation of PDE residuals.

Shape convention: (ncomponents, npts)
- Scalar fields: ncomponents=1
- Vector fields: ncomponents=ndim
"""

from typing import Generic, Optional, Union

from pyapprox.pde.collocation.operators.jacobian_types import (
    DenseJacobian,
    DiagJacobian,
    SparseJacobian,
    ZeroJacobian,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class Field(Generic[Array]):
    """Unified field representation on a collocation mesh.

    A field represents values at mesh points with a specified number
    of components. Scalar fields have ncomponents=1, vector fields
    have ncomponents=ndim.

    Shape convention: (ncomponents, npts)

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis providing mesh and derivative matrices.
    bkd : Backend
        Computational backend.
    ncomponents : int
        Number of field components (1 for scalar, ndim for vector).
    values : Array, optional
        Field values at mesh points. Shape: (ncomponents, npts)
    jacobian : SparseJacobian, optional
        Jacobian with respect to input variables.
    ninput_funs : int
        Number of input functions the Jacobian tracks.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        ncomponents: int = 1,
        values: Optional[Array] = None,
        jacobian: Optional[SparseJacobian[Array]] = None,
        ninput_funs: int = 1,
    ):
        self._basis = basis
        self._bkd = bkd
        self._ncomponents = ncomponents
        self._ninput_funs = ninput_funs
        self._values: Optional[Array] = None
        self._jacobian: Optional[SparseJacobian[Array]] = None

        if values is not None:
            self.set_values(values, jacobian)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> TensorProductBasisProtocol[Array]:
        """Return the collocation basis."""
        return self._basis

    def npts(self) -> int:
        """Return the number of mesh points."""
        return self._basis.npts()

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return self._basis.ndim()

    def ncomponents(self) -> int:
        """Return the number of field components."""
        return self._ncomponents

    def ninput_funs(self) -> int:
        """Return the number of input functions for Jacobian tracking."""
        return self._ninput_funs

    @property
    def is_scalar(self) -> bool:
        """Return True if this is a scalar field (ncomponents=1)."""
        return self._ncomponents == 1

    def set_values(
        self,
        values: Array,
        jacobian: Optional[SparseJacobian[Array]] = None,
    ) -> None:
        """Set field values and optionally the Jacobian.

        Parameters
        ----------
        values : Array
            Field values at mesh points. Shape: (ncomponents, npts)
        jacobian : SparseJacobian, optional
            Jacobian with respect to input variables.
            If None, creates a ZeroJacobian.
        """
        npts = self.npts()
        expected_shape = (self._ncomponents, npts)
        if values.shape != expected_shape:
            raise ValueError(f"values.shape {values.shape} should be {expected_shape}")
        self._values = values

        if jacobian is None:
            total_outputs = self._ncomponents * npts
            jac_shape = (total_outputs, npts * self._ninput_funs)
            self._jacobian = ZeroJacobian(self._bkd, jac_shape)
        else:
            self._jacobian = jacobian

    def values(self) -> Array:
        """Return field values.

        Returns
        -------
        Array
            Field values. Shape: (ncomponents, npts)
        """
        if self._values is None:
            raise RuntimeError("Must first call set_values()")
        return self._values

    def jacobian(self) -> SparseJacobian[Array]:
        """Return the Jacobian.

        Returns
        -------
        SparseJacobian
            Jacobian with respect to input variables.
        """
        if self._jacobian is None:
            raise RuntimeError("Must first call set_values()")
        return self._jacobian

    def get_jacobian(self) -> Array:
        """Return the dense Jacobian matrix.

        Returns
        -------
        Array
            Dense Jacobian. Shape: (ncomponents*npts, ninput_funs*npts)
        """
        return self.jacobian().get_jacobian()

    def as_flat(self) -> Array:
        """Return flattened values.

        Returns
        -------
        Array
            For scalar: shape (npts,)
            For vector: shape (ncomponents * npts,)
        """
        if self.is_scalar:
            return self._values[0, :]
        return self._bkd.flatten(self._values)

    def copy(self) -> "Field[Array]":
        """Return a copy of this field."""
        return Field(
            self._basis,
            self._bkd,
            self._ncomponents,
            self._bkd.copy(self.values()),
            self.jacobian().copy(),
            self._ninput_funs,
        )

    def component(self, idx: int) -> "Field[Array]":
        """Extract a single component as a scalar Field.

        Parameters
        ----------
        idx : int
            Component index.

        Returns
        -------
        Field
            Scalar field (ncomponents=1) for the specified component.
        """
        if idx < 0 or idx >= self._ncomponents:
            raise ValueError(
                f"Component index {idx} out of range [0, {self._ncomponents})"
            )
        values = self._bkd.reshape(self._values[idx, :], (1, self.npts()))
        # Extract corresponding rows from Jacobian
        npts = self.npts()
        start_row = idx * npts
        end_row = start_row + npts
        full_jac = self.get_jacobian()
        component_jac = full_jac[start_row:end_row, :]
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=values,
            jacobian=DenseJacobian(self._bkd, component_jac.shape, component_jac),
            ninput_funs=self._ninput_funs,
        )

    # Arithmetic operations with Jacobian tracking (for scalar fields)

    def __add__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Add two fields or add a scalar."""
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            new_values = self._bkd.copy(self.values())
            new_values[0, :] = self.values()[0, :] + other
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=self.jacobian().copy(),
                ninput_funs=self._ninput_funs,
            )
        if not isinstance(other, Field):
            raise ValueError(f"Cannot add {type(other)} to Field")
        if not other.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        new_values = self._bkd.reshape(
            self.values()[0, :] + other.values()[0, :], (1, self.npts())
        )
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=self.jacobian() + other.jacobian(),
            ninput_funs=self._ninput_funs,
        )

    def __radd__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Right-add (for sum() to work with initial value 0)."""
        if isinstance(other, int):
            other = float(other)
        return self.__add__(other)

    def __sub__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Subtract two fields or subtract a scalar."""
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            new_values = self._bkd.reshape(
                self.values()[0, :] - other, (1, self.npts())
            )
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=self.jacobian().copy(),
                ninput_funs=self._ninput_funs,
            )
        if not isinstance(other, Field):
            raise ValueError(f"Cannot subtract {type(other)} from Field")
        if not other.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        new_values = self._bkd.reshape(
            self.values()[0, :] - other.values()[0, :], (1, self.npts())
        )
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=self.jacobian() - other.jacobian(),
            ninput_funs=self._ninput_funs,
        )

    def __rsub__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Right-subtract (scalar - field)."""
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            new_values = self._bkd.reshape(
                other - self.values()[0, :], (1, self.npts())
            )
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=-self.jacobian(),
                ninput_funs=self._ninput_funs,
            )
        raise ValueError(f"Cannot subtract Field from {type(other)}")

    def __mul__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Multiply two fields or multiply by a scalar.

        Uses product rule for Jacobian: d(f*g) = g*df + f*dg
        """
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            new_values = self._bkd.reshape(
                self.values()[0, :] * other, (1, self.npts())
            )
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=self.jacobian() * other,
                ninput_funs=self._ninput_funs,
            )
        if not isinstance(other, Field):
            raise ValueError(f"Cannot multiply Field by {type(other)}")
        if not other.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")

        # Product rule: d(f*g) = g*df + f*dg
        f_vals = self.values()[0, :]
        g_vals = other.values()[0, :]
        new_values = self._bkd.reshape(f_vals * g_vals, (1, self.npts()))
        jac = g_vals * self.jacobian() + other.jacobian() * f_vals
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=jac,
            ninput_funs=self._ninput_funs,
        )

    def __rmul__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Right-multiply."""
        return self.__mul__(other)

    def __neg__(self) -> "Field[Array]":
        """Negate field."""
        return -1.0 * self

    def __pow__(self, power: Union[int, float]) -> "Field[Array]":
        """Raise field to a power.

        Uses power rule: d(f^n) = n * f^(n-1) * df
        """
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if not isinstance(power, (int, float)) or power == 0:
            raise ValueError("Power must be non-zero int or float")
        power = float(power)
        f_vals = self.values()[0, :]
        new_values = self._bkd.reshape(f_vals**power, (1, self.npts()))
        jac = power * self.jacobian() * f_vals ** (power - 1)
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=jac,
            ninput_funs=self._ninput_funs,
        )

    def __truediv__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Divide field by another field or scalar.

        Uses quotient rule: d(f/g) = (g*df - f*dg) / g^2
        """
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            if other == 0:
                raise ValueError("Cannot divide by zero")
            new_values = self._bkd.reshape(
                self.values()[0, :] / other, (1, self.npts())
            )
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=self.jacobian() / other,
                ninput_funs=self._ninput_funs,
            )
        if not isinstance(other, Field):
            raise ValueError(f"Cannot divide Field by {type(other)}")
        if not other.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")

        # Quotient rule: d(f/g) = (g*df - f*dg) / g^2
        f_vals = self.values()[0, :]
        g_vals = other.values()[0, :]
        new_values = self._bkd.reshape(f_vals / g_vals, (1, self.npts()))
        jac = (g_vals * self.jacobian() - other.jacobian() * f_vals) / g_vals**2
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=jac,
            ninput_funs=self._ninput_funs,
        )

    def __rtruediv__(self, other: Union["Field[Array]", float]) -> "Field[Array]":
        """Right-divide (scalar / field)."""
        if not self.is_scalar:
            raise NotImplementedError("Arithmetic only supported for scalar fields")
        if isinstance(other, (int, float)):
            other = float(other)
            # d(c/f) = -c * df / f^2
            f_vals = self.values()[0, :]
            new_values = self._bkd.reshape(other / f_vals, (1, self.npts()))
            jac = -other * self.jacobian() / f_vals**2
            return Field(
                self._basis,
                self._bkd,
                ncomponents=1,
                values=new_values,
                jacobian=jac,
                ninput_funs=self._ninput_funs,
            )
        raise ValueError(f"Cannot divide {type(other)} by Field")

    def deriv(self, dim: int, order: int = 1) -> "Field[Array]":
        """Compute spatial derivative of scalar field.

        Parameters
        ----------
        dim : int
            Spatial dimension (0 for x, 1 for y, 2 for z).
        order : int
            Derivative order (default 1).

        Returns
        -------
        Field
            Derivative field with updated Jacobian.
        """
        if not self.is_scalar:
            raise NotImplementedError("deriv() only supported for scalar fields")
        D = self._basis.derivative_matrix(order, dim)
        f_vals = self.values()[0, :]
        new_values = self._bkd.reshape(D @ f_vals, (1, self.npts()))
        jac = self.jacobian().rdot(D)
        return Field(
            self._basis,
            self._bkd,
            ncomponents=1,
            values=new_values,
            jacobian=jac,
            ninput_funs=self._ninput_funs,
        )


def scalar_field(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    values: Array,
    jacobian: Optional[SparseJacobian[Array]] = None,
    ninput_funs: int = 1,
) -> Field[Array]:
    """Create a scalar field from 1D values.

    Convenience function that reshapes (npts,) values to (1, npts).

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.
    values : Array
        Field values. Shape: (npts,)
    jacobian : SparseJacobian, optional
        Jacobian with respect to input variables.
    ninput_funs : int
        Number of input functions.

    Returns
    -------
    Field
        Scalar field with shape (1, npts).
    """
    npts = basis.npts()
    if values.shape == (npts,):
        values = bkd.reshape(values, (1, npts))
    return Field(
        basis,
        bkd,
        ncomponents=1,
        values=values,
        jacobian=jacobian,
        ninput_funs=ninput_funs,
    )


def input_field(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    values: Array,
    input_index: int = 0,
    ninput_funs: int = 1,
) -> Field[Array]:
    """Create an input scalar field with diagonal Jacobian.

    The Jacobian is identity for the corresponding input, zero elsewhere.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.
    values : Array
        Field values. Shape: (npts,)
    input_index : int
        Index of this input (0, 1, ..., ninput_funs-1).
    ninput_funs : int
        Total number of input functions.

    Returns
    -------
    Field
        Scalar field with diagonal Jacobian.
    """
    npts = basis.npts()
    if values.shape == (npts,):
        values_2d = bkd.reshape(values, (1, npts))
    else:
        values_2d = values

    # Create diagonal Jacobian: identity for this input, zero elsewhere
    jac_shape = (npts, npts * ninput_funs)
    diag_data = bkd.zeros((npts, ninput_funs))
    diag_data = bkd.copy(diag_data)
    ones = bkd.ones((npts,))
    for ii in range(npts):
        diag_data[ii, input_index] = ones[ii]
    jacobian = DiagJacobian(bkd, jac_shape, diag_data)

    return Field(
        basis,
        bkd,
        ncomponents=1,
        values=values_2d,
        jacobian=jacobian,
        ninput_funs=ninput_funs,
    )


def constant_field(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    value: float,
    ninput_funs: int = 1,
) -> Field[Array]:
    """Create a constant scalar field with zero Jacobian.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.
    value : float
        Constant value at all mesh points.
    ninput_funs : int
        Number of input functions for Jacobian shape.

    Returns
    -------
    Field
        Constant scalar field.
    """
    npts = basis.npts()
    values = bkd.reshape(bkd.full((npts,), value), (1, npts))
    return Field(basis, bkd, ncomponents=1, values=values, ninput_funs=ninput_funs)


def zero_field(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    ninput_funs: int = 1,
) -> Field[Array]:
    """Create a zero scalar field with zero Jacobian.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        The collocation basis.
    bkd : Backend
        Computational backend.
    ninput_funs : int
        Number of input functions for Jacobian shape.

    Returns
    -------
    Field
        Zero scalar field.
    """
    return constant_field(basis, bkd, 0.0, ninput_funs)
