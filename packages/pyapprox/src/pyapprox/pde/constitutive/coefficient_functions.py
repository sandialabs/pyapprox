"""Pointwise coefficient function objects: diffusion, reaction, and velocity laws.

Tiered protocol families mirroring the stress models
(``StressModelProtocol`` / ``WithTangent`` / ``WithSensitivity``): each
capability (spatial DOFs, state dependence, derivative order) is a
protocol tier discovered by ``isinstance`` — never by tuple length or
``callable()`` inspection. Function objects are evaluated pointwise at quadrature
coordinates inside assembly kernels; they are module-level picklable
classes (the ``_ExpTransform`` precedent) and solver-neutral numpy code
(quadrature arrays live outside the backend system).

Representations
---------------
- ``ConstantDiffusion`` — no spatial DOFs; enables stiffness caching.
- ``CoordinateDiffusion`` — kappa(x); no DOFs, evaluated at quadrature.
- ``NodalFieldDiffusion`` — kappa as FEM DOFs on a basis: the
  DIFFERENTIABLE representation (``set_dofs`` is the parameterization
  update path; sensitivities are exact mixed assemblies).
- ``StateDependentDiffusionProtocol`` — kappa(x, u) (shallow-shelf
  style). Declared now as the extension seam; the Jacobian/HVP
  assemblies it requires are future work, and consumers raise
  actionably when handed one.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from skfem import Basis

_Quad = NDArray[np.floating[Any]]


# =====================================================================
# Diffusion functions
# =====================================================================


@runtime_checkable
class DiffusionFunctionProtocol(Protocol):
    """State-independent diffusivity law kappa(x)."""

    def values(self, coords: _Quad) -> _Quad:
        """Evaluate kappa at coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates, shape (ndim, npts) or (ndim, nelems, nquad).

        Returns
        -------
        ndarray
            Diffusivity values with the coordinate trailing shape
            ((npts,) or (nelems, nquad)).
        """
        ...

    def is_constant(self) -> bool:
        """Whether kappa is spatially constant."""
        ...

    def version(self) -> int:
        """Monotone counter, incremented on every mutation.

        Consumers cache assembled operators keyed on this value;
        immutable functions return a constant.
        """
        ...


@runtime_checkable
class StateDependentDiffusionProtocol(Protocol):
    """Diffusivity law kappa(x, u) (e.g. shallow-shelf viscosity).

    Extension seam only: state dependence changes the Jacobian
    structure (an extra ``int kappa'(u) w (grad u . grad v)`` term), so
    consumers must add assemblies before accepting these —
    until then they raise actionably on isinstance.
    """

    def values(self, coords: _Quad, state: _Quad) -> _Quad:
        """Evaluate kappa at coordinates and state values."""
        ...

    def derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        """Evaluate d(kappa)/du at coordinates and state values."""
        ...


class ConstantDiffusion:
    """Spatially constant diffusivity."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def value(self) -> float:
        """Return the constant."""
        return self._value

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        return np.full(coords_np.shape[1:], self._value)

    def is_constant(self) -> bool:
        return True

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"ConstantDiffusion({self._value})"


class CoordinateDiffusion:
    """Diffusivity from a coordinate function kappa(x).

    Parameters
    ----------
    func : Callable
        Accepts coordinates of shape (ndim, npts) and returns (npts,).
    """

    def __init__(self, func: Callable[[_Quad], _Quad]) -> None:
        self._func = func

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._func(flat))
        return values.reshape(coords_np.shape[1:])

    def is_constant(self) -> bool:
        return False

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"CoordinateDiffusion({self._func!r})"


class _BasisEvaluatorProtocol(Protocol):
    """The single basis member NodalFieldDiffusion consumes."""

    def evaluate(self, coeffs: Any, points: Any) -> Any: ...

    def ndofs(self) -> int: ...


class NodalFieldDiffusion:
    """Diffusivity as nodal DOFs on a finite element basis.

    The differentiable representation: parameterizations update the
    DOFs through ``set_dofs`` and physics assemble exact sensitivities
    against them.

    Parameters
    ----------
    basis : _BasisEvaluatorProtocol
        Basis with ``evaluate(coeffs, points)`` and ``ndofs()`` (e.g.
        ``LagrangeBasis``).
    dofs : ndarray, optional
        Initial DOF values. Shape: (ndofs,). Defaults to ones.
    """

    def __init__(
        self,
        basis: _BasisEvaluatorProtocol,
        dofs: Optional[_Quad] = None,
    ) -> None:
        self._basis = basis
        if dofs is None:
            dofs = np.ones(basis.ndofs())
        self.set_dofs(dofs)

    def set_dofs(self, dofs: _Quad) -> None:
        """Set the field DOFs. Shape: (ndofs,)."""
        dofs_np = np.asarray(dofs, dtype=np.float64)
        if dofs_np.shape != (self._basis.ndofs(),):
            raise ValueError(
                f"dofs must have shape ({self._basis.ndofs()},), got "
                f"{dofs_np.shape}"
            )
        self._dofs = dofs_np
        self._version = getattr(self, "_version", 0) + 1

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def is_constant(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"NodalFieldDiffusion(ndofs={self.ndofs()})"


class NodalFieldForcing:
    """Forcing as nodal DOFs on a finite element basis.

    The differentiable representation: parameterizations update the
    DOFs through ``set_dofs``. Callable with the physics forcing
    contract (coordinates ``(ndim, npts)`` -> values ``(npts,)``), so
    it plugs into existing forcing kwargs unchanged; the assembled
    load is then exactly ``M @ dofs`` (the nodal interpolant
    integrated against the test functions), making
    ``residual_forcing_jacobian = M`` exact.

    Parameters
    ----------
    basis : _BasisEvaluatorProtocol
        Basis with ``evaluate(coeffs, points)`` and ``ndofs()`` (e.g.
        ``LagrangeBasis``).
    dofs : ndarray, optional
        Initial DOF values. Shape: (ndofs,). Defaults to zeros.
    """

    def __init__(
        self,
        basis: _BasisEvaluatorProtocol,
        dofs: Optional[_Quad] = None,
    ) -> None:
        self._basis = basis
        if dofs is None:
            dofs = np.zeros(basis.ndofs())
        self.set_dofs(dofs)

    def set_dofs(self, dofs: _Quad) -> None:
        """Set the field DOFs. Shape: (ndofs,)."""
        dofs_np = np.asarray(dofs, dtype=np.float64)
        if dofs_np.shape != (self._basis.ndofs(),):
            raise ValueError(
                f"dofs must have shape ({self._basis.ndofs()},), got "
                f"{dofs_np.shape}"
            )
        self._dofs = dofs_np
        self._version = getattr(self, "_version", 0) + 1

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def __call__(self, coords: _Quad) -> _Quad:
        """Evaluate the nodal interpolant at coordinates (ndim, npts)."""
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def __repr__(self) -> str:
        return f"NodalFieldForcing(ndofs={self.ndofs()})"


# =====================================================================
# Velocity functions
# =====================================================================


@runtime_checkable
class VelocityFunctionProtocol(Protocol):
    """State-independent advection velocity law v(x)."""

    def values(self, coords: _Quad) -> _Quad:
        """Evaluate the velocity at coordinates.

        Parameters
        ----------
        coords : ndarray
            Coordinates, shape (ndim, npts) or (ndim, nelems, nquad).

        Returns
        -------
        ndarray
            Velocity components with the full coordinate shape
            ((ndim, npts) or (ndim, nelems, nquad)).
        """
        ...

    def is_constant(self) -> bool:
        """Whether the velocity is spatially constant."""
        ...

    def version(self) -> int:
        """Monotone counter, incremented on every mutation."""
        ...


class ConstantVelocity:
    """Spatially constant velocity vector."""

    def __init__(self, vec: _Quad) -> None:
        self._vec = np.asarray(vec, dtype=np.float64).reshape(-1)

    def vec(self) -> _Quad:
        """Return the constant vector. Shape: (ndim,)."""
        return self._vec

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        shape = (len(self._vec),) + (1,) * (coords_np.ndim - 1)
        return np.broadcast_to(
            self._vec.reshape(shape), coords_np.shape
        ).copy()

    def is_constant(self) -> bool:
        return True

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"ConstantVelocity({self._vec})"


class CoordinateVelocity:
    """Velocity from a coordinate function v(x).

    Parameters
    ----------
    func : Callable
        Accepts coordinates of shape (ndim, npts) and returns
        (ndim, npts).
    """

    def __init__(self, func: Callable[[_Quad], _Quad]) -> None:
        self._func = func

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._func(flat))
        return values.reshape(coords_np.shape)

    def is_constant(self) -> bool:
        return False

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"CoordinateVelocity({self._func!r})"


class _VectorBasisEvaluatorProtocol(Protocol):
    """The basis members NodalFieldVelocity consumes."""

    def evaluate(self, coeffs: Any, points: Any) -> Any: ...

    def ndofs(self) -> int: ...

    def skfem_basis(self) -> "Basis": ...


class NodalFieldVelocity:
    """Velocity as interleaved DOFs of a vector FEM basis.

    The representation for velocity fields COMPUTED BY ANOTHER PDE
    (e.g. a (Navier-)Stokes solve feeding an advection-diffusion
    transport problem): pass the upstream vector basis and its solution
    DOFs. Evaluation delegates to the basis (exact interpolation).
    One-way-coupled sensitivities (a residual_velocity_jacobian mixed
    assembly) are the declared extension path, not yet implemented.

    Parameters
    ----------
    basis : _VectorBasisEvaluatorProtocol
        Vector basis with ``evaluate(coeffs, points)`` returning
        (ncomponents, npts) (e.g. ``VectorLagrangeBasis``).
    dofs : ndarray
        Interleaved velocity DOFs. Shape: (ndofs,).
    """

    def __init__(
        self, basis: _VectorBasisEvaluatorProtocol, dofs: _Quad
    ) -> None:
        self._basis = basis
        self.set_dofs(dofs)

    def set_dofs(self, dofs: _Quad) -> None:
        """Set the field DOFs. Shape: (ndofs,)."""
        dofs_np = np.asarray(dofs, dtype=np.float64)
        if dofs_np.shape != (self._basis.ndofs(),):
            raise ValueError(
                f"dofs must have shape ({self._basis.ndofs()},), got "
                f"{dofs_np.shape}"
            )
        self._dofs = dofs_np
        self._version = getattr(self, "_version", 0) + 1

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def basis(self) -> _VectorBasisEvaluatorProtocol:
        """Return the vector basis the DOFs live on."""
        return self._basis

    def values(self, coords: _Quad) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape)

    def is_constant(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"NodalFieldVelocity(ndofs={len(self._dofs)})"


# =====================================================================
# Reaction functions
# =====================================================================


@runtime_checkable
class ReactionFunctionProtocol(Protocol):
    """Pointwise reaction law R(x, u) with its u-derivative."""

    def value(self, coords: _Quad, state: _Quad) -> _Quad:
        """Evaluate R at coordinates and state values."""
        ...

    def derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        """Evaluate dR/du at coordinates and state values."""
        ...

    def is_linear(self) -> bool:
        """Whether R is linear in u (enables stiffness caching)."""
        ...


@runtime_checkable
class ReactionFunctionWithSecondDerivativeProtocol(
    ReactionFunctionProtocol, Protocol
):
    """Reaction law additionally exposing d^2R/du^2 (HVP tier)."""

    def second_derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        """Evaluate d^2R/du^2 at coordinates and state values."""
        ...


class LinearReaction:
    """Linear reaction R(u) = coeff * u; second derivative exactly 0."""

    def __init__(self, coeff: float) -> None:
        self._coeff = float(coeff)

    def coeff(self) -> float:
        """Return the linear coefficient."""
        return self._coeff

    def value(self, coords: _Quad, state: _Quad) -> _Quad:
        return self._coeff * np.asarray(state)

    def derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.full_like(np.asarray(state), self._coeff)

    def second_derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.zeros_like(np.asarray(state))

    def is_linear(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"LinearReaction({self._coeff})"


class NodalFieldLinearReaction:
    """Linear reaction R(u) = r(x) * u with r as nodal DOFs.

    The differentiable representation of a spatially-varying linear
    reaction coefficient: parameterizations update the DOFs through
    ``set_dofs`` and physics assemble exact sensitivities against
    them. Satisfies ``ReactionFunctionProtocol`` (linear, zero second
    derivative).

    Parameters
    ----------
    basis : _BasisEvaluatorProtocol
        Basis with ``evaluate(coeffs, points)`` and ``ndofs()`` (e.g.
        ``LagrangeBasis``).
    dofs : ndarray, optional
        Initial DOF values. Shape: (ndofs,). Defaults to zeros.
    """

    def __init__(
        self,
        basis: _BasisEvaluatorProtocol,
        dofs: Optional[_Quad] = None,
    ) -> None:
        self._basis = basis
        if dofs is None:
            dofs = np.zeros(basis.ndofs())
        self.set_dofs(dofs)

    def set_dofs(self, dofs: _Quad) -> None:
        """Set the field DOFs. Shape: (ndofs,)."""
        dofs_np = np.asarray(dofs, dtype=np.float64)
        if dofs_np.shape != (self._basis.ndofs(),):
            raise ValueError(
                f"dofs must have shape ({self._basis.ndofs()},), got "
                f"{dofs_np.shape}"
            )
        self._dofs = dofs_np
        self._version = getattr(self, "_version", 0) + 1

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def values(self, coords: _Quad) -> _Quad:
        """Evaluate the coefficient interpolant at coordinates."""
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def value(self, coords: _Quad, state: _Quad) -> _Quad:
        return self.values(coords) * np.asarray(state)

    def derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.broadcast_to(
            self.values(coords), np.asarray(state).shape
        ).copy()

    def second_derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.zeros_like(np.asarray(state))

    def is_linear(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"NodalFieldLinearReaction(ndofs={self.ndofs()})"


class CallableReaction:
    """Reaction from callables R and R' (and optionally R'').

    ``derivative`` is required — Newton needs the Jacobian term, and
    silently dropping it (the old bare-callable form) produced wrong
    Jacobians. ``second_derivative`` is a capability: supplied, the
    model satisfies ``ReactionFunctionWithSecondDerivativeProtocol``
    (dynamic binding, like the essential-BC time derivative).
    """

    def __init__(
        self,
        value_func: Callable[[_Quad, _Quad], _Quad],
        derivative_func: Callable[[_Quad, _Quad], _Quad],
        second_derivative_func: Optional[
            Callable[[_Quad, _Quad], _Quad]
        ] = None,
    ) -> None:
        self._value_func = value_func
        self._derivative_func = derivative_func
        self._second_derivative_func = second_derivative_func
        if second_derivative_func is not None:
            self.second_derivative = self._second_derivative_impl

    def value(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.asarray(self._value_func(coords, state))

    def derivative(self, coords: _Quad, state: _Quad) -> _Quad:
        return np.asarray(self._derivative_func(coords, state))

    def _second_derivative_impl(
        self, coords: _Quad, state: _Quad
    ) -> _Quad:
        if self._second_derivative_func is None:
            raise RuntimeError("second-derivative capability not bound")
        return np.asarray(self._second_derivative_func(coords, state))

    def is_linear(self) -> bool:
        return False

    def __repr__(self) -> str:
        has_second = self._second_derivative_func is not None
        return f"CallableReaction(second_derivative={has_second})"
