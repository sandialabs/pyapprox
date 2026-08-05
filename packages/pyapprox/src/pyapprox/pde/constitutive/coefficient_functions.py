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

import inspect
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
# Time-awareness declaration
# =====================================================================
#
# Time is threaded as an ARGUMENT, never held as state. The principle:
# bind what is constant across an evaluation, thread what varies within
# it. A parameter vector is constant per residual evaluation, so
# ``set_param`` is state; time is not (Crank-Nicolson assembles at
# t_n and t_{n+1}, multi-stage schemes at every stage time), so time is
# passed in.
#
# Whether a supplier consults that time is DECLARED, never inferred.
# Inference by signature inspection is unreliable (``*args``,
# undecorated wrappers, C builtins) and insufficient: cache-key policy
# needs an answer regardless of what a signature looks like. Runtime
# protocols cannot help either --- they test method presence, so a
# one-argument lambda satisfies a ``__call__(coords, time)`` protocol
# just as well as a two-argument function does.
#
# TODO(time-aware suppliers): require TimeIndependent/TimeDependent for
# EVERY coefficient and forcing supplier, and drop the bare ``f(coords)``
# form entirely.
#
# Bare callables are accepted today for compatibility, and
# ``as_time_aware`` treats them as time-independent. That default is a
# guess, and guesses about time are the failure mode this module exists
# to remove: assume time-independent and a transient source silently
# freezes at t=0; assume time-dependent and a supplier whose second
# parameter means something else silently receives a time. Requiring a
# wrapper everywhere deletes the guess rather than improving it, and
# lets ``as_time_aware`` collapse to a single isinstance check.
#
# DRIVE-BY RULE until then: when you touch a file that constructs a
# coefficient or forcing supplier, wrap the bare callables it passes ---
# ``TimeIndependent(f)`` for a steady field, ``TimeDependent(f)`` for one
# that varies. Small, local, and each one removes a place where the
# default has to be trusted.


@runtime_checkable
class TimeVaryingProtocol(Protocol):
    """Anything that states whether its values vary with time.

    Separate from ``TimeAwareCallableProtocol`` because the two answer
    different questions: this one is about the FIELD, and coefficient
    objects expose their values through ``values``/``value`` rather than
    ``__call__``. Consumers use it to key a cache --- an assembled
    operator built from a time-varying coefficient must not be reused at
    a later time.
    """

    def is_time_dependent(self) -> bool:
        """Whether the values depend on time."""
        ...


@runtime_checkable
class VersionedProtocol(Protocol):
    """Anything carrying a monotone counter bumped on every mutation.

    Consumers key assembled operators on it so a rebind invalidates
    while repeated evaluation reuses. Separate from
    ``TimeVaryingProtocol`` because the two answer different questions:
    a version tracks MUTATION, while time-dependence is a property of
    the values at a fixed version.
    """

    def version(self) -> int:
        """Monotone counter, incremented on every mutation."""
        ...


@runtime_checkable
class TimeAwareCallableProtocol(Protocol):
    """A coefficient supplier that states whether it consults time."""

    def __call__(self, coords: _Quad, time: float) -> _Quad:
        """Evaluate at coordinates and time."""
        ...

    def is_time_dependent(self) -> bool:
        """Whether the values depend on ``time``.

        Describes the FIELD, not the problem: a steady solve of a
        time-dependent coefficient is a well-defined snapshot. Consumers
        use this to choose a cache key, never to decide whether a
        problem is transient.
        """
        ...


class TimeIndependent:
    """Adapt a ``f(coords)`` supplier to the ``f(coords, time)`` call.

    Bare one-argument callables are permanent public API for
    time-independent coefficients; this wrapper is how they reach a
    uniform internal call site. Module-level and picklable, so wrapping
    does not make a picklable supplier unpicklable (it cannot rescue one
    that already was not).
    """

    def __init__(self, func: Callable[[_Quad], _Quad]) -> None:
        self._func = func

    def __call__(self, coords: _Quad, time: float) -> _Quad:
        return self._func(coords)

    def is_time_dependent(self) -> bool:
        return False

    def func(self) -> Callable[[_Quad], _Quad]:
        """Return the wrapped supplier."""
        return self._func

    def __repr__(self) -> str:
        return f"TimeIndependent({self._func!r})"


class TimeDependent:
    """Declare a ``f(coords, time)`` supplier as consulting time.

    Wrap when time is real: a supplier that genuinely varies in time
    must say so, because nothing can detect it reliably and a silently
    dropped time produces wrong numbers rather than an error.
    """

    def __init__(self, func: Callable[[_Quad, float], _Quad]) -> None:
        self._func = func

    def __call__(self, coords: _Quad, time: float) -> _Quad:
        return self._func(coords, time)

    def is_time_dependent(self) -> bool:
        return True

    def func(self) -> Callable[[_Quad, float], _Quad]:
        """Return the wrapped supplier."""
        return self._func

    def __repr__(self) -> str:
        return f"TimeDependent({self._func!r})"


def _accepts_a_time_argument(func: Callable[..., _Quad]) -> bool:
    """Whether ``func`` could be called with a second positional value.

    Used ONCE per construction to reject an ambiguous supplier --- never
    to infer what a supplier means. A signature that cannot be read
    (C builtins) is not treated as ambiguous: refusing something that
    cannot be checked would block legitimate callables for no gain.
    """
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    npositional = 0
    for parameter in signature.parameters.values():
        if parameter.kind is parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
        ):
            npositional += 1
    return npositional >= 2


def as_time_aware(func: Callable[..., _Quad]) -> TimeAwareCallableProtocol:
    """Normalize a supplier to the ``f(coords, time)`` call convention.

    A supplier is exactly one of two things, and the rule that keeps
    that total is: ANY callable that takes a time must say which it is.

    - ``f(coords)`` --- unambiguous, so it stays bare. This is permanent
      public API for time-independent suppliers.
    - ``TimeIndependent(f)`` / ``TimeDependent(f)`` --- declared, so
      ``is_time_dependent`` answers and normalization is idempotent.

    A bare callable that could also accept a time is neither, and is
    rejected here rather than guessed at. Guessing is what makes this
    class of bug silent: assume time-independent and a transient source
    freezes at t=0; assume time-dependent and a supplier whose second
    parameter means something else gets a time in it. Both produce wrong
    numbers with no error, and ``f(coords, time=0.0)`` in particular
    swallows the arity mismatch that would otherwise raise.
    """
    if isinstance(func, TimeAwareCallableProtocol):
        return func
    if _accepts_a_time_argument(func):
        raise TypeError(
            f"{getattr(func, '__name__', func)!r} accepts a second "
            "positional argument, so whether it depends on time is "
            "ambiguous and pyapprox will not guess. Declare it:\n"
            "    TimeDependent(func)    - values change with time\n"
            "    TimeIndependent(func)  - the second argument is not a "
            "time\n"
            "A supplier that takes coordinates alone needs no wrapper."
        )
    return TimeIndependent(func)


# =====================================================================
# Diffusion functions
# =====================================================================


@runtime_checkable
class DiffusionFunctionProtocol(Protocol):
    """State-independent diffusivity law kappa(x, t)."""

    def values(self, coords: _Quad, time: float) -> _Quad:
        """Evaluate kappa at coordinates and time.

        Parameters
        ----------
        coords : ndarray
            Coordinates, shape (ndim, npts) or (ndim, nelems, nquad).
        time : float
            Evaluation time. Time-independent laws ignore it; steady
            solves pass a fixed value that nothing consults.

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

    def is_time_dependent(self) -> bool:
        """Whether kappa varies with time.

        Drives cache-key policy: consumers key assembled operators on
        ``(version, time)`` when this is True and on ``version`` alone
        when it is False.
        """
        ...

    def version(self) -> int:
        """Monotone counter, incremented on every mutation.

        Consumers cache assembled operators keyed on this value;
        immutable functions return a constant. Time is NOT a mutation —
        it is an argument, and time-dependence is expressed through
        ``is_time_dependent`` instead.
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

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        return np.full(coords_np.shape[1:], self._value)

    def is_constant(self) -> bool:
        return True

    def is_time_dependent(self) -> bool:
        return False

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"ConstantDiffusion({self._value})"


class CoordinateDiffusion:
    """Diffusivity from a coordinate function kappa(x) or kappa(x, t).

    Parameters
    ----------
    func : Callable
        Accepts coordinates of shape (ndim, npts) and returns (npts,).
        A bare callable is taken to be time-independent; wrap it in
        ``TimeDependent`` for a law that varies in time.
    """

    def __init__(self, func: Callable[..., _Quad]) -> None:
        self._func = as_time_aware(func)

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._func(flat, time))
        return values.reshape(coords_np.shape[1:])

    def is_constant(self) -> bool:
        return False

    def is_time_dependent(self) -> bool:
        return self._func.is_time_dependent()

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"CoordinateDiffusion({self._func!r})"


@runtime_checkable
class BasisEvaluableFieldProtocol(Protocol):
    """A nodal field that can evaluate itself on its own basis.

    Field-agnostic: diffusivity, forcing, reaction, and velocity fields
    all satisfy it, because the capability concerns HOW a field is
    evaluated rather than WHICH coefficient it supplies.

    ``values(coords)`` accepts arbitrary points, so it must locate the
    element containing each one. During assembly that work is wasted:
    the points are the basis's own quadrature points, whose elements the
    basis already knows. ``values_on_basis`` is the assembly fast path,
    returning bit-identical values without the search.

    Consumers select it with an ``isinstance`` check at CONSTRUCTION and
    pass the bound method onward, so no per-call branching or capability
    sniffing happens during assembly.

    The fast path takes the assembly ``time`` for the same reason
    ``values`` does. A field whose DOFs vary in time would otherwise be
    re-assembled every step from values frozen at its first evaluation
    -- and because the cache would correctly invalidate each step, the
    result looks right while being wrong. Fields with fixed DOFs accept
    the argument and ignore it.
    """

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Evaluate at ``skfem_basis``'s quadrature points.

        Parameters
        ----------
        skfem_basis : Basis
            The basis being assembled on. Must be the basis this field's
            DOFs live on; a mismatch raises.
        time : float
            Assembly time. Ignored by fields with fixed DOFs; consulted
            by fields whose DOFs are a function of time.

        Returns
        -------
        ndarray
            Values with the basis's quadrature trailing shape
            ``(nelems, nquad)``, or ``(ncomponents, nelems, nquad)`` for
            a vector field.
        """
        ...


class _BasisEvaluatorProtocol(Protocol):
    """The basis members the scalar nodal fields consume."""

    def evaluate(self, coeffs: Any, points: Any) -> Any: ...

    def ndofs(self) -> int: ...

    def skfem_basis(self) -> "Basis": ...


def _interpolate_on_basis(
    skfem_basis: "Basis", dofs: _Quad, ndofs: int, owner: str
) -> _Quad:
    """Interpolate DOFs at a basis's quadrature points.

    Shared by the nodal fields. The DOF-count check is what makes the
    fast path safe: interpolating against a basis the DOFs do not belong
    to would silently return values at the wrong locations.
    """
    if int(skfem_basis.N) != int(ndofs):
        raise ValueError(
            f"{owner} holds {ndofs} DOFs but the supplied basis has "
            f"{int(skfem_basis.N)}; values_on_basis requires the basis "
            "the field's DOFs live on"
        )
    return np.asarray(skfem_basis.interpolate(np.asarray(dofs)))


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

    def is_time_dependent(self) -> bool:
        """Fixed values; time is accepted and ignored."""
        return False

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Values at the basis's quadrature points (assembly fast path)."""
        return _interpolate_on_basis(
            skfem_basis, self._dofs, self.ndofs(), "NodalFieldDiffusion"
        )

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

    def is_time_dependent(self) -> bool:
        """Fixed values; time is accepted and ignored."""
        return False

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def __call__(self, coords: _Quad, time: float = 0.0) -> _Quad:
        """Evaluate the nodal interpolant at coordinates (ndim, npts).

        Accepts a time so the field satisfies the time-aware supplier
        contract it declares through ``is_time_dependent``; the DOFs are
        fixed, so the value is ignored. The default keeps the bare
        ``f(coords)`` call working for existing consumers.
        """
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Values at the basis's quadrature points (assembly fast path)."""
        return _interpolate_on_basis(
            skfem_basis, self._dofs, self.ndofs(), "NodalFieldForcing"
        )

    def __repr__(self) -> str:
        return f"NodalFieldForcing(ndofs={self.ndofs()})"


# =====================================================================
# Velocity functions
# =====================================================================


@runtime_checkable
class VelocityFunctionProtocol(Protocol):
    """State-independent advection velocity law v(x, t).

    A velocity computed by an upstream transient flow solve is the
    motivating time-dependent case: the transport problem evaluates it
    with the assembly's time and never learns where it came from.
    """

    def values(self, coords: _Quad, time: float) -> _Quad:
        """Evaluate the velocity at coordinates and time.

        Parameters
        ----------
        coords : ndarray
            Coordinates, shape (ndim, npts) or (ndim, nelems, nquad).
        time : float
            Evaluation time. Time-independent laws ignore it.

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

    def is_time_dependent(self) -> bool:
        """Whether the velocity varies with time."""
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

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        shape = (len(self._vec),) + (1,) * (coords_np.ndim - 1)
        return np.broadcast_to(
            self._vec.reshape(shape), coords_np.shape
        ).copy()

    def is_constant(self) -> bool:
        return True

    def is_time_dependent(self) -> bool:
        """Fixed values; time is accepted and ignored."""
        return False

    def version(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"ConstantVelocity({self._vec})"


class CoordinateVelocity:
    """Velocity from a coordinate function v(x) or v(x, t).

    Parameters
    ----------
    func : Callable
        Accepts coordinates of shape (ndim, npts) and returns
        (ndim, npts). A bare callable is taken to be time-independent;
        wrap a velocity from a transient flow solve in ``TimeDependent``
        so it is evaluated at the assembly's time rather than frozen.
    """

    def __init__(self, func: Callable[..., _Quad]) -> None:
        self._func = as_time_aware(func)

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._func(flat, time))
        return values.reshape(coords_np.shape)

    def is_constant(self) -> bool:
        return False

    def is_time_dependent(self) -> bool:
        return self._func.is_time_dependent()

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

    def is_time_dependent(self) -> bool:
        """Fixed values; time is accepted and ignored."""
        return False

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

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape)

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Values at the basis's quadrature points (assembly fast path).

        A vector basis interpolates to ``(ncomponents, nelems, nquad)``,
        which is the shape ``values`` returns for the same points.
        """
        return _interpolate_on_basis(
            skfem_basis, self._dofs, self.ndofs(), "NodalFieldVelocity"
        )

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
class SpatiallyVaryingReactionProtocol(ReactionFunctionProtocol, Protocol):
    """A linear reaction whose coefficient varies over the domain.

    ``value``/``derivative`` answer the pointwise law R(x, u); this adds
    the coefficient r(x, t) on its own, which is what the LINEAR path
    assembles into the stiffness. A constant reaction has no such
    accessor (its coefficient is a scalar), and a nonlinear one never
    reaches the stiffness at all.

    Declared as a capability so the assembly selects on what a
    coefficient CAN do rather than on which class it is: gating on a
    concrete class silently dropped any other spatially varying linear
    reaction from the stiffness --- no error, just a missing term.
    """

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        """Evaluate r(x, t) at coordinates. Shape follows ``coords``."""
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

    def is_time_dependent(self) -> bool:
        """Fixed coefficient; time is accepted and ignored."""
        return False

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

    def is_time_dependent(self) -> bool:
        """Fixed values; time is accepted and ignored."""
        return False

    def version(self) -> int:
        """Monotone counter; incremented by every set_dofs call."""
        return self._version

    def dofs(self) -> _Quad:
        """Return the field DOFs. Shape: (ndofs,)."""
        return self._dofs

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        """Evaluate the coefficient interpolant at coordinates."""
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(self._basis.evaluate(self._dofs, flat))
        return values.reshape(coords_np.shape[1:])

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Values at the basis's quadrature points (assembly fast path)."""
        return _interpolate_on_basis(
            skfem_basis, self._dofs, self.ndofs(),
            "NodalFieldLinearReaction",
        )

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


class TimeModulatedNodalFieldLinearReaction:
    """Linear reaction whose nodal DOFs vary in time.

    The field is separable,

    .. math::

        r(x, t) = \\sum_k c_k\\, b_k(t)\\, s_k(x),

    with :math:`s_k` the columns of ``spatial_modes`` and :math:`b_k`
    the profiles of a modulation. Interpolating those DOFs gives the
    coefficient at any point and time.

    Knows nothing about parameterizations. It holds NUMBERS --- a mode
    matrix, a modulation, and coefficients --- and answers "what are my
    values here, at this time". It does not know what produced the
    modes, what the coefficients mean, or how to differentiate itself:
    computing a derivative here would put the parameterization's job in
    the constitutive layer, and the caller that owns the mapping is the
    one that can keep the forward field and the gradient describing the
    same control.

    Parameters
    ----------
    basis : _BasisEvaluatorProtocol
        Basis the DOFs live on.
    spatial_modes : ndarray
        One column per coefficient. Shape: (ndofs, nmodes).
    modulation : TimeVaryingProtocol
        Supplies ``values(time)`` of shape ``(nmodes,)``.
    coefficients : ndarray, optional
        Initial coefficients. Shape: (nmodes,). Defaults to zeros.
    """

    def __init__(
        self,
        basis: _BasisEvaluatorProtocol,
        spatial_modes: _Quad,
        modulation: Any,
        coefficients: Optional[_Quad] = None,
    ) -> None:
        modes = np.asarray(spatial_modes, dtype=np.float64)
        if modes.ndim != 2 or modes.shape[0] != basis.ndofs():
            raise ValueError(
                f"spatial_modes must have shape ({basis.ndofs()}, "
                f"nmodes), got {modes.shape}"
            )
        nmodes = int(modulation.nmodes())
        if modes.shape[1] != nmodes:
            raise ValueError(
                f"spatial_modes has {modes.shape[1]} columns but the "
                f"modulation supplies {nmodes} profiles; each mode "
                "needs exactly one profile"
            )
        self._basis = basis
        self._modes = modes
        self._modulation = modulation
        if coefficients is None:
            coefficients = np.zeros(nmodes)
        self.set_coefficients(coefficients)

    def set_coefficients(self, coefficients: _Quad) -> None:
        """Set the mode coefficients. Shape: (nmodes,)."""
        values = np.asarray(coefficients, dtype=np.float64)
        if values.shape != (self._modes.shape[1],):
            raise ValueError(
                f"coefficients must have shape "
                f"({self._modes.shape[1]},), got {values.shape}"
            )
        self._coefficients = values
        self._version = getattr(self, "_version", 0) + 1

    def is_time_dependent(self) -> bool:
        """The DOFs move with time, so assembled operators built from
        this field must not be reused across times."""
        return True

    def version(self) -> int:
        """Monotone counter; incremented by every coefficient update."""
        return self._version

    def nmodes(self) -> int:
        """Return the number of modes."""
        return int(self._modes.shape[1])

    def ndofs(self) -> int:
        """Return the number of field DOFs."""
        return int(self._basis.ndofs())

    def dofs_at(self, time: float) -> _Quad:
        """Return the DOFs realized at ``time``. Shape: (ndofs,)."""
        profiles = np.asarray(self._modulation.values(time))
        realized: _Quad = self._modes @ (self._coefficients * profiles)
        return realized

    def values(self, coords: _Quad, time: float = 0.0) -> _Quad:
        """Evaluate the coefficient interpolant at coordinates."""
        coords_np = np.asarray(coords)
        flat = coords_np.reshape(coords_np.shape[0], -1)
        values = np.asarray(
            self._basis.evaluate(self.dofs_at(time), flat)
        )
        return values.reshape(coords_np.shape[1:])

    def values_on_basis(
        self, skfem_basis: "Basis", time: float = 0.0
    ) -> _Quad:
        """Values at the basis's quadrature points (assembly fast path)."""
        return _interpolate_on_basis(
            skfem_basis,
            self.dofs_at(time),
            self.ndofs(),
            "TimeModulatedNodalFieldLinearReaction",
        )

    # ``value``/``derivative``/``second_derivative`` are the NONLINEAR
    # reaction surface, which a linear reaction never reaches: the
    # physics routes ``is_linear()`` coefficients into the stiffness
    # (where time IS threaded, through ``values_on_basis``) and returns
    # None from the nonlinear forms. They are declared for protocol
    # conformance only, and take a time so that a future caller cannot
    # get a silently frozen field from them.

    def value(self, coords: _Quad, state: _Quad, time: float = 0.0) -> _Quad:
        return self.values(coords, time) * np.asarray(state)

    def derivative(
        self, coords: _Quad, state: _Quad, time: float = 0.0
    ) -> _Quad:
        return np.broadcast_to(
            self.values(coords, time), np.asarray(state).shape
        ).copy()

    def second_derivative(
        self, coords: _Quad, state: _Quad, time: float = 0.0
    ) -> _Quad:
        return np.zeros_like(np.asarray(state))

    def is_linear(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            "TimeModulatedNodalFieldLinearReaction("
            f"ndofs={self.ndofs()}, nmodes={self.nmodes()})"
        )


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

    def is_time_dependent(self) -> bool:
        """Reaction laws are functions of state, not time."""
        return False

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
