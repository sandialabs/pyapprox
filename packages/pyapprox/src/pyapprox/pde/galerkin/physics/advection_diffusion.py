"""Advection-diffusion-reaction physics for Galerkin FEM.

Supports two advection forms:

Non-conservative (default):
    du/dt + v . grad(u) = div(D * grad(u)) + R(u) + f
  Weak form:
    (w, v.grad(u)) + (grad(w), D*grad(u)) - (w, R(u)) = (w, f)

Conservative:
    du/dt + div(v * u) = div(D * grad(u)) + R(u) + f
  Weak form (after integration by parts of div(v*u)):
    -(v*u, grad(w)) + (grad(w), D*grad(u)) - (w, R(u)) = (w, f)

where:
    D = diffusivity (scalar or function of x)
    v = velocity field
    R(u) = reaction term (general nonlinear function of u)
           Positive R(u) = source/production (standard physics convention)
    f = forcing/source term
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from skfem import Basis
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.constitutive.coefficient_functions import (
    BasisEvaluableFieldProtocol,
    ConstantDiffusion,
    ConstantVelocity,
    CoordinateDiffusion,
    CoordinateVelocity,
    DiffusionFunctionProtocol,
    LinearReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
    NodalFieldLinearReaction,
    NodalFieldVelocity,
    ReactionFunctionProtocol,
    ReactionFunctionWithSecondDerivativeProtocol,
    StateDependentDiffusionProtocol,
    TimeVaryingProtocol,
    VelocityFunctionProtocol,
    as_time_aware,
)
from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.util.backends.protocols import Array, Backend

# Import skfem for assembly
try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.helpers import dot, grad
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


# Type alias for reaction functions
# R(x, u) -> values at quadrature points
ReactionFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
# R'(x, u) -> derivative w.r.t. u at quadrature points
ReactionDerivFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _FieldOnBasisEvaluator:
    """Evaluates a nodal field on a fixed basis, ignoring coordinates.

    A nodal field's ``values(coords)`` must locate the element holding
    each coordinate, which during assembly repeats work the basis has
    already done. A field satisfying ``BasisEvaluableFieldProtocol``
    can evaluate itself directly on the basis instead, at identical
    values (to round-off) and a fraction of the cost.

    Substitutable for the coordinate callables the kernels expect: the
    coordinates handed in are the basis's own quadrature points, so
    discarding them loses nothing. Module-level class, not a closure,
    so the forms stay picklable.
    """

    def __init__(
        self,
        field: BasisEvaluableFieldProtocol,
        skfem_basis: "Basis",
    ) -> None:
        self._field = field
        self._skfem_basis = skfem_basis

    def __call__(
        self, coords: NDArray[np.floating[Any]], time: float = 0.0
    ) -> NDArray[np.floating[Any]]:
        # The coordinates are discarded -- they are the basis's own
        # quadrature points, which is the whole point of the fast path.
        # The TIME is forwarded: a field whose DOFs vary in time would
        # otherwise be re-assembled every step from frozen values, and
        # because the cache correctly invalidates each step the result
        # would look right while being wrong.
        values: NDArray[np.floating[Any]] = self._field.values_on_basis(
            self._skfem_basis, time
        )
        return values

    def is_time_dependent(self) -> bool:
        """Defers to the wrapped field.

        The evaluator adds no time dependence of its own and removes
        none: it forwards the assembly time to ``values_on_basis``. A
        field that varies in time keeps saying so through this wrapper,
        which is what keeps the assembly caches keyed correctly.
        """
        field = self._field
        return isinstance(field, TimeVaryingProtocol) and (
            field.is_time_dependent()
        )


class _TimedEvaluatorProtocol(Protocol):
    """The call convention the assembly kernels need: ``f(coords, time)``.

    Weaker than ``TimeAwareCallableProtocol``, which additionally
    requires ``is_time_dependent``. Both a coefficient's bound
    ``values`` method and a normalized supplier satisfy this, and the
    kernels only ever call — they never ask a supplier to describe
    itself, so requiring the declaration here would reject the bound
    methods for a capability nothing at this layer consults.
    """

    def __call__(
        self, coords: NDArray[np.floating[Any]], time: float
    ) -> NDArray[np.floating[Any]]: ...


def _coefficient_evaluator(
    field: object,
    skfem_basis: "Basis",
    coordinate_evaluator: _TimedEvaluatorProtocol,
) -> _TimedEvaluatorProtocol:
    """Return the fast evaluator when the field supports it.

    Called once per assembly, not per quadrature point, so the
    capability check never runs in the hot path.

    Both the fast evaluator and the fallback take ``(coords, time)``, so
    the kernels see one call convention whichever is selected; the basis
    fast path ignores its coordinate argument.

    Time-varying fields are eligible too: ``values_on_basis`` takes the
    assembly time and ``_FieldOnBasisEvaluator`` forwards it. Excluding
    them would be safe but expensive, and expensive on exactly the path
    that pays it -- a fixed field assembles its load once, while a
    time-varying one re-assembles every step.
    """
    if isinstance(field, BasisEvaluableFieldProtocol):
        return _FieldOnBasisEvaluator(field, skfem_basis)
    return coordinate_evaluator


class _DiffusionReactionKernel:
    """Picklable kernel for the diffusion + linear-reaction form.

    Module-level callable class (not a closure) so the forms returned
    by ``stiffness_forms`` remain picklable — the _ExpTransform
    precedent in pde.field_maps.
    """

    __name__ = "diffusion_reaction"

    def __init__(
        self,
        diff_const: Optional[float],
        diff_callable: Optional[_TimedEvaluatorProtocol],
        react_coeff: Optional[float],
        react_callable: Optional[_TimedEvaluatorProtocol] = None,
        time: float = 0.0,
    ) -> None:
        self._diff_const = diff_const
        self._diff_callable = diff_callable
        self._react_coeff = react_coeff
        self._react_callable = react_callable
        # Bound at construction: kernels are built per assembly, so the
        # time is immutable for the life of this one. Crank-Nicolson's
        # two assemblies each carry their own.
        self._time = time

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        # Diffusion coefficient
        diff: Union[float, NDArray[np.floating[Any]]]
        if self._diff_const is not None:
            diff = self._diff_const
        else:
            assert self._diff_callable is not None
            diff = self._diff_callable(np.asarray(w.x), self._time)

        # Diffusion term: (grad(w), D*grad(u)) contributes D*grad(u).grad(v)
        result: NDArray[np.floating[Any]] = diff * dot(grad(u), grad(v))

        # Linear reaction term: -(w, r*u) contributes -r*u*v
        # (negative because it's moved to LHS of weak form)
        if self._react_coeff is not None:
            result = result - self._react_coeff * u * v
        elif self._react_callable is not None:
            result = result - (
                self._react_callable(np.asarray(w.x), self._time) * u * v
            )

        return result


class _DiffusivitySensitivityKernel:
    """Mixed bilinear kernel for dF/d(diffusivity DOFs).

    Trial function is the diffusivity basis function, test function the
    state basis function; the state enters interpolated:
    ``-grad(u_prev) . grad(v) * k``.
    """

    __name__ = "diffusivity_sensitivity"

    def __call__(
        self,
        k: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        return np.asarray(-dot(w["u_prev"].grad, grad(v)) * k)


class _VelocitySensitivityKernel:
    """Mixed bilinear kernel for dF/d(velocity DOFs).

    Trial function is the VECTOR velocity basis function, test
    function the scalar state basis function; the state gradient
    enters interpolated: ``-dot(a, grad(u_prev)) * v``
    (non-conservative advection enters the residual as
    ``-(v, vel . grad u)``).
    """

    __name__ = "velocity_sensitivity"

    def __call__(
        self,
        a: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        grad_u = w["u_prev"].grad
        ndim = grad_u.shape[0]
        return np.asarray(
            -sum(a[dim] * grad_u[dim] for dim in range(ndim)) * v
        )


class _ReactionSensitivityKernel:
    """Mixed bilinear kernel for dF/d(reaction DOFs).

    Trial function is the reaction-coefficient basis function, test
    function the state basis function; the state enters interpolated:
    ``+u_prev * r * v`` (the linear reaction enters the residual as
    ``+(v, r*u)``).
    """

    __name__ = "reaction_sensitivity"

    def __call__(
        self,
        r: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        return np.asarray(w["u_prev"] * r * v)


class _ReactionMixedStateKernel:
    """Bilinear kernel for the reaction mixed state Jacobian.

    ``A(delta_r) = d/du [dF/d(reaction) delta_r]``: the reaction mass
    matrix with the GIVEN coefficient field,
    ``+delta * u * v``.
    """

    __name__ = "reaction_mixed_state"

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        return np.asarray(w["delta_prev"] * u * v)


class _DiffusivityMixedStateKernel:
    """Bilinear kernel for the diffusivity mixed state Jacobian.

    ``A(delta_g) = d/du [dF/d(diffusivity) delta_g]``: the
    diffusion-only stiffness assembled with the GIVEN coefficient
    field, ``-delta * dot(grad(u), grad(v))``.
    """

    __name__ = "diffusivity_mixed_state"

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        return np.asarray(-w["delta_prev"] * dot(grad(u), grad(v)))


class _ReactionHVPKernel:
    """Linear kernel for the reaction state-state HVP contraction.

    Assembles ``R''(u) * adj * wdir * v`` with all three fields
    interpolated at the quadrature points (positive sign: the reaction
    enters the residual as ``+(w, R(u))``).
    """

    __name__ = "reaction_hvp"

    def __init__(self, reaction_deriv2: ReactionDerivFunc) -> None:
        self._reaction_deriv2 = reaction_deriv2

    def __call__(
        self, v: "DiscreteField", w: "FormExtraParams"
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        deriv2 = self._reaction_deriv2(x_np, np.asarray(w["u_prev"]))
        return np.asarray(
            deriv2 * np.asarray(w["adj_prev"]) * np.asarray(w["dir_prev"]) * v
        )


class _AdvectionKernel:
    """Picklable kernel for the advection form."""

    __name__ = "advection"

    def __init__(
        self,
        vel_np: Optional[NDArray[np.floating[Any]]],
        vel_callable: Optional[_TimedEvaluatorProtocol],
        conservative: bool,
        time: float = 0.0,
    ) -> None:
        self._vel_np = vel_np
        self._vel_callable = vel_callable
        self._conservative = conservative
        # Bound at construction; see _DiffusionReactionKernel.
        self._time = time

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        if self._vel_np is not None:
            vel = self._vel_np
        else:
            assert self._vel_callable is not None
            vel = self._vel_callable(np.asarray(w.x), self._time)
        if self._conservative:
            # Conservative: -(v*u, grad(w)) from div(v*u)
            ret: NDArray[np.floating[Any]] = -u * dot(vel, grad(v))
            return ret
        else:
            # Non-conservative: (w, v.grad(u))
            ret2: NDArray[np.floating[Any]] = dot(vel, grad(u)) * v
            return ret2


class _ForcingKernel:
    """Picklable kernel for the forcing load form (w, f).

    Takes a supplier already normalized to ``f(coords, time)``, so the
    time it was built with is bound for the whole assembly and the call
    is unconditional. A TypeError raised inside a forcing now
    propagates instead of being mistaken for a wrong-arity call and
    silently retried without the time.
    """

    __name__ = "forcing"

    def __init__(
        self,
        forcing_func: _TimedEvaluatorProtocol,
        time: float,
    ) -> None:
        self._forcing_func = forcing_func
        self._time = time

    def __call__(
        self, v: "DiscreteField", w: "FormExtraParams"
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            forc_flat = self._forcing_func(x_flat, self._time)
            forc = forc_flat.reshape(nelem, nquad)
        else:
            forc = self._forcing_func(x_np, self._time)
        ret: NDArray[np.floating[Any]] = forc * v
        return ret


class _ReactionKernel:
    """Picklable kernel for the nonlinear reaction load (w, R(u)).

    Assembled with the interpolated state as the ``u_prev`` form
    parameter.
    """

    __name__ = "reaction"

    def __init__(self, reaction_func: ReactionFunc) -> None:
        self._reaction_func = reaction_func

    def __call__(
        self, v: "DiscreteField", w: "FormExtraParams"
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        u_prev = w.u_prev  # Interpolated state values

        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            u_flat = np.asarray(u_prev).reshape(-1)
            react_flat = self._reaction_func(x_flat, u_flat)
            react = react_flat.reshape(nelem, nquad)
        else:
            react = self._reaction_func(x_np, u_prev)
        ret: NDArray[np.floating[Any]] = react * v
        return ret


class _ReactionJacobianKernel:
    """Picklable kernel for the reaction Jacobian (w, R'(u)*du).

    Assembled with the interpolated state as the ``u_prev`` form
    parameter.
    """

    __name__ = "reaction_jacobian"

    def __init__(self, reaction_deriv: ReactionDerivFunc) -> None:
        self._reaction_deriv = reaction_deriv

    def __call__(
        self,
        u: "DiscreteField",
        v: "DiscreteField",
        w: "FormExtraParams",
    ) -> np.ndarray:
        x_np = np.asarray(w.x)
        u_prev = w.u_prev

        x_shape = x_np.shape
        if len(x_shape) == 3:
            ndim, nelem, nquad = x_shape
            x_flat = x_np.reshape(ndim, -1)
            u_flat = np.asarray(u_prev).reshape(-1)
            react_deriv_flat = self._reaction_deriv(x_flat, u_flat)
            react_deriv = react_deriv_flat.reshape(nelem, nquad)
        else:
            react_deriv = self._reaction_deriv(x_np, u_prev)

        # Derivative of the load term (w, R(u)) with respect to the
        # state: F = load - K*u, so dF/du gains +(w, R'(u)*du)
        ret: NDArray[np.floating[Any]] = react_deriv * u * v
        return ret


class AdvectionDiffusionReaction(GalerkinPhysicsBase[Array]):
    """Advection-diffusion-reaction physics with general reaction term.

    Solves:
        du/dt + v . grad(u) = div(D * grad(u)) + R(u) + f

    where R(u) is a general (possibly nonlinear) reaction term.
    Positive R(u) represents a source/production term.

    TIME THREADING. Diffusivity, velocity, and forcing are evaluated at
    the assembly's time, so any of them may be declared
    ``TimeDependent``. The stiffness cache keys on time whenever a
    contributing coefficient declares time-dependence, and on
    coefficient versions alone otherwise --- so a steady problem still
    assembles once and reuses it across every time step.

    The REACTION is the exception, by construction rather than
    omission: it is R(x, u), a function of state supplying
    ``value``/``derivative``/``is_linear``, so a coefficient-style
    ``TimeDependent`` supplier is not a valid reaction and is rejected.
    A time-varying reaction would need that protocol widened first.

    The SENSITIVITY surface is not time-threaded:
    ``residual_<field>_jacobian`` and the mixed slots differentiate with
    respect to nodal-field DOFs, and a nodal field is time-independent
    by construction. Differentiating through a time-varying coefficient
    (a separable control ``f(x, t) = sum_k p_k b_k(t) s_k(x)``, say)
    needs time threaded through those assemblies first.

    The weak form is:
        (w, du/dt) + (w, v.grad(u)) + (grad(w), D*grad(u)) = (w, R(u)) + (w, f)

    For the residual F in M * du/dt = F:
        F_i = integral(f * phi_i) + integral(R(u) * phi_i)
              - integral(D * grad(u) . grad(phi_i))
              - integral(v . grad(u) * phi_i)

    In steady state (F=0), the linear system K*u = b is solved where the
    reaction term contributes to both the stiffness matrix (for linear R)
    or is evaluated at each Newton iteration (for nonlinear R).

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    diffusivity : float or Callable
        Diffusion coefficient D. If callable, takes coordinates (ndim, npts)
        and returns values (npts,).
    bkd : Backend
        Computational backend.
    velocity : Array or Callable, optional
        Velocity field v. If array, shape (ndim,). If callable, takes
        coordinates and returns (ndim, npts).
    reaction : float, Callable, or Tuple[Callable, Callable], optional
        Reaction term R(u). Can be:
        - float: Linear reaction R(u) = coeff * u (positive = source)
        - Callable: R(x, u) returning reaction values
        - Tuple[R, R']: (reaction function, derivative function)
          where R(x, u) and R'(x, u) are callables
        Default is None (no reaction).
    forcing : Callable, optional
        Forcing/source term f. Takes coordinates and returns (npts,).
        For time-dependent problems, takes (coordinates, time).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.
    conservative : bool, default=False
        If True, use conservative advection form div(v*u) with weak form
        bilinear term -(v*u, grad(w)). If False, use non-conservative
        form v.grad(u) with weak form bilinear term (w, v.grad(u)).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>>
    >>> # Pure diffusion
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, bkd=bkd
    ... )
    >>>
    >>> # Linear reaction R(u) = 2*u (source term)
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01, reaction=2.0, bkd=bkd
    ... )
    >>>
    >>> # Nonlinear reaction R(u) = u^2 (typed reaction function object)
    >>> from pyapprox.pde.constitutive.coefficient_functions import (
    ...     CallableReaction,
    ... )
    >>> def R(x, u): return u**2
    >>> def R_prime(x, u): return 2*u
    >>> physics = AdvectionDiffusionReaction(
    ...     basis=basis, diffusivity=0.01,
    ...     reaction=CallableReaction(R, R_prime), bkd=bkd
    ... )
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        diffusivity: Union[float, Callable[..., Any], DiffusionFunctionProtocol],
        bkd: Backend[Array],
        velocity: Optional[
            Union[Array, Callable[..., Any], VelocityFunctionProtocol]
        ] = None,
        reaction: Optional[Union[float, ReactionFunctionProtocol]] = None,
        forcing: Optional[Callable[..., Any]] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
        conservative: bool = False,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._mass = ScalarMassAssembler(basis, bkd)

        # Coerce coefficient inputs to function objects at the boundary;
        # kernels and assemblies see only the function protocols.
        self._diffusion_function = self._coerce_diffusion(diffusivity)
        self._velocity_function = self._coerce_velocity(velocity)
        self._reaction_function = self._coerce_reaction(reaction)
        # The raw supplier stays put: forcing_function() hands it to the
        # parameterization facade, and the load cache keys on its being a
        # NodalFieldForcing. Assembly evaluates the normalized companion.
        self._forcing = forcing
        self._forcing_eval = (
            None if forcing is None else as_time_aware(forcing)
        )
        # Whether the assembled forcing load may be cached across times.
        # Decided ONCE here rather than per assembly, so no capability
        # sniffing happens in the hot path. Two conditions, and the
        # second is not implied by the first: the supplier must carry a
        # version() to key the cache on, AND it must declare that its
        # values do not depend on time. A field that satisfies the
        # nodal-DOF interface but whose DOFs vary in time would
        # otherwise have the first step's load reused for the whole
        # trajectory -- silently, since nothing raises.
        self._forcing_load_cacheable = isinstance(
            forcing, NodalFieldForcing
        ) and not (
            isinstance(forcing, TimeVaryingProtocol)
            and forcing.is_time_dependent()
        )
        self._conservative = conservative

        # Version-keyed assembly caches: coefficient objects carry a
        # version() bumped on mutation (set_dofs), and each cached
        # assembly product stores the versions it was built from —
        # rebinds invalidate, Newton iterations and time steps reuse.
        # The same pattern serves the stiffness (below) and the
        # time-invariant forcing load (_assemble_forcing_load).
        self._stiffness_cached: Optional[Array] = None
        # Three coefficient versions, plus the assembly time when any of
        # them depends on it: a time-dependent stiffness must not be
        # served from a key that cannot distinguish two times.
        self._stiffness_versions: Optional[
            Union[Tuple[int, int, int], Tuple[int, int, int, float]]
        ] = None
        self._load_cached: Optional[Array] = None
        self._forcing_load_cached: Optional[np.ndarray] = None
        self._forcing_load_version: Optional[int] = None

    @staticmethod
    def _coerce_diffusion(
        diffusivity: Union[float, Callable[..., Any], DiffusionFunctionProtocol],
    ) -> DiffusionFunctionProtocol:
        """Coerce legacy float/callable diffusivity to a function object."""
        if isinstance(diffusivity, StateDependentDiffusionProtocol):
            raise NotImplementedError(
                "state-dependent diffusion kappa(x, u) requires Jacobian "
                "assemblies that do not exist yet; supply a "
                "state-independent DiffusionFunctionProtocol"
            )
        if isinstance(diffusivity, DiffusionFunctionProtocol):
            return diffusivity
        if callable(diffusivity):
            return CoordinateDiffusion(diffusivity)
        return ConstantDiffusion(float(diffusivity))

    def _coerce_velocity(
        self,
        velocity: Optional[
            Union[Array, Callable[..., Any], VelocityFunctionProtocol]
        ],
    ) -> Optional[VelocityFunctionProtocol]:
        """Coerce legacy array/callable velocity to a function object."""
        if velocity is None:
            return None
        if isinstance(velocity, VelocityFunctionProtocol):
            return velocity
        if callable(velocity):
            return CoordinateVelocity(velocity)
        return ConstantVelocity(self._bkd.to_numpy(velocity))

    @staticmethod
    def _coerce_reaction(
        reaction: Optional[Union[float, ReactionFunctionProtocol]],
    ) -> Optional[ReactionFunctionProtocol]:
        """Coerce a legacy float reaction to LinearReaction."""
        if reaction is None:
            return None
        if isinstance(reaction, ReactionFunctionProtocol):
            return reaction
        if isinstance(reaction, (int, float)):
            return LinearReaction(float(reaction))
        raise TypeError(
            "reaction must be a float or a ReactionFunctionProtocol (see "
            "pde.constitutive.coefficient_functions — the tuple form was "
            f"replaced by CallableReaction), got {type(reaction).__name__}"
        )

    def diffusion_function(self) -> DiffusionFunctionProtocol:
        """Return the diffusion model."""
        return self._diffusion_function

    def forcing_function(self) -> Optional[Callable[..., Any]]:
        """Return the forcing (callable or ``NodalFieldForcing``)."""
        return self._forcing

    def velocity_function(self) -> Optional[VelocityFunctionProtocol]:
        """Return the velocity model, or None."""
        return self._velocity_function

    def reaction_function(self) -> Optional[ReactionFunctionProtocol]:
        """Return the reaction model, or None."""
        return self._reaction_function

    def is_linear(self) -> bool:
        """Return True if the problem is linear (linear or no reaction)."""
        return (
            self._reaction_function is None
            or self._reaction_function.is_linear()
        )

    def _diffusion_reaction_form(
        self, time: float = 0.0
    ) -> "BilinearForm":
        """Bilinear form for diffusion plus linear reaction.

        All diffusion functions are consumed uniformly through
        ``values`` (constant-coefficient problems assemble once and hit
        the stiffness cache, so no scalar fast path is warranted). A
        LINEAR reaction structurally belongs in this bilinear form
        (nonlinear reactions enter the load instead), which is what the
        ``is_linear`` capability selects.
        """
        reaction = self._reaction_function
        react_coeff = (
            reaction.coeff()
            if isinstance(reaction, LinearReaction)
            else None
        )
        skfem_basis = self._basis.skfem_basis()
        react_callable = (
            _coefficient_evaluator(reaction, skfem_basis, reaction.values)
            if isinstance(reaction, NodalFieldLinearReaction)
            else None
        )

        return BilinearForm(
            _DiffusionReactionKernel(
                None,
                _coefficient_evaluator(
                    self._diffusion_function,
                    skfem_basis,
                    self._diffusion_function.values,
                ),
                react_coeff,
                react_callable,
                time,
            )
        )

    def _advection_form(
        self, time: float = 0.0
    ) -> Optional["BilinearForm"]:
        """Bilinear form for advection, or None when velocity is absent."""
        velocity = self._velocity_function
        if velocity is None:
            return None
        # A NodalFieldVelocity carries its OWN (vector) basis, which is
        # generally not this physics' basis — an upstream flow solve
        # supplies it. Its quadrature points must therefore come from
        # that basis, so the fast path is only valid when the two
        # discretize the same elements. Comparing quadrature shapes is
        # what establishes that.
        evaluator: Callable[
            ..., NDArray[np.floating[Any]]
        ] = velocity.values
        if isinstance(velocity, NodalFieldVelocity):
            vel_skfem = velocity.basis().skfem_basis()
            own_shape = self._basis.skfem_basis().global_coordinates().shape
            if vel_skfem.global_coordinates().shape == own_shape:
                evaluator = _coefficient_evaluator(
                    velocity, vel_skfem, velocity.values
                )
        return BilinearForm(
            _AdvectionKernel(
                None, evaluator, self._conservative, time
            )
        )

    def forcing_form(self, time: float) -> Optional["LinearForm"]:
        """Linear form for the forcing contribution (w, f), or None.

        A nodal forcing takes the same basis fast path as the other
        coefficients: evaluating it by coordinate makes the element
        search the basis has already done, once per quadrature point.
        """
        if self._forcing_eval is None:
            return None
        # Both branches of the selector satisfy the time-aware call
        # convention: _FieldOnBasisEvaluator accepts (coords, time), and
        # the fallback is the already-normalized _forcing_eval.
        evaluator: _TimedEvaluatorProtocol = _coefficient_evaluator(
            self._forcing, self._basis.skfem_basis(), self._forcing_eval
        )
        return LinearForm(_ForcingKernel(evaluator, time))

    def reaction_form(self) -> Optional["LinearForm"]:
        """Linear form for the nonlinear reaction (w, R(u)), or None.

        Assemble with the interpolated state as a form parameter,
        ``asm(form, basis, u_prev=basis.interpolate(state))``; a raw
        (nelems, nquad) array of state values at the quadrature points
        also works, enabling element-restricted assembly.
        """
        reaction = self._reaction_function
        if reaction is None or reaction.is_linear():
            return None
        return LinearForm(_ReactionKernel(reaction.value))

    def reaction_jacobian_form(self) -> Optional["BilinearForm"]:
        """Bilinear form (w, R'(u)*du) for the reaction Jacobian, or None.

        Assemble with the interpolated state as a form parameter, as in
        :meth:`reaction_form`.
        """
        reaction = self._reaction_function
        if reaction is None or reaction.is_linear():
            return None
        return BilinearForm(_ReactionJacobianKernel(reaction.derivative))

    def stiffness_forms(
        self, time: float = 0.0
    ) -> List["BilinearForm"]:
        """Return the bilinear forms whose sum assembles the stiffness.

        Each form can be assembled on any compatible skfem basis — in
        particular a ``basis.with_elements(...)``-restricted basis — so
        consumers such as hyper-reduction can extract per-element
        contributions without changing the global assembly path.

        Parameters
        ----------
        time : float
            Bound into the returned forms, which evaluate their
            coefficients at it. A form is therefore a snapshot: build a
            new one to assemble at a different time. Steady problems can
            ignore this, which is why it defaults.

        Returns
        -------
        List[BilinearForm]
            Diffusion (+ linear reaction) form, followed by the
            advection form when a velocity is present.
        """
        forms = [self._diffusion_reaction_form(time)]
        advection = self._advection_form(time)
        if advection is not None:
            forms.append(advection)
        return forms

    def _stiffness_is_time_dependent(self) -> bool:
        """Whether any coefficient in the stiffness varies with time.

        Drives the cache key, not the assembly: a declared
        time-dependent coefficient forces re-assembly when the time
        moves, while a steady problem keeps the single-key fast path.
        """
        contributors = (
            self._diffusion_function,
            self._velocity_function,
            self._reaction_function,
        )
        return any(
            isinstance(contributor, TimeVaryingProtocol)
            and contributor.is_time_dependent()
            for contributor in contributors
        )

    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble stiffness matrix K.

        For the weak form, K includes:
        - Diffusion: (grad(w), D*grad(u))
        - Advection: (w, v.grad(u))
        - Linear reaction (if applicable): -(w, r*u) where R(u) = r*u

        Note: For nonlinear reaction, the contribution is handled in
        the residual and Jacobian separately.
        """
        # Check cache for linear problems
        # The stiffness (diffusion + advection + linear reaction) is
        # state-independent; the cache is keyed on the coefficient
        # function versions so field updates (set_dofs) invalidate it
        # while Newton iterations reuse it.
        #
        # Time joins the key only when a contributing coefficient
        # declares that it varies with time. Without that, a
        # time-dependent diffusivity or velocity would assemble once and
        # be served stale at every later time --- silently, and the
        # adjoint would inherit the stale Jacobian. Time-independent
        # coefficients keep the single-key fast path, so a steady
        # problem reuses one assembly across all time steps as before.
        versions: Union[
            Tuple[int, int, int], Tuple[int, int, int, float]
        ] = (
            self._diffusion_function.version(),
            self._velocity_function.version()
            if self._velocity_function is not None
            else 0,
            self._reaction_function.version()
            if isinstance(self._reaction_function, NodalFieldLinearReaction)
            else 0,
        )
        if self._stiffness_is_time_dependent():
            versions = versions[:3] + (time,)
        if (
            self._stiffness_cached is not None
            and self._stiffness_versions == versions
        ):
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()

        self._check_diffusivity_positive(skfem_basis, time)

        forms = self.stiffness_forms(time)
        stiffness = asm(forms[0], skfem_basis)
        # Add advection if present
        for form in forms[1:]:
            stiffness = stiffness + asm(form, skfem_basis)

        self._stiffness_cached = stiffness
        self._stiffness_versions = versions

        result: Array = stiffness
        return result

    def _check_diffusivity_positive(
        self, skfem_basis: "Basis", time: float
    ) -> None:
        """Raise if the diffusivity is not positive where it is used.

        A property of THIS operator, not of whatever produced the
        field: :math:`-\\nabla\\cdot(D\\nabla u)` is elliptic only while
        :math:`D > 0`. Where D dips negative the local operator changes
        character and the linear solve returns a plausible field with no
        error, so the check has to exist somewhere -- and the only place
        that sees every field, however it arrived, is the assembly that
        consumes it. A parameterization can enforce it for the fields it
        writes, but not for a physics built directly, a field mutated
        through ``set_dofs``, or a manufactured-solution setup.

        Evaluated at the quadrature points the assembly will integrate
        over, so it also covers interpolation between DOFs. Runs once
        per assembly, inside the cache miss: a cached stiffness was
        already checked when it was built.
        """
        values = _coefficient_evaluator(
            self._diffusion_function,
            skfem_basis,
            self._diffusion_function.values,
        )(np.asarray(skfem_basis.global_coordinates()), time)
        smallest = float(np.min(np.asarray(values)))
        if smallest <= 0.0:
            raise ValueError(
                "diffusivity must be positive everywhere it is "
                f"evaluated; found {smallest:.3e} at time {time:.6g}. "
                "A non-positive diffusivity makes the operator "
                "non-elliptic there, which yields a plausible-looking "
                "solution rather than a solver failure"
            )

    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble load vector b.

        For the weak form, b includes:
        - Forcing: (w, f)
        - Nonlinear reaction: (w, R(u)) evaluated at current state

        For linear reaction, the term is already in the stiffness matrix.
        """
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)

        # Start with forcing contribution
        if self._forcing is None and (
            self._reaction_function is None
            or self._reaction_function.is_linear()
        ):
            # No forcing and no nonlinear reaction - use cached zero vector
            if self._load_cached is not None:
                return self._load_cached
            load_np = np.zeros(self.nstates())
            self._load_cached = self._bkd.asarray(load_np.astype(np.float64))
            return self._load_cached

        load_np = np.zeros(self.nstates())

        # Forcing contribution: (w, f)
        forcing_load = self._assemble_forcing_load(time)
        if forcing_load is not None:
            load_np += forcing_load

        # Nonlinear reaction contribution: (w, R(u))
        reaction = self.reaction_form()
        if reaction is not None:
            # Interpolate state to get u values at quadrature points
            state_interp = skfem_basis.interpolate(state_np)
            load_np += asm(reaction, skfem_basis, u_prev=state_interp)

        return self._bkd.asarray(load_np.astype(np.float64))

    def _assemble_forcing_load(self, time: float) -> Optional[np.ndarray]:
        """Assemble the forcing contribution (w, f) to the load.

        A forcing whose values do not depend on time and that carries a
        ``version()`` has its assembled load cached on that version —
        rebinds (``set_dofs``) invalidate, while Newton iterations and
        time steps reuse it. Everything else is assembled fresh every
        call, which is what keeps a time-varying forcing correct: the
        cache key has no time component, so caching one would freeze it
        at its first evaluation. ``_forcing_load_cacheable`` records
        that decision, made once at construction.
        """
        forcing_form = self.forcing_form(time)
        if forcing_form is None:
            return None
        if not self._forcing_load_cacheable:
            return np.asarray(asm(forcing_form, self._basis.skfem_basis()))
        if not isinstance(self._forcing, NodalFieldForcing):
            raise TypeError(
                "forcing load marked cacheable but the forcing is "
                f"{type(self._forcing).__name__}, which carries no "
                "version() to key the cache on"
            )
        version = self._forcing.version()
        if (
            self._forcing_load_cached is not None
            and self._forcing_load_version == version
        ):
            return self._forcing_load_cached
        forcing_load = np.asarray(
            asm(forcing_form, self._basis.skfem_basis())
        )
        self._forcing_load_cached = forcing_load
        self._forcing_load_version = version
        return forcing_load

    def _assemble_reaction_jacobian(self, state: Array, time: float) -> Array:
        """Assemble Jacobian contribution from nonlinear reaction.

        For R(u), the Jacobian term is: (w, R'(u) * du)
        where du is the trial function.

        This returns the matrix J where J_ij = integral(R'(u) * phi_j * phi_i)
        """
        form = self.reaction_jacobian_form()
        if form is None:
            # No nonlinear reaction or linear reaction (already in stiffness)
            empty: Array = csr_matrix((self.nstates(), self.nstates()))
            return empty

        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)

        # Interpolate state
        state_interp = skfem_basis.interpolate(state_np)

        jacobian: Array = asm(form, skfem_basis, u_prev=state_interp)
        return jacobian

    def mass_matrix(self) -> Array:
        """Return the scalar mass matrix."""
        return self._mass.mass_matrix()

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return self._mass.mass_solve(rhs)

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F = b - K*u without Dirichlet enforcement.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        stiffness = self._assemble_stiffness(state, time)
        load = self._assemble_load(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        load = self._apply_bc_to_load(load, time)
        return load - stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF/du without Dirichlet enforcement.

        Includes nonlinear reaction Jacobian if applicable.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        jacobian = -stiffness
        if (
            self._reaction_function is not None
            and not self._reaction_function.is_linear()
        ):
            jacobian = jacobian + self._assemble_reaction_jacobian(state, time)
        return jacobian

    def residual_diffusivity_jacobian(self, state: Array) -> Array:
        """Compute dF/d(diffusivity DOFs) at the given state.

        Requires the diffusivity to be a ``NodalFieldDiffusion`` (the
        differentiable representation). The mixed assembly is exact:
        ``B(u)[j, k] = -int phi_k (grad u . grad phi_j) dx`` — the
        minus sign because diffusion sits inside K and F = b - K u.
        B is LINEAR in the state with zero constant part, and its state
        derivative tensor is symmetric in (state, residual) indices —
        the properties the parameterization's HVP contractions rely on.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)

        Returns
        -------
        Array
            Sensitivity matrix (scipy sparse).
            Shape: (nstates, nfield_dofs)
        """
        diffusion = self._diffusion_function
        if not isinstance(diffusion, NodalFieldDiffusion):
            raise TypeError(
                "residual_diffusivity_jacobian requires a "
                "NodalFieldDiffusion diffusivity (nodal DOFs are the "
                f"differentiable representation), got "
                f"{type(diffusion).__name__}"
            )
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        sensitivity = asm(
            BilinearForm(_DiffusivitySensitivityKernel()),
            skfem_basis,
            skfem_basis,
            u_prev=skfem_basis.interpolate(state_np),
        )
        result: Array = sensitivity
        return result

    def residual_forcing_jacobian(self) -> Array:
        r"""Compute :math:`dF/d(\text{forcing DOFs}) = +M`.

        The forcing enters the load as :math:`(v, f_h)` with
        :math:`f_h` the nodal interpolant, so the load is exactly
        :math:`M f` and the sensitivity is the (cached) scalar mass
        matrix — state-independent. Requires the forcing to be a
        ``NodalFieldForcing`` (the differentiable representation).

        Returns
        -------
        Array
            Sensitivity matrix (scipy sparse). Shape: (nstates, nstates)
        """
        if not isinstance(self._forcing, NodalFieldForcing):
            raise TypeError(
                "residual_forcing_jacobian requires a NodalFieldForcing "
                "forcing (nodal DOFs are the differentiable "
                f"representation), got {type(self._forcing).__name__}"
            )
        return self.mass_matrix()

    def residual_velocity_jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dF/d(\text{velocity DOFs})` at the given state.

        The mixed rectangular assembly is exact:
        :math:`S(u)[j, k] = -\int (\psi_k \cdot \nabla u) \, \phi_j`
        with :math:`\psi_k` the VECTOR velocity basis functions
        (columns follow the velocity basis DOF ordering) —
        non-conservative advection enters the residual as
        :math:`-(v, \text{vel} \cdot \nabla u)`. S is LINEAR in the
        state with zero constant part. Requires the velocity to be a
        ``NodalFieldVelocity``; the conservative form is a follow-up.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)

        Returns
        -------
        Array
            Sensitivity matrix (scipy sparse).
            Shape: (nstates, nvel_dofs)
        """
        velocity = self._velocity_function
        if not isinstance(velocity, NodalFieldVelocity):
            raise TypeError(
                "residual_velocity_jacobian requires a "
                "NodalFieldVelocity velocity (nodal DOFs are the "
                "differentiable representation), got "
                f"{type(velocity).__name__}"
            )
        if self._conservative:
            raise NotImplementedError(
                "velocity sensitivities are implemented for the "
                "non-conservative advection form only"
            )
        vel_skfem = velocity.basis().skfem_basis()
        scalar_skfem = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        sensitivity = asm(
            BilinearForm(_VelocitySensitivityKernel()),
            vel_skfem,
            scalar_skfem,
            u_prev=scalar_skfem.interpolate(state_np),
        )
        result: Array = sensitivity
        return result

    def residual_velocity_state_jacobian(
        self, delta_dofs: Array, state: Array
    ) -> Array:
        r"""Compute :math:`A(\delta a) = d/du \, [dF/d(a)\,\delta a]`.

        The advection operator assembled with the GIVEN velocity
        field: :math:`-\int (\delta a \cdot \nabla u) \, v`. The
        non-conservative advection term is linear in both the velocity
        and the state, so the result is state-independent; the
        ``state`` argument is kept for the typed field-derivative
        signature.

        Parameters
        ----------
        delta_dofs : Array
            Velocity-field direction (vector-basis DOFs).
            Shape: (nvel_dofs,)
        state : Array
            Solution state (unused here). Shape: (nstates,)

        Returns
        -------
        Array
            Mixed Jacobian (scipy sparse). Shape: (nstates, nstates)
        """
        velocity = self._velocity_function
        if not isinstance(velocity, NodalFieldVelocity):
            raise TypeError(
                "residual_velocity_state_jacobian requires a "
                "NodalFieldVelocity velocity, got "
                f"{type(velocity).__name__}"
            )
        if self._conservative:
            raise NotImplementedError(
                "velocity sensitivities are implemented for the "
                "non-conservative advection form only"
            )
        delta_velocity = NodalFieldVelocity(
            velocity.basis(), self._bkd.to_numpy(delta_dofs)
        )
        scalar_skfem = self._basis.skfem_basis()
        advection = asm(
            BilinearForm(
                _AdvectionKernel(None, delta_velocity.values, False)
            ),
            scalar_skfem,
        )
        result: Array = -advection
        return result

    def residual_reaction_jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dF/d(\text{reaction DOFs})` at the given state.

        The mixed assembly is exact:
        :math:`S(u)[j, k] = +\int \phi_k \, u \, \phi_j \, dx` — the
        linear reaction enters the residual as :math:`+(v, r u)`. S is
        LINEAR in the state with zero constant part and its
        state-derivative tensor is the reaction mass structure.
        Requires the reaction to be a ``NodalFieldLinearReaction``.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)

        Returns
        -------
        Array
            Sensitivity matrix (scipy sparse). Shape: (nstates, nstates)
        """
        if not isinstance(self._reaction_function, NodalFieldLinearReaction):
            raise TypeError(
                "residual_reaction_jacobian requires a "
                "NodalFieldLinearReaction reaction (nodal DOFs are the "
                "differentiable representation), got "
                f"{type(self._reaction_function).__name__}"
            )
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        sensitivity = asm(
            BilinearForm(_ReactionSensitivityKernel()),
            skfem_basis,
            skfem_basis,
            u_prev=skfem_basis.interpolate(state_np),
        )
        result: Array = sensitivity
        return result

    def residual_reaction_state_jacobian(
        self, delta_dofs: Array, state: Array
    ) -> Array:
        r"""Compute :math:`A(\delta r) = d/du \, [dF/d(r)\,\delta r]`.

        The reaction mass matrix with the GIVEN coefficient field:
        :math:`+\int \delta r \, u \, v`. The linear reaction term is
        linear in both :math:`r` and the state, so the result is
        state-independent; the ``state`` argument is kept for the
        typed field-derivative signature.

        Parameters
        ----------
        delta_dofs : Array
            Reaction-field direction (nodal DOFs). Shape: (nstates,)
        state : Array
            Solution state (unused here). Shape: (nstates,)

        Returns
        -------
        Array
            Mixed Jacobian (scipy sparse). Shape: (nstates, nstates)
        """
        skfem_basis = self._basis.skfem_basis()
        delta_np = self._bkd.to_numpy(delta_dofs)
        mixed = asm(
            BilinearForm(_ReactionMixedStateKernel()),
            skfem_basis,
            delta_prev=skfem_basis.interpolate(delta_np),
        )
        result: Array = mixed
        return result

    def residual_diffusivity_state_jacobian(
        self, delta_dofs: Array, state: Array
    ) -> Array:
        r"""Compute :math:`A(\delta g) = d/du \, [dF/d(\kappa)\,\delta g]`.

        The diffusion-only stiffness assembled with the GIVEN
        coefficient field: :math:`-\int \delta g \, \nabla u \cdot
        \nabla v`. The diffusion term is linear in both :math:`\kappa`
        and the state, so the result is state-independent; the
        ``state`` argument is kept for the typed field-derivative
        signature (quasilinear physics need it).

        Parameters
        ----------
        delta_dofs : Array
            Diffusivity-field direction (nodal DOFs). Shape: (nstates,)
        state : Array
            Solution state (unused here). Shape: (nstates,)

        Returns
        -------
        Array
            Mixed Jacobian (scipy sparse). Shape: (nstates, nstates)
        """
        skfem_basis = self._basis.skfem_basis()
        delta_np = self._bkd.to_numpy(delta_dofs)
        mixed = asm(
            BilinearForm(_DiffusivityMixedStateKernel()),
            skfem_basis,
            delta_prev=skfem_basis.interpolate(delta_np),
        )
        result: Array = mixed
        return result

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T (d^2F/du^2) w of the RAW spatial residual.

        Only the reaction is nonlinear in the state (diffusion and
        advection are linear), so the contraction is
        ``+int R''(u) adj w phi_i dx`` — positive because the reaction
        enters the residual as ``+(w, R(u))`` — and exactly zero for
        linear problems.

        Raises
        ------
        TypeError
            If the reaction is nonlinear but does not provide the
            analytic second derivative (supply it via
            ``CallableReaction(..., second_derivative_func=...)``).
        """
        reaction = self._reaction_function
        if reaction is None or reaction.is_linear():
            return self._bkd.full_like(state, 0.0)
        if not isinstance(
            reaction, ReactionFunctionWithSecondDerivativeProtocol
        ):
            raise TypeError(
                "state_state_hvp with a nonlinear reaction requires the "
                "analytic second derivative "
                "(ReactionFunctionWithSecondDerivativeProtocol); supply "
                "second_derivative_func on CallableReaction"
            )
        skfem_basis = self._basis.skfem_basis()
        contraction = asm(
            LinearForm(_ReactionHVPKernel(reaction.second_derivative)),
            skfem_basis,
            u_prev=skfem_basis.interpolate(self._bkd.to_numpy(state)),
            adj_prev=skfem_basis.interpolate(self._bkd.to_numpy(adj_state)),
            dir_prev=skfem_basis.interpolate(self._bkd.to_numpy(wvec)),
        )
        return self._bkd.asarray(np.asarray(contraction).astype(np.float64))

    def initial_condition(self, func: Callable[..., Any]) -> Array:
        """Create initial condition by interpolating a function.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Takes coordinates (ndim, npts)
            and returns (npts,).

        Returns
        -------
        Array
            Initial DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        return (
            f"AdvectionDiffusionReaction("
            f"nstates={self.nstates()}, "
            f"diffusivity={self._diffusion_function!r}, "
            f"reaction={self._reaction_function!r})"
        )


# Backwards compatibility alias
LinearAdvectionDiffusionReaction = AdvectionDiffusionReaction
