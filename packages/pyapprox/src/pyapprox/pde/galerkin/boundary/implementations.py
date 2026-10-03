"""Concrete boundary condition implementations for Galerkin FEM.

Provides implementations of Dirichlet, Neumann, and Robin boundary conditions
that integrate with scikit-fem for assembly.
"""

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse, spmatrix

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary.signal import BoundarySignal, DofSignal
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
    TimeIndependent,
    as_time_aware,
)
from pyapprox.pde.galerkin.protocols.basis import (
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
)
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.sparse_utils import apply_dirichlet_rows
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import Basis, BilinearForm, LinearForm, asm
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class _ConstantBoundaryValue:
    """Constant boundary value as a picklable callable.

    Boundary conditions store their value functions; wrapping constants
    in a lambda would make every BC (and everything holding one —
    physics, parameterizations) unpicklable.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(
        self, x: np.ndarray, t: Optional[float] = None
    ) -> np.ndarray:
        return np.full(x.shape[1], self._value)


class DirichletBC(Generic[Array]):
    """Dirichlet boundary condition: u = g(x, t) on boundary.

    Enforces the constraint by modifying the residual and Jacobian
    at boundary DOFs.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary (e.g., "left", "right", "bottom", "top").
    value_func : float, Callable or BoundarySignal
        The boundary values g: a constant; a bare ``g(coords)``
        (time-independent); a declared ``TimeIndependent(g)`` /
        ``TimeDependent(g)`` for ``g(coords, time)``; or a
        ``BoundarySignal`` carrying g with its analytic time derivatives
        by order. Only a signal carries derivatives: a time-dependent g
        passed without one cannot be used with stage-based steppers on a
        consistent mass matrix, while a time-independent g has exact
        zero derivatives automatically. A bare callable that could take
        a time is rejected, so time dependence is never guessed.
        Coordinates have shape (ndim, npts); g returns either per-DOF
        values (npts,) or, for vector bases, per-component values
        (ncomponents, npts) from which each DOF's component is selected
        automatically.
    bkd : Backend[Array]
        Computational backend.
    components : tuple of int, optional
        For vector bases, constrain only these displacement components on
        the boundary (e.g. ``components=(2,)`` fixes u_z only — a
        symmetry/roller condition). Default is None (all components).
        Requires a basis whose ``get_dofs`` supports component selection.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> from pyapprox.pde.galerkin.basis import LagrangeBasis
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> bc = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        value_func: Union[Callable[..., Any], float, BoundarySignal],
        bkd: Backend[Array],
        components: Optional[tuple[int, ...]] = None,
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        self._bkd = bkd
        self._signal = (
            value_func
            if isinstance(value_func, BoundarySignal)
            else BoundarySignal(value_func)
        )

        if isinstance(basis, ComponentDofsBasisProtocol):
            self._ncomponents = basis.ncomponents()
        else:
            self._ncomponents = 1

        # Get and cache boundary DOF indices
        if components is None:
            self._boundary_dofs = basis.get_dofs(boundary_name)
        else:
            if not isinstance(basis, ComponentDofsBasisProtocol):
                raise TypeError(
                    "components requires a vector basis supporting "
                    "per-component DOF selection, got "
                    f"{type(basis).__name__}"
                )
            self._boundary_dofs = basis.get_dofs(
                boundary_name, components=components
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary.

        Returns
        -------
        Array
            Integer DOF indices. Shape: (nboundary_dofs,)
        """
        return self._boundary_dofs

    def constrained_dofs(self) -> Array:
        """Return constrained DOF indices (EssentialBCProtocol)."""
        return self._boundary_dofs

    def constrained_values(self, time: float) -> Array:
        """Return prescribed values at ``time`` (EssentialBCProtocol)."""
        return self.boundary_values(time)

    def is_time_invariant(self) -> bool:
        """Whether the signal is declared time-independent."""
        return not self._signal.is_time_dependent()

    def signal(self) -> BoundarySignal:
        """Return the boundary signal (values and time derivatives)."""
        return self._signal

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Dirichlet boundary values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Boundary values. Shape: (nboundary_dofs,)
        """
        return self._evaluate_on_boundary(self._signal.values(), time)

    def constrained_values_derivative(
        self, order: int
    ) -> Optional[Callable[[float], Array]]:
        """Return ``t -> d^k g/dt^k`` at the DOFs, or ``None``.

        Exact zeros for a time-independent signal; ``None`` when a
        time-dependent signal was not given that order.
        """
        derivative = self._signal.time_derivative(order)
        if derivative is None:
            return None
        return partial(self._evaluate_on_boundary, derivative)

    def _evaluate_on_boundary(
        self, func: TimeAwareCallableProtocol, time: float
    ) -> Array:
        """Evaluate a boundary function at the constrained DOF coords."""
        # Get DOF coordinates on boundary
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)

        # Extract boundary coordinates
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        # Evaluate boundary function
        values_np = np.asarray(func(bndry_coords, time))

        if values_np.ndim == 2:
            # Vector-valued return (ncomponents, nboundary_dofs): select
            # each DOF's own component. skfem interleaves vector-element
            # DOFs, so DOF d belongs to component d % ncomponents.
            if self._ncomponents == 1:
                raise ValueError(
                    "value_func must return a 1D array for scalar bases, "
                    f"got shape {values_np.shape}"
                )
            if values_np.shape != (
                self._ncomponents,
                bndry_dofs_np.shape[0],
            ):
                raise ValueError(
                    "vector value_func must return shape "
                    f"({self._ncomponents}, {bndry_dofs_np.shape[0]}), "
                    f"got {values_np.shape}"
                )
            dof_components = bndry_dofs_np % self._ncomponents
            values_np = values_np[
                dof_components, np.arange(bndry_dofs_np.shape[0])
            ]

        return self._bkd.asarray(values_np.astype(np.float64))

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply Dirichlet BC to residual.

        Sets residual[dof] = state[dof] - g(x, t) for boundary DOFs.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        res_np = self._bkd.to_numpy(residual).copy()
        state_np = self._bkd.to_numpy(state)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_vals_np = self._bkd.to_numpy(self.boundary_values(time))

        # Set residual to constraint violation: u - g
        res_np[bndry_dofs_np] = state_np[bndry_dofs_np] - bndry_vals_np

        return self._bkd.asarray(res_np)

    def apply_to_jacobian(
        self,
        jacobian: Union[spmatrix, Array],
        state: Array,
        time: float,
    ) -> Union[spmatrix, Array]:
        """Apply Dirichlet BC to Jacobian.

        Sets Jacobian rows to identity for boundary DOFs.
        Accepts both sparse matrices and dense arrays.

        Parameters
        ----------
        jacobian : sparse matrix or Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        sparse matrix or Array
            Modified Jacobian (same type as input).
        """
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)

        if issparse(jacobian):
            return apply_dirichlet_rows(jacobian, bndry_dofs_np)
        else:
            jac_np = self._bkd.to_numpy(jacobian).copy()
            for dof in bndry_dofs_np:
                jac_np[dof, :] = 0.0
                jac_np[dof, dof] = 1.0
            return self._bkd.asarray(jac_np)

    def apply_to_param_jacobian(
        self,
        param_jacobian: Array,
        state: Array,
        time: float,
    ) -> Array:
        """Apply Dirichlet BC to parameter Jacobian.

        Dirichlet constraint u = g(x, t) does not depend on material
        parameters, so the parameter Jacobian rows at boundary DOFs
        are set to zero.

        Parameters
        ----------
        param_jacobian : Array
            Parameter Jacobian. Shape: (nstates, nparams)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified parameter Jacobian. Shape: (nstates, nparams)
        """
        pj_np = self._bkd.to_numpy(param_jacobian).copy()
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        pj_np[bndry_dofs_np, :] = 0.0
        return self._bkd.asarray(pj_np)

    def __repr__(self) -> str:
        return (
            f"DirichletBC(boundary='{self._boundary_name}', "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class NeumannBC(Generic[Array]):
    """Neumann boundary condition: flux . n = g(x, t) on boundary.

    In weak form, contributes to the load vector via boundary integral:
        integral_{Gamma} g * phi ds

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary.
    flux_func : Callable or float
        Flux values: a constant; a bare ``g(coords)`` (time-independent);
        or a declared ``TimeIndependent(g)`` / ``TimeDependent(g)`` for
        ``g(coords, time)``. A bare callable that could take a time is
        rejected, so time dependence is never guessed. Coordinates have
        shape (ndim, npts); values (npts,) for scalar and (ndim, npts)
        for vector bases.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        flux_func: Union[Callable[..., Any], float],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        self._bkd = bkd

        # Time dependence is declared, never inferred: a bare f(coords) is
        # time-independent, a declared TimeIndependent/TimeDependent says
        # which it is, and a bare callable that could take a time is
        # rejected by as_time_aware.
        self._flux_func: TimeAwareCallableProtocol = (
            as_time_aware(flux_func)
            if callable(flux_func)
            else TimeIndependent(_ConstantBoundaryValue(float(flux_func)))
        )

        # Get boundary DOFs
        self._boundary_dofs = basis.get_dofs(boundary_name)

        # Cache boundary basis
        self._boundary_basis: Optional[Basis] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._boundary_dofs

    def is_time_dependent(self) -> bool:
        """Whether the flux data varies in time (declared, not inferred)."""
        return self._flux_func.is_time_dependent()

    def is_time_invariant(self) -> bool:
        """Whether the term is declared time-independent."""
        return not self.is_time_dependent()

    def set_flux_func(self, flux_func: Union[Callable[..., Any], float]) -> None:
        """Replace the flux data, under the same declaration rule as the
        constructor (a bare callable that could take a time is rejected).

        Lets a caller vary the data between solves (e.g. a load per
        sample) without rebuilding the physics that holds this BC.
        """
        self._flux_func = (
            as_time_aware(flux_func)
            if callable(flux_func)
            else TimeIndependent(_ConstantBoundaryValue(float(flux_func)))
        )

    def flux_values(self, time: float = 0.0) -> Array:
        """Return Neumann flux values at given time.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Array
            Flux values. Shape: (nboundary_dofs,)
        """
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        values_np = self._flux_func(bndry_coords, time)
        return self._bkd.asarray(values_np.astype(np.float64))

    def _get_boundary_basis(self) -> Basis:
        """Get or create the boundary basis for assembly."""
        if self._boundary_basis is None:
            skfem_basis = self._basis.skfem_basis()
            self._boundary_basis = skfem_basis.boundary(self._boundary_name)
        return self._boundary_basis

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Neumann BC contribution to load vector.

        Adds boundary integral: integral_{Gamma} g . phi ds

        For scalar elements, flux_func returns (npts,).
        For vector elements, flux_func returns (ndim, npts) and the
        form computes sum_i(flux_i * v_i).

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector. Shape: (nstates,)
        """
        load_np = self._bkd.to_numpy(load).copy()
        bndry_basis = self._get_boundary_basis()

        # Store flux function for closure
        flux_func = self._flux_func
        current_time = time

        def neumann_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                flux_flat = np.asarray(flux_func(x_flat, current_time))
                # Detect vector flux: shape (ndim, npts) vs scalar (npts,)
                if flux_flat.ndim == 2 and flux_flat.shape[0] == ndim:
                    # Vector: flux_flat is (ndim, npts)
                    flux_3d = flux_flat.reshape(ndim, nelem, nquad)
                    return np.asarray(sum(flux_3d[i] * v[i] for i in range(ndim)))
                else:
                    flux_2d = flux_flat.reshape(nelem, nquad)
                    return np.asarray(flux_2d * v)
            else:
                flux = np.asarray(flux_func(x_np, current_time))
                if flux.ndim == 2:
                    ndim = flux.shape[0]
                    return np.asarray(sum(flux[i] * v[i] for i in range(ndim)))
                return np.asarray(flux * v)

        contribution = asm(LinearForm(neumann_form), bndry_basis)
        load_np += contribution

        return self._bkd.asarray(load_np.astype(np.float64))

    def apply_to_stiffness(
        self,
        stiffness: Union[spmatrix, Array],
        time: float,
    ) -> Union[spmatrix, Array]:
        """Return the stiffness matrix unchanged (WeakFormBCProtocol).

        A pure Neumann BC adds no state-dependent boundary term.
        """
        return stiffness

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Add the Neumann term c = integral_{Gamma} g . phi ds.

        The physics' sign convention F = b - K u: the term is the load
        (see ``WeakFormBCProtocol``).
        """
        zero = self._bkd.full_like(residual, 0.0)
        return residual + self.apply_to_load(zero, time)

    def apply_to_jacobian(
        self,
        jacobian: Union[spmatrix, Array],
        state: Array,
        time: float,
    ) -> Union[spmatrix, Array]:
        """Return the Jacobian unchanged (WeakFormBCProtocol).

        The Neumann contribution is state-independent.
        """
        return jacobian

    def state_derivatives(self) -> StateDerivatives[Array]:
        """Exact zero curvature: the term is independent of u."""
        return StateDerivatives.linear(self._bkd)

    def __repr__(self) -> str:
        return (
            f"NeumannBC(boundary='{self._boundary_name}', "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class RobinBC(Generic[Array]):
    """Robin boundary condition: alpha(x) * u + beta * (flux . n) = g(x, t).

    This is a mixed boundary condition that combines Dirichlet and Neumann.
    Special cases:
    - alpha=1, beta=0: Dirichlet BC
    - alpha=0, beta=1: Neumann BC

    In weak form for diffusion: -D * du/dn = alpha(x) * u - g
    Contributes:
    - To stiffness matrix: integral_{Gamma} alpha(x) * u * phi ds
    - To load vector: integral_{Gamma} g * phi ds

    A spatially varying coefficient supports conditions whose strength
    follows a profile along the boundary — e.g. the Danckwerts inflow
    condition kappa*grad(u).n = (v.n)(u - u_in), whose alpha is the
    boundary-normal velocity.

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Finite element basis.
    boundary_name : str
        Name of the boundary.
    alpha : float or Callable
        Coefficient for the u term: a constant, or ``alpha(x)`` taking
        coordinates of shape ``(ndim, npts)`` and returning ``(npts,)``
        values at boundary quadrature points (time-independent; time
        dependence belongs to ``value_func``).
    value_func : Callable or float
        Robin data g: a constant; a bare ``g(coords)`` (time-independent);
        or a declared ``TimeIndependent(g)`` / ``TimeDependent(g)`` for
        ``g(coords, time)``. A bare callable that could take a time is
        rejected, so time dependence is never guessed.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        boundary_name: str,
        alpha: Union[float, Callable[[np.ndarray], np.ndarray]],
        value_func: Union[Callable[..., Any], float],
        bkd: Backend[Array],
    ):
        self._basis = basis
        self._boundary_name = boundary_name
        if not callable(alpha):
            alpha = float(alpha)
        self._alpha = alpha
        self._bkd = bkd

        # Time dependence is declared, never inferred (see NeumannBC).
        self._value_func: TimeAwareCallableProtocol = (
            as_time_aware(value_func)
            if callable(value_func)
            else TimeIndependent(_ConstantBoundaryValue(float(value_func)))
        )

        # Get boundary DOFs
        self._boundary_dofs = basis.get_dofs(boundary_name)

        # Cache boundary basis
        self._boundary_basis: Optional[Basis] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_name(self) -> str:
        """Return the boundary name."""
        return self._boundary_name

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._boundary_dofs

    def is_time_dependent(self) -> bool:
        """Whether the Robin data g varies in time (declared, not inferred).

        The coefficient alpha is time-independent by construction.
        """
        return self._value_func.is_time_dependent()

    def is_time_invariant(self) -> bool:
        """Whether the term is declared time-independent."""
        return not self.is_time_dependent()

    def alpha(self) -> Union[float, Callable[[np.ndarray], np.ndarray]]:
        """Return the coefficient for the u term (constant or callable)."""
        return self._alpha

    def _alpha_at_quadrature(
        self, w: "FormExtraParams"
    ) -> Union[float, np.ndarray]:
        """Evaluate alpha at the form's boundary quadrature points.

        Constants pass through (broadcasting handles them); callables
        are evaluated on the flattened coordinates and reshaped to the
        skfem ``(nelem, nquad)`` layout.
        """
        if not callable(self._alpha):
            return self._alpha
        x_np = np.asarray(w.x)
        if x_np.ndim == 3:
            ndim, nelem, nquad = x_np.shape
            vals = np.asarray(self._alpha(x_np.reshape(ndim, -1)))
            return vals.reshape(nelem, nquad)
        return np.asarray(self._alpha(x_np))

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Robin boundary values g at given time."""
        dof_coords = self._basis.dof_coordinates()
        dof_coords_np = self._bkd.to_numpy(dof_coords)
        bndry_dofs_np = self._bkd.to_numpy(self._boundary_dofs)
        bndry_coords = dof_coords_np[:, bndry_dofs_np]

        values_np = self._value_func(bndry_coords, time)
        return self._bkd.asarray(values_np.astype(np.float64))

    def _get_boundary_basis(self) -> Basis:
        """Get or create the boundary basis for assembly."""
        if self._boundary_basis is None:
            skfem_basis = self._basis.skfem_basis()
            self._boundary_basis = skfem_basis.boundary(self._boundary_name)
        return self._boundary_basis

    def _stiffness_contribution(self) -> spmatrix:
        """Assemble K_Gamma = alpha * integral_{Gamma} u . phi ds (sparse).

        For vector elements uses sum_i(u_i * v_i); for scalar uses u * v.
        """
        bndry_basis = self._get_boundary_basis()
        alpha_at_quadrature = self._alpha_at_quadrature
        ncomps = getattr(self._basis, "ncomponents", lambda: 1)()

        if ncomps > 1:

            def robin_bilinear(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                return np.asarray(
                    alpha_at_quadrature(w)
                    * sum(u[i] * v[i] for i in range(ncomps))
                )
        else:

            def robin_bilinear(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                return np.asarray(alpha_at_quadrature(w) * u * v)

        contribution: spmatrix = asm(BilinearForm(robin_bilinear), bndry_basis)
        return contribution

    def apply_to_stiffness(
        self,
        stiffness: Union[spmatrix, Array],
        time: float,
    ) -> Union[spmatrix, Array]:
        """Apply Robin BC contribution to stiffness matrix.

        Adds: alpha * integral_{Gamma} u . phi ds

        For vector elements uses sum_i(u_i * v_i); for scalar uses u * v.
        Accepts both sparse matrices and dense arrays.

        Parameters
        ----------
        stiffness : sparse matrix or Array
            Stiffness matrix. Shape: (nstates, nstates)
        time : float
            Current time.

        Returns
        -------
        sparse matrix or Array
            Modified stiffness matrix (same type as input).
        """
        contribution_sparse = self._stiffness_contribution()

        if issparse(stiffness):
            return stiffness + contribution_sparse
        else:
            stiff_np = self._bkd.to_numpy(stiffness).copy()
            stiff_np += contribution_sparse.toarray()

        return self._bkd.asarray(stiff_np.astype(np.float64))

    def apply_to_load(self, load: Array, time: float) -> Array:
        """Apply Robin BC contribution to load vector.

        Adds: integral_{Gamma} g . phi ds

        For vector elements, value_func returns (ndim, npts) and the
        form computes sum_i(g_i * v_i). For scalar, returns (npts,).

        Parameters
        ----------
        load : Array
            Load vector. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified load vector.
        """
        load_np = self._bkd.to_numpy(load).copy()
        bndry_basis = self._get_boundary_basis()
        value_func = self._value_func
        current_time = time

        def robin_linear(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_shape = x_np.shape
            if len(x_shape) == 3:
                ndim, nelem, nquad = x_shape
                x_flat = x_np.reshape(ndim, -1)
                vals_flat = np.asarray(value_func(x_flat, current_time))
                if vals_flat.ndim == 2 and vals_flat.shape[0] == ndim:
                    vals_3d = vals_flat.reshape(ndim, nelem, nquad)
                    return np.asarray(sum(vals_3d[i] * v[i] for i in range(ndim)))
                else:
                    vals_2d = vals_flat.reshape(nelem, nquad)
                    return np.asarray(vals_2d * v)
            else:
                vals = np.asarray(value_func(x_np, current_time))
                if vals.ndim == 2:
                    ndim = vals.shape[0]
                    return np.asarray(sum(vals[i] * v[i] for i in range(ndim)))
                return np.asarray(vals * v)

        contribution = asm(LinearForm(robin_linear), bndry_basis)
        load_np += contribution

        return self._bkd.asarray(load_np.astype(np.float64))

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Add the Robin term c = integral_{Gamma} (g - alpha u) . phi ds.

        The physics' sign convention F = b - K u: the term is the Robin
        load minus K_Gamma u (see ``WeakFormBCProtocol``). Scalar and
        vector bases alike.

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual.
        """
        load = self.apply_to_load(self._bkd.full_like(residual, 0.0), time)
        stiffness_times_state = self._bkd.asarray(
            self._stiffness_contribution() @ self._bkd.to_numpy(state)
        )
        return residual + load - stiffness_times_state

    def apply_to_jacobian(
        self,
        jacobian: Union[spmatrix, Array],
        state: Array,
        time: float,
    ) -> Union[spmatrix, Array]:
        """Add the Robin term's Jacobian, -K_Gamma (same type as input)."""
        contribution = self._stiffness_contribution()
        if issparse(jacobian):
            return jacobian - contribution
        jac_np = self._bkd.to_numpy(jacobian) - contribution.toarray()
        return self._bkd.asarray(jac_np.astype(np.float64))

    def state_derivatives(self) -> StateDerivatives[Array]:
        """Exact zero curvature: ``c = b_Gamma - K_Gamma u`` is linear."""
        return StateDerivatives.linear(self._bkd)

    def __repr__(self) -> str:
        alpha_repr = (
            "callable" if callable(self._alpha) else repr(self._alpha)
        )
        return (
            f"RobinBC(boundary='{self._boundary_name}', "
            f"alpha={alpha_repr}, "
            f"ndofs={len(self._bkd.to_numpy(self._boundary_dofs))})"
        )


class BoundaryConditionSet(Generic[Array]):
    """Typed builder/container for a problem's boundary conditions.

    Collects BCs by type and exposes role accessors
    (``weak_form_bcs``/``essential_bcs``) and ``all_conditions()`` for
    passing to physics classes, which own all BC APPLICATION (via
    their mixin and ``DirichletConstraintSet``).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._dirichlet_bcs: List[DirichletBC[Array]] = []
        self._neumann_bcs: List[NeumannBC[Array]] = []
        self._robin_bcs: List[RobinBC[Array]] = []

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def add_dirichlet(self, bc: DirichletBC[Array]) -> None:
        """Add a Dirichlet boundary condition."""
        self._dirichlet_bcs.append(bc)

    def add_neumann(self, bc: NeumannBC[Array]) -> None:
        """Add a Neumann boundary condition."""
        self._neumann_bcs.append(bc)

    def add_robin(self, bc: RobinBC[Array]) -> None:
        """Add a Robin boundary condition."""
        self._robin_bcs.append(bc)

    def ndirichlet(self) -> int:
        """Return number of Dirichlet BCs."""
        return len(self._dirichlet_bcs)

    def nneumann(self) -> int:
        """Return number of Neumann BCs."""
        return len(self._neumann_bcs)

    def nrobin(self) -> int:
        """Return number of Robin BCs."""
        return len(self._robin_bcs)

    def weak_form_bcs(
        self,
    ) -> List[Union[NeumannBC[Array], RobinBC[Array]]]:
        """Return the natural (Neumann/Robin) BCs, in insertion order."""
        return list(self._neumann_bcs) + list(self._robin_bcs)

    def essential_bcs(self) -> List[DirichletBC[Array]]:
        """Return the essential (Dirichlet) BCs, in insertion order."""
        return list(self._dirichlet_bcs)

    def all_conditions(self) -> List[BoundaryConditionProtocol[Array]]:
        """Return all boundary conditions as a flat list.

        The order is: Dirichlet, then Neumann, then Robin.
        This can be passed directly to physics classes that accept
        a list of BoundaryConditionProtocol objects.

        Returns
        -------
        List
            All boundary conditions.
        """
        return (
            list(self._dirichlet_bcs)
            + list(self._neumann_bcs)
            + list(self._robin_bcs)
        )

    def dirichlet_dofs(self) -> Array:
        """Return all Dirichlet DOF indices."""
        if not self._dirichlet_bcs:
            return self._bkd.asarray(
                np.array([], dtype=np.int64), dtype=self._bkd.int64_dtype()
            )

        all_dofs = []
        for bc in self._dirichlet_bcs:
            all_dofs.append(self._bkd.to_numpy(bc.boundary_dofs()))

        return self._bkd.asarray(
            np.concatenate(all_dofs).astype(np.int64),
            dtype=self._bkd.int64_dtype(),
        )

    def dirichlet_values(self, time: float = 0.0) -> Array:
        """Return all Dirichlet values at given time."""
        if not self._dirichlet_bcs:
            return self._bkd.asarray(np.array([], dtype=np.float64))

        all_vals = []
        for bc in self._dirichlet_bcs:
            all_vals.append(self._bkd.to_numpy(bc.boundary_values(time)))

        return self._bkd.asarray(np.concatenate(all_vals).astype(np.float64))

    def __repr__(self) -> str:
        return (
            f"BoundaryConditionSet("
            f"dirichlet={self.ndirichlet()}, "
            f"neumann={self.nneumann()}, "
            f"robin={self.nrobin()})"
        )


class DirectDirichletBC(Generic[Array]):
    """Dirichlet BC from pre-computed DOF indices and values.

    Lightweight alternative to ``DirichletBC`` for problems where DOF
    indices and values are known directly (e.g., Euler-Bernoulli beams
    with hardcoded clamped DOFs).

    Satisfies ``DirichletBCProtocol``.

    Parameters
    ----------
    dof_indices : Array or array-like
        Global DOF indices. Shape: (nboundary_dofs,)
    values : Array or array-like
        Dirichlet values at those DOFs. Shape: (nboundary_dofs,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        dof_indices: npt.ArrayLike,
        values: npt.ArrayLike,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._dof_indices = bkd.asarray(
            np.asarray(dof_indices, dtype=np.int64), dtype=bkd.int64_dtype()
        )
        self._values = bkd.asarray(np.asarray(values, dtype=np.float64))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._dof_indices

    def constrained_dofs(self) -> Array:
        """Return constrained DOF indices (EssentialBCProtocol)."""
        return self._dof_indices

    def constrained_values(self, time: float) -> Array:
        """Return prescribed values (EssentialBCProtocol)."""
        return self.boundary_values(time)

    def constrained_values_derivative(
        self, order: int
    ) -> Optional[Callable[[float], Array]]:
        """Return the time derivative: exactly zero (static values)."""
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        return self._zero_derivative

    def _zero_derivative(self, time: float) -> Array:
        """Exact zero time derivative of the static values."""
        return self._bkd.full_like(self._values, 0.0)

    def is_time_invariant(self) -> bool:
        """Values are fixed at construction."""
        return True

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Dirichlet values (constant, ignores time)."""
        return self._values

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply Dirichlet BC to residual.

        Sets residual[dof] = state[dof] - value for boundary DOFs.
        """
        res_np = self._bkd.to_numpy(residual).copy()
        state_np = self._bkd.to_numpy(state)
        dofs_np = self._bkd.to_numpy(self._dof_indices)
        vals_np = self._bkd.to_numpy(self._values)
        res_np[dofs_np] = state_np[dofs_np] - vals_np
        return self._bkd.asarray(res_np)

    def apply_to_jacobian(
        self,
        jacobian: Union[spmatrix, Array],
        state: Array,
        time: float,
    ) -> Union[spmatrix, Array]:
        """Apply Dirichlet BC to Jacobian.

        Sets Jacobian rows to identity for boundary DOFs.
        Accepts both sparse matrices and dense arrays.
        """
        dofs_np = self._bkd.to_numpy(self._dof_indices)
        if issparse(jacobian):
            return apply_dirichlet_rows(jacobian, dofs_np)
        else:
            jac_np = self._bkd.to_numpy(jacobian).copy()
            for dof in dofs_np:
                jac_np[dof, :] = 0.0
                jac_np[dof, dof] = 1.0
            return self._bkd.asarray(jac_np)

    def __repr__(self) -> str:
        n = len(self._bkd.to_numpy(self._dof_indices))
        return f"DirectDirichletBC(ndofs={n})"


class CallableDirichletBC(Generic[Array]):
    """Dirichlet BC with time-dependent values from a callable.

    Like ``DirectDirichletBC`` but the values are recomputed at each
    time step via a user-supplied callable.

    Satisfies ``DirichletBCProtocol``.

    Parameters
    ----------
    dof_indices : array-like
        Global DOF indices. Shape: (nboundary_dofs,)
    value_func : Callable[[float], np.ndarray] or DofSignal
        The values at the DOFs: a function of time alone returning shape
        (nboundary_dofs,), or a ``DofSignal`` carrying that function
        with its analytic time derivatives by order. A plain function
        has no time derivatives, so it cannot be used with stage-based
        steppers on a consistent mass matrix.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        dof_indices: npt.ArrayLike,
        value_func: Union[Callable[[float], np.ndarray], DofSignal],
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._dof_indices = bkd.asarray(
            np.asarray(dof_indices, dtype=np.int64), dtype=bkd.int64_dtype()
        )
        self._signal = (
            value_func
            if isinstance(value_func, DofSignal)
            else DofSignal(value_func)
        )

    def signal(self) -> DofSignal:
        """Return the DOF signal (values and time derivatives)."""
        return self._signal

    def constrained_values_derivative(
        self, order: int
    ) -> Optional[Callable[[float], Array]]:
        """Return the analytic ``order``-th time derivative, or ``None``.

        ``None`` for orders the signal was not given.
        """
        derivative = self._signal.time_derivative(order)
        if derivative is None:
            return None
        return partial(self._evaluate, derivative)

    def _evaluate(
        self, func: Callable[[float], np.ndarray], time: float
    ) -> Array:
        """Evaluate a time-only supplier at the DOFs as a backend array."""
        return self._bkd.asarray(np.asarray(func(time), dtype=np.float64))

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_dofs(self) -> Array:
        """Return indices of DOFs on this boundary."""
        return self._dof_indices

    def constrained_dofs(self) -> Array:
        """Return constrained DOF indices (EssentialBCProtocol)."""
        return self._dof_indices

    def constrained_values(self, time: float) -> Array:
        """Return prescribed values at ``time`` (EssentialBCProtocol)."""
        return self.boundary_values(time)

    def is_time_invariant(self) -> bool:
        """Whether the DOF signal is declared time-independent."""
        return not self._signal.is_time_dependent()

    def boundary_values(self, time: float = 0.0) -> Array:
        """Return Dirichlet values at given time."""
        return self._evaluate(self._signal.values(), time)

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply Dirichlet BC to residual.

        Sets residual[dof] = state[dof] - value(time) for boundary DOFs.
        """
        res_np = self._bkd.to_numpy(residual).copy()
        state_np = self._bkd.to_numpy(state)
        dofs_np = self._bkd.to_numpy(self._dof_indices)
        vals_np = np.asarray(self._signal.values()(time), dtype=np.float64)
        res_np[dofs_np] = state_np[dofs_np] - vals_np
        return self._bkd.asarray(res_np)

    def apply_to_jacobian(
        self,
        jacobian: Union[spmatrix, Array],
        state: Array,
        time: float,
    ) -> Union[spmatrix, Array]:
        """Apply Dirichlet BC to Jacobian.

        Sets Jacobian rows to identity for boundary DOFs.
        Accepts both sparse matrices and dense arrays.
        """
        dofs_np = self._bkd.to_numpy(self._dof_indices)
        if issparse(jacobian):
            return apply_dirichlet_rows(jacobian, dofs_np)
        else:
            jac_np = self._bkd.to_numpy(jacobian).copy()
            for dof in dofs_np:
                jac_np[dof, :] = 0.0
                jac_np[dof, dof] = 1.0
            return self._bkd.asarray(jac_np)

    def __repr__(self) -> str:
        n = len(self._bkd.to_numpy(self._dof_indices))
        return f"CallableDirichletBC(ndofs={n})"
