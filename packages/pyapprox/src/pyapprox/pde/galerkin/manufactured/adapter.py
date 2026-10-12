"""Galerkin boundary conditions and suppliers from a manufactured solution.

Bridges the manufactured solutions in ``pyapprox.pde.manufactured``, which
both discretizations share, to Galerkin physics and boundary conditions.
Natural boundary data comes from the physics' own flux law
(``FluxLawProviderProtocol``) evaluated at the exact solution and gradient
with the mesh's facet normals, so the data is what the physics' weak form
integrates by parts, on any boundary shape.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    NonConservativeAdvection,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeVaryingProtocol,
)
from pyapprox.pde.constitutive.protocols import (
    SymbolicStressModelProtocol,
)
from pyapprox.pde.galerkin.boundary import (
    DirichletBC,
    NeumannBC,
    NormalFluxLawProtocol,
    RobinBC,
)
from pyapprox.pde.galerkin.protocols.basis import (
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
)
from pyapprox.pde.manufactured import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
)
from pyapprox.pde.manufactured.hyperelasticity import (
    ManufacturedHyperelasticityEquations,
)
from pyapprox.pde.manufactured.linear_elasticity import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.util.backends.protocols import Array, Backend

_Pts = NDArray[np.floating[Any]]
_Alpha = Union[float, Callable[[_Pts], _Pts]]


def _declares_time(func: Callable[..., Any]) -> bool:
    """Whether a manufactured function is declared time-dependent.

    Manufactured functions of a transient solution are ``TimeDependent``;
    those of a steady one take coordinates only.
    """
    return isinstance(func, TimeVaryingProtocol) and func.is_time_dependent()


class _ManufacturedField:
    """One manufactured function, called with or without time as declared."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self._time_dependent = _declares_time(func)

    def __call__(self, coords: _Pts, time: float) -> _Pts:
        if self._time_dependent:
            return np.asarray(self._func(coords, time))
        return np.asarray(self._func(coords))

    def is_time_dependent(self) -> bool:
        return self._time_dependent


def _components_first(values: _Pts) -> _Pts:
    """``(npts,)`` or ``(npts, ncomp)`` to ``(ncomp, npts)``."""
    if values.ndim == 1:
        return values[None, :]
    return values.T


def _gradient_components_first(values: _Pts) -> _Pts:
    """Manufactured gradient to the flux law's ``(ncomp, ndim, npts)``.

    Scalar solutions give ``(npts, ndim)``, vector ones
    ``(ncomp, npts, ndim)``.
    """
    if values.ndim == 2:
        return values.T[None, :, :]
    return np.transpose(values, (0, 2, 1))


class _BasisLayout:
    """The value layout a basis' suppliers and conditions use.

    A vector basis takes ``(ncomp, npts)``; a scalar basis ``(npts,)``.
    Essential values are ``(npts,)`` whenever there is one component,
    which is what ``DirichletBC`` selects from.
    """

    def __init__(self, basis: GalerkinBasisProtocol[Array]) -> None:
        self._vector = isinstance(basis, ComponentDofsBasisProtocol)
        self._ncomponents = (
            basis.ncomponents()
            if isinstance(basis, ComponentDofsBasisProtocol)
            else 1
        )

    def data(self, values: _Pts) -> _Pts:
        """Natural-BC data and forcing from ``(ncomp, npts)``."""
        return values if self._vector else values[0]

    def essential(self, values: _Pts) -> _Pts:
        """Dirichlet values from ``(ncomp, npts)``."""
        return values if self._ncomponents > 1 else values[0]


class _EssentialValues:
    """Exact values (or a time derivative) at the boundary DOFs."""

    def __init__(self, field: _ManufacturedField, layout: _BasisLayout) -> None:
        self._field = field
        self._layout = layout

    def __call__(self, coords: _Pts, time: float) -> _Pts:
        return self._layout.essential(
            _components_first(self._field(coords, time))
        )

    def is_time_dependent(self) -> bool:
        return self._field.is_time_dependent()


class _NaturalData:
    """``g = alpha u + law(x, u, grad u, n, t)`` at the exact solution.

    With ``alpha`` zero this is Neumann data, otherwise Robin data. It
    satisfies ``NormalDependentDataProtocol``, so the assembly supplies
    the facet normals.
    """

    def __init__(
        self,
        solution: _ManufacturedField,
        gradient: _ManufacturedField,
        law: NormalFluxLawProtocol,
        alpha: _Alpha,
        layout: _BasisLayout,
    ) -> None:
        self._solution = solution
        self._gradient = gradient
        self._law = law
        self._alpha = alpha
        self._layout = layout

    def evaluate(self, coords: _Pts, normal: _Pts, time: float) -> _Pts:
        u = _components_first(self._solution(coords, time))
        grad_u = _gradient_components_first(self._gradient(coords, time))
        flux = np.asarray(
            self._law.normal_flux(coords, u, grad_u, normal, time)
        )
        alpha = (
            np.asarray(self._alpha(coords))
            if callable(self._alpha)
            else self._alpha
        )
        return self._layout.data(alpha * u + flux)

    def is_time_dependent(self) -> bool:
        return (
            self._solution.is_time_dependent()
            or self._law.is_time_dependent()
        )


class _Forcing:
    """The manufactured forcing in the basis' layout."""

    def __init__(self, field: _ManufacturedField, layout: _BasisLayout) -> None:
        self._field = field
        self._layout = layout

    def __call__(self, coords: _Pts, time: float) -> _Pts:
        return self._layout.data(_components_first(self._field(coords, time)))

    def is_time_dependent(self) -> bool:
        return self._field.is_time_dependent()


class _Velocity:
    """The manufactured velocity as ``(ndim, ...)``, matching ``coords``.

    The manufactured velocity takes flat ``(ndim, npts)`` points and
    returns ``(npts, ndim)``; skfem evaluates at ``(ndim, nelem, nquad)``.
    """

    def __init__(self, field: _ManufacturedField) -> None:
        self._field = field

    def __call__(self, coords: _Pts, time: float) -> _Pts:
        ndim = coords.shape[0]
        values = self._field(coords.reshape(ndim, -1), time)
        return values.T.reshape(ndim, *coords.shape[1:])

    def is_time_dependent(self) -> bool:
        return self._field.is_time_dependent()


class GalerkinManufacturedSolution(Generic[Array]):
    """A manufactured solution's boundary conditions and suppliers on a basis.

    Each condition is built on request, so a test lists exactly the
    conditions it composes::

        ms = GalerkinManufacturedSolution(basis, man_sol.functions, bkd)
        physics = AdvectionDiffusionReaction(
            basis, ..., forcing=ms.forcing(), bkd=bkd
        )
        system = compose_galerkin_system(physics, [
            ms.dirichlet("left"),
            ms.robin("right", 2.0, physics.flux_law()),
        ])

    Natural data takes the physics' flux law because the physics, which
    needs the forcing, exists only after this object. Build the physics
    from the exact coefficients, so the law carries no discretization
    error. Values and suppliers are declared time-dependent exactly when
    the manufactured functions are.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis; a vector basis gets component-first values.
    functions : Mapping[str, Callable]
        The manufactured functions. Uses ``"solution"``, and as needed
        ``"gradient"`` (natural data), ``"solution_time_derivative"``
        (transient Dirichlet data), ``"forcing"`` and ``"velocity"``.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        functions: Mapping[str, Callable[..., Any]],
        bkd: Backend[Array],
    ) -> None:
        if "solution" not in functions:
            raise ValueError("functions must contain 'solution'")
        self._basis = basis
        self._functions = functions
        self._bkd = bkd
        self._layout = _BasisLayout(basis)
        self._solution = _ManufacturedField(functions["solution"])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> GalerkinBasisProtocol[Array]:
        """Return the finite element basis."""
        return self._basis

    def _field(self, key: str, needed_for: str) -> _ManufacturedField:
        if key not in self._functions:
            raise ValueError(
                f"functions must contain {key!r} for {needed_for}"
            )
        return _ManufacturedField(self._functions[key])

    def solution(self) -> _ManufacturedField:
        """Return the exact solution, called as ``u(coords, time)``.

        Values are in the manufactured layout: ``(npts,)`` scalar,
        ``(npts, ncomp)`` vector.
        """
        return self._solution

    def dirichlet(self, boundary_name: str) -> DirichletBC[Array]:
        """Return ``u = u*`` on the boundary.

        A transient solution's signal carries the exact first time
        derivative, which stage-based steppers need.
        """
        values = _EssentialValues(self._solution, self._layout)
        if not self._solution.is_time_dependent():
            signal = BoundarySignal(values)
        else:
            derivative = _EssentialValues(
                self._field(
                    "solution_time_derivative", "transient Dirichlet data"
                ),
                self._layout,
            )
            signal = BoundarySignal(values, [derivative])
        return DirichletBC(self._basis, boundary_name, signal, self._bkd)

    def neumann(
        self, boundary_name: str, law: NormalFluxLawProtocol
    ) -> NeumannBC[Array]:
        """Return the Neumann condition with data ``law(u*, grad u*, n)``.

        Parameters
        ----------
        boundary_name : str
            Name of the boundary.
        law : NormalFluxLawProtocol
            The physics' ``flux_law()``.
        """
        return NeumannBC(
            self._basis,
            boundary_name,
            self._natural_data(law, 0.0),
            self._bkd,
        )

    def robin(
        self,
        boundary_name: str,
        alpha: _Alpha,
        law: NormalFluxLawProtocol,
    ) -> RobinBC[Array]:
        """Return the Robin condition with data
        ``alpha u* + law(u*, grad u*, n)``.

        Parameters
        ----------
        boundary_name : str
            Name of the boundary.
        alpha : float or Callable
            The Robin coefficient: a constant, or ``alpha(coords)``.
        law : NormalFluxLawProtocol
            The physics' ``flux_law()``.
        """
        return RobinBC(
            self._basis,
            boundary_name,
            alpha,
            self._natural_data(law, alpha),
            self._bkd,
        )

    def _natural_data(
        self, law: NormalFluxLawProtocol, alpha: _Alpha
    ) -> _NaturalData:
        if not isinstance(law, NormalFluxLawProtocol):
            raise TypeError(
                "law must satisfy NormalFluxLawProtocol, got "
                f"{type(law).__name__}"
            )
        return _NaturalData(
            self._solution,
            self._field("gradient", "natural boundary data"),
            law,
            alpha,
            self._layout,
        )

    def forcing(self) -> _Forcing:
        """Return the forcing as a declared supplier ``f(coords, time)``.

        ``(npts,)`` for a scalar basis, ``(ncomp, npts)`` for a vector one.
        """
        return _Forcing(self._field("forcing", "the forcing"), self._layout)

    def velocity(self) -> _Velocity:
        """Return the velocity as a declared supplier ``v(coords, time)``,
        shaped ``(ndim, ...)`` like ``coords``."""
        return _Velocity(self._field("velocity", "the velocity"))

    def __repr__(self) -> str:
        return (
            f"GalerkinManufacturedSolution(ndofs={self._basis.ndofs()}, "
            "time_dependent="
            f"{self._solution.is_time_dependent()})"
        )


def create_adr_manufactured_test(
    bounds: List[float],
    sol_str: str,
    diff_str: str,
    react_str: str,
    vel_strs: List[str],
    bkd: Backend[Array],
    time_dependent: bool = False,
    advection_form: Optional[AdvectionFormProtocol] = None,
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create ADR manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_str : str
        Solution string (uses 'x', 'y', 'z' for coords, 'T' for time).
    diff_str : str
        Diffusion coefficient string.
    react_str : str
        Reaction term string (uses 'u' for solution).
    vel_strs : List[str]
        Velocity component strings.
    bkd : Backend
        Computational backend.
    time_dependent : bool
        Whether solution is time-dependent.
    advection_form : AdvectionFormProtocol, optional
        How the advection term is written; it must match the physics'.
        Defaults to ``NonConservativeAdvection()``, the Galerkin ADR
        default.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedAdvectionDiffusionReaction(
        sol_str=sol_str,
        nvars=nvars,
        diff_str=diff_str,
        react_str=react_str,
        vel_strs=vel_strs,
        bkd=bkd,
        oned=True,  # Return 1D arrays for Galerkin
        advection_form=(
            NonConservativeAdvection()
            if advection_form is None
            else advection_form
        ),
    )

    return man_sol.functions, nvars


def create_helmholtz_manufactured_test(
    bounds: List[float],
    sol_str: str,
    sqwavenum_str: str,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create Helmholtz manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_str : str
        Solution string (uses 'x', 'y', 'z' for coords).
    sqwavenum_str : str
        Squared wavenumber string (k²).
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, sqwavenum, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedHelmholtz(
        sol_str=sol_str,
        nvars=nvars,
        sqwavenum_str=sqwavenum_str,
        bkd=bkd,
        oned=True,  # Return 1D arrays for Galerkin
    )

    return man_sol.functions, nvars


def create_elasticity_manufactured_test(
    bounds: List[float],
    sol_strs: List[str],
    lambda_str: str,
    mu_str: str,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create linear elasticity manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds, two entries per dimension (e.g.
        [xmin, xmax, ymin, ymax] in 2D, plus zmin, zmax in 3D).
    sol_strs : List[str]
        Displacement component strings, one per dimension.
        May contain 'x', 'y', 'z' for coords and 'T' for time.
    lambda_str : str
        Lame first parameter string.
    mu_str : str
        Shear modulus string.
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux, etc.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedLinearElasticityEquations(
        sol_strs=sol_strs,
        nvars=nvars,
        lambda_str=lambda_str,
        mu_str=mu_str,
        bkd=bkd,
        oned=True,
    )

    return man_sol.functions, nvars


def create_hyperelasticity_manufactured_test(
    bounds: List[float],
    sol_strs: List[str],
    stress_model: SymbolicStressModelProtocol,
    bkd: Backend[Array],
) -> Tuple[Dict[str, Callable[..., Any]], int]:
    """Create hyperelasticity manufactured solution functions for Galerkin tests.

    Parameters
    ----------
    bounds : List[float]
        Domain bounds [xmin, xmax] or [xmin, xmax, ymin, ymax] etc.
    sol_strs : List[str]
        Displacement component expressions. Length must equal nvars.
        May contain 'x', 'y', 'z' for coords and 'T' for time.
    stress_model : SymbolicStressModelProtocol
        Stress model with sympy expression support (e.g., NeoHookeanStress).
    bkd : Backend
        Computational backend.

    Returns
    -------
    functions : Dict[str, Callable]
        Dictionary containing solution, forcing, flux functions.
        - solution(x): (npts, ncomponents) exact displacement
        - forcing(x): (npts, ncomponents) body force
        - flux(x): (ncomponents, npts, ncomponents) PK1 stress tensor
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    man_sol = ManufacturedHyperelasticityEquations(
        sol_strs=sol_strs,
        nvars=nvars,
        stress_model=stress_model,
        bkd=bkd,
        oned=True,
    )

    return man_sol.functions, nvars

