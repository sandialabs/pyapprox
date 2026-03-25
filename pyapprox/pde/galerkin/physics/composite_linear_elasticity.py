"""Composite linear elasticity with per-element material properties.

Solves the linear elasticity equations:
    -div(sigma) = f

where sigma is the Cauchy stress tensor:
    sigma = lambda(x) * tr(epsilon) * I + 2 * mu(x) * epsilon

and epsilon is the strain tensor:
    epsilon = (grad(u) + grad(u)^T) / 2

The Lame parameters lambda(x) and mu(x) vary element-wise, supporting
multi-material composites. For uniform materials, this reduces to standard
linear elasticity.

When used with a single material covering all elements, this class
is functionally equivalent to the legacy LinearElasticity class.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis
from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.helpers import ddot, eye, sym_grad, trace
    from skfem.models.elasticity import lame_parameters
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


@BilinearForm
def _elastic_form(u, v, w):
    """Elasticity bilinear form with element-wise Lame parameters.

    ``w.lam`` and ``w.mu`` must be ``(nelem, nquad)`` arrays.
    """
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return w.lam * ddot(eye(trace(eps_u), eps_u.shape[0]), eps_v) + 2.0 * w.mu * ddot(
        eps_u, eps_v
    )


class CompositeLinearElasticity(GalerkinPhysicsBase[Array]):
    """Linear elasticity with per-element material properties.

    Supports multi-material composites where each subdomain has its own
    Young's modulus and Poisson's ratio.

    Parameters
    ----------
    basis : VectorLagrangeBasis[Array]
        Vector finite element basis for displacement.
    material_map : Dict[str, Tuple[float, float]]
        Mapping from material name to (E, nu).
    element_materials : Dict[str, np.ndarray]
        Mapping from material name to element index arrays.
    bkd : Backend[Array]
        Computational backend.
    body_force : Callable, optional
        Body force per unit volume. Takes coordinates (ndim, npts) and
        time (float), returns (ndim, npts).
    boundary_conditions : list of BoundaryConditionProtocol, optional
        Boundary conditions (Dirichlet, Neumann, Robin).

    Examples
    --------
    Uniform material (backward-compatible with old LinearElasticity):

    >>> physics = CompositeLinearElasticity.from_uniform(
    ...     basis, youngs_modulus=1.0, poisson_ratio=0.3, bkd=bkd,
    ... )

    Multi-material composite:

    >>> physics = CompositeLinearElasticity(
    ...     basis=basis,
    ...     material_map={"skin": (10.0, 0.3), "core": (1.0, 0.25)},
    ...     element_materials={"skin": skin_elems, "core": core_elems},
    ...     bkd=bkd,
    ... )
    """

    @classmethod
    def from_uniform(
        cls,
        basis: VectorLagrangeBasis[Array],
        youngs_modulus: float,
        poisson_ratio: float,
        bkd: Backend[Array],
        body_force: Optional[Callable[..., Any]] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ) -> "CompositeLinearElasticity[Array]":
        """Create from uniform material properties.

        Convenience constructor that wraps a single material covering
        all elements. Provides the same interface as the legacy
        LinearElasticity class.

        Parameters
        ----------
        basis : VectorLagrangeBasis[Array]
            Vector finite element basis.
        youngs_modulus : float
            Young's modulus E.
        poisson_ratio : float
            Poisson's ratio nu.
        bkd : Backend[Array]
            Computational backend.
        body_force : Callable, optional
            Body force function.
        boundary_conditions : list of BoundaryConditionProtocol, optional
            Boundary conditions.
        """
        nelems = basis.skfem_basis().mesh.nelements
        return cls(
            basis=basis,
            material_map={"uniform": (youngs_modulus, poisson_ratio)},
            element_materials={"uniform": np.arange(nelems)},
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=boundary_conditions,
        )

    def __init__(
        self,
        basis: VectorLagrangeBasis[Array],
        material_map: Dict[str, Tuple[float, float]],
        element_materials: Dict[str, np.ndarray],
        bkd: Backend[Array],
        body_force: Optional[Callable[..., Any]] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._body_force = body_force
        self._material_map = dict(material_map)
        self._element_materials = {
            k: np.asarray(v) for k, v in element_materials.items()
        }
        self._material_names = list(material_map.keys())
        self._nmaterials = len(self._material_names)

        # Validate materials
        for name in self._material_names:
            E, nu = self._material_map[name]
            if not (-1.0 < nu < 0.5):
                raise ValueError(
                    f"Poisson ratio for material '{name}' must satisfy "
                    f"-1 < nu < 0.5, got {nu}"
                )
            if name not in self._element_materials:
                raise ValueError(f"Material '{name}' has no element assignment")

        # Build element-wise Lame parameter arrays
        nelems = basis.skfem_basis().mesh.nelements
        self._lam_per_elem = np.zeros(nelems)
        self._mu_per_elem = np.zeros(nelems)
        for name in self._material_names:
            E, nu = self._material_map[name]
            lam, mu = lame_parameters(E, nu)
            elem_idx = self._element_materials[name]
            self._lam_per_elem[elem_idx] = lam
            self._mu_per_elem[elem_idx] = mu

        # Cache for stiffness and mass matrices
        self._stiffness_cached: Optional[Array] = None
        self._mass_cached: Optional[Array] = None

        # For param_jacobian: build decomposed stiffness per material
        # K_lam_i and K_mu_i are the stiffness contributions from material i
        # when only lambda or mu for that material is 1 and all others are 0
        self._K_lam_per_material: List[Array] = []
        self._K_mu_per_material: List[Array] = []
        skfem_basis = self._basis.skfem_basis()
        nquad = skfem_basis.X.shape[1]
        for name in self._material_names:
            elem_idx = self._element_materials[name]
            # Lambda contribution for this material only
            lam_i = np.zeros((nelems, nquad))
            lam_i[elem_idx, :] = 1.0
            K_lam_i = asm(
                _elastic_form,
                skfem_basis,
                lam=lam_i,
                mu=np.zeros((nelems, nquad)),
            )
            self._K_lam_per_material.append(K_lam_i)

            # Mu contribution for this material only
            mu_i = np.zeros((nelems, nquad))
            mu_i[elem_idx, :] = 1.0
            K_mu_i = asm(
                _elastic_form,
                skfem_basis,
                lam=np.zeros((nelems, nquad)),
                mu=mu_i,
            )
            self._K_mu_per_material.append(K_mu_i)

    def _lame_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (nelem, nquad) Lame parameter arrays for assembly."""
        if self._lam_per_elem.ndim == 2:
            return self._lam_per_elem, self._mu_per_elem
        skfem_basis = self._basis.skfem_basis()
        nquad = skfem_basis.X.shape[1]
        lam = self._lam_per_elem[:, np.newaxis] * np.ones(nquad)
        mu = self._mu_per_elem[:, np.newaxis] * np.ones(nquad)
        return lam, mu

    def ndim(self) -> int:
        """Return spatial dimension."""
        return self._basis.ncomponents()

    def mass_matrix(self) -> Array:
        """Return the vector mass matrix.

        M_ij = integral(phi_i . phi_j)

        Returns
        -------
        Array
            Mass matrix. Shape: (nstates, nstates)
        """
        if self._mass_cached is not None:
            return self._mass_cached

        skfem_basis = self._basis.skfem_basis()

        def mass_form(u, v, w):
            return sum(u[i] * v[i] for i in range(len(u)))

        self._mass_cached = asm(BilinearForm(mass_form), skfem_basis)
        return self._mass_cached

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return solve_maybe_sparse(self._bkd, self.mass_matrix(), rhs)

    def stiffness_matrix(self) -> Array:
        """Return the elasticity stiffness matrix.

        Assembled with element-wise Lame parameters.

        Returns
        -------
        Array
            Stiffness matrix. Shape: (nstates, nstates)
        """
        if self._stiffness_cached is not None:
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()
        lam, mu = self._lame_data()
        self._stiffness_cached = asm(
            _elastic_form,
            skfem_basis,
            lam=lam,
            mu=mu,
        )
        return self._stiffness_cached

    def load_vector(self, time: float = 0.0) -> Array:
        """Return the load vector from body forces and Neumann/Robin BCs.

        b_i = integral(f . phi_i) + boundary integrals

        Parameters
        ----------
        time : float, default=0.0
            Current time.

        Returns
        -------
        Array
            Load vector. Shape: (nstates,)
        """
        skfem_basis = self._basis.skfem_basis()
        ndim = self._basis.ncomponents()

        if self._body_force is None:
            load_np = np.zeros(self.nstates())
        else:
            body_force_func = self._body_force
            current_time = time

            def linear_form(v, w):
                x = np.asarray(w.x)
                x_shape = x.shape
                if len(x_shape) == 3:
                    n, nelem, nquad = x_shape
                    x_flat = x.reshape(n, -1)
                    force_flat = np.asarray(body_force_func(x_flat, current_time))
                    force = force_flat.reshape(ndim, nelem, nquad)
                else:
                    force = np.asarray(body_force_func(x, current_time))
                return sum(force[i] * v[i] for i in range(ndim))

            load_np = asm(LinearForm(linear_form), skfem_basis)

        load = self._bkd.asarray(load_np.astype(np.float64))
        return self._apply_bc_to_load(load, time)

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual F = b(t) - K*u without BC enforcement.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        K = self.stiffness_matrix()
        b = self.load_vector(time)
        return b - K @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF/du = -K without BC enforcement.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        return -self.stiffness_matrix()

    def initial_condition(self, func: Callable[..., Any]) -> Array:
        """Create initial condition by interpolating a displacement field.

        Parameters
        ----------
        func : Callable
            Function taking (ndim, npts) returning (ndim, npts).

        Returns
        -------
        Array
            DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    # -----------------------------------------------------------------
    # Convenience properties for uniform-material access
    # -----------------------------------------------------------------

    def youngs_modulus(self) -> float:
        """Return Young's modulus (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "youngs_modulus is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        return self._material_map[self._material_names[0]][0]

    def poisson_ratio(self) -> float:
        """Return Poisson's ratio (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "poisson_ratio is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        return self._material_map[self._material_names[0]][1]

    def lame_lambda(self) -> float:
        """Return first Lame parameter (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_lambda is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        E, nu = self._material_map[self._material_names[0]]
        return lame_parameters(E, nu)[0]

    def lame_mu(self) -> float:
        """Return second Lame parameter (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_mu is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        E, nu = self._material_map[self._material_names[0]]
        return lame_parameters(E, nu)[1]

    # -----------------------------------------------------------------
    # Material accessors
    # -----------------------------------------------------------------

    def nmaterials(self) -> int:
        """Return the number of distinct materials."""
        return self._nmaterials

    def material_names(self) -> List[str]:
        """Return ordered list of material names."""
        return list(self._material_names)

    def material_params(self, name: str) -> Tuple[float, float]:
        """Return configured (E, nu) for the named material.

        Returns the values from the material map, not reverse-computed
        from current Lame parameters. May be stale if
        ``set_lame_parameters()`` was called directly.
        """
        return self._material_map[name]

    def element_materials(self) -> Dict[str, np.ndarray]:
        """Return element-to-material mapping {name: element_indices}."""
        return dict(self._element_materials)

    # -----------------------------------------------------------------
    # Sensitivity methods
    # -----------------------------------------------------------------

    def residual_lam_sensitivity(
        self,
        state: Array,
        material_idx: int,
    ) -> Array:
        """Sensitivity of spatial residual w.r.t. lambda for a material.

        Returns dF/d(lambda_i) evaluated at the current state, where
        F = b - K*u. This equals ``-K_lam_i @ state``, where K_lam_i is
        the unit-lambda stiffness contribution for material *material_idx*.

        Parameters
        ----------
        state : Array
            Current displacement. Shape: ``(nstates,)``.
        material_idx : int
            Material index (0-based).

        Returns
        -------
        Array
            Sensitivity vector. Shape: ``(nstates,)``.
        """
        return -(self._K_lam_per_material[material_idx] @ state)

    def residual_mu_sensitivity(
        self,
        state: Array,
        material_idx: int,
    ) -> Array:
        """Sensitivity of spatial residual w.r.t. mu for a material.

        Returns dF/d(mu_i) evaluated at the current state, where
        F = b - K*u. This equals ``-K_mu_i @ state``, where K_mu_i is
        the unit-mu stiffness contribution for material *material_idx*.

        Parameters
        ----------
        state : Array
            Current displacement. Shape: ``(nstates,)``.
        material_idx : int
            Material index (0-based).

        Returns
        -------
        Array
            Sensitivity vector. Shape: ``(nstates,)``.
        """
        return -(self._K_mu_per_material[material_idx] @ state)

    # -----------------------------------------------------------------
    # Parameter update methods
    # -----------------------------------------------------------------

    def set_lame_parameters(
        self,
        lam_per_elem: np.ndarray,
        mu_per_elem: np.ndarray,
    ) -> None:
        """Set per-element Lame parameters directly and invalidate cache.

        Parameters
        ----------
        lam_per_elem : np.ndarray
            First Lame parameter per element. Shape: ``(nelems,)`` for
            constant-per-element or ``(nelems, nquad)`` for per-quadrature-
            point values.
        mu_per_elem : np.ndarray
            Shear modulus per element. Same shape as ``lam_per_elem``.
        """
        self._lam_per_elem = np.asarray(lam_per_elem)
        self._mu_per_elem = np.asarray(mu_per_elem)
        self._stiffness_cached = None

    def __repr__(self) -> str:
        materials_str = ", ".join(
            f"{name}: E={E}, nu={nu}" for name, (E, nu) in self._material_map.items()
        )
        return (
            f"CompositeLinearElasticity("
            f"nstates={self.nstates()}, "
            f"materials={{{materials_str}}})"
        )
