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

from typing import Callable, Dict, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.sparse_utils import solve_maybe_sparse
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)
from pyapprox.typing.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis

try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.helpers import sym_grad, ddot, eye, trace
    from skfem.models.elasticity import lame_parameters
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


@BilinearForm
def _elastic_form(u, v, w):
    """Elasticity bilinear form with element-wise Lame parameters.

    ``w.lam`` and ``w.mu`` must be ``(nelem, nquad)`` arrays.
    """
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return (
        w.lam * ddot(eye(trace(eps_u), eps_u.shape[0]), eps_v)
        + 2.0 * w.mu * ddot(eps_u, eps_v)
    )


class CompositeLinearElasticity(Generic[Array]):
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
        body_force: Optional[Callable] = None,
        boundary_conditions: Optional[
            List[BoundaryConditionProtocol[Array]]
        ] = None,
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
        body_force: Optional[Callable] = None,
        boundary_conditions: Optional[
            List[BoundaryConditionProtocol[Array]]
        ] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        self._boundary_conditions = boundary_conditions or []
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
                raise ValueError(
                    f"Material '{name}' has no element assignment"
                )

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
                _elastic_form, skfem_basis,
                lam=lam_i, mu=np.zeros((nelems, nquad)),
            )
            self._K_lam_per_material.append(K_lam_i)

            # Mu contribution for this material only
            mu_i = np.zeros((nelems, nquad))
            mu_i[elem_idx, :] = 1.0
            K_mu_i = asm(
                _elastic_form, skfem_basis,
                lam=np.zeros((nelems, nquad)), mu=mu_i,
            )
            self._K_mu_per_material.append(K_mu_i)

    def _lame_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (nelem, nquad) Lame parameter arrays for assembly."""
        skfem_basis = self._basis.skfem_basis()
        nquad = skfem_basis.X.shape[1]
        lam = self._lam_per_elem[:, np.newaxis] * np.ones(nquad)
        mu = self._mu_per_elem[:, np.newaxis] * np.ones(nquad)
        return lam, mu

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return total number of DOFs."""
        return self._basis.ndofs()

    def ndim(self) -> int:
        """Return spatial dimension."""
        return self._basis.ncomponents()

    def basis(self) -> VectorLagrangeBasis[Array]:
        """Return the vector finite element basis."""
        return self._basis

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
            _elastic_form, skfem_basis, lam=lam, mu=mu,
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
                    force_flat = np.asarray(
                        body_force_func(x_flat, current_time)
                    )
                    force = force_flat.reshape(ndim, nelem, nquad)
                else:
                    force = np.asarray(body_force_func(x, current_time))
                return sum(force[i] * v[i] for i in range(ndim))

            load_np = asm(LinearForm(linear_form), skfem_basis)

        load = self._bkd.asarray(load_np.astype(np.float64))

        # Apply Neumann and Robin BC contributions to load
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                load = bc.apply_to_load(load, time)
            elif isinstance(bc, NeumannBCProtocol):
                load = bc.apply_to_load(load, time)

        return load

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

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and their exact values.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices and dof_values.
        """
        all_dofs = []
        all_vals = []
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                dofs_np = self._bkd.to_numpy(bc.boundary_dofs())
                vals_np = self._bkd.to_numpy(bc.boundary_values(time))
                all_dofs.append(dofs_np)
                all_vals.append(vals_np)
        if all_dofs:
            return (
                self._bkd.asarray(
                    np.concatenate(all_dofs).astype(np.int64)
                ),
                self._bkd.asarray(
                    np.concatenate(all_vals).astype(np.float64)
                ),
            )
        return (
            self._bkd.asarray(np.array([], dtype=np.int64)),
            self._bkd.asarray(np.array([], dtype=np.float64)),
        )

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual F(u, t) = b(t) - K*u with BCs applied.

        Dirichlet BCs replace residual rows with state[dof] - g(x, t).

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        residual = self.spatial_residual(state, time)

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, time)

        return residual

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

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute state Jacobian dF/du = -K with BCs applied.

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
        jac = -self.stiffness_matrix()

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                jac = bc.apply_to_jacobian(jac, state, time)

        return jac

    def apply_boundary_conditions(
        self, residual: Array, jacobian: Array, state: Array
    ) -> Tuple[Array, Array]:
        """Apply boundary conditions to residual and Jacobian."""
        for bc in self._boundary_conditions:
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, 0.0)
                jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)
            elif isinstance(bc, RobinBCProtocol):
                residual = bc.apply_to_residual(residual, state, 0.0)
                jacobian = bc.apply_to_jacobian(jacobian, state, 0.0)
        return residual, jacobian

    def initial_condition(self, func: Callable) -> Array:
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

    @property
    def youngs_modulus(self) -> float:
        """Return Young's modulus (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "youngs_modulus is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        return self._material_map[self._material_names[0]][0]

    @property
    def poisson_ratio(self) -> float:
        """Return Poisson's ratio (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "poisson_ratio is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        return self._material_map[self._material_names[0]][1]

    @property
    def lame_lambda(self) -> float:
        """Return first Lame parameter (only valid for single-material)."""
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_lambda is only defined for single-material problems. "
                f"This problem has {self._nmaterials} materials."
            )
        E, nu = self._material_map[self._material_names[0]]
        return lame_parameters(E, nu)[0]

    @property
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
    # Parameter update methods
    # -----------------------------------------------------------------

    def set_lame_parameters(
        self, lam_per_elem: np.ndarray, mu_per_elem: np.ndarray,
    ) -> None:
        """Set per-element Lame parameters directly and invalidate cache.

        Parameters
        ----------
        lam_per_elem : np.ndarray
            First Lame parameter per element. Shape: (nelems,)
        mu_per_elem : np.ndarray
            Shear modulus per element. Shape: (nelems,)
        """
        self._lam_per_elem[:] = lam_per_elem
        self._mu_per_elem[:] = mu_per_elem
        self._stiffness_cached = None

    def nparams(self) -> int:
        """Return number of material parameters (2 per material: E, nu)."""
        return 2 * self._nmaterials

    def set_param(self, param: Array) -> None:
        """Update material parameters and invalidate stiffness cache.

        Parameters
        ----------
        param : Array
            Parameter vector [E1, nu1, E2, nu2, ...]. Shape: (2*nmaterials,)
        """
        param_np = self._bkd.to_numpy(param)

        for i, name in enumerate(self._material_names):
            E = float(param_np[2 * i])
            nu = float(param_np[2 * i + 1])
            if not (-1.0 < nu < 0.5):
                raise ValueError(
                    f"Poisson ratio for material '{name}' must satisfy "
                    f"-1 < nu < 0.5, got {nu}"
                )
            self._material_map[name] = (E, nu)
            lam, mu = lame_parameters(E, nu)
            elem_idx = self._element_materials[name]
            self._lam_per_elem[elem_idx] = lam
            self._mu_per_elem[elem_idx] = mu

        self._stiffness_cached = None

    def param_jacobian(self, state: Array, time: float) -> Array:
        """Compute parameter Jacobian dF/dp where F = b - K*u.

        Parameters are [E1, nu1, E2, nu2, ...] for each material.
        Uses chain rule through Lame parameters.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, 2*nmaterials)
        """
        nparams = self.nparams()
        n = self.nstates()
        cols = []

        for i, name in enumerate(self._material_names):
            E, nu = self._material_map[name]
            denom1 = (1.0 + nu) * (1.0 - 2.0 * nu)

            # dLambda/dE, dMu/dE
            dLambda_dE = nu / denom1
            dMu_dE = 1.0 / (2.0 * (1.0 + nu))

            # dLambda/dnu, dMu/dnu
            dLambda_dnu = E * (1.0 + 2.0 * nu**2) / denom1**2
            dMu_dnu = -E / (2.0 * (1.0 + nu) ** 2)

            K_lam_u = self._K_lam_per_material[i] @ state
            K_mu_u = self._K_mu_per_material[i] @ state

            # dF/dE_i = -(dLambda/dE * K_lam_i + dMu/dE * K_mu_i) @ u
            col_E = -(dLambda_dE * K_lam_u + dMu_dE * K_mu_u)
            # dF/dnu_i
            col_nu = -(dLambda_dnu * K_lam_u + dMu_dnu * K_mu_u)

            cols.extend([col_E, col_nu])

        pjac = self._bkd.stack(cols, axis=1)

        for bc in self._boundary_conditions:
            if hasattr(bc, "apply_to_param_jacobian"):
                pjac = bc.apply_to_param_jacobian(pjac, state, time)

        return pjac

    def initial_param_jacobian(self) -> Array:
        """Return d(u_0)/dp = 0 (IC does not depend on material params).

        Returns
        -------
        Array
            Zero matrix. Shape: (nstates, 2*nmaterials)
        """
        n = self.nstates()
        return self._bkd.asarray(np.zeros((n, self.nparams())))

    def __repr__(self) -> str:
        materials_str = ", ".join(
            f"{name}: E={E}, nu={nu}"
            for name, (E, nu) in self._material_map.items()
        )
        return (
            f"CompositeLinearElasticity("
            f"nstates={self.nstates()}, "
            f"materials={{{materials_str}}})"
        )
