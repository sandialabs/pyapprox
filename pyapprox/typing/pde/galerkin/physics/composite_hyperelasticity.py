"""Composite hyperelasticity with per-element material properties.

Solves the nonlinear hyperelasticity equations in total Lagrangian form:

    -Div(P(F)) = f    in Omega

where P is the first Piola-Kirchhoff stress computed from a Neo-Hookean
constitutive model with spatially-varying Lame parameters.

Each named subdomain can have different (E, nu) values, enabling
multi-material composite analysis (e.g., sandwich beams, layered structures).

When used with a single material covering all elements, this class is
functionally equivalent to the legacy HyperelasticityPhysics class.
"""

from typing import Callable, Dict, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
    DirichletBCProtocol,
    NeumannBCProtocol,
    RobinBCProtocol,
)
from pyapprox.typing.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.typing.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)

try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.models.elasticity import lame_parameters
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class CompositeHyperelasticityPhysics(Generic[Array]):
    """Nonlinear hyperelasticity with per-element material properties.

    Supports multi-material composites where each subdomain has its own
    Young's modulus and Poisson's ratio. The Neo-Hookean constitutive model
    is evaluated with element-wise Lame parameters at each quadrature point.

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
    Uniform material (backward-compatible with old HyperelasticityPhysics):

    >>> physics = CompositeHyperelasticityPhysics.from_uniform(
    ...     basis, youngs_modulus=1.0, poisson_ratio=0.3, bkd=bkd,
    ... )

    Multi-material composite:

    >>> physics = CompositeHyperelasticityPhysics(
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
    ) -> "CompositeHyperelasticityPhysics[Array]":
        """Create from uniform material properties.

        Convenience constructor that wraps a single material covering
        all elements.
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
        self._numpy_bkd = NumpyBkd()

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

        # Build element-wise Lame parameter arrays: (nelems,)
        nelems = basis.skfem_basis().mesh.nelements
        self._lam_per_elem = np.zeros(nelems)
        self._mu_per_elem = np.zeros(nelems)
        for name in self._material_names:
            E, nu = self._material_map[name]
            lam, mu = lame_parameters(E, nu)
            elem_idx = self._element_materials[name]
            self._lam_per_elem[elem_idx] = lam
            self._mu_per_elem[elem_idx] = mu

        # Stress model (parameters will be set per-evaluation)
        self._stress_model = NeoHookeanStress(lamda=0.0, mu=0.0)

        # Cache mass matrix
        self._mass_cached: Optional[Array] = None

    def _lame_qp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Lame parameters broadcast to (nelem, nquad) shape."""
        skfem_basis = self._basis.skfem_basis()
        nquad = skfem_basis.X.shape[1]
        lam = self._lam_per_elem[:, np.newaxis] * np.ones(nquad)
        mu = self._mu_per_elem[:, np.newaxis] * np.ones(nquad)
        return lam, mu

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def basis(self) -> VectorLagrangeBasis[Array]:
        return self._basis

    def nstates(self) -> int:
        return self._basis.ndofs()

    def ndim(self) -> int:
        return self._basis.ncomponents()

    def mass_matrix(self) -> Array:
        """Return the vector mass matrix M_ij = integral(phi_i . phi_j)."""
        if self._mass_cached is not None:
            return self._mass_cached

        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()

        def mass_form(u, v, w):
            return sum(u[i] * v[i] for i in range(ndim))

        mass_np = asm(BilinearForm(mass_form), skfem_basis).toarray()
        self._mass_cached = self._bkd.asarray(mass_np.astype(np.float64))
        return self._mass_cached

    def mass_solve(self, rhs: Array) -> Array:
        M = self.mass_matrix()
        return self._bkd.solve(M, rhs)

    def _interpolate_state(self, state: Array):
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        return skfem_basis.interpolate(state_np)

    def _assemble_internal_force(
        self, state: Array, time: float
    ) -> Array:
        """Assemble internal force: integral P(F):Grad(v) dX.

        Uses element-wise Lame parameters for the Neo-Hookean stress.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate_state(state)
        ndim = self.ndim()
        lam_qp, mu_qp = self._lame_qp()
        numpy_bkd = self._numpy_bkd

        if ndim == 1:

            def internal_force_1d(v, w):
                F = 1.0 + w.u_prev.grad[0, 0]
                J = F
                ln_J = np.log(J)
                P = w.mu * F + (w.lam * ln_J - w.mu) / J
                return P * v.grad[0, 0]

            f_np = asm(
                LinearForm(internal_force_1d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            )

        elif ndim == 2:

            def internal_force_2d(v, w):
                F11 = 1.0 + w.u_prev.grad[0, 0]
                F12 = w.u_prev.grad[0, 1]
                F21 = w.u_prev.grad[1, 0]
                F22 = 1.0 + w.u_prev.grad[1, 1]

                J = F11 * F22 - F12 * F21
                ln_J = np.log(J)
                coef = w.lam * ln_J - w.mu

                Finv_T_11 = F22 / J
                Finv_T_12 = -F21 / J
                Finv_T_21 = -F12 / J
                Finv_T_22 = F11 / J

                P11 = w.mu * F11 + coef * Finv_T_11
                P12 = w.mu * F12 + coef * Finv_T_12
                P21 = w.mu * F21 + coef * Finv_T_21
                P22 = w.mu * F22 + coef * Finv_T_22

                return (
                    P11 * v.grad[0, 0]
                    + P12 * v.grad[0, 1]
                    + P21 * v.grad[1, 0]
                    + P22 * v.grad[1, 1]
                )

            f_np = asm(
                LinearForm(internal_force_2d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            )

        elif ndim == 3:

            def internal_force_3d(v, w):
                F = tuple(
                    tuple(
                        (1.0 if i == j else 0.0)
                        + w.u_prev.grad[i, j]
                        for j in range(3)
                    )
                    for i in range(3)
                )
                F11, F12, F13 = F[0]
                F21, F22, F23 = F[1]
                F31, F32, F33 = F[2]

                J = (
                    F11 * (F22 * F33 - F23 * F32)
                    - F12 * (F21 * F33 - F23 * F31)
                    + F13 * (F21 * F32 - F22 * F31)
                )
                ln_J = np.log(J)
                coef = w.lam * ln_J - w.mu

                cof11 = F22 * F33 - F23 * F32
                cof12 = -(F21 * F33 - F23 * F31)
                cof13 = F21 * F32 - F22 * F31
                cof21 = -(F12 * F33 - F13 * F32)
                cof22 = F11 * F33 - F13 * F31
                cof23 = -(F11 * F32 - F12 * F31)
                cof31 = F12 * F23 - F13 * F22
                cof32 = -(F11 * F23 - F13 * F21)
                cof33 = F11 * F22 - F12 * F21

                P = (
                    (w.mu * F11 + coef * cof11 / J,
                     w.mu * F12 + coef * cof21 / J,
                     w.mu * F13 + coef * cof31 / J),
                    (w.mu * F21 + coef * cof12 / J,
                     w.mu * F22 + coef * cof22 / J,
                     w.mu * F23 + coef * cof32 / J),
                    (w.mu * F31 + coef * cof13 / J,
                     w.mu * F32 + coef * cof23 / J,
                     w.mu * F33 + coef * cof33 / J),
                )

                result = 0.0
                for i in range(3):
                    for j in range(3):
                        result = result + P[i][j] * v.grad[i, j]
                return result

            f_np = asm(
                LinearForm(internal_force_3d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            )

        else:
            raise ValueError(f"Unsupported dimension: {ndim}")

        return self._bkd.asarray(f_np.astype(np.float64))

    def _assemble_load(self, time: float) -> Array:
        """Assemble external load vector from body forces."""
        if self._body_force is None:
            return self._bkd.asarray(np.zeros(self.nstates()))

        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()
        body_force_func = self._body_force
        current_time = time

        def load_form(v, w):
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
                force = np.asarray(
                    body_force_func(x, current_time)
                )
            return sum(force[i] * v[i] for i in range(ndim))

        load_np = asm(LinearForm(load_form), skfem_basis)
        return self._bkd.asarray(load_np.astype(np.float64))

    def _assemble_tangent_stiffness(
        self, state: Array, time: float
    ) -> Array:
        """Assemble tangent stiffness: K[du, v] = integral A:Grad(du):Grad(v) dX.

        A_iJkL = dP_iJ/dF_kL with element-wise Lame parameters.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate_state(state)
        ndim = self.ndim()
        lam_qp, mu_qp = self._lame_qp()

        if ndim == 1:

            def tangent_1d(u, v, w):
                F = 1.0 + w.u_prev.grad[0, 0]
                J = F
                ln_J = np.log(J)
                dPdF = w.mu + (
                    w.mu + w.lam * (1.0 - ln_J)
                ) / (J ** 2)
                return dPdF * v.grad[0, 0] * u.grad[0, 0]

            K_np = asm(
                BilinearForm(tangent_1d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            ).toarray()

        elif ndim == 2:

            def tangent_2d(u, v, w):
                F11 = 1.0 + w.u_prev.grad[0, 0]
                F12 = w.u_prev.grad[0, 1]
                F21 = w.u_prev.grad[1, 0]
                F22 = 1.0 + w.u_prev.grad[1, 1]

                J = F11 * F22 - F12 * F21
                ln_J = np.log(J)

                beta = (w.lam * ln_J - w.mu) / J
                gamma = (w.mu + w.lam * (1.0 - ln_J)) / (J ** 2)

                # Build tangent A_iJkL inline and contract
                # A_iJkL * v.grad[i,J] * u.grad[k,L]
                A = {
                    "A_1111": w.mu + gamma * F22 ** 2,
                    "A_1112": -gamma * F21 * F22,
                    "A_1121": -gamma * F12 * F22,
                    "A_1122": beta + gamma * F11 * F22,
                    "A_1211": -gamma * F22 * F21,
                    "A_1212": w.mu + gamma * F21 ** 2,
                    "A_1221": -beta + gamma * F12 * F21,
                    "A_1222": -gamma * F11 * F21,
                    "A_2111": -gamma * F22 * F12,
                    "A_2112": -beta + gamma * F21 * F12,
                    "A_2121": w.mu + gamma * F12 ** 2,
                    "A_2122": -gamma * F11 * F12,
                    "A_2211": beta + gamma * F22 * F11,
                    "A_2212": -gamma * F21 * F11,
                    "A_2221": -gamma * F12 * F11,
                    "A_2222": w.mu + gamma * F11 ** 2,
                }

                result = 0.0
                for i in range(2):
                    for Jidx in range(2):
                        for k in range(2):
                            for L in range(2):
                                key = f"A_{i+1}{Jidx+1}{k+1}{L+1}"
                                result = (
                                    result
                                    + A[key]
                                    * v.grad[i, Jidx]
                                    * u.grad[k, L]
                                )
                return result

            K_np = asm(
                BilinearForm(tangent_2d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            ).toarray()

        else:
            raise NotImplementedError(
                f"Tangent stiffness not available for {ndim}D."
            )

        return self._bkd.asarray(K_np.astype(np.float64))

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute R = load - internal_force without Dirichlet enforcement."""
        internal_force = self._assemble_internal_force(state, time)
        load = self._assemble_load(time)

        # Apply Neumann/Robin BC contributions to load
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                load = bc.apply_to_load(load, time)
            elif isinstance(bc, NeumannBCProtocol):
                load = bc.apply_to_load(load, time)

        # Robin stiffness contribution
        n = self.nstates()
        bc_stiffness = self._bkd.asarray(np.zeros((n, n)))
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                bc_stiffness = bc.apply_to_stiffness(bc_stiffness, time)

        return load - internal_force - bc_stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dR/du without Dirichlet enforcement."""
        stiffness = self._assemble_tangent_stiffness(state, time)

        # Robin stiffness contribution
        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                stiffness = bc.apply_to_stiffness(stiffness, time)

        return -stiffness

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and values."""
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
        """Compute residual with Dirichlet BCs enforced."""
        residual = self.spatial_residual(state, time)

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                residual = bc.apply_to_residual(residual, state, time)

        return residual

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian with Dirichlet BCs enforced."""
        jac = self.spatial_jacobian(state, time)

        for bc in self._boundary_conditions:
            if isinstance(bc, RobinBCProtocol):
                continue
            if isinstance(bc, DirichletBCProtocol):
                jac = bc.apply_to_jacobian(jac, state, time)

        return jac

    def initial_condition(self, func: Callable) -> Array:
        """Create initial condition by interpolating a displacement field."""
        return self._basis.interpolate(func)

    # -----------------------------------------------------------------
    # Parameter sensitivity methods
    # -----------------------------------------------------------------

    def nparams(self) -> int:
        """Return number of material parameters (2 per material: E, nu)."""
        return 2 * self._nmaterials

    def set_param(self, param: Array) -> None:
        """Update material parameters.

        Parameters
        ----------
        param : Array
            [E1, nu1, E2, nu2, ...]. Shape: (2*nmaterials,)
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

    def initial_param_jacobian(self) -> Array:
        """Return d(u_0)/dp = 0."""
        n = self.nstates()
        return self._bkd.asarray(np.zeros((n, self.nparams())))

    # -----------------------------------------------------------------
    # Convenience properties for uniform-material access
    # -----------------------------------------------------------------

    @property
    def youngs_modulus(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "youngs_modulus is only defined for single-material problems."
            )
        return self._material_map[self._material_names[0]][0]

    @property
    def poisson_ratio(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "poisson_ratio is only defined for single-material problems."
            )
        return self._material_map[self._material_names[0]][1]

    @property
    def lame_lambda(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_lambda is only defined for single-material problems."
            )
        E, nu = self._material_map[self._material_names[0]]
        return lame_parameters(E, nu)[0]

    @property
    def lame_mu(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_mu is only defined for single-material problems."
            )
        E, nu = self._material_map[self._material_names[0]]
        return lame_parameters(E, nu)[1]

    def __repr__(self) -> str:
        materials_str = ", ".join(
            f"{name}: E={E}, nu={nu}"
            for name, (E, nu) in self._material_map.items()
        )
        return (
            f"CompositeHyperelasticityPhysics("
            f"nstates={self.nstates()}, "
            f"materials={{{materials_str}}})"
        )
