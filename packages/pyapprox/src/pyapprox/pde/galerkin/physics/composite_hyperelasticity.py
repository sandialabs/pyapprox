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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField


import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.constitutive.coefficient_functions import as_time_aware
from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.boundary.flux_law import (
    NeoHookeanTraction,
    NormalFluxLawProtocol,
)
from pyapprox.pde.galerkin.physics.helpers import PerElementField
from pyapprox.pde.time_order import second_order_in_time_refusal
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse

try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.models.elasticity import lame_parameters
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class CompositeHyperelasticityPhysics(Generic[Array]):
    """Nonlinear hyperelasticity with per-element material properties.

    Supports multi-material composites where each subdomain has its own
    Young's modulus and Poisson's ratio. The Neo-Hookean constitutive model
    is evaluated with element-wise Lame parameters at each quadrature point.

    TIME THREADING (partial). The body force is normalized through
    ``as_time_aware`` and evaluated at the assembly's time, so a
    ``TimeDependent`` load works. The MATERIAL PROPERTIES are not:
    ``material_map`` fixes (E, nu) per subdomain at construction and the
    element-wise Lame parameters are precomputed from it, so a modulus
    that degrades with time cannot be expressed. Supporting it means
    making those per-element arrays a function of the assembly time
    rather than constants built once.

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
        Body force per unit volume, returning (ndim, npts). A steady
        force takes coordinates alone, ``f(coords)``. A force that
        varies in time must declare itself by wrapping in
        ``TimeDependent``; ``TimeIndependent`` declares the opposite for
        a callable that could accept a time but ignores it.

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
        body_force: Optional[Callable[..., Any]] = None,
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
        )

    def __init__(
        self,
        basis: VectorLagrangeBasis[Array],
        material_map: Dict[str, Tuple[float, float]],
        element_materials: Dict[str, np.ndarray],
        bkd: Backend[Array],
        body_force: Optional[Callable[..., Any]] = None,
    ):
        self._basis = basis
        self._bkd = bkd
        # Keep the raw supplier (consumers may inspect its type) and a
        # normalized companion that assembly evaluates uniformly as
        # f(coords, time).
        self._body_force = body_force
        self._body_force_eval = (
            None if body_force is None else as_time_aware(body_force)
        )
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
                raise ValueError(f"Material '{name}' has no element assignment")

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
        self._stress_model: NeoHookeanStress[Array] = NeoHookeanStress(
            lamda=0.0,
            mu=0.0,
        )

        # Cache mass matrix
        self._mass_cached: Optional[Array] = None

    def _lame_qp(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Lame parameters broadcast to (nelem, nquad) shape."""
        if self._lam_per_elem.ndim == 2:
            return self._lam_per_elem, self._mu_per_elem
        skfem_basis = self._basis.skfem_basis()
        nquad = skfem_basis.X.shape[1]
        lam = self._lam_per_elem[:, np.newaxis] * np.ones(nquad)
        mu = self._mu_per_elem[:, np.newaxis] * np.ones(nquad)
        return lam, mu

    def ndim(self) -> int:
        return int(self._basis.ncomponents())

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """No curvature supplied: the stress tangent's derivative is not
        assembled, so consumers stay at the Jacobian tier."""
        return StateDerivatives.none()

    def interior_is_time_invariant(self) -> bool:
        """Whether the body force is declared time-independent (the
        material properties cannot depend on time)."""
        return (
            self._body_force_eval is None
            or not self._body_force_eval.is_time_dependent()
        )

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Always: the material properties cannot depend on time, and the
        body force does not enter the Jacobian."""
        return True

    def _lamda_per_element(self) -> np.ndarray:
        """The current first Lame parameter of each element."""
        return self._lam_per_elem

    def _mu_per_element(self) -> np.ndarray:
        """The current shear modulus of each element."""
        return self._mu_per_elem

    def flux_law(self) -> NormalFluxLawProtocol:
        r"""The flux the weak form integrates by parts: the Neo-Hookean
        nominal traction :math:`P(F)\,N`, with each point taking its
        element's material.

        Raises at evaluation if the Lame values were set per quadrature
        point, since those have no value away from the quadrature points.
        """
        mesh = self._basis.skfem_basis().mesh
        return NeoHookeanTraction(
            PerElementField(self._lamda_per_element, mesh),
            PerElementField(self._mu_per_element, mesh),
        )

    def mass_matrix(self) -> Array:
        """Return the vector mass matrix M_ij = integral(phi_i . phi_j)."""
        if self._mass_cached is not None:
            return self._mass_cached

        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()

        def mass_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = sum(u[i] * v[i] for i in range(ndim))
            return ret

        self._mass_cached = asm(BilinearForm(mass_form), skfem_basis)
        return self._mass_cached

    def mass_solve(self, rhs: Array) -> Array:
        return solve_maybe_sparse(self._bkd, self.mass_matrix(), rhs)

    def first_order_transient_refusal(self) -> str:
        """Return why a first-order transient path refuses this physics.

        Satisfies ``SecondOrderInTimeProtocol``: elasticity is second
        order in time, so ``M du/dt = F(u, t)`` is not its dynamics.
        """
        return second_order_in_time_refusal(type(self).__name__)

    def _interpolate_state(self, state: Array) -> object:
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        return skfem_basis.interpolate(state_np)

    def _assemble_internal_force(self, state: Array, time: float) -> Array:
        """Assemble internal force: integral P(F):Grad(v) dX.

        Uses element-wise Lame parameters for the Neo-Hookean stress.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate_state(state)
        ndim = self.ndim()
        lam_qp, mu_qp = self._lame_qp()

        if ndim == 1:

            def internal_force_1d(
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                F = 1.0 + w.u_prev.grad[0, 0]
                J = F
                ln_J = np.log(J)
                P = w.mu * F + (w.lam * ln_J - w.mu) / J
                ret: NDArray[np.floating[Any]] = P * v.grad[0, 0]
                return ret

            f_np = asm(
                LinearForm(internal_force_1d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            )

        elif ndim == 2:

            def internal_force_2d(
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
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

                ret: NDArray[np.floating[Any]] = (
                    P11 * v.grad[0, 0]
                    + P12 * v.grad[0, 1]
                    + P21 * v.grad[1, 0]
                    + P22 * v.grad[1, 1]
                )
                return ret

            f_np = asm(
                LinearForm(internal_force_2d),
                skfem_basis,
                u_prev=state_interp,
                lam=lam_qp,
                mu=mu_qp,
            )

        elif ndim == 3:

            def internal_force_3d(
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                F = tuple(
                    tuple(
                        (1.0 if i == j else 0.0) + w.u_prev.grad[i, j] for j in range(3)
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

                # F^{-1} = cof^T / J, so F^{-T}_{ij} = cof_{ij} / J
                P = (
                    (
                        w.mu * F11 + coef * cof11 / J,
                        w.mu * F12 + coef * cof12 / J,
                        w.mu * F13 + coef * cof13 / J,
                    ),
                    (
                        w.mu * F21 + coef * cof21 / J,
                        w.mu * F22 + coef * cof22 / J,
                        w.mu * F23 + coef * cof23 / J,
                    ),
                    (
                        w.mu * F31 + coef * cof31 / J,
                        w.mu * F32 + coef * cof32 / J,
                        w.mu * F33 + coef * cof33 / J,
                    ),
                )

                # Annotated because skfem's DiscreteField is untyped, so
                # zeros_like() would otherwise make the accumulator Any
                # and leak it through the declared return type.
                result: NDArray[np.floating[Any]] = np.zeros_like(v.grad[0, 0])
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
        if self._body_force_eval is None:
            return self._bkd.asarray(np.zeros(self.nstates()))

        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()
        # Evaluated once at the quadrature points, (ndim, nelem, nquad):
        # skfem calls the form once per element basis function.
        x: NDArray[np.floating[Any]] = np.asarray(
            skfem_basis.global_coordinates().value
        )
        force = np.asarray(
            self._body_force_eval(x.reshape(ndim, -1), time)
        ).reshape(x.shape)

        def load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            ret: NDArray[np.floating[Any]] = sum(force[i] * v[i] for i in range(ndim))
            return ret

        load_np = asm(LinearForm(load_form), skfem_basis)
        return self._bkd.asarray(load_np.astype(np.float64))

    def _assemble_tangent_stiffness(self, state: Array, time: float) -> Array:
        """Assemble tangent stiffness: K[du, v] = integral A:Grad(du):Grad(v) dX.

        A_iJkL = dP_iJ/dF_kL with element-wise Lame parameters.
        """
        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()
        if ndim not in (1, 2, 3):
            raise NotImplementedError(f"Tangent stiffness not available for {ndim}D.")
        lam_qp, mu_qp = self._lame_qp()
        # The modulus depends on the state alone, so it is evaluated once
        # here, shape (ndim, ndim, ndim, ndim, nelem, nquad), in any
        # dimension:
        #   A_iJkL = mu*d_ik*d_JL + (mu - lam*lnJ)*Finv_Jk*Finv_Li
        #            + lam*Finv_Ji*Finv_Lk.
        # skfem calls the form once per pair of element basis functions;
        # evaluating F, J and F^{-1} inside it repeated that work for every
        # pair.
        displacement = skfem_basis.interpolate(self._bkd.to_numpy(state))
        deformation: NDArray[np.floating[Any]] = np.asarray(
            displacement.grad
        ) + np.eye(ndim).reshape(ndim, ndim, 1, 1)
        # Matrix axes last for det and inv, then back to leading.
        matrices = np.moveaxis(deformation, (0, 1), (-2, -1))
        ln_j = np.log(np.linalg.det(matrices))
        finv = np.moveaxis(np.linalg.inv(matrices), (-2, -1), (0, 1))
        identity = np.eye(ndim)
        modulus = (
            mu_qp * np.einsum("ik,JL->iJkL", identity, identity)[..., None, None]
            + (mu_qp - lam_qp * ln_j) * np.einsum("Jk...,Li...->iJkL...", finv, finv)
            + lam_qp * np.einsum("Ji...,Lk...->iJkL...", finv, finv)
        )

        def tangent(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            # K[du, v] = sum_{i,J,k,L} A_{iJkL} * dv_i/dX_J * du_k/dX_L
            ret: NDArray[np.floating[Any]] = np.einsum(
                "ij...,ijkl...,kl...->...", v.grad, modulus, u.grad
            )
            return ret

        K_np: Array = asm(BilinearForm(tangent), skfem_basis)
        return K_np

    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute F_Omega = body load - internal_force (no BCs)."""
        internal_force = self._assemble_internal_force(state, time)
        load = self._assemble_load(time)
        return load - internal_force

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF_Omega/du (no BCs)."""
        stiffness = self._assemble_tangent_stiffness(state, time)
        return -stiffness

    def initial_condition(self, func: Callable[..., Any]) -> Array:
        """Create initial condition by interpolating a displacement field."""
        return self._basis.interpolate(func)

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
    # Parameter update methods
    # -----------------------------------------------------------------

    def set_lame_parameters(
        self,
        lam_per_elem: np.ndarray,
        mu_per_elem: np.ndarray,
    ) -> None:
        """Set per-element Lame parameters directly.

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

    def set_lame_material_values(self, values: np.ndarray) -> None:
        """Set per-material Lame values ``[lam_1, mu_1, lam_2, mu_2, ...]``.

        Expands the interleaved per-material values to per-element arrays
        via the material-to-element mapping and delegates to
        ``set_lame_parameters``.

        Parameters
        ----------
        values : np.ndarray
            Interleaved per-material Lame values.
            Shape: ``(2*nmaterials,)``.
        """
        values = np.asarray(values)
        if values.shape != (2 * self._nmaterials,):
            raise ValueError(
                f"values must have shape ({2 * self._nmaterials},), got {values.shape}"
            )
        nelems = self._lam_per_elem.shape[0]
        lam_per_elem = np.zeros(nelems)
        mu_per_elem = np.zeros(nelems)
        for i, name in enumerate(self._material_names):
            elem_idx = self._element_materials[name]
            lam_per_elem[elem_idx] = values[2 * i]
            mu_per_elem[elem_idx] = values[2 * i + 1]
        self.set_lame_parameters(lam_per_elem, mu_per_elem)

    # -----------------------------------------------------------------
    # Convenience properties for uniform-material access
    # -----------------------------------------------------------------

    def youngs_modulus(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "youngs_modulus is only defined for single-material problems."
            )
        return self._material_map[self._material_names[0]][0]

    def poisson_ratio(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "poisson_ratio is only defined for single-material problems."
            )
        return self._material_map[self._material_names[0]][1]

    def lame_lambda(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_lambda is only defined for single-material problems."
            )
        E, nu = self._material_map[self._material_names[0]]
        return float(lame_parameters(E, nu)[0])

    def lame_mu(self) -> float:
        if self._nmaterials != 1:
            raise AttributeError(
                "lame_mu is only defined for single-material problems."
            )
        E, nu = self._material_map[self._material_names[0]]
        return float(lame_parameters(E, nu)[1])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> VectorLagrangeBasis[Array]:
        """Return the vector finite element basis."""
        return self._basis

    def nstates(self) -> int:
        """Return the number of DOFs."""
        return self._basis.ndofs()

    def __repr__(self) -> str:
        materials_str = ", ".join(
            f"{name}: E={E}, nu={nu}" for name, (E, nu) in self._material_map.items()
        )
        return (
            f"CompositeHyperelasticityPhysics("
            f"nstates={self.nstates()}, "
            f"materials={{{materials_str}}})"
        )
