"""Hyperelasticity physics for Galerkin FEM.

Solves the nonlinear hyperelasticity equations in total Lagrangian form:

    -Div(P(F)) = f    in Omega
    u = g              on Gamma_D
    P.N = t            on Gamma_N

where P is the first Piola-Kirchhoff stress, F = I + grad(u) is the
deformation gradient, and Div is the material divergence.

The weak form is:
    R(u; v) = integral P(F):Grad(v) dX - integral f.v dX - integral t.v dS = 0

The tangent stiffness (Jacobian dR/du) is:
    K(u)[du, v] = integral A:Grad(du):Grad(v) dX

where A_iJkL = dP_iJ/dF_kL is the material tangent modulus.

Uses the NeoHookeanStress model from the collocation module for stress

and tangent computation at quadrature points.
"""

from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.collocation.physics.stress_models.protocols import (
    StressModelProtocol,
    StressModelWithTangentProtocol,
)
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, LinearForm, asm
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class HyperelasticityPhysics(GalerkinPhysicsBase[Array]):
    """Nonlinear hyperelasticity physics for Galerkin FEM.

    Solves the quasi-static balance of linear momentum:
        -Div(P(F)) = f
    using the total Lagrangian formulation with first Piola-Kirchhoff
    stress P computed from a constitutive model (e.g., Neo-Hookean).

    The class supports 1D, 2D, and 3D problems. Analytical tangent
    stiffness (Jacobian) is available when the stress model implements
    StressModelWithTangentProtocol.

    Parameters
    ----------
    basis : VectorLagrangeBasis
        Vector finite element basis for displacement.
    stress_model : StressModelProtocol
        Constitutive model providing PK1 stress from deformation gradient.
    bkd : Backend
        Computational backend.
    body_force : Callable, optional
        Body force per unit volume. Takes coordinates (ndim, npts) and
        time (float), returns (ndim, npts).
    boundary_conditions : list of BoundaryConditionProtocol, optional
        Boundary conditions (Dirichlet, Neumann, Robin).
    """

    def __init__(
        self,
        basis: VectorLagrangeBasis[Array],
        stress_model: StressModelProtocol[Array],
        bkd: Backend[Array],
        body_force: Optional[Callable[..., Any]] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._stress_model = stress_model
        self._body_force = body_force
        self._numpy_bkd = NumpyBkd()

        # Cache mass matrix
        self._mass_cached: Optional[Array] = None

    def ndim(self) -> int:
        """Return spatial dimension."""
        return int(self._basis.ncomponents())

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
        """Solve M * x = rhs for x."""
        return solve_maybe_sparse(self._bkd, self.mass_matrix(), rhs)

    def _interpolate_state(self, state: Array) -> object:
        """Interpolate state onto the skfem basis for use in forms.

        Returns the skfem interpolated DiscreteField with per-component
        access (state_interp[i] for component i).
        """
        skfem_basis = self._basis.skfem_basis()
        state_np = self._bkd.to_numpy(state)
        return skfem_basis.interpolate(state_np)

    def _assemble_internal_force(self, state: Array, time: float) -> Array:
        """Assemble internal force vector.

        Computes: integral P(F):Grad(v) dX

        where F = I + Grad(u) and P is the PK1 stress.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate_state(state)
        ndim = self.ndim()
        stress_model = self._stress_model
        numpy_bkd = self._numpy_bkd

        if ndim == 1:

            def internal_force_1d(
                v: "DiscreteField", w: "FormExtraParams",
            ) -> np.ndarray:
                # F = 1 + du/dx; grad shape (1, 1, nelem, nquad)
                F = 1.0 + w.u_prev.grad[0, 0]
                P = stress_model.compute_stress_1d(F, numpy_bkd)
                ret: NDArray[np.floating[Any]] = P * v.grad[0, 0]
                return ret

            f_np = asm(
                LinearForm(internal_force_1d),
                skfem_basis,
                u_prev=state_interp,
            )

        elif ndim == 2:

            def internal_force_2d(
                v: "DiscreteField", w: "FormExtraParams",
            ) -> np.ndarray:
                # Deformation gradient F = I + grad(u)
                # w.u_prev.grad shape: (2, 2, nelem, nquad)
                F11 = 1.0 + w.u_prev.grad[0, 0]
                F12 = w.u_prev.grad[0, 1]
                F21 = w.u_prev.grad[1, 0]
                F22 = 1.0 + w.u_prev.grad[1, 1]
                P11, P12, P21, P22 = stress_model.compute_stress_2d(
                    F11, F12, F21, F22, numpy_bkd
                )
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
            )

        elif ndim == 3:

            def internal_force_3d(
                v: "DiscreteField", w: "FormExtraParams",
            ) -> np.ndarray:
                # Deformation gradient F = I + grad(u)
                # w.u_prev.grad shape: (3, 3, nelem, nquad)
                F = tuple(
                    tuple(
                        (1.0 if i == j else 0.0) + w.u_prev.grad[i, j] for j in range(3)
                    )
                    for i in range(3)
                )
                P = stress_model.compute_stress_3d(F, numpy_bkd)
                result = 0.0
                for i in range(3):
                    for j in range(3):
                        result = result + P[i][j] * v.grad[i, j]
                return result

            f_np = asm(
                LinearForm(internal_force_3d),
                skfem_basis,
                u_prev=state_interp,
            )

        else:
            raise ValueError(f"Unsupported dimension: {ndim}")

        return self._bkd.asarray(f_np.astype(np.float64))

    def _assemble_load(self, time: float) -> Array:
        """Assemble external load vector from body forces.

        Computes: integral f.v dX
        """
        if self._body_force is None:
            return self._bkd.asarray(np.zeros(self.nstates()))

        skfem_basis = self._basis.skfem_basis()
        ndim = self.ndim()
        body_force_func = self._body_force
        current_time = time

        def load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x = np.asarray(w.x)
            x_shape = x.shape
            if len(x_shape) == 3:
                n, nelem, nquad = x_shape
                x_flat = x.reshape(n, -1)
                force_flat = np.asarray(body_force_func(x_flat, current_time))
                force = force_flat.reshape(ndim, nelem, nquad)
            else:
                force = np.asarray(body_force_func(x, current_time))
            ret: NDArray[np.floating[Any]] = sum(force[i] * v[i] for i in range(ndim))
            return ret

        load_np = asm(LinearForm(load_form), skfem_basis)
        return self._bkd.asarray(load_np.astype(np.float64))

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        R(u) = load - internal_force - robin_stiffness * u

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
        internal_force = self._assemble_internal_force(state, time)
        load = self._assemble_load(time)
        load = self._apply_bc_to_load(load, time)

        # Robin BC stiffness contribution
        n = self.nstates()
        bc_stiffness = csr_matrix((n, n))
        bc_stiffness = self._apply_bc_to_stiffness(bc_stiffness, time)

        return load - internal_force - bc_stiffness @ state

    def _assemble_tangent_stiffness(self, state: Array, time: float) -> Array:
        """Assemble tangent stiffness matrix.

        Computes: K_ij = integral A_iJkL * dv_i/dX_J * du_k/dX_L dX

        where A_iJkL = dP_iJ/dF_kL is the material tangent modulus.

        Raises NotImplementedError if stress model does not implement
        StressModelWithTangentProtocol.
        """
        if not isinstance(self._stress_model, StressModelWithTangentProtocol):
            raise NotImplementedError(
                "Tangent stiffness requires a stress model implementing "
                "StressModelWithTangentProtocol. The provided model "
                f"({type(self._stress_model).__name__}) does not."
            )

        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate_state(state)
        ndim = self.ndim()
        stress_model = self._stress_model
        numpy_bkd = self._numpy_bkd

        if ndim == 1:

            def tangent_1d(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                F = 1.0 + w.u_prev.grad[0, 0]
                dPdF = stress_model.compute_tangent_1d(F, numpy_bkd)
                ret: NDArray[np.floating[Any]] = dPdF * v.grad[0, 0] * u.grad[0, 0]
                return ret

            K_np = asm(
                BilinearForm(tangent_1d),
                skfem_basis,
                u_prev=state_interp,
            )

        elif ndim == 2:

            def tangent_2d(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                F11 = 1.0 + w.u_prev.grad[0, 0]
                F12 = w.u_prev.grad[0, 1]
                F21 = w.u_prev.grad[1, 0]
                F22 = 1.0 + w.u_prev.grad[1, 1]
                A = stress_model.compute_tangent_2d(F11, F12, F21, F22, numpy_bkd)
                # K[du, v] = sum_{i,J,k,L} A_{iJkL} * dv_i/dX_J * du_k/dX_L
                result = 0.0
                for i in range(2):
                    for J in range(2):
                        for k in range(2):
                            for L in range(2):
                                key = f"A_{i + 1}{J + 1}{k + 1}{L + 1}"
                                result = result + A[key] * v.grad[i, J] * u.grad[k, L]
                return result

            K_np = asm(
                BilinearForm(tangent_2d),
                skfem_basis,
                u_prev=state_interp,
            )

        else:
            raise NotImplementedError(
                f"Tangent stiffness not available for {ndim}D. "
                "The stress model does not provide compute_tangent_3d."
            )

        return K_np

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dR/du without Dirichlet enforcement.

        For hyperelasticity, dR/du = -(tangent_stiffness + robin_stiffness).

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dR/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_tangent_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        return -stiffness

    def initial_condition(self, func: Callable[..., Any]) -> Array:
        """Create initial condition by interpolating a displacement field.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Takes coordinates (ndim, npts)
            and returns displacement (ndim, npts).

        Returns
        -------
        Array
            DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        return (
            f"HyperelasticityPhysics("
            f"nstates={self.nstates()}, "
            f"ndim={self.ndim()}, "
            f"stress_model={type(self._stress_model).__name__})"
        )
