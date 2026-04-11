"""Stokes and Navier-Stokes physics for Galerkin FEM.

Solves the Stokes equations:
    -viscosity * div(grad(u)) + grad(p) = f_vel     (momentum)
    -div(u) = f_pres                                 (continuity)

Or Navier-Stokes (adds convective term):
    -viscosity * div(grad(u)) + (u . grad)u + grad(p) = f_vel
    -div(u) = f_pres

Uses Taylor-Hood elements: P2 velocity, P1 pressure.
State vector layout: [vel_dofs | pres_dofs].
"""

from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Tuple

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import block_diag as sp_block_diag
from scipy.sparse import csr_matrix

from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.boundary.implementations import (
    CallableDirichletBC,
)
from pyapprox.pde.galerkin.physics.bc_mixin import GalerkinBCMixin
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse

try:
    from skfem import Basis, BilinearForm, LinearForm, asm, bmat
    from skfem.models.general import divergence
    from skfem.models.poisson import vector_laplace
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class _VelComponentBCValueFunc:
    """Picklable adapter that extracts a single velocity component.

    Wraps a user-supplied BC callable ``func(crds, time) -> vals`` so
    that the returned array is a 1D slice corresponding to component
    ``comp_idx``. Replaces the local ``value_func`` closure previously
    built inside
    :meth:`StokesPhysics._build_boundary_conditions`; keeps
    ``StokesPhysics`` picklable when the underlying BC callable is
    itself picklable.
    """

    def __init__(
        self,
        func: Callable[..., np.ndarray],
        crds: np.ndarray,
        comp_idx: int,
    ) -> None:
        self._func = func
        self._crds = crds
        self._comp_idx = int(comp_idx)

    def __call__(self, time: float) -> np.ndarray:
        try:
            vals = self._func(self._crds, time)
        except TypeError:
            vals = self._func(self._crds)
        if vals.ndim == 2:
            return vals[:, self._comp_idx]
        return vals


class _PresBCValueFunc:
    """Picklable adapter for scalar pressure BC callables.

    Replaces the local ``value_func`` closure previously built inside
    :meth:`StokesPhysics._build_boundary_conditions`.
    """

    def __init__(
        self,
        func: Callable[..., np.ndarray],
        crds: np.ndarray,
    ) -> None:
        self._func = func
        self._crds = crds

    def __call__(self, time: float) -> np.ndarray:
        try:
            vals = self._func(self._crds, time)
        except TypeError:
            vals = self._func(self._crds)
        if vals.ndim == 2:
            return vals.flatten()
        return vals


class StokesPhysics(GalerkinBCMixin[Array], Generic[Array]):
    """Stokes/Navier-Stokes physics for Galerkin FEM.

    Solves the Stokes equations:
        -viscosity * Laplacian(u) + grad(p) = f_vel
        -div(u) = f_pres

    or Navier-Stokes (adds convective term (u . grad)u).

    Uses Taylor-Hood elements (P2 velocity, P1 pressure).
    State vector layout: [vel_dofs | pres_dofs].

    Parameters
    ----------
    vel_basis : VectorLagrangeBasis
        Vector finite element basis for velocity (typically P2).
    pres_basis : LagrangeBasis
        Scalar finite element basis for pressure (typically P1).
    bkd : Backend
        Computational backend.
    navier_stokes : bool
        If True, include convective (nonlinear) terms.
    viscosity : float
        Kinematic viscosity.
    vel_forcing : Callable, optional
        Velocity forcing function. Takes (npts, ndim) points and returns
        (npts, nvars) forcing values. For transient problems, takes
        (npts, ndim) points and float time.
    pres_forcing : Callable, optional
        Pressure (continuity) forcing function. Takes (npts,) or
        (npts, ndim) points and returns (npts,). For transient, takes
        points and float time.
    vel_dirichlet_bcs : list of (str, Callable), optional
        Velocity Dirichlet BCs. Each tuple is (boundary_name, func) where
        func takes (ndim, npts) coordinates (and optional time) and returns
        (npts, nvars) velocity values at boundary nodes.
    pres_dirichlet_bcs : list of (str, Callable), optional
        Pressure Dirichlet BCs. Each tuple is (boundary_name, func) where
        func takes (ndim, npts) coordinates (and optional time) and returns
        (npts,) pressure values.
    """

    def __init__(
        self,
        vel_basis: VectorLagrangeBasis[Array],
        pres_basis: LagrangeBasis[Array],
        bkd: Backend[Array],
        navier_stokes: bool = False,
        viscosity: float = 1.0,
        vel_forcing: Optional[Callable[..., Any]] = None,
        pres_forcing: Optional[Callable[..., Any]] = None,
        vel_dirichlet_bcs: Optional[List[Tuple[str, Callable[..., Any]]]] = None,
        pres_dirichlet_bcs: Optional[List[Tuple[str, Callable[..., Any]]]] = None,
    ):
        self._vel_basis = vel_basis
        self._pres_basis = pres_basis
        self._bkd = bkd
        self._navier_stokes = navier_stokes
        self._viscosity = viscosity
        self._vel_forcing = vel_forcing
        self._pres_forcing = pres_forcing
        self._vel_dirichlet_bcs = vel_dirichlet_bcs or []
        self._pres_dirichlet_bcs = pres_dirichlet_bcs or []

        self._ndim = vel_basis.ncomponents()

        # Create skfem bases with intorder=4 for assembly accuracy
        skfem_mesh = vel_basis.mesh().skfem_mesh()
        self._vel_skfem_basis = Basis(
            skfem_mesh, vel_basis.skfem_basis().elem, intorder=4
        )
        self._pres_skfem_basis = Basis(
            skfem_mesh, pres_basis.skfem_basis().elem, intorder=4
        )

        # Cache assembled operators
        self._A_cached = None  # viscous stiffness (sparse)
        self._B_cached = None  # divergence operator (sparse)
        self._vel_mass_cached: Optional[Array] = None
        self._mass_cached: Optional[Array] = None

        # Convert tuple BCs to CallableDirichletBC objects for mixin
        self._boundary_conditions = self._build_boundary_conditions()

    def _build_boundary_conditions(self) -> List[Any]:
        """Convert tuple-based BCs to CallableDirichletBC objects."""
        bcs = []
        nvars = self._ndim

        for bndry_name, bndry_func in self._vel_dirichlet_bcs:
            for idx in range(nvars):
                # Extract DOF indices for this component
                if nvars >= 2:
                    dofnames = self._vel_skfem_basis.get_dofs().obj.element.dofnames
                    skip = dofnames[nvars - idx - 1]
                    bndry_dofs = self._vel_skfem_basis.get_dofs(bndry_name, skip=skip)
                else:
                    bndry_dofs = self._vel_skfem_basis.get_dofs(bndry_name)

                dof_arr = np.asarray(bndry_dofs).flatten()
                coords = self._vel_skfem_basis.doflocs[:, dof_arr]

                bcs.append(
                    CallableDirichletBC(
                        dof_arr,
                        _VelComponentBCValueFunc(bndry_func, coords, idx),
                        self._bkd,
                    )
                )

        for bndry_name, bndry_func in self._pres_dirichlet_bcs:
            p_dofs = self._pres_skfem_basis.get_dofs(bndry_name)
            p_dofs_arr = np.asarray(p_dofs).flatten()
            shifted_p_dofs = p_dofs_arr + self.vel_ndofs()
            coords = self._pres_skfem_basis.doflocs[:, p_dofs_arr]

            bcs.append(
                CallableDirichletBC(
                    shifted_p_dofs,
                    _PresBCValueFunc(bndry_func, coords),
                    self._bkd,
                )
            )

        return bcs

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return total number of DOFs (velocity + pressure)."""
        return self._vel_basis.ndofs() + self._pres_basis.ndofs()

    def ndim(self) -> int:
        """Return spatial dimension."""
        return self._ndim

    def vel_ndofs(self) -> int:
        """Return number of velocity DOFs."""
        return self._vel_basis.ndofs()

    def pres_ndofs(self) -> int:
        """Return number of pressure DOFs."""
        return self._pres_basis.ndofs()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_viscous_stiffness(self) -> csr_matrix:
        """Assemble viscous stiffness A = viscosity * vector_laplace."""
        if self._A_cached is None:
            self._A_cached = self._viscosity * asm(
                vector_laplace, self._vel_skfem_basis
            )
        return self._A_cached

    def _assemble_divergence(self) -> csr_matrix:
        """Assemble divergence operator B = -divergence(vel, pres)."""
        if self._B_cached is None:
            self._B_cached = -asm(
                divergence, self._vel_skfem_basis, self._pres_skfem_basis
            )
        return self._B_cached

    def _assemble_block_stiffness(
        self, state_np: Optional[np.ndarray] = None,
    ) -> csr_matrix:
        """Assemble the full block stiffness matrix.

        For Stokes: K = [[A, B^T], [B, 0]]
        For Navier-Stokes: K = [[A + A_nl, B^T], [B, 0]]

        Returns sparse CSR matrix.
        """
        A = self._assemble_viscous_stiffness()
        B = self._assemble_divergence()

        if self._navier_stokes and state_np is not None:
            vel_state = state_np[: self.vel_ndofs()]
            vel_interp = self._vel_skfem_basis.interpolate(vel_state)
            A_nl = asm(
                BilinearForm(self._ns_linearized_form),
                self._vel_skfem_basis,
                u_prev=vel_interp,
            )
            return bmat([[A + A_nl, B.T], [B, None]], "csr")

        return bmat([[A, B.T], [B, None]], "csr")

    def _prepare_points(
        self, x_np: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Convert skfem quadrature points to manufactured solution format.

        Parameters
        ----------
        x_np : np.ndarray
            Points from skfem. Shape: (ndim, nelem, nquad) or (ndim, npts).

        Returns
        -------
        x_eval : np.ndarray
            Points for manufactured solution eval.
            1D: (npts,), 2D+: (ndim, npts).
        orig_shape : tuple
            Original shape for reshaping results back.
        """
        if len(x_np.shape) == 3:
            ndim, nelem, nquad = x_np.shape
            x_flat = x_np.reshape(ndim, -1)  # (ndim, npts)
            orig_shape = (nelem, nquad)
        else:
            x_flat = x_np
            orig_shape = None

        # 1D manufactured functions expect (npts,) not (1, npts)
        if self._ndim == 1:
            x_eval = x_flat[0]  # (npts,)
        else:
            x_eval = x_flat  # (ndim, npts)

        return x_eval, orig_shape

    def _assemble_vel_load(self, time: float) -> np.ndarray:
        """Assemble velocity load vector."""
        if self._vel_forcing is None:
            return np.zeros(self.vel_ndofs())

        vel_forcing_func = self._vel_forcing
        nvars = self._ndim
        current_time = time
        prepare_points = self._prepare_points

        def vel_load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_eval, orig_shape = prepare_points(x_np)

            try:
                f = vel_forcing_func(x_eval, current_time)
            except TypeError:
                f = vel_forcing_func(x_eval)
            # f shape: (npts, nvars)

            if orig_shape is not None:
                nelem, nquad = orig_shape
                f = f.T.reshape(nvars, nelem, nquad)
            else:
                f = f.T  # (nvars, npts)
            ret: NDArray[np.floating[Any]] = sum(f[i] * v[i] for i in range(nvars))
            return ret

        vel_load: NDArray[np.floating[Any]] = asm(
            LinearForm(vel_load_form), self._vel_skfem_basis
        )
        return vel_load

    def _assemble_pres_load(self, time: float) -> np.ndarray:
        """Assemble pressure load vector (negated for convention)."""
        if self._pres_forcing is None:
            return np.zeros(self.pres_ndofs())

        pres_forcing_func = self._pres_forcing
        current_time = time
        prepare_points = self._prepare_points

        def pres_load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_eval, orig_shape = prepare_points(x_np)

            try:
                f = pres_forcing_func(x_eval, current_time)
            except TypeError:
                f = pres_forcing_func(x_eval)
            # f shape: (npts,)

            if orig_shape is not None:
                nelem, nquad = orig_shape
                f = f.reshape(nelem, nquad)
            ret: NDArray[np.floating[Any]] = f * v
            return ret

        # Negate: convention is -div(u) = f_p
        pres_load: NDArray[np.floating[Any]] = -asm(
            LinearForm(pres_load_form),
            self._pres_skfem_basis,
        )
        return pres_load

    def _assemble_load(self, time: float) -> np.ndarray:
        """Assemble full load vector [vel_load, pres_load]."""
        vel_load = self._assemble_vel_load(time)
        pres_load = self._assemble_pres_load(time)
        return np.concatenate([vel_load, pres_load])

    # ------------------------------------------------------------------
    # Navier-Stokes bilinear forms
    # ------------------------------------------------------------------

    @staticmethod
    def _ns_linearized_form(
        u: "DiscreteField", v: "DiscreteField", w: "FormExtraParams",
    ) -> np.ndarray:
        """Linearized Navier-Stokes convective term for Jacobian.

        Computes v_i*(u_j*dz_i/dx_j + z_j*du_i/dx_j) where z = u_prev.
        """
        z = w["u_prev"]
        dz = z.grad
        du = u.grad
        if u.shape[0] == 2:
            ret: NDArray[np.floating[Any]] = (
                v[0] * (u[0] * dz[0][0] + u[1] * dz[0][1])
                + v[1] * (u[0] * dz[1][0] + u[1] * dz[1][1])
                + v[0] * (z[0] * du[0][0] + z[1] * du[0][1])
                + v[1] * (z[0] * du[1][0] + z[1] * du[1][1])
            )
            return ret
        if u.shape[0] == 1:
            ret1d: NDArray[np.floating[Any]] = (
                v[0] * (u[0] * dz[0][0])
                + v[0] * (z[0] * du[0][0])
            )
            return ret1d
        raise NotImplementedError("Only 1D and 2D Navier-Stokes supported")

    @staticmethod
    def _ns_nonlinear_residual_form(
        v: "DiscreteField", w: "FormExtraParams",
    ) -> np.ndarray:
        """Nonlinear Navier-Stokes residual contribution.

        Computes v_i*(u_j*du_i/dx_j) where u = u_prev.
        """
        u = w["u_prev"]
        du = u.grad
        if u.shape[0] == 2:
            ret: NDArray[np.floating[Any]] = (
                v[0]
                * (u[0] * du[0][0] + u[1] * du[0][1])
                + v[1]
                * (u[0] * du[1][0] + u[1] * du[1][1])
            )
            return ret
        if u.shape[0] == 1:
            ret1d: NDArray[np.floating[Any]] = v[0] * (u[0] * du[0][0])
            return ret1d
        raise NotImplementedError("Only 1D and 2D Navier-Stokes supported")

    # ------------------------------------------------------------------
    # Physics protocol methods
    # ------------------------------------------------------------------

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        Computes the block assembly:
            F_vel = vel_load - A*vel - B^T*pres [- NS_nonlinear_term]
            F_pres = pres_load - B*vel

        Dirichlet BCs are NOT applied.

        Parameters
        ----------
        state : Array
            Solution state [vel_dofs | pres_dofs]. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        state_np = self._bkd.to_numpy(state)
        vel_state = state_np[: self.vel_ndofs()]
        pres_state = state_np[self.vel_ndofs() :]

        A = self._assemble_viscous_stiffness()
        B = self._assemble_divergence()
        load = self._assemble_load(time)

        # Residual: load - K*state
        b_vel = -(A.dot(vel_state) + B.T.dot(pres_state))
        b_pres = -B.dot(vel_state)

        if self._navier_stokes:
            vel_interp = self._vel_skfem_basis.interpolate(vel_state)
            ns_residual = asm(
                LinearForm(self._ns_nonlinear_residual_form),
                self._vel_skfem_basis,
                u_prev=vel_interp,
            )
            b_vel -= ns_residual

        residual_np = load + np.concatenate([b_vel, b_pres])

        return self._bkd.asarray(residual_np.astype(np.float64))

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian dF/du = -K without Dirichlet enforcement.

        Parameters
        ----------
        state : Array
            Solution state [vel_dofs | pres_dofs]. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        state_np = self._bkd.to_numpy(state)
        K = self._assemble_block_stiffness(state_np)
        return -K

    def residual(self, state: Array, time: float) -> Array:
        """Compute residual F(u, t) = load - K*u with BCs applied.

        Dirichlet BCs replace rows with: state[dof] - exact_value.

        Parameters
        ----------
        state : Array
            Solution state [vel_dofs | pres_dofs]. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual vector. Shape: (nstates,)
        """
        res = self.spatial_residual(state, time)
        return self._apply_dirichlet_to_residual(res, state, time)

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute Jacobian dF/du = -K with BCs applied.

        For Dirichlet BC rows, the Jacobian is set to identity.

        Parameters
        ----------
        state : Array
            Solution state [vel_dofs | pres_dofs]. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        jac = self.spatial_jacobian(state, time)
        return self._apply_dirichlet_to_jacobian(jac, state, time)

    def vel_mass_matrix(self) -> Array:
        """Return velocity mass matrix M_vel.

        Returns
        -------
        Array
            Velocity mass matrix. Shape: (vel_ndofs, vel_ndofs)
        """
        if self._vel_mass_cached is not None:
            return self._vel_mass_cached

        def vector_mass_form(
            u: "DiscreteField", v: "DiscreteField", w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = sum(u[i] * v[i] for i in range(len(u)))
            return ret

        self._vel_mass_cached = asm(
            BilinearForm(vector_mass_form), self._vel_skfem_basis
        )
        return self._vel_mass_cached

    def mass_matrix(self) -> Array:
        """Return block mass matrix [M_vel, 0; 0, 0].

        The pressure block is zero (DAE structure).

        Returns
        -------
        Array
            Block mass matrix. Shape: (nstates, nstates)
        """
        if self._mass_cached is not None:
            return self._mass_cached

        M_vel = self.vel_mass_matrix()
        M_pres = csr_matrix((self.pres_ndofs(), self.pres_ndofs()))
        self._mass_cached = sp_block_diag([M_vel, M_pres], format="csr")
        return self._mass_cached

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        For velocity block: solve M_vel * x_vel = rhs_vel.
        For pressure block: pass through rhs_pres (mass is zero).

        Parameters
        ----------
        rhs : Array
            Right-hand side. Shape: (nstates,) or (nstates, nstates).

        Returns
        -------
        Array
            Solution. Same shape as rhs.
        """
        rhs_np = self._bkd.to_numpy(rhs)
        M_vel = self.vel_mass_matrix()
        n_vel = self.vel_ndofs()

        if rhs_np.ndim == 1:
            result = np.zeros_like(rhs_np)
            result[:n_vel] = solve_maybe_sparse(self._bkd, M_vel, rhs_np[:n_vel])
            result[n_vel:] = rhs_np[n_vel:]
        else:
            result = np.zeros_like(rhs_np)
            result[:n_vel, :] = np.linalg.solve(M_vel.toarray(), rhs_np[:n_vel, :])
            result[n_vel:, :] = rhs_np[n_vel:, :]

        return self._bkd.asarray(result.astype(np.float64))

    def init_guess(self, time: float = 0.0) -> Array:
        """Compute initial guess by solving linear Stokes.

        Solves the linear Stokes system (no NS terms) directly.
        Used as initial guess for Newton iteration on Navier-Stokes.

        Parameters
        ----------
        time : float
            Time for evaluation.

        Returns
        -------
        Array
            Initial guess [vel_dofs | pres_dofs]. Shape: (nstates,)
        """
        from pyapprox.pde.galerkin.solvers import SteadyStateSolver

        # Create a linear Stokes physics (no NS terms)
        linear_physics = StokesPhysics(
            vel_basis=self._vel_basis,
            pres_basis=self._pres_basis,
            bkd=self._bkd,
            navier_stokes=False,
            viscosity=self._viscosity,
            vel_forcing=self._vel_forcing,
            pres_forcing=self._pres_forcing,
            vel_dirichlet_bcs=self._vel_dirichlet_bcs,
            pres_dirichlet_bcs=self._pres_dirichlet_bcs,
        )
        solver = SteadyStateSolver(linear_physics, tol=1e-12)
        result = solver.solve_linear(time=time)
        return result.solution

    def __repr__(self) -> str:
        return (
            f"StokesPhysics("
            f"nstates={self.nstates()}, "
            f"ndim={self.ndim()}, "
            f"navier_stokes={self._navier_stokes}, "
            f"viscosity={self._viscosity})"
        )
