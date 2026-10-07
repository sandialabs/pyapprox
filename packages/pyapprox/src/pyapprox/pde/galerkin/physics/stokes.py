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

from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Tuple

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import block_diag as sp_block_diag
from scipy.sparse import csr_matrix

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.constitutive.coefficient_functions import as_time_aware
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.basis.mixed import MixedBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import (
    VectorLagrangeBasis,
)
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


class StokesPhysics(Generic[Array]):
    """Stokes/Navier-Stokes physics for Galerkin FEM.

    Solves the Stokes equations:
        -viscosity * Laplacian(u) + grad(p) = f_vel
        -div(u) = f_pres

    or Navier-Stokes (adds convective term (u . grad)u).

    Uses Taylor-Hood elements (P2 velocity, P1 pressure).
    State vector layout: [vel_dofs | pres_dofs], the order of the mixed
    basis.

    The physics holds no boundary conditions. Constrain each field
    through its view of the mixed basis and compose:

    .. code-block:: python

        system = compose_galerkin_system(physics, [
            DirichletBC(basis.field("velocity"), "left", inflow, bkd),
            DirichletBC(basis.field("pressure"), "right", 0.0, bkd),
        ])

    A velocity boundary function returns ``(ndim, npts)``, one row per
    component.

    Parameters
    ----------
    basis : MixedBasis
        Fields ``"velocity"`` (``VectorLagrangeBasis``, typically P2) then
        ``"pressure"`` (``LagrangeBasis``, typically P1).
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

    Raises
    ------
    ValueError
        If the mixed basis is not ``[velocity, pressure]`` with a vector
        Lagrange velocity and a scalar Lagrange pressure.
    """

    def __init__(
        self,
        basis: MixedBasis[Array],
        bkd: Backend[Array],
        navier_stokes: bool = False,
        viscosity: float = 1.0,
        vel_forcing: Optional[Callable[..., Any]] = None,
        pres_forcing: Optional[Callable[..., Any]] = None,
    ):
        vel_basis = basis.field_basis("velocity")
        pres_basis = basis.field_basis("pressure")
        if (
            basis.field_names() != ["velocity", "pressure"]
            or not isinstance(vel_basis, VectorLagrangeBasis)
            or not isinstance(pres_basis, LagrangeBasis)
        ):
            raise ValueError(
                "StokesPhysics needs MixedBasis([('velocity', "
                "VectorLagrangeBasis), ('pressure', LagrangeBasis)]), got "
                f"{basis!r}"
            )
        self._basis = basis
        self._vel_basis = vel_basis
        self._pres_basis = pres_basis
        self._bkd = bkd
        self._navier_stokes = navier_stokes
        self._viscosity = viscosity
        # Raw suppliers kept for consumers that inspect them; normalized
        # companions are what assembly evaluates as f(coords, time).
        self._vel_forcing = vel_forcing
        self._pres_forcing = pres_forcing
        self._vel_forcing_eval = (
            None if vel_forcing is None else as_time_aware(vel_forcing)
        )
        self._pres_forcing_eval = (
            None if pres_forcing is None else as_time_aware(pres_forcing)
        )

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

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def basis(self) -> MixedBasis[Array]:
        """Return the mixed velocity-pressure basis."""
        return self._basis

    def nstates(self) -> int:
        """Return total number of DOFs (velocity + pressure)."""
        return self._basis.ndofs()

    def ndim(self) -> int:
        """Return spatial dimension."""
        return self._ndim

    def vel_ndofs(self) -> int:
        """Return number of velocity DOFs."""
        return self._vel_basis.ndofs()

    def pres_ndofs(self) -> int:
        """Return number of pressure DOFs."""
        return self._pres_basis.ndofs()

    def viscosity(self) -> float:
        """Return the kinematic viscosity."""
        return self._viscosity

    def set_viscosity(self, viscosity: float) -> None:
        """Set the kinematic viscosity, invalidating the cached viscous
        stiffness (the divergence and mass operators are
        viscosity-independent)."""
        self._viscosity = viscosity
        self._A_cached = None

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
        vel_forcing_func = self._vel_forcing_eval
        if vel_forcing_func is None:
            return np.zeros(self.vel_ndofs())

        nvars = self._ndim
        current_time = time
        prepare_points = self._prepare_points

        def vel_load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_eval, orig_shape = prepare_points(x_np)

            f = vel_forcing_func(x_eval, current_time)
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
        pres_forcing_func = self._pres_forcing_eval
        if pres_forcing_func is None:
            return np.zeros(self.pres_ndofs())

        current_time = time
        prepare_points = self._prepare_points

        def pres_load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x_np = np.asarray(w.x)
            x_eval, orig_shape = prepare_points(x_np)

            f = pres_forcing_func(x_eval, current_time)
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

    def interior_residual(self, state: Array, time: float) -> Array:
        """Compute the interior residual ``F_Omega``, with no BC in it.

        Computes the block assembly:
            F_vel = vel_load - A*vel - B^T*pres [- NS_nonlinear_term]
            F_pres = pres_load - B*vel

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

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF_Omega/du = -K``, with no BC in it.

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
        # Sparse at the skfem seam, typed Array per the assembly
        # convention.
        jacobian_matrix: Array = -K
        return jacobian_matrix

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Exact zero curvature for Stokes; none supplied for
        Navier-Stokes, whose convective term is not differentiated
        twice."""
        if self._navier_stokes:
            return StateDerivatives.none()
        return StateDerivatives.linear(self._bkd)

    def interior_is_time_invariant(self) -> bool:
        """Whether both forcings are declared time-independent (the
        viscosity cannot depend on time)."""
        return all(
            forcing is None or not forcing.is_time_dependent()
            for forcing in (self._vel_forcing_eval, self._pres_forcing_eval)
        )

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

        # solve_maybe_sparse handles vector and matrix right-hand
        # sides sparse-aware — the velocity mass is never densified.
        result = np.zeros_like(rhs_np)
        vel_solution: Array = solve_maybe_sparse(
            self._bkd, M_vel, self._bkd.asarray(rhs_np[:n_vel])
        )
        result[:n_vel] = self._bkd.to_numpy(vel_solution)
        result[n_vel:] = rhs_np[n_vel:]

        return self._bkd.asarray(result.astype(np.float64))

    def __repr__(self) -> str:
        return (
            f"StokesPhysics("
            f"nstates={self.nstates()}, "
            f"ndim={self.ndim()}, "
            f"navier_stokes={self._navier_stokes}, "
            f"viscosity={self._viscosity})"
        )
