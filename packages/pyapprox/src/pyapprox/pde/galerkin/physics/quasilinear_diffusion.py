"""Quasilinear diffusion physics for Galerkin FEM.

Solves the quasilinear diffusion equation

.. math::

    du/dt - \\nabla \\cdot (a(x) \\, \\kappa(u) \\nabla u) = f

where :math:`a(x)` is a nodal diffusivity field (the parameterized,
differentiable representation) and :math:`\\kappa(u)` a smooth
state-dependent factor supplied with its first two derivatives.

The field-carrying term is NONLINEAR in the state, so the typed
field-derivative surface exposes genuine mixed second-derivative
contractions (``residual_diffusivity_field_state_hvp`` /
``residual_diffusivity_state_field_hvp``) instead of the linearity
identities that suffice for the ADR coefficients — this physics is the
exemplar for the engine's callable mixed slots.
"""

from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField


import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
)
from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, LinearForm, asm
    from skfem.helpers import dot, grad
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )

_KappaFn = Callable[
    ["NDArray[np.floating[Any]]"], "NDArray[np.floating[Any]]"
]


class QuasilinearDiffusion(GalerkinPhysicsBase[Array], Generic[Array]):
    """Quasilinear diffusion :math:`-\\nabla\\cdot(a(x)\\kappa(u)\\nabla u) = f`.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Scalar finite element basis.
    diffusivity : NodalFieldDiffusion
        Nodal diffusivity field :math:`a(x)` on the same basis (the
        differentiable representation; parameterizations update its
        DOFs).
    bkd : Backend
        Computational backend.
    kappa : Callable
        :math:`\\kappa(u)`: values at quadrature points -> values.
    kappa_deriv : Callable
        :math:`\\kappa'(u)`.
    kappa_second_deriv : Callable
        :math:`\\kappa''(u)` (needed by ``state_state_hvp``).
    forcing : Callable, optional
        Forcing f. Takes coordinates (ndim, npts) and returns (npts,);
        time-dependent variants take (coordinates, time).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        Boundary conditions.
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        diffusivity: NodalFieldDiffusion,
        bkd: Backend[Array],
        kappa: _KappaFn,
        kappa_deriv: _KappaFn,
        kappa_second_deriv: _KappaFn,
        forcing: Optional[Callable[..., Any]] = None,
        boundary_conditions: Optional[
            List[BoundaryConditionProtocol[Array]]
        ] = None,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        if not isinstance(diffusivity, NodalFieldDiffusion):
            raise TypeError(
                "diffusivity must be a NodalFieldDiffusion, got "
                f"{type(diffusivity).__name__}"
            )
        if diffusivity.ndofs() != basis.ndofs():
            raise ValueError(
                f"diffusivity has {diffusivity.ndofs()} DOFs but the "
                f"basis has {basis.ndofs()}"
            )
        self._mass = ScalarMassAssembler(basis, bkd)
        self._diffusivity = diffusivity
        self._kappa = kappa
        self._kappa_deriv = kappa_deriv
        self._kappa_second_deriv = kappa_second_deriv
        self._forcing = forcing

    def is_linear(self) -> bool:
        """Quasilinear diffusion is nonlinear in the state."""
        return False

    def diffusion_function(self) -> NodalFieldDiffusion:
        """Return the nodal diffusivity field."""
        return self._diffusivity

    def mass_matrix(self) -> Array:
        """Return the scalar mass matrix."""
        return self._mass.mass_matrix()

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return self._mass.mass_solve(rhs)

    def _get_forcing(
        self, coords: np.ndarray, time: float = 0.0
    ) -> np.ndarray:
        """Get forcing values at given coordinates."""
        if self._forcing is None:
            return np.zeros(coords.shape[-1])
        try:
            ret: NDArray[np.floating[Any]] = self._forcing(coords, time)
            return ret
        except TypeError:
            ret2: NDArray[np.floating[Any]] = self._forcing(coords)
            return ret2

    def _interpolate(self, dofs: np.ndarray) -> "DiscreteField":
        """Interpolate DOFs at quadrature points."""
        interp: "DiscreteField" = self._basis.skfem_basis().interpolate(
            np.asarray(dofs, dtype=np.float64)
        )
        return interp

    def _newton_stiffness(self, weight_dofs: np.ndarray, state: Array) -> Array:
        """Assemble :math:`K(w)` — the u-derivative of the field-carrying
        term with weight field :math:`w_h` in place of :math:`a(x)`:

        .. math::

            K(w)_{ji} = \\int w_h [\\kappa'(u) \\phi_i \\nabla u
                + \\kappa(u) \\nabla\\phi_i] \\cdot \\nabla\\phi_j

        With :math:`w = a` this is the Newton stiffness (spatial
        jacobian is :math:`-K(a)`); with :math:`w = \\delta g` it is the
        mixed assembly :math:`-A(\\delta g, u)`.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate(self._bkd.to_numpy(state))
        weight_interp = self._interpolate(weight_dofs)
        kappa, kappa_deriv = self._kappa, self._kappa_deriv

        def bilinear_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            uvals = np.asarray(w.u_prev)
            ret: NDArray[np.floating[Any]] = w.weight * (
                kappa_deriv(uvals) * u * dot(grad(w.u_prev), grad(v))
                + kappa(uvals) * dot(grad(u), grad(v))
            )
            return ret

        stiffness: Array = asm(
            BilinearForm(bilinear_form),
            skfem_basis,
            u_prev=state_interp,
            weight=weight_interp,
        )
        return stiffness

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute the spatial residual without Dirichlet enforcement.

        .. math::

            R_j = \\int f \\phi_j
                - \\int a(x) \\kappa(u_h) \\nabla u_h \\cdot \\nabla\\phi_j

        plus Neumann/Robin boundary contributions.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate(self._bkd.to_numpy(state))
        diff_interp = self._interpolate(self._diffusivity.dofs())
        kappa = self._kappa
        current_time = time

        def linear_form(
            v: "DiscreteField", w: "FormExtraParams"
        ) -> np.ndarray:
            x_np = np.asarray(w.x)
            if len(x_np.shape) == 3:
                ndim, nelem, nquad = x_np.shape
                forc = self._get_forcing(
                    x_np.reshape(ndim, -1), current_time
                ).reshape(nelem, nquad)
            else:
                forc = self._get_forcing(x_np, current_time)
            ret: NDArray[np.floating[Any]] = forc * v - w.a_field * kappa(
                np.asarray(w.u_prev)
            ) * dot(grad(w.u_prev), grad(v))
            return ret

        load_np = asm(
            LinearForm(linear_form),
            skfem_basis,
            u_prev=state_interp,
            a_field=diff_interp,
        )
        load = self._bkd.asarray(load_np.astype(np.float64))
        load = self._apply_bc_to_load(load, time)

        # Zero stiffness — only BC contributions (Robin alpha*M_bnd) matter
        n = self.nstates()
        bc_stiffness = csr_matrix((n, n))
        bc_stiffness = self._apply_bc_to_stiffness(bc_stiffness, time)
        return load - bc_stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dR/du without Dirichlet enforcement: :math:`-K(a)`."""
        stiffness = self._newton_stiffness(self._diffusivity.dofs(), state)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        return -stiffness

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T (d^2R/du^2) w of the RAW spatial residual.

        Differentiating the field-carrying term twice in the state:

        .. math::

            [\\lambda^T \\partial^2 R/\\partial u^2 w]_i =
            -\\int a [\\kappa''(u) w_h (\\nabla u \\cdot \\nabla\\lambda_h)
            + \\kappa'(u) (\\nabla w_h \\cdot \\nabla\\lambda_h)] \\phi_i
            - \\int a \\kappa'(u) w_h
              (\\nabla\\lambda_h \\cdot \\nabla\\phi_i)
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate(self._bkd.to_numpy(state))
        adj_interp = self._interpolate(self._bkd.to_numpy(adj_state))
        w_interp = self._interpolate(self._bkd.to_numpy(wvec))
        diff_interp = self._interpolate(self._diffusivity.dofs())
        kappa_deriv = self._kappa_deriv
        kappa_second_deriv = self._kappa_second_deriv

        def linear_form(
            v: "DiscreteField", w: "FormExtraParams"
        ) -> np.ndarray:
            uvals = np.asarray(w.u_prev)
            kp, kpp = kappa_deriv(uvals), kappa_second_deriv(uvals)
            ret: NDArray[np.floating[Any]] = -w.a_field * (
                (
                    kpp * w.wdir * dot(grad(w.u_prev), grad(w.adj))
                    + kp * dot(grad(w.wdir), grad(w.adj))
                )
                * v
                + kp * w.wdir * dot(grad(w.adj), grad(v))
            )
            return ret

        out_np = asm(
            LinearForm(linear_form),
            skfem_basis,
            u_prev=state_interp,
            adj=adj_interp,
            wdir=w_interp,
            a_field=diff_interp,
        )
        return self._bkd.asarray(out_np.astype(np.float64))

    # -----------------------------------------------------------------
    # Typed diffusivity-derivative assemblies
    # -----------------------------------------------------------------

    def residual_diffusivity_jacobian(self, state: Array) -> Array:
        """Jacobian of the spatial residual w.r.t. the diffusivity DOFs.

        .. math::

            S(u)_{jk} = -\\int \\psi_k \\kappa(u_h)
                \\nabla u_h \\cdot \\nabla\\phi_j

        Shape: ``(nstates, nfield_dofs)`` (sparse).
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate(self._bkd.to_numpy(state))
        kappa = self._kappa

        def bilinear_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = (
                -u * kappa(np.asarray(w.u_prev)) * dot(grad(w.u_prev), grad(v))
            )
            return ret

        jacobian: Array = asm(
            BilinearForm(bilinear_form), skfem_basis, u_prev=state_interp
        )
        return jacobian

    def residual_diffusivity_field_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """Field-shaped mixed contraction
        :math:`\\lambda^T (\\partial^2 R/\\partial g \\partial u) w`.

        .. math::

            \\text{out}_k = -\\int \\psi_k [\\kappa'(u) w_h \\nabla u_h
                + \\kappa(u) \\nabla w_h] \\cdot \\nabla\\lambda_h

        Shape: ``(nfield_dofs,)``.
        """
        skfem_basis = self._basis.skfem_basis()
        state_interp = self._interpolate(self._bkd.to_numpy(state))
        adj_interp = self._interpolate(self._bkd.to_numpy(adj_state))
        w_interp = self._interpolate(self._bkd.to_numpy(wvec))
        kappa, kappa_deriv = self._kappa, self._kappa_deriv

        def linear_form(
            v: "DiscreteField", w: "FormExtraParams"
        ) -> np.ndarray:
            uvals = np.asarray(w.u_prev)
            ret: NDArray[np.floating[Any]] = -v * (
                kappa_deriv(uvals)
                * w.wdir
                * dot(grad(w.u_prev), grad(w.adj))
                + kappa(uvals) * dot(grad(w.wdir), grad(w.adj))
            )
            return ret

        out_np = asm(
            LinearForm(linear_form),
            skfem_basis,
            u_prev=state_interp,
            adj=adj_interp,
            wdir=w_interp,
        )
        return self._bkd.asarray(out_np.astype(np.float64))

    def residual_diffusivity_state_field_hvp(
        self, state: Array, adj_state: Array, delta_field: Array
    ) -> Array:
        """State-shaped mixed contraction
        :math:`[\\partial^2 R/\\partial u \\partial g \\, \\delta g]^T
        \\lambda = A(\\delta g, u)^T \\lambda` with
        :math:`A(\\delta g, u) = -K(\\delta g)` (the Newton stiffness
        weighted by :math:`\\delta g` instead of :math:`a`).

        Shape: ``(nstates,)``.
        """
        mixed = -self._newton_stiffness(
            self._bkd.to_numpy(delta_field), state
        )
        return self._bkd.asarray(mixed.T @ self._bkd.to_numpy(adj_state))

    def initial_condition(self, func: Callable[..., Any]) -> Array:
        """Create initial condition by interpolating a function."""
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        return f"QuasilinearDiffusion(nstates={self.nstates()})"
