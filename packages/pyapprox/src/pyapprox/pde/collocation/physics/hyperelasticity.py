"""Hyperelasticity physics for spectral collocation.

Implements the equilibrium equation for hyperelastic materials:
    div(P) + f = 0

where P is the first Piola-Kirchhoff stress computed by a pluggable
stress model. Supports 1D, 2D, and 3D with any constitutive law that
satisfies StressModelWithSensitivityProtocol.
"""

from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalCoefficientProtocol,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
)
from pyapprox.pde.constitutive.protocols import (
    StressModelWithSensitivityProtocol,
    StressModelWithTangentProtocol,
)
from pyapprox.pde.constitutive.registry import (
    create_stress_model,
)
from pyapprox.pde.time_order import second_order_in_time_refusal
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class HyperelasticLawProtocol(Protocol, Generic[Array]):
    """The constitutive law this physics needs: the first Piola-Kirchhoff
    stress and its tangent, from the deformation gradient and the Lame
    parameters at the same points.

    ``F`` is nested ``F[i][J]``, ``ndim`` by ``ndim``; ``mu``, ``lamda``
    and every entry of ``F`` share one shape, (npts,) at the nodes.
    """

    def stress(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        """``P[i][J]``, nested as ``F``."""
        ...

    def tangent(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[List[List[Array]]]]:
        """:math:`A_{iJkL} = \\partial P_{iJ}/\\partial F_{kL}`, nested
        ``A[i][J][k][L]``."""
        ...


@runtime_checkable
class LameSensitiveLawProtocol(Protocol, Generic[Array]):
    """What the residual's Lame-parameter sensitivities need: the
    pointwise derivatives of the stress with respect to mu and lamda.
    Arguments as for :class:`HyperelasticLawProtocol`."""

    def stress_mu_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        """:math:`\\partial P_{iJ}/\\partial \\mu`, nested as ``F``."""
        ...

    def stress_lamda_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        """:math:`\\partial P_{iJ}/\\partial \\lambda`, nested as ``F``."""
        ...


class HyperelasticityPhysics(AbstractVectorPhysics[Array], Generic[Array]):
    """Hyperelastic physics with pluggable stress model.

    Implements the equilibrium equation:
        div(P) + f = 0

    where P is the first Piola-Kirchhoff stress tensor computed by the
    provided stress model from the deformation gradient F = I + grad(u).

    Supports 1D, 2D, and 3D. The analytical Jacobian requires the stress
    model to provide a tangent modulus (compute_tangent methods).

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (1D, 2D, or 3D).
    bkd : Backend
        Computational backend.
    stress_model : StressModelWithSensitivityProtocol
        Constitutive model for PK1 stress with Lame-parameter setters
        and stress sensitivities.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Body force with the components stacked, in the forms
        ``as_nodal_coefficient`` takes but of length nstates. Absent
        means none.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        stress_model: StressModelWithSensitivityProtocol[Array],
        forcing: Optional[CoefficientInput[Array]] = None,
    ):
        ndim = basis.ndim()
        if ndim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {ndim}")
        if not isinstance(stress_model, StressModelWithSensitivityProtocol):
            raise TypeError(
                f"stress_model must satisfy "
                f"StressModelWithSensitivityProtocol, "
                f"got {type(stress_model).__name__}"
            )

        super().__init__(basis, bkd, ncomponents=ndim)

        self._stress_model = stress_model
        # The tangent is a capability of the model, decided once here.
        self._tangent_model: Optional[StressModelWithTangentProtocol[Array]] = (
            stress_model
            if isinstance(stress_model, StressModelWithTangentProtocol)
            else None
        )
        self._forcing = optional_nodal_coefficient(forcing, ndim * basis.npts(), bkd)
        self._ndim = ndim

        # Precompute first-order derivative matrices
        self._D: List[Array] = [basis.derivative_matrix(1, dim) for dim in range(ndim)]

    def stress_model(self) -> StressModelWithSensitivityProtocol[Array]:
        """Return the stress model."""
        return self._stress_model

    def first_order_transient_refusal(self) -> str:
        """Return why a first-order transient path refuses this physics.

        Satisfies ``SecondOrderInTimeProtocol``: elasticity is second
        order in time, so ``du/dt = div(P) + f`` is not its dynamics.
        """
        return second_order_in_time_refusal(type(self).__name__)

    # ------------------------------------------------------------------
    # Material property setters
    # ------------------------------------------------------------------

    def set_mu(self, mu_values: Union[float, Array]) -> None:
        """Set shear modulus (scalar or per-point array).

        Parameters
        ----------
        mu_values : float or Array
            Shear modulus values. Must be positive.
        """
        mu_arr = self._bkd.ravel(self._bkd.asarray(mu_values))
        min_val = self._bkd.to_float(self._bkd.min(mu_arr))
        if min_val <= 0.0:
            raise ValueError(f"mu must be positive; found min {min_val:.2e}")
        self._stress_model.set_mu(mu_values)

    def set_lamda(self, lamda_values: Union[float, Array]) -> None:
        """Set Lame's first parameter (scalar or per-point array).

        Parameters
        ----------
        lamda_values : float or Array
            Lame's first parameter values. Must be non-negative.
        """
        lam_arr = self._bkd.ravel(self._bkd.asarray(lamda_values))
        min_val = self._bkd.to_float(self._bkd.min(lam_arr))
        if min_val < 0.0:
            raise ValueError(f"lamda must be non-negative; found min {min_val:.2e}")
        self._stress_model.set_lamda(lamda_values)

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The stacked body force; None without one."""
        return self._forcing

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        if self._forcing is None:
            return self._bkd.zeros((self.nstates(),))
        return self._forcing.values(time)

    def _deformation_gradient(self, state: Array) -> List[List[Array]]:
        """:math:`F_{iJ} = \\delta_{iJ} + D_J u_i` at every node, nested
        ``ndim`` by ``ndim``, each entry shape (npts,)."""
        comps = self._extract_components(state)
        return [
            [
                (1.0 + self._D[J] @ comps[i]) if i == J else self._D[J] @ comps[i]
                for J in range(self._ndim)
            ]
            for i in range(self._ndim)
        ]

    def _tangent(self, F: List[List[Array]]) -> List[List[List[List[Array]]]]:
        """The tangent :math:`A_{iJkL} = \\partial P_{iJ}/\\partial F_{kL}`
        nested as ``A[i][J][k][L]``, each entry shape (npts,).

        Raises
        ------
        NotImplementedError
            If the stress model provides no tangent modulus.
        """
        model = self._tangent_model
        if model is None:
            raise NotImplementedError(
                f"{type(self._stress_model).__name__} provides no tangent "
                "modulus, so the analytical Jacobian is unavailable"
            )
        bkd = self._bkd
        ndim = self._ndim
        if ndim == 1:
            return [[[[model.compute_tangent_1d(F[0][0], bkd)]]]]
        if ndim == 2:
            A2 = model.compute_tangent_2d(F[0][0], F[0][1], F[1][0], F[1][1], bkd)
            return [
                [
                    [
                        [A2[f"A_{i + 1}{J + 1}{k + 1}{L + 1}"] for L in range(2)]
                        for k in range(2)
                    ]
                    for J in range(2)
                ]
                for i in range(2)
            ]
        F3 = tuple(tuple(row) for row in F)
        A3 = model.compute_tangent_3d(F3, bkd)
        return [
            [
                [[A3[i, J, k, L] for L in range(3)] for k in range(3)]
                for J in range(3)
            ]
            for i in range(3)
        ]

    def _stress_entry_jacobian(
        self, A_iJ: List[List[Array]]
    ) -> Array:
        """:math:`\\partial P_{iJ}/\\partial u` from its tangent row
        ``A_iJ[k][L]``: block k is :math:`\\sum_L \\operatorname{diag}
        (A_{iJkL}) D_L`, formed by row scaling. Shape: (npts, nstates)"""
        blocks = []
        for k in range(self._ndim):
            block = A_iJ[k][0][:, None] * self._D[0]
            for L in range(1, self._ndim):
                block = block + A_iJ[k][L][:, None] * self._D[L]
            blocks.append(block)
        return self._bkd.concatenate(blocks, axis=1)

    def _extract_components(self, state: Array) -> Tuple[Array, ...]:
        """Extract displacement components from state vector.

        State is ordered as [u0_0, ..., u0_{n-1}, u1_0, ..., u1_{n-1}, ...]

        Parameters
        ----------
        state : Array
            Full state vector. Shape: (ndim * npts,)

        Returns
        -------
        Tuple[Array, ...]
            Displacement components, each shape (npts,).
        """
        npts = self.npts()
        return tuple(state[i * npts : (i + 1) * npts] for i in range(self._ndim))

    # ------------------------------------------------------------------
    # Flux supply (1D) and PK1 stress
    # ------------------------------------------------------------------

    def _require_scalar_flux(self) -> None:
        if self._ndim != 1:
            raise NotImplementedError(
                "the flux of a vector physics is a tensor; in "
                f"{self._ndim}D use HyperelasticTractionNormalOperator for "
                "boundary traction, or pk1_stress for the stress tensor"
            )

    def compute_flux(self, state: Array, time: float) -> List[Array]:
        """Compute the flux in 1D: the PK1 stress ``[P]``.

        Parameters
        ----------
        state : Array
            Displacement field. Shape: (npts,).
        time : float
            Unused: the stress model is fixed.

        Returns
        -------
        List[Array]
            ``[P]``, shape (npts,).

        Raises
        ------
        NotImplementedError
            In 2D or 3D, where the flux is a tensor.
        """
        self._require_scalar_flux()
        return self.pk1_stress(state)[0]

    def compute_flux_jacobian(self, state: Array, time: float) -> List[Array]:
        """Compute the Jacobian of the 1D flux w.r.t. the state.

        Parameters
        ----------
        state : Array
            Displacement field. Shape: (npts,).
        time : float
            Unused: the stress model is fixed.

        Returns
        -------
        List[Array]
            ``[dP/du]``, shape (npts, npts).

        Raises
        ------
        NotImplementedError
            In 2D or 3D, where the flux is a tensor.
        """
        self._require_scalar_flux()
        return self.pk1_stress_jacobian(state)[0]

    def pk1_stress(self, state: Array) -> List[List[Array]]:
        """Compute the first Piola-Kirchhoff stress tensor at every node.

        Parameters
        ----------
        state : Array
            Displacement field. Shape: (nstates,).

        Returns
        -------
        List[List[Array]]
            ``P[i][J]``, an ``ndim`` by ``ndim`` nesting, each entry of
            shape (npts,): ``[[P]]`` in 1D, ``[[P11, P12], [P21, P22]]``
            in 2D.
        """
        bkd = self._bkd
        F = self._deformation_gradient(state)
        if self._ndim == 1:
            return [[self._stress_model.compute_stress_1d(F[0][0], bkd)]]
        if self._ndim == 2:
            P11, P12, P21, P22 = self._stress_model.compute_stress_2d(
                F[0][0], F[0][1], F[1][0], F[1][1], bkd
            )
            return [[P11, P12], [P21, P22]]
        P3 = self._stress_model.compute_stress_3d(
            tuple(tuple(row) for row in F), bkd
        )
        return [list(row) for row in P3]

    def pk1_stress_jacobian(self, state: Array) -> List[List[Array]]:
        """Compute the Jacobian of each PK1 stress entry w.r.t. the state.

        Parameters
        ----------
        state : Array
            Displacement field. Shape: (nstates,).

        Returns
        -------
        List[List[Array]]
            ``dP[i][J]``, nested as in :meth:`pk1_stress`, each entry of
            shape (npts, nstates).

        Raises
        ------
        NotImplementedError
            If the stress model provides no tangent modulus.
        """
        A = self._tangent(self._deformation_gradient(state))
        return [
            [self._stress_entry_jacobian(A[i][J]) for J in range(self._ndim)]
            for i in range(self._ndim)
        ]

    # ------------------------------------------------------------------
    # Residual sensitivity to material parameters
    # ------------------------------------------------------------------

    # -- full-matrix field-derivative assemblies (engine slots)

    def residual_mu_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\mu(u) = \\partial R/\\partial
        \\mu_{\\text{field}}` from the stress model's mu sensitivity.

        State-nonlinear (the PK1 sensitivity depends on the
        deformation gradient), so no linearity identities apply — the
        parameterization tier stays first order until stress-model
        tangent assemblies (:math:`\\partial^2 P/\\partial F
        \\partial \\mu`) provide the mixed contractions.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)

        Returns
        -------
        Array
            Assembly. Shape: (nstates, npts)
        """
        bkd = self._bkd
        if self._ndim == 1:
            Dx = self._D[0]
            F = 1.0 + Dx @ state
            dP_dmu = self._stress_model.stress_sensitivity_mu_1d(F, bkd)
            return Dx * dP_dmu[None, :]
        if self._ndim == 2:
            Dx, Dy = self._D
            u, v = self._extract_components(state)
            F11 = 1.0 + Dx @ u
            F12 = Dy @ u
            F21 = Dx @ v
            F22 = 1.0 + Dy @ v
            dP11, dP12, dP21, dP22 = self._stress_model.stress_sensitivity_mu_2d(
                F11, F12, F21, F22, bkd
            )
            top = Dx * dP11[None, :] + Dy * dP12[None, :]
            bottom = Dx * dP21[None, :] + Dy * dP22[None, :]
            return bkd.concatenate([top, bottom], axis=0)
        raise NotImplementedError("residual_mu_jacobian only implemented for 1D and 2D")

    def residual_lamda_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\lambda(u) = \\partial R/\\partial
        \\lambda_{\\text{field}}` from the stress model's lambda
        sensitivity.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)

        Returns
        -------
        Array
            Assembly. Shape: (nstates, npts)
        """
        bkd = self._bkd
        if self._ndim == 1:
            Dx = self._D[0]
            F = 1.0 + Dx @ state
            dP_dlam = self._stress_model.stress_sensitivity_lamda_1d(F, bkd)
            return Dx * dP_dlam[None, :]
        if self._ndim == 2:
            Dx, Dy = self._D
            u, v = self._extract_components(state)
            F11 = 1.0 + Dx @ u
            F12 = Dy @ u
            F21 = Dx @ v
            F22 = 1.0 + Dy @ v
            dP11, dP12, dP21, dP22 = self._stress_model.stress_sensitivity_lamda_2d(
                F11, F12, F21, F22, bkd
            )
            top = Dx * dP11[None, :] + Dy * dP12[None, :]
            bottom = Dx * dP21[None, :] + Dy * dP22[None, :]
            return bkd.concatenate([top, bottom], axis=0)
        raise NotImplementedError(
            "residual_lamda_jacobian only implemented for 1D and 2D"
        )

    def boundary_traction_lame_jacobian(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Assemble :math:`\\partial(P n)/\\partial [\\mu; \\lambda]`
        at the boundary rows of one traction BC.

        Consumer convention (the BC supplies its own row indices and
        normals): ``bc_indices`` are the BC's replaced STATE rows —
        one traction component per BC in 2D (``mesh_idx + comp *
        npts``) — and each returned row pairs one nonzero in the mu
        block with one in the lambda block at that row's mesh column.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)
        time : float
            Current time (unused; kept for signature uniformity).
        bc_indices : Array
            Replaced state-row indices of the BC. Shape: (n_bc,)
        normals : Array
            Outward unit normals. Shape: (n_bc, ndim)

        Returns
        -------
        Array
            Assembly over the stacked [mu; lambda] field.
            Shape: (n_bc, 2*npts)
        """
        bkd = self._bkd
        npts = self.npts()
        nbnd = bc_indices.shape[0]
        result = bkd.copy(bkd.zeros((nbnd, 2 * npts)))
        if self._ndim == 1:
            Dx = self._D[0]
            F = 1.0 + Dx @ state
            dP_dmu = self._stress_model.stress_sensitivity_mu_1d(F, bkd)
            dP_dlam = self._stress_model.stress_sensitivity_lamda_1d(F, bkd)
            for i in range(nbnd):
                idx = bkd.to_int(bc_indices[i])
                result[i, idx] = normals[i, 0] * dP_dmu[idx]
                result[i, npts + idx] = normals[i, 0] * dP_dlam[idx]
            return result
        if self._ndim == 2:
            comp = bkd.to_int(bc_indices[0]) // npts
            mesh_idx = bc_indices - comp * npts
            Dx, Dy = self._D
            u, v = self._extract_components(state)
            F11 = (1.0 + Dx @ u)[mesh_idx]
            F12 = (Dy @ u)[mesh_idx]
            F21 = (Dx @ v)[mesh_idx]
            F22 = (1.0 + Dy @ v)[mesh_idx]
            dP_mu = self._stress_model.stress_sensitivity_mu_2d(F11, F12, F21, F22, bkd)
            dP_lam = self._stress_model.stress_sensitivity_lamda_2d(
                F11, F12, F21, F22, bkd
            )
            i1 = 2 * comp
            i2 = 2 * comp + 1
            dt_dmu = dP_mu[i1] * normals[:, 0] + dP_mu[i2] * normals[:, 1]
            dt_dlam = dP_lam[i1] * normals[:, 0] + dP_lam[i2] * normals[:, 1]
            for i in range(nbnd):
                idx = bkd.to_int(mesh_idx[i])
                result[i, idx] = dt_dmu[i]
                result[i, npts + idx] = dt_dlam[i]
            return result
        raise NotImplementedError(
            "boundary_traction_lame_jacobian only implemented for 1D and 2D"
        )

    # ------------------------------------------------------------------
    # Residual computation
    # ------------------------------------------------------------------

    def residual(self, state: Array, time: float) -> Array:
        """Compute :math:`R_i = \\sum_J D_J P_{iJ} + f_i`.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (nstates,)
        """
        bkd = self._bkd
        npts = self.npts()
        P = self.pk1_stress(state)
        forcing = self._get_forcing(time)
        residuals = []
        for i in range(self._ndim):
            div_P_i = self._D[0] @ P[i][0]
            for J in range(1, self._ndim):
                div_P_i = div_P_i + self._D[J] @ P[i][J]
            residuals.append(div_P_i + forcing[i * npts : (i + 1) * npts])
        return bkd.concatenate(residuals)

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute d(residual)/d(state).

        Block :math:`(a, b)` is :math:`\\sum_{J, L} D_J
        \\operatorname{diag}(A_{aJbL}) D_L`, with the tangent
        :math:`A_{iJkL} = \\partial P_{iJ}/\\partial F_{kL}` of the stress
        model; the same assembly serves 1D, 2D and 3D.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)

        Raises
        ------
        NotImplementedError
            If the stress model provides no tangent modulus.
        """
        A = self._tangent(self._deformation_gradient(state))
        rows = []
        for a in range(self._ndim):
            row = self._D[0] @ self._stress_entry_jacobian(A[a][0])
            for J in range(1, self._ndim):
                row = row + self._D[J] @ self._stress_entry_jacobian(A[a][J])
            rows.append(row)
        return self._bkd.concatenate(rows, axis=0)

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, t)`, no boundary condition applied.
        Shape: (nstates,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        """:math:`\\partial f/\\partial u`. Shape: (nstates, nstates)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """No second derivatives: the stress models supply none."""
        return StateDerivatives.none()

    def interior_jacobian_is_time_invariant(self) -> bool:
        """True: the stress model does not depend on time."""
        return True

    def interior_is_time_invariant(self) -> bool:
        """Whether the body force is declared time-invariant."""
        return declared_time_invariant(self._forcing)


def create_hyperelasticity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    stress_model_name: str = "neo_hookean",
    forcing: Optional[CoefficientInput[Array]] = None,
    **stress_kwargs: float,
) -> HyperelasticityPhysics[Array]:
    """Create hyperelasticity physics with a named stress model.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (1D, 2D, or 3D).
    bkd : Backend
        Computational backend.
    stress_model_name : str
        Name of registered stress model. Default: "neo_hookean".
    forcing : float, NodalCoefficientProtocol or callable, optional
        Body force, components stacked.
    **stress_kwargs
        Material parameters passed to the stress model factory.

    Returns
    -------
    HyperelasticityPhysics
        Hyperelasticity physics instance.
    """
    stress_model = create_stress_model(stress_model_name, **stress_kwargs)
    return HyperelasticityPhysics(basis, bkd, stress_model, forcing)
