"""Hyperelasticity physics for spectral collocation.

Implements the equilibrium equation for hyperelastic materials:
    div(P) + f = 0

where P is the first Piola-Kirchhoff stress computed by a pluggable
constitutive law from the deformation gradient and the Lame parameters
the physics holds. Supports 1D, 2D, and 3D with any law that satisfies
HyperelasticLawProtocol.
"""

from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.physics.base import AbstractVectorPhysics
from pyapprox.pde.collocation.physics.coefficients import (
    CheckedValues,
    CoefficientInput,
    NodalCoefficientProtocol,
    as_nodal_coefficient,
    declared_time_invariant,
    optional_nodal_coefficient,
)
from pyapprox.pde.collocation.protocols.basis import (
    TensorProductBasisProtocol,
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
    """Hyperelastic physics with a pluggable constitutive law.

    Implements the equilibrium equation:
        div(P) + f = 0

    where P is the first Piola-Kirchhoff stress the law computes from
    the deformation gradient F = I + grad(u) and the Lame parameters,
    which the physics holds as nodal coefficients.

    Supports 1D, 2D, and 3D.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        Collocation basis (1D, 2D, or 3D).
    bkd : Backend
        Computational backend.
    law : HyperelasticLawProtocol
        The constitutive law, e.g. ``NeoHookeanLaw``. The Lame
        sensitivities a parameterization needs are available when it
        also satisfies ``LameSensitiveLawProtocol``.
    mu, lamda : float or NodalCoefficientProtocol
        The Lame parameters, in the forms ``as_nodal_coefficient``
        takes; a parameterized one must be a ``NodalField``. mu must be
        positive and lamda non-negative, checked whenever they change.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Body force with the components stacked, in the forms
        ``as_nodal_coefficient`` takes but of length nstates. Absent
        means none.

    Raises
    ------
    TypeError
        If ``law`` does not satisfy ``HyperelasticLawProtocol``.
    ValueError
        If mu or lamda is declared time-dependent: the physics is
        steady and its stress takes no time.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        law: HyperelasticLawProtocol[Array],
        mu: CoefficientInput[Array],
        lamda: CoefficientInput[Array],
        forcing: Optional[CoefficientInput[Array]] = None,
    ):
        ndim = basis.ndim()
        if ndim not in (1, 2, 3):
            raise ValueError(f"Unsupported dimension: {ndim}")
        if not isinstance(law, HyperelasticLawProtocol):
            raise TypeError(
                f"law must satisfy HyperelasticLawProtocol, got {type(law).__name__}"
            )

        super().__init__(basis, bkd, ncomponents=ndim)

        self._law = law
        # The Lame sensitivities are a capability of the law, decided
        # once here.
        self._sensitive_law: Optional[LameSensitiveLawProtocol[Array]] = (
            law if isinstance(law, LameSensitiveLawProtocol) else None
        )
        npts = basis.npts()
        self._mu = as_nodal_coefficient(mu, npts, bkd)
        self._lamda = as_nodal_coefficient(lamda, npts, bkd)
        for name, coefficient in (("mu", self._mu), ("lamda", self._lamda)):
            if coefficient.is_time_dependent():
                raise ValueError(
                    f"{name} must be time-invariant: hyperelasticity is "
                    "steady and its stress takes no time"
                )
        self._checked_mu = CheckedValues(self._mu, self._check_mu)
        self._checked_lamda = CheckedValues(self._lamda, self._check_lamda)
        self._forcing = optional_nodal_coefficient(forcing, ndim * npts, bkd)
        self._ndim = ndim

        # Precompute first-order derivative matrices
        self._D: List[Array] = [basis.derivative_matrix(1, dim) for dim in range(ndim)]

    def law(self) -> HyperelasticLawProtocol[Array]:
        """Return the constitutive law."""
        return self._law

    def first_order_transient_refusal(self) -> str:
        """Return why a first-order transient path refuses this physics.

        Satisfies ``SecondOrderInTimeProtocol``: elasticity is second
        order in time, so ``du/dt = div(P) + f`` is not its dynamics.
        """
        return second_order_in_time_refusal(type(self).__name__)

    # ------------------------------------------------------------------
    # Material
    # ------------------------------------------------------------------

    def mu_function(self) -> NodalCoefficientProtocol[Array]:
        """The shear modulus coefficient."""
        return self._mu

    def lamda_function(self) -> NodalCoefficientProtocol[Array]:
        """Lame's first parameter coefficient."""
        return self._lamda

    def lame_values(self, time: float) -> Tuple[Array, Array]:
        """``(lamda, mu)`` at the nodes, each checked when it changes.
        The time is accepted for uniformity; the material is
        time-invariant. Shapes: (npts,)"""
        mu = self._checked_mu.values(time)
        return self._checked_lamda.values(time), mu

    def _check_mu(self, mu: Array, time: float) -> None:
        min_val = self._bkd.to_float(self._bkd.min(mu))
        if min_val <= 0.0:
            raise ValueError(f"mu must be positive; found min {min_val:.2e}")

    def _check_lamda(self, lamda: Array, time: float) -> None:
        min_val = self._bkd.to_float(self._bkd.min(lamda))
        if min_val < 0.0:
            raise ValueError(f"lamda must be non-negative; found min {min_val:.2e}")

    def _sensitivity_law(self) -> LameSensitiveLawProtocol[Array]:
        """The law's Lame sensitivities.

        Raises
        ------
        NotImplementedError
            If the law does not satisfy ``LameSensitiveLawProtocol``.
        """
        if self._sensitive_law is None:
            raise NotImplementedError(
                f"{type(self._law).__name__} provides no Lame-parameter sensitivities"
            )
        return self._sensitive_law

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
        nested as ``A[i][J][k][L]``, each entry shape (npts,)."""
        lamda, mu = self.lame_values(0.0)
        return self._law.tangent(F, mu, lamda)

    def _stress_entry_jacobian(self, A_iJ: List[List[Array]]) -> Array:
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
            Unused: the material is time-invariant.

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
            Unused: the material is time-invariant.

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
        lamda, mu = self.lame_values(0.0)
        return self._law.stress(self._deformation_gradient(state), mu, lamda)

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

    def _stress_lame_sensitivity(self, state: Array, which: str) -> List[List[Array]]:
        """:math:`\\partial P_{iJ}/\\partial \\mu` (``which="mu"``) or
        :math:`\\partial P_{iJ}/\\partial \\lambda` at every node."""
        law = self._sensitivity_law()
        F = self._deformation_gradient(state)
        lamda, mu = self.lame_values(0.0)
        if which == "mu":
            return law.stress_mu_sensitivity(F, mu, lamda)
        return law.stress_lamda_sensitivity(F, mu, lamda)

    def _residual_field_jacobian(self, dP: List[List[Array]]) -> Array:
        """:math:`\\partial R_i/\\partial c = \\sum_J D_J
        \\operatorname{diag}(\\partial P_{iJ}/\\partial c)`, formed by
        column scaling and stacked over i. Shape: (nstates, npts)"""
        rows = []
        for i in range(self._ndim):
            row = self._D[0] * dP[i][0][None, :]
            for J in range(1, self._ndim):
                row = row + self._D[J] * dP[i][J][None, :]
            rows.append(row)
        return self._bkd.concatenate(rows, axis=0)

    def residual_mu_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\mu(u) = \\partial R/\\partial
        \\mu_{\\text{field}}` from the law's mu sensitivity.

        State-nonlinear (the PK1 sensitivity depends on the
        deformation gradient), so no linearity identities apply — the
        parameterization tier stays first order until law tangent
        assemblies (:math:`\\partial^2 P/\\partial F
        \\partial \\mu`) provide the mixed contractions.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)

        Returns
        -------
        Array
            Assembly. Shape: (nstates, npts)

        Raises
        ------
        NotImplementedError
            If the law provides no Lame sensitivities (or none in this
            dimension).
        """
        return self._residual_field_jacobian(self._stress_lame_sensitivity(state, "mu"))

    def residual_lamda_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\lambda(u) = \\partial R/\\partial
        \\lambda_{\\text{field}}` from the law's lambda sensitivity.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (nstates,)

        Returns
        -------
        Array
            Assembly. Shape: (nstates, npts)

        Raises
        ------
        NotImplementedError
            If the law provides no Lame sensitivities (or none in this
            dimension).
        """
        return self._residual_field_jacobian(
            self._stress_lame_sensitivity(state, "lamda")
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
        one traction component per BC (``mesh_idx + comp * npts``) — and
        each returned row pairs one nonzero in the mu block with one in
        the lambda block at that row's mesh column.

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

        Raises
        ------
        NotImplementedError
            If the law provides no Lame sensitivities (or none in this
            dimension).
        """
        bkd = self._bkd
        npts = self.npts()
        comp = bkd.to_int(bc_indices[0]) // npts
        mesh_idx = bc_indices - comp * npts
        # Row r selects mesh node mesh_idx[r].
        select = bkd.eye(npts)[mesh_idx]
        blocks = []
        for which in ("mu", "lamda"):
            dP = self._stress_lame_sensitivity(state, which)[comp]
            dt = dP[0][mesh_idx] * normals[:, 0]
            for J in range(1, self._ndim):
                dt = dt + dP[J][mesh_idx] * normals[:, J]
            blocks.append(dt[:, None] * select)
        return bkd.concatenate(blocks, axis=1)

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
        :math:`A_{iJkL} = \\partial P_{iJ}/\\partial F_{kL}` of the law;
        the same assembly serves 1D, 2D and 3D.

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
        """No second derivatives: the law supplies none."""
        return StateDerivatives.none()

    def interior_jacobian_is_time_invariant(self) -> bool:
        """True: the law takes no time and the material is
        time-invariant."""
        return True

    def flux_is_time_dependent(self) -> bool:
        """False: the flux is the PK1 stress, whose law takes no time."""
        return False

    def flux_is_linear_in_state(self) -> bool:
        """False: the PK1 stress is nonlinear in the displacement."""
        return False

    def stress_is_time_dependent(self) -> bool:
        """False: the PK1 stress law takes no time."""
        return False

    def interior_is_time_invariant(self) -> bool:
        """Whether the body force is declared time-invariant."""
        return declared_time_invariant(self._forcing)
