r"""Linear elasticity physics for spectral collocation.

Implements the linear elasticity equations in 2D and 3D:

.. math::

    -\nabla \cdot \sigma + f = 0

where:

- :math:`\sigma = \lambda \operatorname{tr}(\varepsilon) I + 2\mu \varepsilon`
  is the stress tensor,
- :math:`\varepsilon_{ij} = \tfrac{1}{2}(\partial u_i/\partial x_j
  + \partial u_j/\partial x_i)` is the strain tensor,
- :math:`u = (u, v)` or :math:`(u, v, w)` is the displacement field,
- :math:`\lambda, \mu` are the Lamé parameters.
"""

from typing import Generic, List, Optional, Tuple

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


class LinearElasticityPhysics(AbstractVectorPhysics[Array], Generic[Array]):
    r"""Linear elasticity physics in 2D or 3D.

    Implements the equilibrium equation

    .. math::

        -\nabla \cdot \sigma + f = 0

    where the stress tensor is

    .. math::

        \sigma = \lambda \operatorname{tr}(\varepsilon) I + 2\mu \varepsilon

    and the strain tensor is

    .. math::

        \varepsilon_{ij} = \tfrac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
        + \frac{\partial u_j}{\partial x_i}\right)

    The residual is formulated as :math:`\nabla \cdot \sigma + f`, so
    that it vanishes at equilibrium. The state stacks one
    displacement component per spatial dimension, each of length npts.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D or 3D collocation basis (provides nodes, derivative
        matrices). The number of displacement components equals
        ``basis.ndim()``.
    bkd : Backend
        Computational backend.
    lamda : float, NodalCoefficientProtocol or callable
        Lamé's first parameter :math:`\lambda`, in the forms
        ``as_nodal_coefficient`` takes (nodal values arrive as
        ``NodalField``). Must be non-negative.
    mu : float, NodalCoefficientProtocol or callable
        Shear modulus :math:`\mu`, in the same forms. Must be positive.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Forcing with the force components stacked, in the same forms but
        of length ndim*npts. Absent means none.
    density : float, NodalCoefficientProtocol or callable, optional
        The density :math:`\rho` of the dynamics :math:`\rho \ddot u =
        \nabla \cdot \sigma + f`, in the same forms as the Lamé
        parameters; read only by a second-order transient path. Absent
        means none declared, and such a path refuses the physics.
    """

    def __init__(
        self,
        basis: TensorProductBasisProtocol[Array],
        bkd: Backend[Array],
        lamda: CoefficientInput[Array],
        mu: CoefficientInput[Array],
        forcing: Optional[CoefficientInput[Array]] = None,
        density: Optional[CoefficientInput[Array]] = None,
    ):
        ndim = basis.ndim()
        if ndim not in (2, 3):
            raise ValueError("LinearElasticityPhysics requires a 2D or 3D basis")

        super().__init__(basis, bkd, ncomponents=ndim)

        self._ndim = ndim
        npts = basis.npts()
        self._lamda = as_nodal_coefficient(lamda, npts, bkd)
        self._mu = as_nodal_coefficient(mu, npts, bkd)
        self._checked_lamda = CheckedValues(self._lamda, self._check_lamda)
        self._checked_mu = CheckedValues(self._mu, self._check_mu)
        self._forcing = optional_nodal_coefficient(forcing, ndim * npts, bkd)
        self._density = optional_nodal_coefficient(density, npts, bkd)

        # Precompute first-derivative matrices per dimension
        self._D: List[Array] = [basis.derivative_matrix(1, dim) for dim in range(ndim)]
        if ndim == 2:
            self._Dx = self._D[0]  # d/dx
            self._Dy = self._D[1]  # d/dy

    # ------------------------------------------------------------------
    # Material properties
    # ------------------------------------------------------------------

    def lamda_function(self) -> NodalCoefficientProtocol[Array]:
        """Lamé's first parameter."""
        return self._lamda

    def mu_function(self) -> NodalCoefficientProtocol[Array]:
        """The shear modulus."""
        return self._mu

    def forcing_function(self) -> Optional[NodalCoefficientProtocol[Array]]:
        """The stacked forcing; None without a forcing term."""
        return self._forcing

    def lame_values(self, time: float) -> Tuple[Array, Array]:
        """:math:`(\\lambda, \\mu)` at the nodes at ``time``.

        Raises
        ------
        ValueError
            If :math:`\\mu` is not positive or :math:`\\lambda` is
            negative somewhere.
        """
        mu = self._checked_mu.values(time)
        return self._checked_lamda.values(time), mu

    def _check_mu(self, mu: Array, time: float) -> None:
        """Raise unless the shear modulus is positive everywhere."""
        mu_min = self._bkd.to_float(self._bkd.min(mu))
        if mu_min <= 0.0:
            raise ValueError(f"mu must be positive; found min {mu_min:.2e}")

    def _check_lamda(self, lam: Array, time: float) -> None:
        """Raise unless Lamé's first parameter is non-negative everywhere."""
        lam_min = self._bkd.to_float(self._bkd.min(lam))
        if lam_min < 0.0:
            raise ValueError(f"lamda must be non-negative; found min {lam_min:.2e}")

    def _get_forcing(self, time: float) -> Array:
        """Get forcing array at given time."""
        if self._forcing is None:
            return self._bkd.zeros((self.nstates(),))
        return self._forcing.values(time)

    def _extract_components(self, state: Array) -> Tuple[Array, ...]:
        """Extract displacement components from state vector.

        State is component-major: [u_0, ..., u_{n-1}, v_0, ..., v_{n-1}]
        in 2D, with a trailing w block in 3D.

        Parameters
        ----------
        state : Array
            Full state vector. Shape: (ndim*npts,)

        Returns
        -------
        Tuple[Array, ...]
            One component per spatial dimension, each shape (npts,)
        """
        npts = self.npts()
        return tuple(state[i * npts : (i + 1) * npts] for i in range(self._ndim))

    def residual(self, state: Array, time: float) -> Array:
        r"""Compute spatial residual f(u, t).

        The residual is :math:`R_i = \sum_j D_j \sigma_{ij} + f_i`, where
        :math:`\sigma = \lambda \operatorname{tr}(\varepsilon) I
        + 2\mu \varepsilon`.

        Parameters
        ----------
        state : Array
            Displacement components stacked. Shape: (ndim*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual with one block per component. Shape: (ndim*npts,)
        """
        bkd = self._bkd
        npts = self.npts()
        ndim = self._ndim
        lam, mu = self.lame_values(time)
        eps = self._strain_tensor(state)
        trace_e = eps[0][0]
        for k in range(1, ndim):
            trace_e = trace_e + eps[k][k]
        two_mu = 2.0 * mu
        forcing = self._get_forcing(time)
        residuals = []
        for i in range(ndim):
            div_sigma_i = bkd.zeros((npts,))
            for j in range(ndim):
                sigma_ij = two_mu * eps[i][j]
                if i == j:
                    sigma_ij = sigma_ij + lam * trace_e
                div_sigma_i = div_sigma_i + self._D[j] @ sigma_ij
            residuals.append(div_sigma_i + forcing[i * npts : (i + 1) * npts])
        return bkd.concatenate(residuals)

    def _strain_tensor(self, state: Array) -> List[List[Array]]:
        """Symmetric strain tensor fields eps[i][j] at all mesh points.

        Returns the full ndim x ndim nested list with eps[i][j] ==
        eps[j][i], each entry shape (npts,).
        """
        ndim = self._ndim
        comps = self._extract_components(state)
        grads = [[self._D[j] @ comps[i] for j in range(ndim)] for i in range(ndim)]
        return [
            [
                grads[i][j] if i == j else 0.5 * (grads[i][j] + grads[j][i])
                for j in range(ndim)
            ]
            for i in range(ndim)
        ]

    def stress(self, state: Array, time: float) -> List[List[Array]]:
        r"""Cauchy stress :math:`\sigma_{ij} = \lambda
        \operatorname{tr}(\varepsilon) \delta_{ij} + 2\mu
        \varepsilon_{ij}` at the nodes, for the Lame parameters held at
        ``time``.

        Returns
        -------
        List[List[Array]]
            ``sigma[i][j]``, each shape (npts,).
        """
        ndim = self._ndim
        lam, mu = self.lame_values(time)
        eps = self._strain_tensor(state)
        trace_e = eps[0][0]
        for k in range(1, ndim):
            trace_e = trace_e + eps[k][k]
        sigma: List[List[Array]] = []
        for i in range(ndim):
            row: List[Array] = []
            for j in range(ndim):
                sigma_ij = 2.0 * mu * eps[i][j]
                if i == j:
                    sigma_ij = sigma_ij + lam * trace_e
                row.append(sigma_ij)
            sigma.append(row)
        return sigma

    def stress_jacobian(self, state: Array, time: float) -> List[List[Array]]:
        r""":math:`\partial \sigma_{ij}/\partial u`, independent of the
        state: the block for displacement component :math:`k` is
        :math:`\mu (\delta_{ik} D_j + \delta_{jk} D_i) + \delta_{ij}
        \lambda D_k`, with the material scaling rows.

        Returns
        -------
        List[List[Array]]
            ``dsigma[i][j]``, each shape (npts, ndim*npts).
        """
        bkd = self._bkd
        ndim = self._ndim
        D = self._D
        npts = self.npts()
        lam, mu = self.lame_values(time)
        zero = bkd.zeros((npts, npts))
        result: List[List[Array]] = []
        for i in range(ndim):
            row: List[Array] = []
            for j in range(ndim):
                blocks = []
                for k in range(ndim):
                    block = zero
                    if i == k:
                        block = block + mu[:, None] * D[j]
                    if j == k:
                        block = block + mu[:, None] * D[i]
                    if i == j:
                        block = block + lam[:, None] * D[k]
                    blocks.append(block)
                row.append(bkd.concatenate(blocks, axis=1))
            result.append(row)
        return result

    def jacobian(self, state: Array, time: float) -> Array:
        r"""Compute the state Jacobian, independent of the state.

        Derived from :math:`R_i = \sum_j D_j \sigma_{ij}` (the divergence
        operator is always the LEFT factor, so the ordering is correct on
        curvilinear meshes where :math:`D_i D_j \neq D_j D_i`):

        .. math::

            J_{ii} &= D_i \operatorname{diag}(\lambda + 2\mu) D_i
            + \sum_{k \neq i} D_k \operatorname{diag}(\mu) D_k, \\
            J_{ij} &= D_i \operatorname{diag}(\lambda) D_j
            + D_j \operatorname{diag}(\mu) D_i \quad (i \neq j).

        This is the exact Jacobian of ``residual`` for constant and
        variable Lamé parameters alike. Products with diagonal matrices
        are formed by row scaling.

        Parameters
        ----------
        state : Array
            Displacement components stacked. Shape: (ndim*npts,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (ndim*npts, ndim*npts)
        """
        bkd = self._bkd
        ndim = self._ndim
        D = self._D
        lam, mu = self.lame_values(time)
        lam_D = [lam[:, None] * D[j] for j in range(ndim)]
        mu_D = [mu[:, None] * D[j] for j in range(ndim)]
        rows = []
        for i in range(ndim):
            blocks = []
            for j in range(ndim):
                if i == j:
                    block = D[i] @ (lam_D[i] + 2.0 * mu_D[i])
                    for k in range(ndim):
                        if k != i:
                            block = block + D[k] @ mu_D[k]
                else:
                    block = D[i] @ lam_D[j] + D[j] @ mu_D[i]
                blocks.append(block)
            rows.append(bkd.concatenate(blocks, axis=1))
        return bkd.concatenate(rows, axis=0)

    def state_state_hvp(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        """Compute lambda^T (d^2R/du^2) w = 0 (residual linear in state).

        Linear elasticity's residual is linear in the displacement, so
        the contraction is exactly zero.
        """
        return self._bkd.zeros((self.nstates(),))

    def first_order_transient_refusal(self) -> str:
        """Return why a first-order transient path refuses this physics.

        Satisfies ``SecondOrderInTimeProtocol``: elasticity is second
        order in time, so ``du/dt = div(sigma) + f`` is not its dynamics.
        """
        return second_order_in_time_refusal(type(self).__name__)

    def interior_density_weights(self) -> Array:
        r"""The nodal density :math:`\rho(x_i)`, repeated for each
        displacement component: the weights of :math:`\ddot u` in
        :math:`\mathrm{diag}(\rho)\, \ddot u = f(u, t)`. Shape: (ndim*npts,)

        Raises
        ------
        ValueError
            If no density was declared, it is declared time-dependent (the
            mass is fixed for a system's life), or it is not positive
            everywhere.
        """
        bkd = self._bkd
        if self._density is None:
            raise ValueError(
                f"{type(self).__name__} declares no density; pass density= to "
                "integrate its dynamics"
            )
        if self._density.is_time_dependent():
            raise ValueError(
                "the density is declared time-dependent, but the mass of a "
                "second-order system is fixed for its life"
            )
        rho = self._density.values(0.0)
        if not bkd.to_bool(bkd.all(rho > 0.0)):
            raise ValueError("the density must be positive at every node")
        return bkd.concatenate([rho] * self._ndim)

    # -- the interior operator (CollocationInteriorOperatorProtocol)

    def interior_residual(self, state: Array, time: float) -> Array:
        """:math:`f(u, t)`, no boundary condition applied.
        Shape: (ndim*npts,)"""
        return self.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        r""":math:`\partial f/\partial u`. Shape: (ndim*npts, ndim*npts)"""
        return self.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Linear in the displacement: the second derivative is zero."""
        return StateDerivatives.linear(self._bkd)

    def interior_jacobian_is_time_invariant(self) -> bool:
        """Whether both Lamé parameters are declared time-invariant."""
        return declared_time_invariant(self._lamda, self._mu)

    def stress_is_time_dependent(self) -> bool:
        """Whether either Lamé parameter, which the stress reads, is
        declared time-dependent."""
        return not declared_time_invariant(self._lamda, self._mu)

    def stress_is_linear_in_state(self) -> bool:
        """True: Hooke's law with Lamé parameters that do not read the
        state."""
        return True

    def interior_is_time_invariant(self) -> bool:
        """Whether, besides the Lamé parameters, the forcing is declared
        time-invariant."""
        return declared_time_invariant(self._lamda, self._mu, self._forcing)

    # -- full-matrix field-derivative assemblies (engine slots)

    def _strains(self, state: Array) -> Tuple[Array, Array, Array]:
        """Strain fields (exx, exy, eyy) at all mesh points."""
        u, v = self._extract_components(state)
        exx = self._Dx @ u
        exy = 0.5 * ((self._Dy @ u) + (self._Dx @ v))
        eyy = self._Dy @ v
        return exx, exy, eyy

    def residual_mu_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\mu(u) = \\partial R/\\partial
        \\mu_{\\text{field}}`.

        With :math:`\\partial\\sigma/\\partial\\mu = 2\\varepsilon`,

        .. math::

            S_\\mu = \\begin{bmatrix}
            D_x \\mathrm{diag}(2\\varepsilon_{xx})
            + D_y \\mathrm{diag}(2\\varepsilon_{xy}) \\\\
            D_x \\mathrm{diag}(2\\varepsilon_{xy})
            + D_y \\mathrm{diag}(2\\varepsilon_{yy})
            \\end{bmatrix}

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            eps = self._strain_tensor(state)
            row_blocks = []
            for i in range(3):
                block = self._D[0] @ bkd.diag(2.0 * eps[i][0])
                for j in range(1, 3):
                    block = block + self._D[j] @ bkd.diag(2.0 * eps[i][j])
                row_blocks.append(block)
            return bkd.concatenate(row_blocks, axis=0)
        exx, exy, eyy = self._strains(state)
        top = self._Dx @ bkd.diag(2.0 * exx) + self._Dy @ bkd.diag(2.0 * exy)
        bottom = self._Dx @ bkd.diag(2.0 * exy) + self._Dy @ bkd.diag(2.0 * eyy)
        return bkd.concatenate([top, bottom], axis=0)

    def residual_lamda_jacobian(self, state: Array) -> Array:
        """Assemble :math:`S_\\lambda(u) = \\partial R/\\partial
        \\lambda_{\\text{field}}`.

        With :math:`\\partial\\sigma/\\partial\\lambda =
        \\mathrm{tr}(\\varepsilon) I`,

        .. math::

            S_\\lambda = \\begin{bmatrix}
            D_x \\mathrm{diag}(\\mathrm{tr}\\,\\varepsilon) \\\\
            D_y \\mathrm{diag}(\\mathrm{tr}\\,\\varepsilon)
            \\end{bmatrix}

        Parameters
        ----------
        state : Array
            Displacement state [u, v]. Shape: (2*npts,)

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            eps = self._strain_tensor(state)
            diag_trace = bkd.diag(eps[0][0] + eps[1][1] + eps[2][2])
            return bkd.concatenate([self._D[i] @ diag_trace for i in range(3)], axis=0)
        exx, _, eyy = self._strains(state)
        diag_trace = bkd.diag(exx + eyy)
        return bkd.concatenate([self._Dx @ diag_trace, self._Dy @ diag_trace], axis=0)

    def residual_mu_state_jacobian(self, delta: Array, state: Array) -> Array:
        """Assemble the mixed operator :math:`A_\\mu(\\delta) =
        \\partial/\\partial(u,v)\\,[S_\\mu(u,v)\\,\\delta]`.

        The stress is bilinear in :math:`(\\varepsilon, \\mu)`, so the
        assembly is state-independent with blocks (:math:`\\delta` the
        diagonal of the field direction):

        .. math::

            A_\\mu = \\begin{bmatrix}
            D_x 2\\delta D_x + D_y \\delta D_y & D_y \\delta D_x \\\\
            D_x \\delta D_y & D_x \\delta D_x + D_y 2\\delta D_y
            \\end{bmatrix}

        Parameters
        ----------
        delta : Array
            Mu-field direction. Shape: (npts,)
        state : Array
            Displacement state (unused; kept for the engine's
            mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, ndim*npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            diag_d = bkd.diag(delta)
            diag_2d = bkd.diag(2.0 * delta)
            rows = []
            for i in range(3):
                blocks = []
                for j in range(3):
                    if i == j:
                        block = self._D[i] @ diag_2d @ self._D[i]
                        for k in range(3):
                            if k != i:
                                block = block + self._D[k] @ diag_d @ self._D[k]
                    else:
                        block = self._D[j] @ diag_d @ self._D[i]
                    blocks.append(block)
                rows.append(bkd.concatenate(blocks, axis=1))
            return bkd.concatenate(rows, axis=0)
        Dx, Dy = self._Dx, self._Dy
        diag_d = bkd.diag(delta)
        diag_2d = bkd.diag(2.0 * delta)
        a_uu = Dx @ diag_2d @ Dx + Dy @ diag_d @ Dy
        a_uv = Dy @ diag_d @ Dx
        a_vu = Dx @ diag_d @ Dy
        a_vv = Dx @ diag_d @ Dx + Dy @ diag_2d @ Dy
        top = bkd.concatenate([a_uu, a_uv], axis=1)
        bottom = bkd.concatenate([a_vu, a_vv], axis=1)
        return bkd.concatenate([top, bottom], axis=0)

    def residual_lamda_state_jacobian(self, delta: Array, state: Array) -> Array:
        """Assemble the mixed operator :math:`A_\\lambda(\\delta) =
        \\partial/\\partial(u,v)\\,[S_\\lambda(u,v)\\,\\delta]`.

        .. math::

            A_\\lambda = \\begin{bmatrix}
            D_x \\delta D_x & D_x \\delta D_y \\\\
            D_y \\delta D_x & D_y \\delta D_y
            \\end{bmatrix}

        Parameters
        ----------
        delta : Array
            Lambda-field direction. Shape: (npts,)
        state : Array
            Displacement state (unused; kept for the engine's
            mixed-assembly signature).

        Returns
        -------
        Array
            Assembly. Shape: (ndim*npts, ndim*npts)
        """
        bkd = self._bkd
        if self._ndim == 3:
            diag_d = bkd.diag(delta)
            rows = [
                bkd.concatenate(
                    [self._D[i] @ diag_d @ self._D[j] for j in range(3)],
                    axis=1,
                )
                for i in range(3)
            ]
            return bkd.concatenate(rows, axis=0)
        Dx, Dy = self._Dx, self._Dy
        diag_d = bkd.diag(delta)
        top = bkd.concatenate([Dx @ diag_d @ Dx, Dx @ diag_d @ Dy], axis=1)
        bottom = bkd.concatenate([Dy @ diag_d @ Dx, Dy @ diag_d @ Dy], axis=1)
        return bkd.concatenate([top, bottom], axis=0)

    def boundary_traction_lame_jacobian(
        self,
        state: Array,
        time: float,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Assemble :math:`\\partial t/\\partial [\\mu; \\lambda]` at
        the boundary rows of one traction BC.

        Consumer convention (the BC supplies its own row indices and
        normals): ``bc_indices`` are the BC's replaced STATE rows —
        one traction component per BC (``mesh_idx + comp * npts``).
        The traction at mesh point :math:`i` depends on the Lame
        fields only through their local values:

        .. math::

            \\partial t_x/\\partial\\mu_i
            = 2\\varepsilon_{xx} n_x + 2\\varepsilon_{xy} n_y,
            \\quad
            \\partial t_x/\\partial\\lambda_i
            = \\mathrm{tr}(\\varepsilon)\\, n_x

        (similarly for :math:`t_y`), so each row pairs one nonzero in
        the mu block with one in the lambda block.

        Parameters
        ----------
        state : Array
            Displacement state. Shape: (ndim*npts,)
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
        comp = bkd.to_int(bc_indices[0]) // npts
        mesh_idx = bc_indices - comp * npts
        if self._ndim == 3:
            eps = self._strain_tensor(state)
            trace_b = (eps[0][0] + eps[1][1] + eps[2][2])[mesh_idx]
            dt_dmu = 2.0 * eps[comp][0][mesh_idx] * normals[:, 0]
            for j in range(1, 3):
                dt_dmu = dt_dmu + 2.0 * eps[comp][j][mesh_idx] * normals[:, j]
            dt_dlam = trace_b * normals[:, comp]
            return self._one_nonzero_per_block(mesh_idx, dt_dmu, dt_dlam)
        exx, exy, eyy = self._strains(state)
        exx_b = exx[mesh_idx]
        exy_b = exy[mesh_idx]
        eyy_b = eyy[mesh_idx]
        trace_b = exx_b + eyy_b
        nx = normals[:, 0]
        ny = normals[:, 1]
        if comp == 0:
            dt_dmu = 2.0 * exx_b * nx + 2.0 * exy_b * ny
            dt_dlam = trace_b * nx
        else:
            dt_dmu = 2.0 * exy_b * nx + 2.0 * eyy_b * ny
            dt_dlam = trace_b * ny
        return self._one_nonzero_per_block(mesh_idx, dt_dmu, dt_dlam)

    def _one_nonzero_per_block(
        self, mesh_idx: Array, dt_dmu: Array, dt_dlam: Array
    ) -> Array:
        r"""Rows :math:`[\,\mathrm{diag}(t_\mu) E \;|\; \mathrm{diag}
        (t_\lambda) E\,]`, :math:`E` selecting each row's node: one nonzero
        in each of the mu and lambda blocks. Shape: (n_bc, 2*npts)"""
        select = self._bkd.eye(self.npts())[mesh_idx]
        return self._bkd.concatenate(
            [dt_dmu[:, None] * select, dt_dlam[:, None] * select], axis=1
        )


def create_linear_elasticity(
    basis: TensorProductBasisProtocol[Array],
    bkd: Backend[Array],
    lamda: CoefficientInput[Array],
    mu: CoefficientInput[Array],
    forcing: Optional[CoefficientInput[Array]] = None,
    density: Optional[CoefficientInput[Array]] = None,
) -> LinearElasticityPhysics[Array]:
    r"""Create linear elasticity physics.

    Parameters
    ----------
    basis : TensorProductBasisProtocol
        2D or 3D collocation basis.
    bkd : Backend
        Computational backend.
    lamda : float, NodalCoefficientProtocol or callable
        Lamé's first parameter :math:`\lambda`.
    mu : float, NodalCoefficientProtocol or callable
        Shear modulus :math:`\mu`.
    forcing : float, NodalCoefficientProtocol or callable, optional
        Source term, components stacked.
    density : float, NodalCoefficientProtocol or callable, optional
        Density :math:`\rho`, read only by a second-order transient path.

    Returns
    -------
    LinearElasticityPhysics
        Linear elasticity physics instance.
    """
    return LinearElasticityPhysics(basis, bkd, lamda, mu, forcing, density)
