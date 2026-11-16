"""2D elasticity QoI functionals for collocation-based PDE solutions.

Provides three QoI functionals for 2D elastic problems:
1. OuterWallRadialDisplacementFunctional - average u_r on a boundary
2. AverageHoopStressFunctional - average sigma_tt over a subdomain
3. StrainEnergyFunctional2D - total strain energy integral

All satisfy SteadyFunctionalProtocol. The stress-based ones are functions
of the state and the parameters alone: they evaluate the Lame
parameters from the parameters with the same map the forward model's
parameterization uses, so the material they read is the one the state
was solved with, and their parameter Jacobian carries the material's
dependence on the parameters.
"""
# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Generic, Literal, Protocol, Tuple, runtime_checkable

from pyapprox.implicitfunction.functionals.derivatives import (
    SteadyFunctionalDerivatives,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LameFieldMapProtocol(Protocol, Generic[Array]):
    """The parameters-to-material map the stress functionals evaluate:
    the stacked Lame fields ``[mu; lamda]`` and their Jacobian, as
    ``FixedPoissonRatioLameMap`` provides."""

    def nvars(self) -> int:
        """The number of parameters."""
        ...

    def __call__(self, params_1d: Array) -> Array:
        """``[mu; lamda]``. Shape: (2*npts,)"""
        ...

    def jacobian(self, params_1d: Array) -> Array:
        """Shape: (2*npts, nvars)"""
        ...


@runtime_checkable
class HoopStressSourceProtocol(Protocol, Generic[Array]):
    """What the hoop-stress functional reads: the hoop stress for given
    Lame parameters, its state Jacobian and its pointwise Lame
    derivatives. Both 2D stress post-processors satisfy it."""

    def npts(self) -> int: ...

    def hoop_stress(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def hoop_stress_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def hoop_stress_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...


@runtime_checkable
class StrainEnergySourceProtocol(Protocol, Generic[Array]):
    """What the strain-energy functional reads, as for
    :class:`HoopStressSourceProtocol`."""

    def npts(self) -> int: ...

    def strain_energy_density(self, state: Array, mu: Array, lamda: Array) -> Array: ...

    def strain_energy_density_state_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Array: ...

    def strain_energy_density_lame_jacobian(
        self, state: Array, mu: Array, lamda: Array
    ) -> Tuple[Array, Array]: ...


class _LameOfParams(Generic[Array]):
    """The Lame parameters a stress functional reads at ``param``, and
    the chain rule through the map."""

    def __init__(
        self, lame_map: LameFieldMapProtocol[Array], npts: int, bkd: Backend[Array]
    ) -> None:
        if not isinstance(lame_map, LameFieldMapProtocol):
            raise TypeError(
                "lame_map must satisfy LameFieldMapProtocol, got "
                f"{type(lame_map).__name__}"
            )
        self._map = lame_map
        self._npts = npts
        self._bkd = bkd

    def nparams(self) -> int:
        return self._map.nvars()

    def values(self, param: Array) -> Tuple[Array, Array]:
        """``(mu, lamda)`` at ``param`` (shape (nparams, 1))."""
        stacked = self._map(param[:, 0])
        if stacked.shape[0] != 2 * self._npts:
            raise ValueError(
                f"lame_map must give 2*npts = {2 * self._npts} values, got "
                f"{stacked.shape[0]}"
            )
        return stacked[: self._npts], stacked[self._npts :]

    def param_jacobian(self, param: Array, dq_dmu: Array, dq_dlamda: Array) -> Array:
        """:math:`[\\partial Q/\\partial \\mu, \\partial Q/\\partial
        \\lambda] \\, \\partial [\\mu; \\lambda]/\\partial p`.
        Shape: (1, nparams)"""
        dq = self._bkd.concatenate([dq_dmu, dq_dlamda])[None, :]
        return dq @ self._map.jacobian(param[:, 0])


class OuterWallRadialDisplacementFunctional(Generic[Array]):
    """Average radial displacement on a boundary.

    Computes Q = (1/n_bnd) * sum_k u_r(x_k) where
    u_r = u_x * cos(theta_k) + u_y * sin(theta_k).

    This is a linear functional of the state, so the state Jacobian
    is constant (independent of state).

    Parameters
    ----------
    outer_indices : Array, shape (n_bnd,)
        Mesh indices of boundary nodes (scalar indices, not DOF indices).
    cos_theta : Array, shape (n_bnd,)
        cos(theta) at each boundary node.
    sin_theta : Array, shape (n_bnd,)
        sin(theta) at each boundary node.
    npts : int
        Number of scalar mesh points (total DOFs = 2*npts).
    nparams : int
        Number of parameters in the forward model.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        outer_indices: Array,
        cos_theta: Array,
        sin_theta: Array,
        npts: int,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._npts = npts
        self._nparams = nparams
        self._outer_indices = outer_indices
        n_bnd = outer_indices.shape[0]

        # Pre-compute weight vector: w^T @ state = Q
        w = bkd.zeros((2 * npts,))
        inv_n = 1.0 / n_bnd
        for k in range(n_bnd):
            idx = bkd.to_int(outer_indices[k])
            w[idx] = w[idx] + cos_theta[k] * inv_n
            w[idx + npts] = w[idx + npts] + sin_theta[k] * inv_n
        self._weight_vector = w  # (2*npts,)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return 2 * self._npts

    def nparams(self) -> int:
        return self._nparams

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return SteadyFunctionalDerivatives.first_order(
            self.state_jacobian, self.param_jacobian
        )

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate Q = (1/n_bnd) * sum u_r on boundary.

        Parameters
        ----------
        state : Array, shape (2*npts, 1)
        param : Array, shape (nparams, 1)

        Returns
        -------
        Array, shape (1, 1)
        """
        bkd = self._bkd
        val = bkd.sum(self._weight_vector * state[:, 0])
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Constant Jacobian dQ/d(state) = w^T.

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        return self._weight_vector[None, :]

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0.

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))


class AverageHoopStressFunctional(Generic[Array]):
    """Average hoop stress over a subdomain.

    Computes Q = (1/A) * integral_weld sigma_tt dA using pre-computed
    quadrature weights and a stress post-processor, for the Lame
    parameters ``lame_map`` gives at the parameters. Linear and
    hyperelastic materials alike: the post-processor decides how the
    stress depends on the state.

    Parameters
    ----------
    stress_processor : HoopStressSourceProtocol
        Computes hoop stress, its state Jacobian and its Lame
        derivatives.
    lame_map : LameFieldMapProtocol
        The parameters-to-``[mu; lamda]`` map of the forward model's
        parameterization.
    quad_weights : Array, shape (npts,)
        2D quadrature weights for the subdomain.
    area : float
        Area of the subdomain (for normalization).
    bkd : Backend
        Computational backend.

    Raises
    ------
    TypeError
        If ``stress_processor`` or ``lame_map`` does not satisfy its
        protocol.
    """

    def __init__(
        self,
        stress_processor: HoopStressSourceProtocol[Array],
        lame_map: LameFieldMapProtocol[Array],
        quad_weights: Array,
        area: float,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(stress_processor, HoopStressSourceProtocol):
            raise TypeError(
                "stress_processor must satisfy HoopStressSourceProtocol, got "
                f"{type(stress_processor).__name__}"
            )
        self._bkd = bkd
        self._proc = stress_processor
        self._npts = stress_processor.npts()
        self._lame = _LameOfParams(lame_map, self._npts, bkd)
        self._weights = quad_weights
        self._inv_area = 1.0 / area

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return 2 * self._npts

    def nparams(self) -> int:
        return self._lame.nparams()

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return SteadyFunctionalDerivatives.first_order(
            self.state_jacobian, self.param_jacobian
        )

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate Q = (1/A) * integral sigma_tt dA.

        Parameters
        ----------
        state : Array, shape (2*npts, 1)
        param : Array, shape (nparams, 1)

        Returns
        -------
        Array, shape (1, 1)
        """
        bkd = self._bkd
        mu, lamda = self._lame.values(param)
        sigma_tt = self._proc.hoop_stress(state[:, 0], mu, lamda)
        val = self._inv_area * bkd.sum(self._weights * sigma_tt)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/d(state) = (1/A) * w^T @ d(sigma_tt)/d(state).

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        mu, lamda = self._lame.values(param)
        dstt_dstate = self._proc.hoop_stress_state_jacobian(state[:, 0], mu, lamda)
        # (1, npts) @ (npts, 2*npts) -> (1, 2*npts)
        return self._inv_area * (self._weights[None, :] @ dstt_dstate)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/dp at fixed state, through the material:
        (1/A) * w^T d(sigma_tt)/d[mu; lamda] @ d[mu; lamda]/dp.

        Returns
        -------
        Array, shape (1, nparams)
        """
        mu, lamda = self._lame.values(param)
        dmu, dlamda = self._proc.hoop_stress_lame_jacobian(state[:, 0], mu, lamda)
        scale = self._inv_area * self._weights
        return self._lame.param_jacobian(param, scale * dmu, scale * dlamda)


class StrainEnergyFunctional2D(Generic[Array]):
    """Total strain energy over the domain.

    Computes Q = integral_Omega psi dA for the strain energy density psi
    of the post-processor (0.5 * sigma : epsilon for linear elasticity,
    the neo-Hookean stored energy for hyperelasticity), at the Lame
    parameters ``lame_map`` gives at the parameters.

    Parameters
    ----------
    stress_processor : StrainEnergySourceProtocol
        Computes strain energy density, its state Jacobian and its Lame
        derivatives.
    lame_map : LameFieldMapProtocol
        The parameters-to-``[mu; lamda]`` map of the forward model's
        parameterization.
    quad_weights : Array, shape (npts,)
        2D quadrature weights for the full domain.
    bkd : Backend
        Computational backend.

    Raises
    ------
    TypeError
        If ``stress_processor`` or ``lame_map`` does not satisfy its
        protocol.
    """

    def __init__(
        self,
        stress_processor: StrainEnergySourceProtocol[Array],
        lame_map: LameFieldMapProtocol[Array],
        quad_weights: Array,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(stress_processor, StrainEnergySourceProtocol):
            raise TypeError(
                "stress_processor must satisfy StrainEnergySourceProtocol, got "
                f"{type(stress_processor).__name__}"
            )
        self._bkd = bkd
        self._proc = stress_processor
        self._npts = stress_processor.npts()
        self._lame = _LameOfParams(lame_map, self._npts, bkd)
        self._weights = quad_weights

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return 2 * self._npts

    def nparams(self) -> int:
        return self._lame.nparams()

    def nunique_params(self) -> int:
        return 0

    def time_dependence(self) -> Literal["steady"]:
        """Return the kind of state this functional reads.

        One steady state, shape ``(nstates, 1)``.
        """
        return "steady"

    def derivatives(self) -> SteadyFunctionalDerivatives[Array]:
        """Return the optional derivatives this functional provides."""
        return SteadyFunctionalDerivatives.first_order(
            self.state_jacobian, self.param_jacobian
        )

    def __call__(self, state: Array, param: Array) -> Array:
        """Evaluate Q = integral psi dA.

        Parameters
        ----------
        state : Array, shape (2*npts, 1)
        param : Array, shape (nparams, 1)

        Returns
        -------
        Array, shape (1, 1)
        """
        bkd = self._bkd
        mu, lamda = self._lame.values(param)
        psi = self._proc.strain_energy_density(state[:, 0], mu, lamda)
        val = bkd.sum(self._weights * psi)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/d(state) = w^T @ d(psi)/d(state).

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        mu, lamda = self._lame.values(param)
        dpsi = self._proc.strain_energy_density_state_jacobian(
            state[:, 0], mu, lamda
        )  # (npts, 2*npts)
        return self._weights[None, :] @ dpsi

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/dp at fixed state, through the material:
        w^T d(psi)/d[mu; lamda] @ d[mu; lamda]/dp.

        Returns
        -------
        Array, shape (1, nparams)
        """
        mu, lamda = self._lame.values(param)
        dmu, dlamda = self._proc.strain_energy_density_lame_jacobian(
            state[:, 0], mu, lamda
        )
        return self._lame.param_jacobian(
            param, self._weights * dmu, self._weights * dlamda
        )
