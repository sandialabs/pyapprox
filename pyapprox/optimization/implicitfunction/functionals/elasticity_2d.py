"""2D elasticity QoI functionals for collocation-based PDE solutions.

Provides three QoI functionals for 2D linear elastic problems:
1. OuterWallRadialDisplacementFunctional - average u_r on a boundary
2. AverageHoopStressFunctional - average sigma_tt over a subdomain
3. StrainEnergyFunctional2D - total strain energy integral

All satisfy ParameterizedFunctionalWithJacobianProtocol.
"""
# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

from typing import Generic

from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)
from pyapprox.util.backends.protocols import Array, Backend


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
    quadrature weights and a stress post-processor.

    For linear elasticity, sigma_tt is linear in the state, so the
    state Jacobian is constant.

    Parameters
    ----------
    stress_processor : StressPostProcessor2D
        Computes hoop stress and its state Jacobian.
    quad_weights : Array, shape (npts,)
        2D quadrature weights for the subdomain.
    area : float
        Area of the subdomain (for normalization).
    nparams : int
        Number of parameters.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        stress_processor: StressPostProcessor2D[Array],
        quad_weights: Array,
        area: float,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._proc = stress_processor
        self._weights = quad_weights
        self._inv_area = 1.0 / area
        self._nparams = nparams
        self._npts = stress_processor.npts()

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
        sigma_tt = self._proc.hoop_stress(state[:, 0])
        val = self._inv_area * bkd.sum(self._weights * sigma_tt)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/d(state) = (1/A) * w^T @ d(sigma_tt)/d(state).

        State-independent for linear elasticity.

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        dstt_dstate = self._proc.hoop_stress_state_jacobian()  # (npts, 2*npts)
        # (1, npts) @ (npts, 2*npts) -> (1, 2*npts)
        return self._inv_area * (self._weights[None, :] @ dstt_dstate)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0.

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))


class HyperelasticAverageHoopStressFunctional(Generic[Array]):
    """Average hoop stress over a subdomain for hyperelastic materials.

    Identical to AverageHoopStressFunctional except the state Jacobian
    is state-dependent (nonlinear Cauchy stress).

    Parameters
    ----------
    stress_processor : HyperelasticStressPostProcessor2D
        Computes hoop stress and its state-dependent Jacobian.
    quad_weights : Array, shape (npts,)
        2D quadrature weights for the subdomain.
    area : float
        Area of the subdomain (for normalization).
    nparams : int
        Number of parameters.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        stress_processor: HyperelasticStressPostProcessor2D[Array],
        quad_weights: Array,
        area: float,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._proc = stress_processor
        self._weights = quad_weights
        self._inv_area = 1.0 / area
        self._nparams = nparams
        self._npts = stress_processor.npts()

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
        sigma_tt = self._proc.hoop_stress(state[:, 0])
        val = self._inv_area * bkd.sum(self._weights * sigma_tt)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/d(state) = (1/A) * w^T @ d(sigma_tt)/d(state).

        State-dependent for hyperelastic materials.

        Parameters
        ----------
        state : Array, shape (2*npts, 1)
        param : Array, shape (nparams, 1)

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        dstt_dstate = self._proc.hoop_stress_state_jacobian(
            state[:, 0],
        )  # (npts, 2*npts)
        return self._inv_area * (self._weights[None, :] @ dstt_dstate)

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0.

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))


class StrainEnergyFunctional2D(Generic[Array]):
    """Total strain energy over the domain.

    Computes Q = integral_Omega 0.5 * sigma : epsilon dA.

    For linear elasticity, psi = 0.5 * sigma : epsilon is quadratic
    in the state, so the state Jacobian depends on state.

    Parameters
    ----------
    stress_processor : StressPostProcessor2D
        Computes strain energy density and its state Jacobian.
    quad_weights : Array, shape (npts,)
        2D quadrature weights for the full domain.
    nparams : int
        Number of parameters.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        stress_processor: StressPostProcessor2D[Array],
        quad_weights: Array,
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._proc = stress_processor
        self._weights = quad_weights
        self._nparams = nparams
        self._npts = stress_processor.npts()

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
        psi = self._proc.strain_energy_density(state[:, 0])
        val = bkd.sum(self._weights * psi)
        return bkd.reshape(val, (1, 1))

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute dQ/d(state) = w^T @ d(psi)/d(state).

        State-dependent (psi is quadratic in state).

        Parameters
        ----------
        state : Array, shape (2*npts, 1)
        param : Array, shape (nparams, 1)

        Returns
        -------
        Array, shape (1, 2*npts)
        """
        dpsi = self._proc.strain_energy_density_state_jacobian(
            state[:, 0],
        )  # (npts, 2*npts)
        return self._weights[None, :] @ dpsi

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Return dQ/dp = 0.

        Returns
        -------
        Array, shape (1, nparams)
        """
        return self._bkd.zeros((1, self._nparams))
