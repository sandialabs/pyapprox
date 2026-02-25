"""Transient forward model for parameterized collocation PDEs.

Provides TransientForwardModel which satisfies FunctionProtocol with
adjoint-based Jacobian (scalar QoI) or forward sensitivity Jacobian
(vector QoI).
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.time.config import TimeIntegrationConfig
from pyapprox.pde.time.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.pde.time.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)


class TransientForwardModel(Generic[Array]):
    """Transient parameterized PDE forward model.

    Maps PDE parameters to quantities of interest extracted from the
    transient solution. Satisfies FunctionProtocol with Jacobian
    computation via the adjoint method (scalar QoI) or forward
    sensitivities (vector QoI).

    Parameters
    ----------
    physics : object
        Parameterized collocation physics (must have set_param,
        param_jacobian, nparams, and satisfy PhysicsProtocol).
    bkd : Backend
        Computational backend.
    init_state : Array
        Initial condition for the transient solve. Shape: (nstates,).
    time_config : TimeIntegrationConfig
        Time integration configuration.
    functional : TransientFunctionalWithJacobianAndHVPProtocol, optional
        QoI functional. If None, uses AllStatesEndpointFunctional
        (nqoi = nstates, returns full solution at final time).
    """

    def __init__(
        self,
        physics,
        bkd: Backend[Array],
        init_state: Array,
        time_config: TimeIntegrationConfig,
        functional=None,
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        if parameterization is not None and not isinstance(
            parameterization, ParameterizationProtocol
        ):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._bkd = bkd
        self._physics = physics
        self._init_state = init_state
        self._time_config = time_config
        self._parameterization = parameterization

        if parameterization is not None:
            self._nparams = parameterization.nparams()
        else:
            self._nparams = physics.nparams()

        if functional is None:
            functional = AllStatesEndpointFunctional(
                physics.nstates(), self._nparams, bkd
            )
        self._functional = functional

        # Dynamic binding for jacobian
        has_param_jac = (
            (parameterization is not None
             and hasattr(parameterization, "param_jacobian"))
            or (parameterization is None
                and hasattr(physics, "param_jacobian"))
        )
        if has_param_jac:
            self.jacobian = self._jacobian_dispatch

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._nparams

    def nqoi(self) -> int:
        """Return number of output quantities of interest."""
        return self._functional.nqoi()

    def _forward_solve(
        self, param_2d: Array
    ) -> Tuple[CollocationModel[Array], Array, Array]:
        """Set parameter, solve transient problem.

        Parameters
        ----------
        param_2d : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        model : CollocationModel
            The collocation model (stores last integrator).
        solutions : Array
            Solution trajectory. Shape: (nstates, ntimes).
        times : Array
            Time points. Shape: (ntimes,).
        """
        if self._parameterization is not None:
            self._parameterization.apply(self._physics, param_2d[:, 0])
        else:
            self._physics.set_param(param_2d[:, 0])
        model = CollocationModel(
            self._physics, self._bkd,
            parameterization=self._parameterization,
        )
        # Store params on adapter so param_jacobian can access them
        model.adapter().set_param(param_2d[:, 0])
        solutions, times = model.solve_transient(
            self._init_state, self._time_config
        )
        return model, solutions, times

    def __call__(self, samples: Array) -> Array:
        """Evaluate forward model for multiple parameter samples.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nvars, nsamples).

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, nsamples).
        """
        bkd = self._bkd
        nsamples = samples.shape[1]
        nqoi = self.nqoi()
        result = bkd.zeros((nqoi, nsamples))
        result = bkd.copy(result)
        for ii in range(nsamples):
            param_2d = samples[:, ii:ii+1]
            _, fwd_sols, times = self._forward_solve(param_2d)
            qoi = self._functional(fwd_sols, param_2d)
            if qoi.ndim == 2:
                result[:, ii:ii+1] = qoi
            else:
                result[:, ii] = qoi
        return result

    def _jacobian_dispatch(self, sample: Array) -> Array:
        """Compute Jacobian of QoI w.r.t. parameters.

        For scalar QoI (nqoi == 1), uses the adjoint method via
        TimeAdjointOperatorWithHVP.

        For vector QoI (nqoi > 1), uses forward sensitivities to compute
        the full dy(T)/dp matrix.

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars).
        """
        if self._functional.nqoi() == 1:
            return self._jacobian_adjoint(sample)
        return self._jacobian_sensitivity(sample)

    def _jacobian_adjoint(self, sample: Array) -> Array:
        """Compute Jacobian via adjoint method (scalar QoI only).

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1).

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars).
        """
        # TODO: _forward_solve computes the trajectory, then
        # TimeAdjointOperatorWithHVP.jacobian redoes the forward solve
        # internally. Refactor to pass the precomputed trajectory.
        model, fwd_sols, times = self._forward_solve(sample)
        integrator = model.last_integrator()
        adjoint_op = TimeAdjointOperatorWithHVP(
            integrator, self._functional
        )
        return adjoint_op.jacobian(self._init_state, sample)

    def _jacobian_sensitivity(self, sample: Array) -> Array:
        """Compute Jacobian via forward sensitivities (vector QoI).

        Solves the tangent linear model for the full sensitivity matrix
        W = dy/dp, then applies the functional Jacobian:
            dQ/dp = dQ/dy(T) @ dy(T)/dp + dQ/dp_direct

        For AllStatesEndpointFunctional, dQ/dy(T) = I, so dQ/dp = W_T.

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1).

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars).
        """
        model, fwd_sols, times = self._forward_solve(sample)
        integrator = model.last_integrator()
        time_residual = integrator.time_residual()

        W_T = self._solve_full_forward_sensitivity(
            fwd_sols, times, time_residual
        )
        # W_T shape: (nstates, nparams)
        return W_T

    def _solve_full_forward_sensitivity(
        self,
        fwd_sols: Array,
        times: Array,
        time_residual,
    ) -> Array:
        """Solve tangent linear model for full sensitivity matrix.

        Computes W_n = dy_n/dp at each time step via:
            W_n = -(dR_n/dy_n)^{-1} @ [dR_n/dy_{n-1} @ W_{n-1} + dR_n/dp]

        Only returns W at the final time.

        Parameters
        ----------
        fwd_sols : Array
            Forward solutions. Shape: (nstates, ntimes).
        times : Array
            Time points. Shape: (ntimes,).
        time_residual : object
            Time stepping residual with param_jacobian and
            sensitivity_off_diag_jacobian methods.

        Returns
        -------
        Array
            Sensitivity matrix at final time. Shape: (nstates, nparams).
        """
        bkd = self._bkd
        nstates = fwd_sols.shape[0]
        nparams = self._nparams
        ntimes = fwd_sols.shape[1]

        W_prev = bkd.zeros((nstates, nparams))

        if hasattr(time_residual, "initial_param_jacobian"):
            deltat_0 = float(times[1] - times[0])
            time_residual.set_time(
                float(times[0]), deltat_0, fwd_sols[:, 0]
            )
            W_prev = time_residual.initial_param_jacobian()

        for nn in range(1, ntimes):
            deltat_n = float(times[nn] - times[nn - 1])
            time_residual.set_time(
                float(times[nn - 1]), deltat_n, fwd_sols[:, nn - 1]
            )

            drdy_n = time_residual.jacobian(fwd_sols[:, nn])
            drdy_nm1 = time_residual.sensitivity_off_diag_jacobian(
                fwd_sols[:, nn - 1], fwd_sols[:, nn], deltat_n
            )
            drdp_n = time_residual.param_jacobian(
                fwd_sols[:, nn - 1], fwd_sols[:, nn]
            )

            rhs = bkd.dot(drdy_nm1, W_prev) + drdp_n
            W_prev = -bkd.solve(drdy_n, rhs)

        return W_prev

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nqoi={self.nqoi()}, "
            f"nvars={self.nvars()})"
        )
