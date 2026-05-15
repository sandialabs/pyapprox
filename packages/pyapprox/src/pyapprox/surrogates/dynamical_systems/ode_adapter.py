"""Adapter bridging ParametricVectorField to ODEResidualProtocol.

Translates between surrogates 2D convention (nstates, nsamples)
and ODE 1D convention (nstates,). All shape translations are
localized in this single class.
"""

from typing import Generic

from pyapprox.surrogates.dynamical_systems.protocols import (
    ParametricVectorFieldProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class VectorFieldODEAdapter(Generic[Array]):
    """Adapts a ParametricVectorField to ODEResidualWithParamJacobianProtocol.

    Shape translations (all in one place):
    - __call__(state: (n,)) -> vf(state[:, None])[:, 0] -> (n,)
    - jacobian(state: (n,)) -> vf.state_jacobian(state[:, None])[0] -> (n, n)
    - param_jacobian(state: (n,)) -> vf.param_jacobian(state[:, None])[0]
    - mass_matrix(n) -> eye(n) (standard ODE)
    - set_param(param) -> vf.hyp_list().set_active_values(param)
    - initial_param_jacobian() -> zeros(n, nparams)

    Derivative methods are dynamically bound based on what the wrapped
    vector field supports.

    Parameters
    ----------
    vector_field : ParametricVectorFieldProtocol[Array]
        Vector field to adapt.
    """

    def __init__(self, vector_field: ParametricVectorFieldProtocol[Array]):
        if not isinstance(vector_field, ParametricVectorFieldProtocol):
            raise TypeError(
                f"vector_field must satisfy ParametricVectorFieldProtocol, "
                f"got {type(vector_field).__name__}"
            )
        self._vf = vector_field
        self._bkd = vector_field.bkd()
        self._time: float = 0.0
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if hasattr(self._vf, "param_jacobian"):
            self.param_jacobian = self._param_jacobian
            self.nparams = self._nparams
            self.set_param = self._set_param
            self.initial_param_jacobian = self._initial_param_jacobian

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float) -> None:
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate f(y, t).

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Residual f(y, t). Shape: (nstates,)
        """
        states_2d = state[:, None]
        result_2d = self._vf(states_2d)
        return result_2d[:, 0]

    def jacobian(self, state: Array) -> Array:
        """Compute df/dy.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian df/dy. Shape: (nstates, nstates)
        """
        if not hasattr(self._vf, "state_jacobian"):
            raise AttributeError(
                "Wrapped vector field does not have state_jacobian"
            )
        states_2d = state[:, None]
        jac_batch = self._vf.state_jacobian(states_2d)
        return jac_batch[0]

    def mass_matrix(self, nstates: int) -> Array:
        """Return identity mass matrix.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Identity matrix. Shape: (nstates, nstates)
        """
        return self._bkd.eye(nstates)

    def apply_mass_matrix(self, vec: Array) -> Array:
        """Apply identity mass matrix (no-op).

        Parameters
        ----------
        vec : Array
            Shape: (nstates,)

        Returns
        -------
        Array
            Shape: (nstates,)
        """
        return vec

    def _nparams(self) -> int:
        return self._vf.hyp_list().nactive_params()

    def _set_param(self, param: Array) -> None:
        """Set parameter values.

        Parameters
        ----------
        param : Array
            Shape: (nparams,) or (nparams, 1)
        """
        if param.ndim == 2:
            param = param[:, 0]
        self._vf.hyp_list().set_active_values(param)

    def _param_jacobian(self, state: Array) -> Array:
        """Compute df/dp.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        states_2d = state[:, None]
        jac_batch = self._vf.param_jacobian(states_2d)
        return jac_batch[0]

    def _initial_param_jacobian(self) -> Array:
        """Return dy0/dp = 0 (IC independent of VF params).

        Returns
        -------
        Array
            Shape: (nstates, nparams)
        """
        n = self._vf.nstates()
        nparams = self._vf.hyp_list().nactive_params()
        return self._bkd.zeros((n, nparams))
