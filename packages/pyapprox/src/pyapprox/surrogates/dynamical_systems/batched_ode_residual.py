"""BatchedBoundODEResidual — ODE residual wrapper for learned functions.

Wraps a LearnedFunctionProtocol as an ImplicitODEResidualWithParamJacobianProtocol,
enabling integration via BackwardEuler/CrankNicolson with adjoint gradients.

Supports batching k trajectories in a single system by interleaving states:
flat state = [traj_0_state_0, ..., traj_0_state_{n-1}, traj_1_state_0, ...].

The Newton Jacobian is block-diagonal (k blocks of size n_dynamic x n_dynamic),
giving O(k * n^3) solve cost instead of O((kn)^3).
"""

from typing import Generic, Optional

from pyapprox.ode.linear_operator import BlockDiagonalLinearOperator
from pyapprox.ode.mass_matrix import IdentityMassMatrix, MassMatrixProtocol
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class BatchedBoundODEResidual(Generic[Array]):
    """ODE residual wrapping a learned function for batched integration.

    Satisfies ImplicitODEResidualWithParamJacobianProtocol. The flat state
    vector has shape (n_dynamic * k,) where k is the number of trajectories.

    Parameters
    ----------
    learned_function : LearnedFunctionProtocol[Array]
        The surrogate providing __call__, jacobian_batch, jacobian_wrt_params.
    n_dynamic : int
        Number of dynamic state variables per trajectory.
    mu_batch : Array or None
        Parameter values for each trajectory. Shape: (n_params, k).
        None for parameter-free systems (nvars == n_dynamic).
    has_time_input : bool
        Whether the learned function takes time as an input variable.
    """

    def __init__(
        self,
        learned_function: LearnedFunctionProtocol[Array],
        n_dynamic: int,
        mu_batch: Optional[Array] = None,
        has_time_input: bool = False,
    ) -> None:
        self._lf = learned_function
        self._n_dynamic = n_dynamic
        self._has_time_input = has_time_input
        self._bkd = learned_function.bkd()

        self._mu_batch: Optional[Array] = mu_batch
        if mu_batch is not None:
            self._n_params = mu_batch.shape[0]
            self._k = mu_batch.shape[1]
        else:
            self._n_params = 0
            self._k = 1

        expected_nvars = n_dynamic + int(has_time_input) + self._n_params
        if learned_function.nvars() != expected_nvars:
            raise ValueError(
                f"learned_function.nvars()={learned_function.nvars()} != "
                f"n_dynamic({n_dynamic}) + has_time_input({int(has_time_input)}) "
                f"+ n_params({self._n_params}) = {expected_nvars}"
            )
        if learned_function.nqoi() != n_dynamic:
            raise ValueError(
                f"learned_function.nqoi()={learned_function.nqoi()} != "
                f"n_dynamic={n_dynamic}"
            )

        self._mass = IdentityMassMatrix(n_dynamic * self._k, self._bkd)
        self._time: float = 0.0

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_time(self, time: float) -> None:
        self._time = time

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        return self._mass

    def nparams(self) -> int:
        return self._lf.hyp_list().nactive_params()

    def get_param(self) -> Array:
        return self._lf.hyp_list().get_active_values()

    def set_param(self, param: Array) -> None:
        self._lf.hyp_list().set_active_values(param)

    def learned_function(self) -> "LearnedFunctionProtocol[Array]":
        return self._lf

    def __call__(self, state: Array) -> Array:
        """Evaluate f(state). Shape: (n_dynamic*k,) -> (n_dynamic*k,)."""
        augmented = self._assemble_augmented_input(state)
        output = self._lf(augmented)
        # output: (n_dynamic, k). Flatten column-major: [col0, col1, ...col_{k-1}]
        return self._bkd.reshape(
            self._bkd.transpose(output), (self._n_dynamic * self._k,)
        )

    def jacobian(self, state: Array) -> Array:
        """Compute df/dy as dense matrix. Shape: (n_dynamic*k, n_dynamic*k)."""
        blocks = self._state_jacobian_blocks(state)
        op = BlockDiagonalLinearOperator(blocks, self._bkd)
        return op.as_matrix()

    def newton_jacobian(
        self, state: Array, coefficient: float
    ) -> BlockDiagonalLinearOperator[Array]:
        """Compute M - coefficient*J as BlockDiagonalLinearOperator.

        Returns
        -------
        BlockDiagonalLinearOperator
            Blocks are I - coefficient * J_block^(i) for each trajectory.
        """
        jac_blocks = self._state_jacobian_blocks(state)
        eye = self._bkd.eye(self._n_dynamic)
        newton_blocks = self._bkd.stack([
            eye - coefficient * jac_blocks[i]
            for i in range(self._k)
        ])
        return BlockDiagonalLinearOperator(newton_blocks, self._bkd)

    def param_jacobian(self, state: Array) -> Array:
        """Compute df/d_eta. Shape: (n_dynamic*k, nactive_params)."""
        augmented = self._assemble_augmented_input(state)
        pjac = self._lf.jacobian_wrt_params(augmented)
        nactive = pjac.shape[2]
        return self._bkd.reshape(pjac, (self._n_dynamic * self._k, nactive))

    def initial_param_jacobian(self) -> Array:
        """dy0/d_eta = 0: the IC is user-supplied, not a function of eta.

        The parameters eta are the surrogate's learned coefficients
        controlling the RHS f(y; eta). The initial condition y0 is set
        externally via integrator.solve(init_state).
        """
        return self._bkd.zeros(
            (self._n_dynamic * self._k, self.nparams())
        )

    def _assemble_augmented_input(self, state: Array) -> Array:
        """Build augmented input (nvars, k) from flat state."""
        # Flat layout: [traj0_s0, ..., traj0_s_{n-1}, traj1_s0, ...]
        # Reshape to (k, n_dynamic) then transpose to (n_dynamic, k)
        dynamic = self._bkd.transpose(
            self._bkd.reshape(state, (self._k, self._n_dynamic))
        )
        parts = [dynamic]
        if self._has_time_input:
            time_row = self._bkd.full((1, self._k), self._time)
            parts.append(time_row)
        if self._mu_batch is not None:
            parts.append(self._mu_batch)
        return self._bkd.vstack(parts)

    def _state_jacobian_blocks(self, state: Array) -> Array:
        """Compute per-trajectory Jacobian blocks. Shape: (k, n_dynamic, n_dynamic)."""
        augmented = self._assemble_augmented_input(state)
        full_jac = self._lf.jacobian_batch(augmented)
        return full_jac[:, :, :self._n_dynamic]
