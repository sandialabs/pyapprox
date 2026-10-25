"""Small ODE models for tests, written against the backend.

``LotkaVolterra`` supplies its residual, its Jacobian matrix and its
Jacobian actions, so the matrix and matrix-free wrappers can be checked
against each other on one model. Its four coefficients are its
parameters, with their Jacobian and second derivatives, so adjoint
gradients and Hessian-vector products can be checked on it too.
"""

from typing import Generic, Optional

from pyapprox.ode.mass_matrix import (
    ConstantDenseMassMatrix,
    IdentityMassMatrix,
    MassMatrixProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class LotkaVolterra(Generic[Array]):
    r"""Predator-prey: :math:`x' = a x - b x z`, :math:`z' = c x z - d z`.

    The state is :math:`(x, z)` and the parameters are
    :math:`p = (a, b, c, d)`, held as a backend array so a torch graph
    through them survives. The Jacobian is
    :math:`J = [[a - b z, -b x], [c z, c x - d]]` and
    :math:`\partial f/\partial p = [[x, -x z, 0, 0], [0, 0, x z, -z]]`;
    :math:`f` is linear in :math:`p`, so its second parameter derivative
    is zero. The model does not own an initial state: callers pass
    :math:`y_0` as a fixed input, so :math:`dy_0/dp = 0` is exact and
    ``initial_param_jacobian`` / ``initial_param_hvp`` return zeros. A
    test whose :math:`y_0` depends on :math:`p` needs a model that states
    those derivatives (as in ``test_initial_state_parameters.py``); with
    this one, a derivative check of such a test fails at O(1).
    States have shape (2,); the actions take and return blocks of shape
    (2, ncols).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: float = 1.0,
        b: float = 0.1,
        c: float = 0.075,
        d: float = 1.5,
        mass: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._param = bkd.asarray([a, b, c, d])
        self._mass: MassMatrixProtocol[Array] = (
            IdentityMassMatrix(2, bkd)
            if mass is None
            else ConstantDenseMassMatrix(mass, bkd)
        )

    def bkd(self) -> Backend[Array]:
        """The computational backend."""
        return self._bkd

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        r"""The identity, or the constant mass given, so that
        :math:`M \dot y = f` can be tested with a nontrivial mass."""
        return self._mass

    def nparams(self) -> int:
        return 4

    def set_param(self, param: Array) -> None:
        """Set :math:`(a, b, c, d)`. Shape: (4,) or (4, 1)"""
        self._param = self._bkd.flatten(param)

    def __call__(self, state: Array, time: float) -> Array:
        a, b, c, d = self._param
        x, z = state[0], state[1]
        return self._bkd.stack([a * x - b * x * z, c * x * z - d * z])

    def jacobian(self, state: Array, time: float) -> Array:
        a, b, c, d = self._param
        x, z = state[0], state[1]
        bkd = self._bkd
        return bkd.stack(
            [bkd.stack([a - b * z, -b * x]), bkd.stack([c * z, c * x - d])]
        )

    def apply_jacobian(self, state: Array, time: float, v: Array) -> Array:
        a, b, c, d = self._param
        x, z = state[0], state[1]
        v_x, v_z = v[0:1], v[1:2]
        return self._bkd.vstack(
            [(a - b * z) * v_x - b * x * v_z, c * z * v_x + (c * x - d) * v_z]
        )

    def apply_jacobian_transpose(self, state: Array, time: float, w: Array) -> Array:
        a, b, c, d = self._param
        x, z = state[0], state[1]
        w_x, w_z = w[0:1], w[1:2]
        return self._bkd.vstack(
            [(a - b * z) * w_x + c * z * w_z, -b * x * w_x + (c * x - d) * w_z]
        )

    def param_jacobian(self, state: Array, time: float) -> Array:
        x, z = state[0], state[1]
        bkd = self._bkd
        zero = bkd.zeros(())
        return bkd.stack(
            [bkd.stack([x, -x * z, zero, zero]), bkd.stack([zero, zero, x * z, -z])]
        )

    def initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((2, 4))

    def state_state_hvp(
        self, state: Array, time: float, adj_state: Array, wvec: Array
    ) -> Array:
        _, b, c, _ = self._param
        w = self._bkd.flatten(wvec)
        weight = -b * adj_state[0] + c * adj_state[1]
        return self._bkd.stack([weight * w[1], weight * w[0]])

    def state_param_hvp(
        self, state: Array, time: float, adj_state: Array, vvec: Array
    ) -> Array:
        x, z = state[0], state[1]
        lx, lz = adj_state[0], adj_state[1]
        v = self._bkd.flatten(vvec)
        return self._bkd.stack(
            [
                lx * (v[0] - z * v[1]) + lz * z * v[2],
                -lx * x * v[1] + lz * (x * v[2] - v[3]),
            ]
        )

    def param_state_hvp(
        self, state: Array, time: float, adj_state: Array, wvec: Array
    ) -> Array:
        x, z = state[0], state[1]
        lx, lz = adj_state[0], adj_state[1]
        w = self._bkd.flatten(wvec)
        return self._bkd.stack(
            [
                lx * w[0],
                -lx * (z * w[0] + x * w[1]),
                lz * (z * w[0] + x * w[1]),
                -lz * w[1],
            ]
        )

    def param_param_hvp(
        self, state: Array, time: float, adj_state: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((4,))

    def initial_param_hvp(self, weight: Array, vvec: Array) -> Array:
        return self._bkd.zeros((4,))
