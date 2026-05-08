"""Flow matching objective for optimizer binding.

FlowMatchingObjective wraps the CFM loss as an ObjectiveProtocol so it
can be bound to ScipyTrustConstrOptimizer or similar. Uses a generic
chain-rule gradient via ``vf.jacobian_wrt_params()``.
"""

from typing import Generic, Optional

from pyapprox.generative.flowmatching.protocols import (
    DifferentiableVFProtocol,
    ParameterizedVFProtocol,
    ProbabilityPathProtocol,
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class FlowMatchingObjective(Generic[Array]):
    """Wraps CFM loss as ObjectiveProtocol for optimizer binding.

    Satisfies the FunctionProtocol interface so it can be bound to
    ScipyTrustConstrOptimizer or similar.

    Parameters
    ----------
    vf : ParameterizedVFProtocol[Array]
        Vector field with hyp_list() and sync_params().
        NOT cloned -- caller's responsibility to pass a clone.
    path : ProbabilityPathProtocol[Array]
        Probability path.
    quad_data : FlowMatchingQuadData[Array]
        Pre-assembled quadrature data.
    bkd : Backend[Array]
        Computational backend.
    time_weight : TimeWeightProtocol[Array], optional
        Time-dependent weight. Defaults to UniformWeight.
    """

    def __init__(
        self,
        vf: ParameterizedVFProtocol[Array],
        path: ProbabilityPathProtocol[Array],
        quad_data: FlowMatchingQuadData[Array],
        bkd: Backend[Array],
        time_weight: Optional[TimeWeightProtocol[Array]] = None,
    ) -> None:
        self._vf = vf
        self._path = path
        self._qd = quad_data
        self._bkd = bkd

        tw: TimeWeightProtocol[Array] = (
            time_weight if time_weight is not None else UniformWeight(bkd)
        )

        x_t = path.interpolate(quad_data.t(), quad_data.x0(), quad_data.x1())
        self._u_t = path.target_field(
            quad_data.t(), quad_data.x0(), quad_data.x1()
        )

        c = quad_data.c()
        if c is not None:
            self._vf_input = bkd.vstack([quad_data.t(), x_t, c])
        else:
            self._vf_input = bkd.vstack([quad_data.t(), x_t])

        w_t = tw(quad_data.t())  # (1, n_quad)
        self._combined_weights = quad_data.weights() * w_t[0, :]

        if isinstance(vf, DifferentiableVFProtocol):
            self._differentiable_vf: Optional[
                DifferentiableVFProtocol[Array]
            ] = vf
            self.jacobian = self._jacobian_generic
        else:
            self._differentiable_vf = None

    def nvars(self) -> int:
        """Number of active parameters."""
        return int(self._vf.hyp_list().nactive_params())

    def nqoi(self) -> int:
        """Always 1 (scalar loss)."""
        return 1

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the VF's hyperparameter list."""
        return self._vf.hyp_list()

    def __call__(self, params: Array) -> Array:
        """Evaluate loss at given parameter values.

        Parameters
        ----------
        params : Array
            Active parameter values, shape ``(nactive,)`` or
            ``(nactive, 1)``.

        Returns
        -------
        Array
            Loss value, shape ``(1, 1)``.
        """
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._vf.hyp_list().set_active_values(params)
        self._vf.sync_params()

        v_t = self._vf(self._vf_input)
        diff = v_t - self._u_t
        pointwise = self._bkd.sum(diff * diff, axis=0)  # (n_quad,)
        loss_val = self._bkd.sum(self._combined_weights * pointwise)

        return self._bkd.reshape(loss_val, (1, 1))

    def _jacobian_generic(self, params: Array) -> Array:
        """Generic gradient of the CFM loss w.r.t. active parameters.

        Uses the chain rule: dL/dp = 2 * sum_i w_i * (v - u)_i * dv/dp_i,
        where dv/dp comes from ``vf.jacobian_wrt_params()``.

        Parameters
        ----------
        params : Array
            Active parameter values, shape ``(nactive,)`` or
            ``(nactive, 1)``.

        Returns
        -------
        Array
            Gradient, shape ``(1, nactive)``.
        """
        if self._differentiable_vf is None:
            raise RuntimeError(
                "jacobian called but VF does not satisfy "
                "DifferentiableVFProtocol"
            )

        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._differentiable_vf.hyp_list().set_active_values(params)
        self._differentiable_vf.sync_params()

        v_t = self._differentiable_vf(self._vf_input)
        residual = (v_t - self._u_t).T  # (ns, nqoi)

        jac_p = self._differentiable_vf.jacobian_wrt_params(
            self._vf_input
        )  # (ns, nqoi, nactive)

        w_res = (
            self._bkd.reshape(self._combined_weights, (-1, 1)) * residual
        )  # (ns, nqoi)

        grad = 2.0 * self._bkd.einsum("sq,sqp->p", w_res, jac_p)
        return self._bkd.reshape(grad, (1, -1))
