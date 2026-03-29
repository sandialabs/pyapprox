"""Conditional flow matching loss and objective.

Provides the CFM loss function for training vector fields, along with
a uniform time weight and a FlowMatchingObjective that wraps the loss
as an ObjectiveProtocol for optimizer binding.
"""

from typing import Generic, Optional

from pyapprox.surrogates.flowmatching.protocols import (
    ProbabilityPathProtocol,
    TimeWeightProtocol,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class UniformWeight(Generic[Array]):
    """Uniform time weight: w(t) = 1.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, t: Array) -> Array:
        """Evaluate weight. t: (1, ns) -> (1, ns) of ones."""
        return self._bkd.ones_like(t)


class CFMLoss(Generic[Array]):
    """Conditional flow matching loss.

    Computes the weighted squared error between a learned vector field
    and the conditional target field:

        L = sum_i w_i * w(t_i) * ||v(t_i, x_t_i) - u_t_i||^2

    where x_t_i = path.interpolate(t_i, x0_i, x1_i) and
    u_t_i = path.target_field(t_i, x0_i, x1_i).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    weight : TimeWeightProtocol[Array], optional
        Time-dependent weight. Defaults to UniformWeight.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        weight: Optional[TimeWeightProtocol[Array]] = None,
    ) -> None:
        self._bkd = bkd
        self._weight: TimeWeightProtocol[Array] = (
            weight if weight is not None else UniformWeight(bkd)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def weight(self) -> TimeWeightProtocol[Array]:
        """Return the time weight function."""
        return self._weight

    def integrand(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        t: Array,
        x0: Array,
        x1: Array,
        c: Optional[Array] = None,
    ) -> Array:
        """Pointwise CFM loss integrand.

        Parameters
        ----------
        vf : callable
            Vector field, ``(nvars_in, ns) -> (d, ns)``.
        path : ProbabilityPathProtocol[Array]
            Probability path.
        t : Array
            Time values, shape ``(1, ns)``.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.
        c : Array, optional
            Conditioning variables, shape ``(m, ns)``.

        Returns
        -------
        Array
            Per-sample loss, shape ``(ns,)``.
        """
        x_t = path.interpolate(t, x0, x1)
        u_t = path.target_field(t, x0, x1)

        if c is not None:
            vf_input = self._bkd.vstack([t, x_t, c])
        else:
            vf_input = self._bkd.vstack([t, x_t])

        v_t = vf(vf_input)  # type: ignore[operator]
        diff = v_t - u_t
        w_t = self._weight(t)  # (1, ns)
        return w_t[0, :] * self._bkd.sum(diff * diff, axis=0)

    def __call__(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        t: Array,
        x0: Array,
        x1: Array,
        weights: Array,
        c: Optional[Array] = None,
    ) -> Array:
        """Weighted CFM loss (quadrature approximation).

        Parameters
        ----------
        vf : callable
            Vector field.
        path : ProbabilityPathProtocol[Array]
            Probability path.
        t : Array
            Time values, shape ``(1, ns)``.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.
        weights : Array
            Quadrature weights, shape ``(ns,)``.
        c : Array, optional
            Conditioning variables, shape ``(m, ns)``.

        Returns
        -------
        Array
            Scalar loss value.
        """
        return self._bkd.sum(weights * self.integrand(vf, path, t, x0, x1, c))


class FlowMatchingObjective(Generic[Array]):
    """Wraps CFMLoss as ObjectiveProtocol for optimizer binding.

    Satisfies the FunctionProtocol interface so it can be bound to
    ScipyTrustConstrOptimizer or similar.

    Parameters
    ----------
    vf : object
        Vector field (e.g. BasisExpansion). NOT cloned — caller's
        responsibility to pass a clone.
    path : ProbabilityPathProtocol[Array]
        Probability path.
    loss : CFMLoss[Array]
        CFM loss function.
    quad_data : FlowMatchingQuadData[Array]
        Pre-assembled quadrature data.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        loss: CFMLoss[Array],
        quad_data: FlowMatchingQuadData[Array],
        bkd: Backend[Array],
    ) -> None:
        self._vf = vf
        self._path = path
        self._loss = loss
        self._qd = quad_data
        self._bkd = bkd

        # Pre-compute vf_input and target field (they don't depend on params)
        x_t = path.interpolate(quad_data.t(), quad_data.x0(), quad_data.x1())
        self._u_t = path.target_field(quad_data.t(), quad_data.x0(), quad_data.x1())

        if quad_data.c() is not None:
            self._vf_input = bkd.vstack([quad_data.t(), x_t, quad_data.c()])
        else:
            self._vf_input = bkd.vstack([quad_data.t(), x_t])

        # Pre-compute combined weights: quad_weights * time_weight
        w_t = loss.weight()(quad_data.t())  # (1, n_quad)
        self._combined_weights = quad_data.weights() * w_t[0, :]

        # Bind analytical jacobian
        self.jacobian = self._jacobian_analytical

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
        self._vf._sync_from_hyp_list()

        v_t = self._vf(self._vf_input)  # type: ignore[operator]
        diff = v_t - self._u_t
        pointwise = self._bkd.sum(diff * diff, axis=0)  # (n_quad,)
        loss_val = self._bkd.sum(self._combined_weights * pointwise)

        return self._bkd.reshape(loss_val, (1, 1))

    def _jacobian_analytical(self, params: Array) -> Array:
        """Analytical gradient of the CFM loss w.r.t. active parameters.

        Exploits linearity of BasisExpansion:
            L = sum_i w_i ||Phi_i @ coef - u_i||^2
            dL/dcoef = 2 * Phi^T @ diag(w) @ (Phi @ coef - U)

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
        if len(params.shape) == 2 and params.shape[1] == 1:
            params = params[:, 0]

        self._vf.hyp_list().set_active_values(params)
        self._vf._sync_from_hyp_list()

        # Phi: (n_quad, nterms), coef: (nterms, d), U: (d, n_quad)
        Phi = self._vf.basis_matrix(self._vf_input)
        coef = self._vf.get_coefficients()
        Phi.shape[1]
        self._u_t.shape[0]

        # residual: (n_quad, d) = Phi @ coef - U^T
        residual = Phi @ coef - self._u_t.T  # (n_quad, d)

        # weighted residual: diag(w) @ residual
        w = self._combined_weights  # (n_quad,)
        w_residual = self._bkd.reshape(w, (-1, 1)) * residual  # (n_quad, d)

        # Full gradient over all params: 2 * Phi^T @ w_residual
        # grad_full shape: (nterms, d)
        grad_full = 2.0 * (Phi.T @ w_residual)

        # Flatten to (nterms * d,) matching hyp_list layout
        grad_flat = self._bkd.flatten(grad_full)

        # Extract active subset
        active_grad = self._vf.hyp_list().extract_active(grad_flat)
        return self._bkd.reshape(active_grad, (1, -1))
