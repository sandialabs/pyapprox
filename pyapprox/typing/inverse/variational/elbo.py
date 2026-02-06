"""
ELBO objective for variational inference.

Provides a single unified ELBOObjective class that uses joint (label, base)
quadrature nodes, plus convenience constructors for common VI setups.
"""

from typing import Any, Callable, Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.inverse.variational.protocols import (
    VariationalFamilyProtocol,
    AmortizationFunctionProtocol,
)
from pyapprox.typing.inverse.variational.amortization import (
    ConstantAmortization,
)


class ELBOObjective(Generic[Array]):
    """Negative ELBO for minimization. Satisfies FunctionProtocol.

    Uses joint (label, base) quadrature nodes and an amortization function
    to compute the ELBO in a single weighted sum.

    Parameters
    ----------
    variational_family : VariationalFamilyProtocol
        The variational distribution family.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z, labels) -> (1, N)`` where z is
        ``(nvars, N)`` and labels is ``(nlabel_dims, N)``.
    prior : object
        Prior distribution with ``logpdf(z) -> (1, N)``.
    joint_nodes : Array
        Joint quadrature nodes, shape ``(nlabel_dims + nbase_dims, N)``.
    joint_weights : Array
        Quadrature weights, shape ``(1, N)``.
    amortization_fn : AmortizationFunctionProtocol
        Maps labels to per-sample variational parameters.
    nlabel_dims : int
        Number of label dimensions in joint_nodes.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        variational_family: VariationalFamilyProtocol,
        log_likelihood_fn: Callable,
        prior: Any,
        joint_nodes: Array,
        joint_weights: Array,
        amortization_fn: AmortizationFunctionProtocol,
        nlabel_dims: int,
        bkd: Backend[Array],
    ) -> None:
        self._family = variational_family
        self._log_lik_fn = log_likelihood_fn
        self._prior = prior
        self._joint_nodes = joint_nodes
        self._joint_weights = joint_weights
        self._amort = amortization_fn
        self._nlabel_dims = nlabel_dims
        self._bkd = bkd
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if hasattr(self._bkd, 'jacobian'):
            self.jacobian = self._jacobian_autograd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._amort.hyp_list().nactive_params()

    def nqoi(self) -> int:
        return 1

    def __call__(self, params: Array) -> Array:
        """Evaluate negative ELBO.

        Parameters
        ----------
        params : Array
            Amortization parameters, shape ``(nvars, 1)`` per
            FunctionProtocol convention.

        Returns
        -------
        Array
            Negative ELBO, shape ``(1, 1)``.
        """
        self._amort.hyp_list().set_active_values(params[:, 0])

        label_nodes = self._joint_nodes[:self._nlabel_dims, :]
        base_nodes = self._joint_nodes[self._nlabel_dims:, :]

        var_params = self._amort(label_nodes)
        z = self._family.reparameterize(base_nodes, var_params)
        log_lik = self._log_lik_fn(z, label_nodes)

        try:
            kl_terms = self._family.kl_divergence(
                self._prior, var_params
            )
        except NotImplementedError:
            log_q = self._family.logpdf(z, var_params)
            log_prior = self._prior.logpdf(z)
            kl_terms = log_q - log_prior

        elbo = self._bkd.sum(self._joint_weights * (log_lik - kl_terms))
        return self._bkd.reshape(-elbo, (1, 1))

    def push_params_to_family(self) -> None:
        """Push current amortization params back to the variational family.

        After optimization, the fitted params live in the amortization's
        hyp_list. This method evaluates the amortization at a dummy label
        to get the variational params and sets them on the family's
        hyp_list, making the family the source of truth.

        For ConstantAmortization (single-problem VI), this copies the
        optimized params to the family. For non-constant amortization
        (amortized VI), this sets the family to the params at a zero label
        — the amortization function remains the primary output for
        label-dependent params.
        """
        dummy_label = self._bkd.zeros((max(self._nlabel_dims, 1), 1))
        if self._nlabel_dims == 0:
            dummy_label = self._bkd.zeros((1, 1))
        var_params = self._amort(dummy_label)  # (nactive, 1)
        self._family.hyp_list().set_active_values(var_params[:, 0])

    def _jacobian_autograd(self, params: Array) -> Array:
        """Compute Jacobian via autograd.

        Parameters
        ----------
        params : Array
            Shape ``(nvars, 1)``.

        Returns
        -------
        Array
            Jacobian, shape ``(1, nvars)``.
        """
        if params.shape[1] == 1:
            p = params[:, 0]
        else:
            p = params

        def loss_func(p_flat: Array) -> Array:
            return self(self._bkd.reshape(p_flat, (len(p_flat), 1)))[0, 0]

        jac = self._bkd.jacobian(loss_func, p)
        return self._bkd.reshape(jac, (1, self.nvars()))


def make_single_problem_elbo(
    family: VariationalFamilyProtocol,
    log_likelihood_fn: Callable,
    prior: Any,
    base_samples: Array,
    weights: Array,
    bkd: Backend,
) -> ELBOObjective:
    """Create ELBO for single-problem VI (no labels).

    Parameters
    ----------
    family : VariationalFamilyProtocol
        Variational distribution family.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z) -> (1, N)``.
    prior : object
        Prior distribution.
    base_samples : Array
        Base samples, shape ``(nbase, nsamples)``.
    weights : Array
        Quadrature weights, shape ``(1, nsamples)``.
    bkd : Backend
        Computational backend.

    Returns
    -------
    ELBOObjective
    """
    amort = ConstantAmortization(
        family.hyp_list().nactive_params(), bkd,
        init_values=list(
            bkd.to_numpy(family.hyp_list().get_active_values())
        ),
    )

    def wrapped_log_lik(z: Array, labels: Array) -> Array:
        return log_likelihood_fn(z)

    # nlabel_dims=0: joint_nodes = base_samples
    return ELBOObjective(
        family, wrapped_log_lik, prior,
        base_samples, weights, amort,
        nlabel_dims=0, bkd=bkd,
    )
