"""
ELBO objective for variational inference.

Provides ELBOObjective that uses conditional distributions directly
(no VariationalFamilyProtocol wrappers), plus convenience constructors
for common VI setups.
"""

from typing import Any, Callable, Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class ELBOObjective(Generic[Array]):
    """Negative ELBO for minimization. Satisfies FunctionProtocol.

    Uses a conditional distribution (with ``reparameterize`` and
    ``kl_divergence``) as the variational distribution. The distribution's
    ``hyp_list()`` holds the optimization parameters (e.g., BasisExpansion
    coefficients).

    Parameters
    ----------
    var_distribution : object
        Conditional distribution with ``reparameterize(x, base_samples)``,
        ``kl_divergence(x, prior)``, and ``hyp_list()``.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z, labels) -> (1, N)`` where z is
        ``(nqoi, N)`` and labels is ``(nlabel_dims, N)``.
    prior : object
        Prior distribution (marginal or IndependentJoint).
    joint_nodes : Array
        Joint quadrature nodes, shape ``(nlabel_dims + nbase_dims, N)``.
    joint_weights : Array
        Quadrature weights, shape ``(1, N)``.
    nlabel_dims : int
        Number of label dimensions in joint_nodes.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        var_distribution: Any,
        log_likelihood_fn: Callable,
        prior: Any,
        joint_nodes: Array,
        joint_weights: Array,
        nlabel_dims: int,
        bkd: Backend[Array],
    ) -> None:
        self._var_dist = var_distribution
        self._log_lik_fn = log_likelihood_fn
        self._prior = prior
        self._joint_nodes = joint_nodes
        self._joint_weights = joint_weights
        self._nlabel_dims = nlabel_dims
        self._bkd = bkd
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if hasattr(self._bkd, 'jacobian'):
            self.jacobian = self._jacobian_autograd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._var_dist.hyp_list().nactive_params()

    def nqoi(self) -> int:
        return 1

    def __call__(self, params: Array) -> Array:
        """Evaluate negative ELBO.

        Parameters
        ----------
        params : Array
            Variational parameters, shape ``(nparams, 1)`` per
            FunctionProtocol convention.

        Returns
        -------
        Array
            Negative ELBO, shape ``(1, 1)``.
        """
        self._var_dist.hyp_list().set_active_values(params[:, 0])

        label_nodes = self._joint_nodes[:self._nlabel_dims, :]
        base_nodes = self._joint_nodes[self._nlabel_dims:, :]

        z = self._var_dist.reparameterize(label_nodes, base_nodes)
        log_lik = self._log_lik_fn(z, label_nodes)

        if hasattr(self._var_dist, 'kl_divergence'):
            kl_terms = self._var_dist.kl_divergence(
                label_nodes, self._prior
            )
        else:
            # MC fallback
            log_q = self._var_dist.logpdf(label_nodes, z)
            log_prior = self._prior.logpdf(z)
            kl_terms = log_q - log_prior

        elbo = self._bkd.sum(self._joint_weights * (log_lik - kl_terms))
        return self._bkd.reshape(-elbo, (1, 1))

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
    var_distribution: Any,
    log_likelihood_fn: Callable,
    prior: Any,
    base_nodes: Array,
    base_weights: Array,
    bkd: Backend,
) -> ELBOObjective:
    """Create ELBO for single-problem VI (no labels).

    For conditional distributions that require a conditioning input x
    (e.g., ConditionalGaussian with BasisExpansion), dummy zero-valued
    label rows are prepended to the joint nodes. The degree-0 expansion
    ignores the input value but needs the correct shape.

    Parameters
    ----------
    var_distribution : object
        Conditional distribution with ``reparameterize``, ``kl_divergence``,
        ``hyp_list()``, and ``nvars()``.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z) -> (1, N)``.
    prior : object
        Prior distribution.
    base_nodes : Array
        Base samples, shape ``(nbase, nsamples)``.
    base_weights : Array
        Quadrature weights, shape ``(1, nsamples)``.
    bkd : Backend
        Computational backend.

    Returns
    -------
    ELBOObjective
    """
    def wrapped_log_lik(z: Array, labels: Array) -> Array:
        return log_likelihood_fn(z)

    # Conditional distributions need a conditioning input x of shape
    # (nvars_cond, nsamples). For single-problem VI we use dummy zeros.
    nlabel_dims = var_distribution.nvars()
    nsamples = base_nodes.shape[1]
    dummy_labels = bkd.zeros((nlabel_dims, nsamples))
    joint_nodes = bkd.concatenate([dummy_labels, base_nodes], axis=0)

    return ELBOObjective(
        var_distribution, wrapped_log_lik, prior,
        joint_nodes, base_weights, nlabel_dims=nlabel_dims, bkd=bkd,
    )
