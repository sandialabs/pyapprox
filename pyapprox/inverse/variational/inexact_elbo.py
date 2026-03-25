"""Inexact ELBO objective for variational inference with adaptive accuracy.

Provides InexactELBOObjective that uses an InexactGradientStrategy to
adaptively control the number of quadrature/MC samples based on the
tolerance passed by ROL's trust-region algorithm.
"""

from typing import Any, Callable, Generic, List, Optional, Tuple

from pyapprox.inverse.variational.summary import SummaryStatistic
from pyapprox.util.backends.protocols import Array, Backend


class InexactELBOObjective(Generic[Array]):
    """Negative ELBO with tolerance-dependent sample count.

    Like ``ELBOObjective`` but uses an ``InexactGradientStrategy`` to
    provide tol-dependent base samples. Satisfies ``ObjectiveProtocol``
    and provides ``inexact_value`` / ``inexact_jacobian`` for ROL
    integration.

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
    strategy : object
        An ``InexactGradientStrategyProtocol`` with
        ``samples_and_weights(tol)`` returning ``(samples, weights)``.
    nlabel_dims : int
        Number of label dimensions (prepended to joint nodes).
    bkd : Backend[Array]
        Computational backend.
    label_nodes : Array, optional
        Pre-computed label nodes, shape ``(nlabel_dims, K)``.
        For single-problem VI, leave as None (dummy zeros used).
        For discrete-group VI, provide normalized labels.
    """

    def __init__(
        self,
        var_distribution: Any,
        log_likelihood_fn: Callable[..., Any],
        prior: Any,
        strategy: Any,
        nlabel_dims: int,
        bkd: Backend[Array],
        label_nodes: Optional[Array] = None,
    ) -> None:
        self._var_dist = var_distribution
        self._log_lik_fn = log_likelihood_fn
        self._prior = prior
        self._strategy = strategy
        self._nlabel_dims = nlabel_dims
        self._bkd = bkd
        self._label_nodes = label_nodes
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if hasattr(self._bkd, "jacobian"):
            self.jacobian = self._jacobian_autograd
            self.inexact_jacobian = self._inexact_jacobian_autograd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._var_dist.hyp_list().nactive_params()

    def nqoi(self) -> int:
        return 1

    def bounds(self) -> Array:
        """Return optimization bounds from the variational distribution."""
        return self._var_dist.hyp_list().get_active_bounds()

    def _build_joint(
        self,
        tol: float,
    ) -> Tuple[Array, Array]:
        """Build joint_nodes and joint_weights from strategy samples.

        Parameters
        ----------
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        joint_nodes : Array
            Shape ``(nlabel_dims + nbase, N)`` or ``(nlabel_dims + nbase, K*M)``.
        joint_weights : Array
            Shape ``(1, N)`` or ``(1, K*M)``.
        """
        bkd = self._bkd
        base_nodes, base_weights = self._strategy.samples_and_weights(tol)
        M = base_nodes.shape[1]

        if self._label_nodes is None:
            # Single-problem: dummy zero labels
            dummy_labels = bkd.zeros((self._nlabel_dims, M))
            joint_nodes = bkd.concatenate([dummy_labels, base_nodes], axis=0)
            joint_weights = bkd.reshape(base_weights, (1, M))
        else:
            # Discrete-group: tile labels M times, base K times
            K = self._label_nodes.shape[1]
            labels_tiled = bkd.repeat(self._label_nodes, M, axis=1)
            base_tiled = bkd.tile(base_nodes, (1, K))
            joint_nodes = bkd.concatenate([labels_tiled, base_tiled], axis=0)
            joint_weights = (
                bkd.tile(
                    bkd.reshape(base_weights, (1, M)),
                    (1, K),
                )
                / K
            )

        return joint_nodes, joint_weights

    def _evaluate_elbo(self, params: Array, tol: float) -> Array:
        """Core ELBO computation with tol-dependent samples.

        Parameters
        ----------
        params : Array
            Variational parameters, shape ``(nparams, 1)``.
        tol : float
            Accuracy tolerance.

        Returns
        -------
        Array
            Negative ELBO, shape ``(1, 1)``.
        """
        bkd = self._bkd
        self._var_dist.hyp_list().set_active_values(params[:, 0])

        joint_nodes, joint_weights = self._build_joint(tol)
        label_nodes = joint_nodes[: self._nlabel_dims, :]
        base_nodes = joint_nodes[self._nlabel_dims :, :]

        z = self._var_dist.reparameterize(label_nodes, base_nodes)
        log_lik = self._log_lik_fn(z, label_nodes)

        if hasattr(self._var_dist, "kl_divergence"):
            kl_terms = self._var_dist.kl_divergence(label_nodes, self._prior)
        else:
            log_q = self._var_dist.logpdf(label_nodes, z)
            log_prior = self._prior.logpdf(z)
            kl_terms = log_q - log_prior

        elbo = bkd.sum(joint_weights * (log_lik - kl_terms))
        return bkd.reshape(-elbo, (1, 1))

    def __call__(self, params: Array) -> Array:
        """Evaluate negative ELBO using all available samples (tol=0).

        Parameters
        ----------
        params : Array
            Variational parameters, shape ``(nparams, 1)``.

        Returns
        -------
        Array
            Negative ELBO, shape ``(1, 1)``.
        """
        return self.inexact_value(params, 0.0)

    def inexact_value(self, params: Array, tol: float) -> Array:
        """Evaluate negative ELBO with tolerance-dependent accuracy.

        Parameters
        ----------
        params : Array
            Variational parameters, shape ``(nparams, 1)``.
        tol : float
            Accuracy tolerance from ROL.

        Returns
        -------
        Array
            Negative ELBO, shape ``(1, 1)``.
        """
        return self._evaluate_elbo(params, tol)

    def _jacobian_autograd(self, params: Array) -> Array:
        """Compute Jacobian via autograd (tol=0).

        Parameters
        ----------
        params : Array
            Shape ``(nvars, 1)``.

        Returns
        -------
        Array
            Jacobian, shape ``(1, nvars)``.
        """
        return self._inexact_jacobian_autograd(params, 0.0)

    def _inexact_jacobian_autograd(
        self,
        params: Array,
        tol: float,
    ) -> Array:
        """Compute Jacobian via autograd with tol-dependent accuracy.

        Parameters
        ----------
        params : Array
            Shape ``(nvars, 1)``.
        tol : float
            Accuracy tolerance from ROL.

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
            p_col = self._bkd.reshape(p_flat, (len(p_flat), 1))
            return self._evaluate_elbo(p_col, tol)[0, 0]

        jac = self._bkd.jacobian(loss_func, p)
        return self._bkd.reshape(jac, (1, self.nvars()))


def make_inexact_single_problem_elbo(
    var_distribution: Any,
    log_likelihood_fn: Callable[..., Any],
    prior: Any,
    strategy: Any,
    bkd: Backend,
) -> InexactELBOObjective:
    """Create inexact ELBO for single-problem VI (no labels).

    Like ``make_single_problem_elbo`` but uses a strategy for
    tol-dependent base samples instead of fixed quadrature.

    Parameters
    ----------
    var_distribution : object
        Conditional distribution with ``reparameterize``,
        ``kl_divergence``, ``hyp_list()``, and ``nvars()``.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z) -> (1, N)``.
    prior : object
        Prior distribution.
    strategy : object
        An ``InexactGradientStrategyProtocol`` with
        ``samples_and_weights(tol)``.
    bkd : Backend
        Computational backend.

    Returns
    -------
    InexactELBOObjective
    """

    def wrapped_log_lik(z: Array, labels: Array) -> Array:
        return log_likelihood_fn(z)

    nlabel_dims = var_distribution.nvars()

    return InexactELBOObjective(
        var_distribution,
        wrapped_log_lik,
        prior,
        strategy,
        nlabel_dims=nlabel_dims,
        bkd=bkd,
    )


def make_inexact_discrete_group_elbo(
    var_distribution: Any,
    log_likelihood_fns: List[Callable[..., Any]],
    prior: Any,
    strategy: Any,
    bkd: Backend,
    *,
    observations: Optional[List[Array]] = None,
    summary: Optional[SummaryStatistic] = None,
    labels: Optional[Array] = None,
) -> InexactELBOObjective:
    """Create inexact ELBO for discrete-group amortized VI.

    Like ``make_discrete_group_elbo`` but uses a strategy for
    tol-dependent base samples. The joint nodes are rebuilt on each
    evaluation with tol-dependent sample count.

    Provide either (``observations`` + ``summary``) or ``labels``.

    Parameters
    ----------
    var_distribution : object
        Conditional distribution with ``reparameterize``,
        ``kl_divergence``, ``hyp_list()``, and ``nvars()``.
    log_likelihood_fns : list of callables
        One per group. Each callable has signature
        ``log_lik_fn(z) -> (1, N)`` where z is ``(nqoi, N)``.
    prior : object
        Prior distribution.
    strategy : object
        An ``InexactGradientStrategyProtocol`` with
        ``samples_and_weights(tol)``.
    bkd : Backend
        Computational backend.
    observations : list of Array, optional
        Raw observations per group.
    summary : SummaryStatistic, optional
        Summary statistic mapping observations to labels.
    labels : Array, optional
        Pre-computed normalized labels, shape ``(nlabel_dims, K)``.

    Returns
    -------
    InexactELBOObjective
    """
    from pyapprox.inverse.variational.elbo import _compute_normalized_labels

    K = len(log_likelihood_fns)

    if labels is None:
        if observations is None or summary is None:
            raise ValueError(
                "Either 'labels' or both 'observations' and 'summary' must be provided"
            )
        if len(observations) != K:
            raise ValueError(
                f"Expected {K} observation arrays (one per group), "
                f"got {len(observations)}"
            )
        labels, _, _ = _compute_normalized_labels(observations, summary, bkd)

    nlabel_dims = labels.shape[0]

    if labels.shape[1] != K:
        raise ValueError(
            f"Expected {K} label columns (one per group), got {labels.shape[1]}"
        )

    # The log-likelihood dispatcher needs to know M (samples per group)
    # at evaluation time, which varies with tol. We capture K and dispatch
    # based on the total number of columns divided by K.
    def joint_log_lik(z: Array, label_nodes: Array) -> Array:
        total = z.shape[1]
        M = total // K
        parts = []
        for k in range(K):
            z_k = z[:, k * M : (k + 1) * M]
            parts.append(log_likelihood_fns[k](z_k))
        return bkd.concatenate(parts, axis=1)

    return InexactELBOObjective(
        var_distribution,
        joint_log_lik,
        prior,
        strategy,
        nlabel_dims=nlabel_dims,
        bkd=bkd,
        label_nodes=labels,
    )
