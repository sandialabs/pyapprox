"""Importance-weighted convergence diagnostics for variational inference.

Provides importance sampling diagnostics that estimate the gap between
the ELBO and the log evidence, and assess the quality of the variational
approximation via effective sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Union

import numpy as np

from pyapprox.inverse.variational.convergence_protocols import (
    ConvergenceCheckResult,
)
from pyapprox.inverse.variational.protocols import (
    VariationalDistributionProtocol,
)

if TYPE_CHECKING:
    from pyapprox.inverse.variational.elbo import ELBOObjective
    from pyapprox.inverse.variational.inexact_elbo import InexactELBOObjective
from pyapprox.util.backends.protocols import Array, Backend


def _logsumexp(bkd: Backend[Array], x: Array) -> Array:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    x : Array
        1D array of values.

    Returns
    -------
    Array
        Scalar log(sum(exp(x))).
    """
    x_max = bkd.max(x)
    return x_max + bkd.log(bkd.sum(bkd.exp(x - x_max)))


@dataclass
class ImportanceWeightedMetrics:
    """Diagnostic metrics from importance-weighted estimation.

    Attributes
    ----------
    evidence_bound : float
        Lower bound on log evidence via importance sampling.
    elbo_estimate : float
        ELBO estimate from the same samples.
    evidence_gap : float
        Gap between evidence bound and ELBO (>= 0).
    ess : float
        Effective sample size from importance weights.
    ess_ratio : float
        ESS / n_samples, in [0, 1].
    var_log_weights : float
        Variance of log importance weights.
    n_samples : int
        Number of diagnostic samples used.
    """

    evidence_bound: float
    elbo_estimate: float
    evidence_gap: float
    ess: float
    ess_ratio: float
    var_log_weights: float
    n_samples: int


class ImportanceWeightedCheck(Generic[Array]):
    """Convergence check using importance-weighted evidence estimation.

    Draws fresh samples from the variational distribution and computes
    importance weights to estimate the gap between the ELBO and the
    log marginal likelihood. When this gap is large relative to recent
    ELBO improvement, further optimization will not meaningfully reduce
    the total error.

    Parameters
    ----------
    var_distribution : object
        Conditional distribution with ``reparameterize``, ``logpdf``,
        and ``hyp_list()``.
    log_likelihood_fn : callable
        ``log_likelihood_fn(z, labels) -> (1, N)``.
    log_prior_fn : callable
        ``log_prior_fn(z) -> (1, N)``.
    nlabel_dims : int
        Number of label dimensions.
    label_nodes : Array or None
        Label nodes for evaluation. None for single-problem VI.
    bkd : Backend[Array]
        Computational backend.
    n_diagnostic_samples : int
        Number of fresh samples for diagnostics.
    gap_ratio_threshold : float
        Stop when ``evidence_gap > threshold * |recent_improvement|``.
    """

    def __init__(
        self,
        var_distribution: VariationalDistributionProtocol[Array],
        log_likelihood_fn: Callable[..., Array],
        log_prior_fn: Callable[..., Array],
        nlabel_dims: int,
        label_nodes: Optional[Array],
        bkd: Backend[Array],
        n_diagnostic_samples: int = 200,
        gap_ratio_threshold: float = 10.0,
    ) -> None:
        if not isinstance(var_distribution, VariationalDistributionProtocol):
            raise TypeError(
                f"var_distribution must satisfy VariationalDistributionProtocol, "
                f"got {type(var_distribution).__name__}"
            )
        self._var_dist = var_distribution
        self._log_lik_fn = log_likelihood_fn
        self._log_prior_fn = log_prior_fn
        self._nlabel_dims = nlabel_dims
        self._label_nodes = label_nodes
        self._bkd = bkd
        self._n_samples = n_diagnostic_samples
        self._gap_ratio_threshold = gap_ratio_threshold

    def compute_log_weights(self, params: Array) -> Array:
        """Compute importance weights log w_i = log p(y|z_i) + log p(z_i) - log q(z_i).

        Parameters
        ----------
        params : Array
            Current variational parameters, shape ``(nvars, 1)``.

        Returns
        -------
        Array
            Log importance weights, shape ``(N,)``.
        """
        bkd = self._bkd
        self._var_dist.hyp_list().set_active_values(params[:, 0])

        # Determine nbase_dims from the variational distribution
        nqoi = self._var_dist.nqoi()

        # Draw fresh base samples
        base_samples = bkd.asarray(np.random.normal(0, 1, (nqoi, self._n_samples)))

        # Build label nodes
        if self._label_nodes is None:
            label_nodes = bkd.zeros((self._nlabel_dims, self._n_samples))
        else:
            label_nodes = bkd.repeat(self._label_nodes, self._n_samples, axis=1)

        # Reparameterize to get z samples
        z = self._var_dist.reparameterize(label_nodes, base_samples)

        # Compute log weights (all return shape (1, N))
        log_lik = self._log_lik_fn(z, label_nodes)
        log_prior = self._log_prior_fn(z)
        log_q = self._var_dist.logpdf(label_nodes, z)

        log_w = log_lik + log_prior - log_q  # (1, N)
        return log_w[0, :]  # (N,)

    def compute_metrics(self, params: Array) -> ImportanceWeightedMetrics:
        """Compute importance-weighted diagnostic metrics.

        Parameters
        ----------
        params : Array
            Current variational parameters, shape ``(nvars, 1)``.

        Returns
        -------
        ImportanceWeightedMetrics
            Diagnostic metrics including evidence bound, ESS, etc.
        """
        bkd = self._bkd

        # Optionally disable gradient tracking
        _no_grad = _get_no_grad_context(bkd)
        with _no_grad():
            log_w = self.compute_log_weights(params)

        n = self._n_samples

        # Evidence bound: log(1/N * sum(w_i)) = logsumexp(log_w) - log(N)
        log_n = bkd.log(bkd.asarray(float(n)))
        evidence_bound = float(
            bkd.to_numpy(_logsumexp(bkd, log_w) - log_n).flatten()[0]
        )

        # ELBO estimate: 1/N * sum(log_w_i) = mean(log_w)
        elbo_estimate = float(bkd.to_numpy(bkd.mean(log_w)).flatten()[0])

        # Evidence gap
        evidence_gap = evidence_bound - elbo_estimate

        # ESS: exp(2*logsumexp(log_w) - logsumexp(2*log_w))
        log_ess = 2.0 * _logsumexp(bkd, log_w) - _logsumexp(bkd, 2.0 * log_w)
        ess = float(bkd.to_numpy(bkd.exp(log_ess)).flatten()[0])
        ess_ratio = ess / n

        # Variance of log weights
        var_log_w = float(bkd.to_numpy(bkd.var(log_w)).flatten()[0])

        return ImportanceWeightedMetrics(
            evidence_bound=evidence_bound,
            elbo_estimate=elbo_estimate,
            evidence_gap=evidence_gap,
            ess=ess,
            ess_ratio=ess_ratio,
            var_log_weights=var_log_w,
            n_samples=n,
        )

    def check(
        self,
        params: Array,
        recent_elbo_improvement: float,
    ) -> ConvergenceCheckResult:
        """Check if optimization should stop.

        Stops when the evidence gap (approximation error) exceeds
        ``gap_ratio_threshold * |recent_elbo_improvement|`` (optimization
        progress), indicating further iterations won't meaningfully
        reduce total error.

        Parameters
        ----------
        params : Array
            Current variational parameters, shape ``(nvars, 1)``.
        recent_elbo_improvement : float
            Recent ELBO improvement (positive = improving).

        Returns
        -------
        ConvergenceCheckResult
        """
        metrics = self.compute_metrics(params)

        abs_improvement = abs(recent_elbo_improvement)
        if abs_improvement > 0:
            should_stop = (
                metrics.evidence_gap > self._gap_ratio_threshold * abs_improvement
            )
        else:
            should_stop = False

        quality = min(1.0, max(0.0, metrics.ess_ratio))

        return ConvergenceCheckResult(
            should_stop=should_stop,
            approximation_quality=quality,
            detail={
                "evidence_bound": metrics.evidence_bound,
                "elbo_estimate": metrics.elbo_estimate,
                "evidence_gap": metrics.evidence_gap,
                "ess": metrics.ess,
                "ess_ratio": metrics.ess_ratio,
                "var_log_weights": metrics.var_log_weights,
                "gap_ratio": (
                    metrics.evidence_gap / abs_improvement
                    if abs_improvement > 0
                    else float("inf")
                ),
            },
            check_type="importance_weighted",
        )


def make_importance_check_from_elbo(
    elbo_objective: Union[ELBOObjective[Array], InexactELBOObjective[Array]],
    log_prior_fn: Callable[..., Array],
    n_diagnostic_samples: int = 200,
    gap_ratio_threshold: float = 10.0,
) -> ImportanceWeightedCheck[Array]:
    """Create an ImportanceWeightedCheck from an existing ELBO objective.

    Extracts the variational distribution, log-likelihood function,
    label dimensions, and backend from the ELBO objective.

    Parameters
    ----------
    elbo_objective : ELBOObjective or InexactELBOObjective
        An existing ELBO objective.
    log_prior_fn : callable
        ``log_prior_fn(z) -> (1, N)`` where z is ``(nqoi, N)``.
    n_diagnostic_samples : int
        Number of fresh samples for diagnostics.
    gap_ratio_threshold : float
        Stop when ``evidence_gap > threshold * |recent_improvement|``.

    Returns
    -------
    ImportanceWeightedCheck
    """
    label_nodes = getattr(elbo_objective, "_label_nodes", None)

    return ImportanceWeightedCheck(
        var_distribution=elbo_objective._var_dist,
        log_likelihood_fn=elbo_objective._log_lik_fn,
        log_prior_fn=log_prior_fn,
        nlabel_dims=elbo_objective._nlabel_dims,
        label_nodes=label_nodes,
        bkd=elbo_objective.bkd(),
        n_diagnostic_samples=n_diagnostic_samples,
        gap_ratio_threshold=gap_ratio_threshold,
    )


class _NullContext:
    """No-op context manager."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        pass


def _get_no_grad_context(bkd: Backend[Array]) -> type:
    """Return torch.no_grad if using TorchBkd, else a no-op context."""
    if hasattr(bkd, "jacobian"):
        import torch

        return torch.no_grad
    return _NullContext
