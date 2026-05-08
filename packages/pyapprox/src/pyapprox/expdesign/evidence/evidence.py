"""
Evidence computation for Bayesian OED.

The evidence (marginal likelihood) is:
    p(obs | design) = integral p(obs | theta, design) p(theta) d theta

For numerical computation with quadrature:
    evidence[j] = sum_i quad_weights[i] * likelihood[i, j]

where likelihood[i, j] = p(obs_j | theta_i, design).
"""

from typing import Generic, Optional

from pyapprox.expdesign.protocols.likelihood import (
    OEDInnerLoopLikelihoodProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class Evidence(Generic[Array]):
    """
    Compute evidence for different realizations of observational data.

    Evidence is the marginal likelihood computed by integrating the likelihood
    over the prior distribution using quadrature:

        evidence[j] = sum_i quad_weights[i] * exp(loglike[i, j])

    Parameters
    ----------
    inner_likelihood : OEDInnerLoopLikelihoodProtocol[Array]
        Inner loop likelihood providing logpdf_matrix.
    quad_weights : Array, optional
        Quadrature weights for prior integration. Shape: (ninner,)
        If None, uses uniform weights 1/ninner (Monte Carlo).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        inner_likelihood: OEDInnerLoopLikelihoodProtocol[Array],
        quad_weights: Optional[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._loglike = inner_likelihood
        self._has_fused_evidence_jacobian = hasattr(
            inner_likelihood, "evidence_jacobian"
        )
        self._ninner = inner_likelihood.ninner()
        self._nouter = inner_likelihood.nouter()

        # Set quadrature weights
        if quad_weights is None:
            # Uniform weights for Monte Carlo
            quad_weights = bkd.ones((self._ninner,)) / self._ninner
        if quad_weights.shape != (self._ninner,):
            raise ValueError(
                f"quad_weights has shape {quad_weights.shape}, "
                f"expected ({self._ninner},)"
            )
        self._quad_weights = quad_weights

        # Cached values for Jacobian computation
        self._cached_weights: Optional[Array] = None
        self._cached_loglike_matrix: Optional[Array] = None
        self._cached_like_matrix: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        return self._ninner

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        return self._nouter

    def _compute_likelihood_matrix(self, design_weights: Array) -> None:
        """Compute and cache the likelihood matrix.

        Skips recomputation when ``design_weights`` is elementwise identical
        to the value used for the last compute (common during a single
        objective/jacobian call, which touches the evidence 3–5 times with
        the same weights).
        """
        if self._cached_like_matrix is not None and self._weights_unchanged(
            design_weights
        ):
            return
        self._cached_weights = self._bkd.copy(design_weights)
        self._cached_loglike_matrix = self._loglike.logpdf_matrix(design_weights)
        self._cached_like_matrix = self._bkd.exp(self._cached_loglike_matrix)

    def _weights_unchanged(self, design_weights: Array) -> bool:
        """True if ``design_weights`` matches the last cached value bitwise.

        Returns False whenever ``design_weights`` carries an autograd graph
        (``requires_grad=True``) — caching across autograd calls would
        re-use a tensor whose ``grad_fn`` belongs to a stale graph.
        """
        if getattr(design_weights, "requires_grad", False):
            return False
        cached = self._cached_weights
        if cached is None or cached.shape != design_weights.shape:
            return False
        return bool(self._bkd.all_bool(cached == design_weights))

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute evidence for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Evidence values. Shape: (1, nouter)
        """
        self._compute_likelihood_matrix(design_weights)

        # evidence[j] = sum_i quad_weights[i] * like[i, j]
        # like_matrix: (ninner, nouter), quad_weights: (ninner,)
        evidence = self._bkd.sum(
            self._quad_weights[:, None] * self._cached_like_matrix, axis=0
        )
        return self._bkd.reshape(evidence, (1, -1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of evidence w.r.t. design weights.

        d/dw evidence[j] = sum_i quad_weights[i] * like[i,j] * d/dw loglike[i,j]

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nouter, nobs)
        """
        self._compute_likelihood_matrix(design_weights)

        quad_weighted_like = self._quad_weights[:, None] * self._cached_like_matrix

        if self._has_fused_evidence_jacobian:
            return self._loglike.evidence_jacobian(
                design_weights,
                quad_weighted_like,
            )

        # Fallback: separate jacobian_matrix + einsum
        loglike_jac = self._loglike.jacobian_matrix(design_weights)
        return self._bkd.einsum("io,iok->ok", quad_weighted_like, loglike_jac)

    @property
    def quad_weighted_like_vals(self) -> Array:
        """
        Get the quadrature-weighted likelihood values.

        This is quad_weights[:, None] * like_matrix, where like_matrix[i, j]
        is the likelihood of outer sample j given inner sample i.

        Returns
        -------
        Array
            Quadrature-weighted likelihood. Shape: (ninner, nouter)

        Raises
        ------
        RuntimeError
            If called before __call__ or jacobian.
        """
        if self._cached_like_matrix is None:
            raise RuntimeError(
                "Must call __call__ or jacobian before accessing "
                "quad_weighted_like_vals"
            )
        return self._quad_weights[:, None] * self._cached_like_matrix

    def has_fused_weighted_jacobian(self) -> bool:
        """Whether a fused weighted-jacobian kernel is available.

        Requires the inner likelihood to expose ``weighted_jacobian`` and
        ``has_weighted_jacobian`` (currently only the Gaussian likelihood
        with the numpy+numba backend satisfies this).
        """
        return (
            hasattr(self._loglike, "weighted_jacobian")
            and hasattr(self._loglike, "has_weighted_jacobian")
            and self._loglike.has_weighted_jacobian()
        )

    def fused_weighted_jacobian(
        self,
        design_weights: Array,
        weights_a: Array,
        weights_b: Array,
    ):
        """Fused jacobian contractions with two independent inner-weight
        matrices.

        Computes, in a single numba-parallel pass over the outer samples::

            full_a[q, j, k] = d/dw_k sum_i weights_a[i, q] * norm_like[i, j]
            full_b[q, j, k] = d/dw_k sum_i weights_b[i, q] * norm_like[i, j]

        where ``norm_like[i, j] = qwl[i, j] / evid[j]`` is the normalized
        quadrature-weighted likelihood, using the decomposition::

            normalized_like_jac = qwl_ratio * Jlog
                                  - (qwl / evid**2) * evid_jac[None]

        The first term is contracted inside the fused kernel
        (``part_a`` / ``part_b``); the second term — a rank-1 outer
        product over (j, k) — is applied here:

            full_a = part_a - M_a[:, :, None] * evid_jac[None]
            full_b = part_b - M_b[:, :, None] * evid_jac[None]

        with ``M[q, j] = sum_i weights[i, q] * qwl[i, j] / evid[j]**2``.

        The ``(ninner, nouter, nobs)`` intermediate in the legacy path is
        never materialized.

        Callers:

        * ``StandardDeviationMeasure`` passes ``(qoi, qoi**2)`` to get
          ``(first_mom_jac, second_mom_jac)``.
        * ``EntropicDeviationMeasure`` passes ``(exp(alpha*qoi), qoi)``
          to get ``(d_expectation, mean_jac)``.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1).
        weights_a : Array
            First inner-weight matrix. Shape: (ninner, npred).
        weights_b : Array
            Second inner-weight matrix. Shape: (ninner, npred).

        Returns
        -------
        full_a, full_b : Array, Array
            Each of shape (npred, nouter, nobs).

        Raises
        ------
        RuntimeError
            If no fused kernel is available. Check
            ``has_fused_weighted_jacobian()`` first.
        """
        if not self.has_fused_weighted_jacobian():
            raise RuntimeError(
                "No fused weighted_jacobian kernel for this backend / "
                "likelihood; check has_fused_weighted_jacobian() and fall "
                "back to the legacy path."
            )

        # Populates _cached_like_matrix.
        self._compute_likelihood_matrix(design_weights)
        bkd = self._bkd

        # evidence[j] = sum_i quad_weights[i] * like[i, j]
        qwl = self._quad_weights[:, None] * self._cached_like_matrix
        evid = bkd.sum(qwl, axis=0)                           # (nouter,)
        qwl_ratio = qwl / evid                                 # (ninner, nouter)

        # First term via fused kernel.
        part_a, part_b = self._loglike.weighted_jacobian(
            design_weights, qwl_ratio, weights_a, weights_b,
        )

        # Second term: rank-1 outer product over (j, k).
        #   M[q, j] = sum_i weights[i, q] * qwl[i, j] / evid[j]**2
        qwl_by_evid2 = qwl / (evid ** 2)                       # (ninner, nouter)
        M_a = bkd.einsum("iq,io->qo", weights_a, qwl_by_evid2)  # (npred, nouter)
        M_b = bkd.einsum("iq,io->qo", weights_b, qwl_by_evid2)

        evid_jac = self._loglike.evidence_jacobian(
            design_weights, qwl,
        )                                                       # (nouter, nobs)

        full_a = part_a - M_a[:, :, None] * evid_jac[None, :, :]
        full_b = part_b - M_b[:, :, None] * evid_jac[None, :, :]
        return full_a, full_b

    def quad_weighted_likelihood_jacobian(self, design_weights: Array) -> Array:
        """
        Compute the Jacobian of quad_weighted_like_vals w.r.t. design weights.

        This computes:
            d/dw [quad_weights[i] * like[i,j]] = quad_weights[i] * like[i,j] * d/dw
            loglike[i,j]

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (ninner, nouter, nobs)
        """
        self._compute_likelihood_matrix(design_weights)

        # Get jacobian of log-likelihood matrix
        # Shape: (ninner, nouter, nobs)
        loglike_jac = self._loglike.jacobian_matrix(design_weights)

        # d/dw [quad_weights[i] * like[i,j]] = quad_weights[i] * like[i,j] *
        # loglike_jac[i,j,k]
        quad_weighted_like = self.quad_weighted_like_vals
        return quad_weighted_like[:, :, None] * loglike_jac

    def effective_sample_size(self, design_weights: Array) -> Array:
        """
        Compute effective sample size for importance sampling.

        ESS measures the quality of the quadrature approximation.
        For uniform weights (MC), ESS = (sum like)^2 / sum(like^2).

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            ESS for each outer sample. Shape: (nouter,)
        """
        self._compute_likelihood_matrix(design_weights)

        # For MC (uniform weights): ESS = (sum like)^2 / sum(like^2)
        like_sum = self._bkd.sum(self._cached_like_matrix, axis=0)
        like_sq_sum = self._bkd.sum(self._cached_like_matrix**2, axis=0)

        # Avoid division by zero
        ess = like_sum**2 / (like_sq_sum + 1e-300)
        return ess


class LogEvidence(Generic[Array]):
    """
    Compute log-evidence for Bayesian OED.

    Log-evidence is computed with numerical stability using log-sum-exp:

        log_evidence[j] = log(sum_i quad_weights[i] * exp(loglike[i, j]))
                        = log_sum_exp(log_quad_weights + loglike[:, j])

    This is the primary quantity used in the KL-OED objective.

    Parameters
    ----------
    inner_likelihood : OEDInnerLoopLikelihoodProtocol[Array]
        Inner loop likelihood providing logpdf_matrix.
    quad_weights : Array, optional
        Quadrature weights for prior integration. Shape: (ninner,)
        If None, uses uniform weights 1/ninner (Monte Carlo).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        inner_likelihood: OEDInnerLoopLikelihoodProtocol[Array],
        quad_weights: Optional[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._loglike = inner_likelihood
        self._has_fused_evidence_jacobian = hasattr(
            inner_likelihood, "evidence_jacobian"
        )
        self._ninner = inner_likelihood.ninner()
        self._nouter = inner_likelihood.nouter()

        # Set quadrature weights
        if quad_weights is None:
            quad_weights = bkd.ones((self._ninner,)) / self._ninner
        if quad_weights.shape != (self._ninner,):
            raise ValueError(
                f"quad_weights has shape {quad_weights.shape}, "
                f"expected ({self._ninner},)"
            )
        self._quad_weights = quad_weights
        self._log_quad_weights = bkd.log(quad_weights)

        # Cached values
        self._cached_weights: Optional[Array] = None
        self._cached_loglike_matrix: Optional[Array] = None
        self._cached_log_evidence: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        return self._ninner

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        return self._nouter

    def _compute_log_evidence(self, design_weights: Array) -> None:
        """Compute and cache log-evidence using log-sum-exp.

        Skips recomputation when ``design_weights`` is elementwise identical
        to the value used for the last compute.
        """
        if self._cached_log_evidence is not None and self._weights_unchanged(
            design_weights
        ):
            return
        self._cached_weights = self._bkd.copy(design_weights)
        self._cached_loglike_matrix = self._loglike.logpdf_matrix(design_weights)
        self._compute_log_evidence_from_cached()

    def _weights_unchanged(self, design_weights: Array) -> bool:
        """True if ``design_weights`` matches the last cached value bitwise.

        Returns False whenever ``design_weights`` carries an autograd graph
        (``requires_grad=True``) — caching across autograd calls would
        re-use a tensor whose ``grad_fn`` belongs to a stale graph.
        """
        if getattr(design_weights, "requires_grad", False):
            return False
        cached = self._cached_weights
        if cached is None or cached.shape != design_weights.shape:
            return False
        return bool(self._bkd.all_bool(cached == design_weights))

    def _compute_log_evidence_from_cached(self) -> None:
        # log_evidence[j] = log_sum_exp(log_weights + loglike[:, j])
        # Use log-sum-exp trick for stability:
        # log(sum exp(x)) = max(x) + log(sum exp(x - max(x)))
        log_terms = self._log_quad_weights[:, None] + self._cached_loglike_matrix
        max_log = self._bkd.max(log_terms, axis=0, keepdims=True)
        self._cached_log_evidence = max_log[0] + self._bkd.log(
            self._bkd.sum(self._bkd.exp(log_terms - max_log), axis=0)
        )

    def __call__(self, design_weights: Array) -> Array:
        """
        Compute log-evidence for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-evidence values. Shape: (1, nouter)
        """
        self._compute_log_evidence(design_weights)
        return self._bkd.reshape(self._cached_log_evidence, (1, -1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of log-evidence w.r.t. design weights.

        Uses chain rule: d/dw log(E) = (1/E) * dE/dw

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nouter, nobs)
        """
        self._compute_log_evidence(design_weights)

        # Compute evidence from log-evidence
        evidence = self._bkd.exp(self._cached_log_evidence)  # (nouter,)

        # Compute likelihood matrix for jacobian
        like_matrix = self._bkd.exp(self._cached_loglike_matrix)  # (ninner, nouter)

        # quad_weighted_like: (ninner, nouter) = quad_weights[i] * like[i, j]
        quad_weighted_like = self._quad_weights[:, None] * like_matrix

        if self._has_fused_evidence_jacobian:
            evidence_jac = self._loglike.evidence_jacobian(
                design_weights,
                quad_weighted_like,
            )
        else:
            # Fallback: separate jacobian_matrix + einsum
            loglike_jac = self._loglike.jacobian_matrix(design_weights)
            evidence_jac = self._bkd.einsum(
                "io,iok->ok", quad_weighted_like, loglike_jac
            )

        # d/dw log(evidence) = (1/evidence) * d/dw evidence
        log_evidence_jac = evidence_jac / evidence[:, None]

        return log_evidence_jac
