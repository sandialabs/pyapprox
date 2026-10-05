r"""Reference design criteria for nonlinear models, by weighted rules.

The data are the blended observation of the noise-free model
:math:`g(\xi)`,

.. math::

    z = W^{1/2} g(\xi) + \Sigma_z^{1/2} \varepsilon, \qquad
    \Sigma_z = W^{1/2} \Gamma_e W^{1/2} + \Lambda_w,
    \qquad \varepsilon \sim N(0, I),

so :math:`z \mid \xi \sim N(W^{1/2} g(\xi), \Sigma_z)` with a covariance
that does not depend on :math:`\xi`, and a zero weight removes its datum
exactly. Three integrals are discretized by rules the caller chooses:

- the **inner rule** :math:`(\xi_k, \omega_k)` over the whole input space
  gives the evidence :math:`p(z) = \sum_k \omega_k p(z \mid \xi_k)` and the
  posterior weights :math:`\pi_k(z) = \omega_k p(z \mid \xi_k) / p(z)`;
- the **outer rule** :math:`((\xi_n, \varepsilon_n), v_n)` over the inputs
  and the standard-normal noise gives the expectation over data
  :math:`z_n = W^{1/2} g(\xi_n) + \Sigma_z^{1/2} \varepsilon_n`;
- an optional **nuisance rule** :math:`(a_l, v_l)` gives
  :math:`p(z \mid m) = \sum_l v_l p(z \mid m, a_l)` for the information
  gain about parameters in the presence of nuisances.

Every model evaluation happens once, at construction; a design costs only
linear algebra. Unlike the moment-Gaussian criteria these make no Gaussian
closure, so they are exact up to the rules' discretization error.
"""

import math
from typing import Generic, List, Optional, Sequence, Tuple

from pyapprox.expdesign.protocols.relaxation import ObservationRelaxationProtocol
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.joint import JointEvaluatorProtocol
from pyapprox.probability.moments import WeightedRuleProtocol
from pyapprox.probability.protocols.covariance import CovarianceOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend


def _require(obj: object, protocol: type, name: str) -> None:
    if not isinstance(obj, protocol):
        raise TypeError(
            f"{name} must satisfy {protocol.__name__}, got {type(obj).__name__}"
        )


class _BlendedGaussianData(Generic[Array]):
    """Model outputs at the rules' points, and the blended data per design."""

    def __init__(
        self,
        evaluator: JointEvaluatorProtocol[Array],
        inner_rule: WeightedRuleProtocol[Array],
        outer_rule: WeightedRuleProtocol[Array],
        noise: CovarianceOperatorProtocol[Array],
        relaxation: ObservationRelaxationProtocol[Array],
        max_entries: int,
    ) -> None:
        _require(evaluator, JointEvaluatorProtocol, "evaluator")
        _require(inner_rule, WeightedRuleProtocol, "inner_rule")
        _require(outer_rule, WeightedRuleProtocol, "outer_rule")
        _require(noise, CovarianceOperatorProtocol, "noise")
        _require(relaxation, ObservationRelaxationProtocol, "relaxation")
        nvars, nobs = evaluator.nvars(), evaluator.nobs()
        if inner_rule.nvars() != nvars:
            raise ValueError(
                f"inner_rule has {inner_rule.nvars()} variables, the evaluator "
                f"takes {nvars}"
            )
        if outer_rule.nvars() != nvars + nobs:
            raise ValueError(
                f"outer_rule must span the {nvars} inputs and {nobs} noise "
                f"variables, {nvars + nobs} in all; it has {outer_rule.nvars()}"
            )
        if noise.nvars() != nobs or relaxation.nobs() != nobs:
            raise ValueError(
                f"noise ({noise.nvars()}) and relaxation ({relaxation.nobs()}) "
                f"must both have the evaluator's {nobs} observations"
            )
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._bkd = evaluator.bkd()
        self._evaluator = evaluator
        self._nobs = nobs
        self._noise_cov = noise.covariance()
        self._relaxation = relaxation
        self._max_entries = max_entries
        inner_points, self._inner_weights = inner_rule()
        self._inner = evaluator.evaluate(inner_points)
        outer_points, self._outer_weights = outer_rule()
        self._outer_inputs = outer_points[:nvars]
        self._noise_draws = outer_points[nvars:]
        self._outer_obs = evaluator.evaluate(self._outer_inputs).observations

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nobs(self) -> int:
        return self._nobs

    def evaluator(self) -> JointEvaluatorProtocol[Array]:
        return self._evaluator

    def ninner(self) -> int:
        """Number of inner points."""
        return int(self._inner_weights.shape[0])

    def inner_targets(self, index: int) -> Array:
        """Target block ``index`` at the inner points. Shape: (n_t, K)"""
        return self._inner.targets[index]

    def outer_inputs(self) -> Array:
        """Inputs of the outer points. Shape: (nvars, N)"""
        return self._outer_inputs

    def outer_weights(self) -> Array:
        """Shape: (N,)"""
        return self._outer_weights

    def noise_draws(self) -> Array:
        """Standard-normal noise of the outer points. Shape: (nobs, N)"""
        return self._noise_draws

    def check_design(self, design_weights: Array) -> None:
        if design_weights.ndim != 2 or design_weights.shape[0] != self._nobs:
            raise ValueError(
                f"design_weights must have shape ({self._nobs}, n), got "
                f"{tuple(design_weights.shape)}"
            )

    def scale_and_factor(self, weights: Array) -> Tuple[Array, Array, float]:
        """``sqrt(w)`` (nobs, 1), the Cholesky factor of ``Sigma_z`` and
        ``log det(2 pi Sigma_z)``."""
        bkd = self._bkd
        scale = bkd.sqrt(weights)
        cov = scale * self._noise_cov * scale.T + bkd.diag(
            self._relaxation.variances(weights)[:, 0]
        )
        factor = bkd.cholesky(cov)
        logdet = 2.0 * bkd.to_float(bkd.sum(bkd.log(bkd.diag(factor))))
        return scale, factor, logdet + self._nobs * math.log(2.0 * math.pi)

    def data(self, scale: Array, factor: Array) -> Array:
        """The outer data ``z_n``. Shape: (nobs, N)"""
        return scale * self._outer_obs + self._bkd.dot(factor, self._noise_draws)

    def batches(self, nper: int) -> List[slice]:
        """Slices of the outer points with ``nper`` entries each at most
        ``max_entries``."""
        nouter = self._outer_weights.shape[0]
        size = max(1, self._max_entries // max(1, nper))
        return [slice(ii, min(ii + size, nouter)) for ii in range(0, nouter, size)]

    def log_likelihoods(
        self, data: Array, means: Array, factor: Array, log_norm: float
    ) -> Array:
        """``log N(z_n; mu_k, Sigma_z)`` for data (nobs, Nb) and means
        shared by every datum (nobs, K) or per datum (nobs, Nb, K).
        Shape: (Nb, K)"""
        bkd = self._bkd
        nbatch, nmeans = data.shape[1], means.shape[-1]
        if means.ndim == 2:
            means = means[:, None, :]
        diff = data[:, :, None] - means
        white = bkd.solve_triangular(
            factor, bkd.reshape(diff, (self._nobs, nbatch * nmeans)), lower=True
        )
        quad = bkd.reshape(bkd.sum(white**2, axis=0), (nbatch, nmeans))
        return -0.5 * quad - 0.5 * log_norm

    def weighted_log_sum(
        self, log_terms: Array, weights: Array, what: str
    ) -> Tuple[Array, Array]:
        """``log sum_k weights_k exp(log_terms_nk)`` and the normalized
        terms, allowing signed weights. Shapes: (Nb,) and (Nb, K)"""
        bkd = self._bkd
        shift = bkd.max(log_terms, axis=1, keepdims=True)
        terms = weights[None, :] * bkd.exp(log_terms - shift)
        total = bkd.sum(terms, axis=1)
        if bkd.to_float(bkd.min(total)) <= 0.0:
            raise ValueError(
                f"the rule gives a non-positive {what} at some outer point; "
                "use a rule with positive weights or refine it"
            )
        return shift[:, 0] + bkd.log(total), terms / total[:, None]

    def inner_posterior(
        self, data: Array, scale: Array, factor: Array, log_norm: float
    ) -> Tuple[Array, Array]:
        """``log p(z_n)`` (Nb,) and posterior weights ``pi_nk`` (Nb, K)."""
        log_lik = self.log_likelihoods(
            data, scale * self._inner.observations, factor, log_norm
        )
        return self.weighted_log_sum(log_lik, self._inner_weights, "evidence")


class ReferenceAOptimal(Generic[Array]):
    r"""Expected posterior trace of a target.

    The value is
    :math:`E_z[\mathrm{tr}\, C\,\mathrm{Cov}(t \mid z)\,C^\top]`.

    Exact for any target :math:`t(\xi)`, parameters or predictions, with or
    without nuisances: the posterior is the inner rule reweighted, so
    nuisances are marginalized by the same sum. Satisfies
    ``OEDObjectiveProtocol``; the value is minimized, and the derivative
    bundle is empty.

    Parameters
    ----------
    evaluator : JointEvaluatorProtocol[Array]
        Noise-free observations and targets of the inputs ``xi``.
    inner_rule : WeightedRuleProtocol[Array]
        Points and weights over the inputs, representing their prior.
        Dense or adaptive in the directions the likelihood resolves: for
        small noise the likelihood is a narrow spike.
    outer_rule : WeightedRuleProtocol[Array]
        Points and weights over the stacked inputs and standard-normal
        noise, ``nvars + nobs`` variables.
    noise : CovarianceOperatorProtocol[Array]
        The observation noise covariance ``Gamma_e``.
    relaxation : ObservationRelaxationProtocol[Array]
        Independent-noise variances ``nu(w)`` of the blended observation.
    index : int
        Which target block of the evaluator.
    target_map : Array, optional
        ``C``, shape (n_c, n_t). Default the identity.
    max_entries : int
        Largest intermediate array, in entries; the outer points are
        processed in batches below it. Default ``2**22``.
    """

    def __init__(
        self,
        evaluator: JointEvaluatorProtocol[Array],
        inner_rule: WeightedRuleProtocol[Array],
        outer_rule: WeightedRuleProtocol[Array],
        noise: CovarianceOperatorProtocol[Array],
        relaxation: ObservationRelaxationProtocol[Array],
        index: int,
        target_map: Optional[Array] = None,
        max_entries: int = 2**22,
    ) -> None:
        self._data = _BlendedGaussianData(
            evaluator, inner_rule, outer_rule, noise, relaxation, max_entries
        )
        sizes = evaluator.target_sizes()
        if not 0 <= index < len(sizes):
            raise ValueError(
                f"index {index} is out of range for {len(sizes)} target blocks"
            )
        targets = self._data.inner_targets(index)
        if target_map is not None:
            if target_map.ndim != 2 or target_map.shape[1] != sizes[index]:
                raise ValueError(
                    f"target_map must have shape (n_c, {sizes[index]}), got "
                    f"{tuple(target_map.shape)}"
                )
            targets = evaluator.bkd().dot(target_map, targets)
        self._targets = targets

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._data.bkd()

    def nvars(self) -> int:
        """Number of design weights, one per observation."""
        return self._data.nobs()

    def nqoi(self) -> int:
        """Always 1."""
        return 1

    def _value(self, weights: Array) -> float:
        bkd = self.bkd()
        scale, factor, log_norm = self._data.scale_and_factor(weights)
        data = self._data.data(scale, factor)
        outer_weights = self._data.outer_weights()
        nper = self._data.ninner() * max(self._data.nobs(), self._targets.shape[0])
        total = 0.0
        for batch in self._data.batches(nper):
            _, post = self._data.inner_posterior(
                data[:, batch], scale, factor, log_norm
            )
            means = bkd.dot(self._targets, post.T)
            centered = self._targets[:, None, :] - means[:, :, None]
            traces = bkd.sum(bkd.sum(centered**2, axis=0) * post, axis=1)
            total += bkd.to_float(bkd.sum(outer_weights[batch] * traces))
        return total

    def __call__(self, design_weights: Array) -> Array:
        """Values for weights (nobs, n). Shape: (1, n)"""
        self._data.check_design(design_weights)
        values = [
            self._value(design_weights[:, ii : ii + 1])
            for ii in range(design_weights.shape[1])
        ]
        return self.bkd().reshape(self.bkd().asarray(values), (1, -1))

    def derivatives(self) -> Derivatives[Array]:
        """Empty: the reference values are not differentiated."""
        return Derivatives.none()


class ReferenceExpectedInformationGain(Generic[Array]):
    r"""Expected information gain, as a value to minimize.

    The value is :math:`-\mathrm{EIG}`, with
    :math:`\mathrm{EIG} = E_z[\log p(z \mid m) - \log p(z)]`.

    - Without a nuisance rule, :math:`m` is all the inputs and
      :math:`\log p(z_n \mid \xi_n) = -\tfrac12 \|\varepsilon_n\|^2
      - \tfrac12 \log\det(2\pi\Sigma_z)` exactly.
    - With a nuisance rule over the coordinates ``nuisance_indices``,
      :math:`m` is the remaining inputs and
      :math:`p(z_n \mid m_n) = \sum_l v_l\, p(z_n \mid m_n, a_l)`, which
      costs ``N x L`` model evaluations at construction.

    The information gain about a prediction is not available: it needs
    :math:`p(z \mid q)`, a conditional on the level set of :math:`q`, which
    point rules cannot give in general. Use ``ReferenceAOptimal`` for
    predictions.

    Satisfies ``OEDObjectiveProtocol`` with an empty derivative bundle.

    Parameters
    ----------
    evaluator, inner_rule, outer_rule, noise, relaxation
        As for ``ReferenceAOptimal``; the targets are not used.
    nuisance_rule : WeightedRuleProtocol[Array], optional
        Points and weights over the nuisances, representing their prior,
        which must be independent of the parameters.
    nuisance_indices : Sequence[int]
        Input coordinates that are nuisances. Required with
        ``nuisance_rule`` and empty without it.
    max_entries : int
        Largest intermediate array, in entries. Default ``2**22``.
    """

    def __init__(
        self,
        evaluator: JointEvaluatorProtocol[Array],
        inner_rule: WeightedRuleProtocol[Array],
        outer_rule: WeightedRuleProtocol[Array],
        noise: CovarianceOperatorProtocol[Array],
        relaxation: ObservationRelaxationProtocol[Array],
        nuisance_rule: Optional[WeightedRuleProtocol[Array]] = None,
        nuisance_indices: Sequence[int] = (),
        max_entries: int = 2**22,
    ) -> None:
        self._data = _BlendedGaussianData(
            evaluator, inner_rule, outer_rule, noise, relaxation, max_entries
        )
        self._nuisance_obs: Optional[Array] = None
        self._nuisance_weights: Optional[Array] = None
        indices = list(nuisance_indices)
        if nuisance_rule is None:
            if indices:
                raise ValueError("nuisance_indices given without a nuisance_rule")
            return
        _require(nuisance_rule, WeightedRuleProtocol, "nuisance_rule")
        nvars = evaluator.nvars()
        if (
            not indices
            or len(set(indices)) != len(indices)
            or not all(0 <= ii < nvars for ii in indices)
        ):
            raise ValueError(
                f"nuisance_indices must be distinct inputs in [0, {nvars}), got "
                f"{indices}"
            )
        if nuisance_rule.nvars() != len(indices):
            raise ValueError(
                f"nuisance_rule has {nuisance_rule.nvars()} variables for "
                f"{len(indices)} nuisance_indices"
            )
        self._nuisance_obs, self._nuisance_weights = self._evaluate_nuisances(
            nuisance_rule, indices
        )

    def _evaluate_nuisances(
        self, nuisance_rule: WeightedRuleProtocol[Array], indices: List[int]
    ) -> Tuple[Array, Array]:
        """Observations at every outer input with its nuisances replaced by
        each nuisance node. Shapes: (nobs, N, L) and (L,)"""
        bkd = self.bkd()
        nodes, weights = nuisance_rule()
        inputs = self._data.outer_inputs()
        nouter, nnodes = inputs.shape[1], nodes.shape[1]
        replaced = bkd.repeat(inputs, nnodes, axis=1)
        tiled = bkd.tile(nodes, (1, nouter))
        rows = [
            tiled[indices.index(ii)] if ii in indices else replaced[ii]
            for ii in range(inputs.shape[0])
        ]
        obs = self._data.evaluator().evaluate(bkd.stack(rows, axis=0)).observations
        return bkd.reshape(obs, (self.nvars(), nouter, nnodes)), weights

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._data.bkd()

    def nvars(self) -> int:
        """Number of design weights, one per observation."""
        return self._data.nobs()

    def nqoi(self) -> int:
        """Always 1."""
        return 1

    def _log_conditional(
        self,
        batch: slice,
        data: Array,
        scale: Array,
        factor: Array,
        log_norm: float,
    ) -> Array:
        """``log p(z_n | m_n)`` for a batch of outer points. Shape: (Nb,)"""
        bkd = self.bkd()
        if self._nuisance_obs is None or self._nuisance_weights is None:
            draws = self._data.noise_draws()[:, batch]
            return -0.5 * bkd.sum(draws**2, axis=0) - 0.5 * log_norm
        means = scale[:, :, None] * self._nuisance_obs[:, batch, :]
        log_lik = self._data.log_likelihoods(data, means, factor, log_norm)
        log_sum, _ = self._data.weighted_log_sum(
            log_lik, self._nuisance_weights, "conditional density"
        )
        return log_sum

    def expected_information_gain(self, design_weights: Array) -> float:
        """The expected information gain at weights (nobs, 1)."""
        self._data.check_design(design_weights)
        if design_weights.shape[1] != 1:
            raise ValueError("expected_information_gain takes one design")
        bkd = self.bkd()
        scale, factor, log_norm = self._data.scale_and_factor(design_weights)
        data = self._data.data(scale, factor)
        outer_weights = self._data.outer_weights()
        nnodes = (
            1 if self._nuisance_weights is None else self._nuisance_weights.shape[0]
        )
        nper = self._data.nobs() * max(self._data.ninner(), nnodes)
        total = 0.0
        for batch in self._data.batches(nper):
            log_evidence, _ = self._data.inner_posterior(
                data[:, batch], scale, factor, log_norm
            )
            log_cond = self._log_conditional(
                batch, data[:, batch], scale, factor, log_norm
            )
            total += bkd.to_float(
                bkd.sum(outer_weights[batch] * (log_cond - log_evidence))
            )
        return total

    def __call__(self, design_weights: Array) -> Array:
        """``-EIG`` for weights (nobs, n). Shape: (1, n)"""
        self._data.check_design(design_weights)
        values = [
            -self.expected_information_gain(design_weights[:, ii : ii + 1])
            for ii in range(design_weights.shape[1])
        ]
        return self.bkd().reshape(self.bkd().asarray(values), (1, -1))

    def derivatives(self) -> Derivatives[Array]:
        """Empty: the reference values are not differentiated."""
        return Derivatives.none()
