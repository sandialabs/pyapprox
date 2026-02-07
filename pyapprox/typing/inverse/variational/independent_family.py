"""
Independent marginal variational family.

A thin compositor that satisfies VariationalFamilyProtocol by delegating
to marginals that have ``reparameterize()`` and ``base_distribution()``
methods.  Adding a new distribution to VI requires only adding those two
methods to the marginal — zero changes here or in the ELBO.
"""

from typing import Any, Generic, List, Optional, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.probability.joint.independent import IndependentJoint


class IndependentMarginalVariationalFamily(Generic[Array]):
    """Independent marginal variational family.

    Composes any marginals that implement ``reparameterize()`` and
    ``base_distribution()`` into a joint variational family.  The family
    satisfies ``VariationalFamilyProtocol``.

    Parameters
    ----------
    marginals : Sequence
        Marginals, each with ``reparameterize()``, ``base_distribution()``,
        ``hyp_list()``, and ``logpdf()`` methods.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        marginals: Sequence,
        bkd: Backend[Array],
    ) -> None:
        if len(marginals) == 0:
            raise ValueError("At least one marginal is required")
        for i, m in enumerate(marginals):
            if not hasattr(m, "reparameterize"):
                raise TypeError(
                    f"Marginal {i} ({type(m).__name__}) missing reparameterize()"
                )
            if not hasattr(m, "base_distribution"):
                raise TypeError(
                    f"Marginal {i} ({type(m).__name__}) missing base_distribution()"
                )
        self._marginals: List = list(marginals)
        self._bkd = bkd
        self._nvars = len(marginals)
        # Aggregate hyp_lists
        self._hyp_list = marginals[0].hyp_list()
        for m in marginals[1:]:
            self._hyp_list = self._hyp_list + m.hyp_list()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def reparameterize(
        self, base_samples: Array, params: Optional[Array] = None
    ) -> Array:
        """Transform base samples to variational samples.

        Parameters
        ----------
        base_samples : Array
            Shape ``(nvars, nsamples)`` drawn from ``base_distribution()``.
        params : Array or None
            Per-sample unconstrained active params. Not yet supported.

        Returns
        -------
        Array
            Shape ``(nvars, nsamples)``.
        """
        if params is not None:
            raise NotImplementedError(
                "Per-sample params not yet supported for "
                "IndependentMarginalVariationalFamily"
            )
        parts: List[Array] = []
        for i, m in enumerate(self._marginals):
            row = self._bkd.reshape(base_samples[i], (1, -1))
            result = m.reparameterize(row)
            parts.append(result[0])
        return self._bkd.stack(parts, axis=0)

    def logpdf(
        self, samples: Array, params: Optional[Array] = None
    ) -> Array:
        """Evaluate log-pdf of the variational distribution.

        Parameters
        ----------
        samples : Array
            Shape ``(nvars, nsamples)``.
        params : Array or None
            Per-sample unconstrained active params. Not yet supported.

        Returns
        -------
        Array
            Shape ``(1, nsamples)``.
        """
        if params is not None:
            raise NotImplementedError(
                "Per-sample params not yet supported for "
                "IndependentMarginalVariationalFamily"
            )
        joint = IndependentJoint(self._marginals, self._bkd)
        return joint.logpdf(samples)

    def kl_divergence(
        self, prior: Any, params: Optional[Array] = None
    ) -> Any:
        """Compute KL(q || prior) as sum of marginal KLs.

        Requires ``prior`` to be an ``IndependentJoint`` with the same
        number of marginals, and every marginal pair must support
        ``kl_divergence``.  Falls back to ``NotImplementedError`` (ELBO
        quadrature fallback) otherwise.

        Parameters
        ----------
        prior : IndependentJoint or similar
            Prior distribution.
        params : Array or None
            Per-sample params (not yet supported).
        """
        if params is not None:
            raise NotImplementedError(
                "Per-sample KL not yet supported for "
                "IndependentMarginalVariationalFamily"
            )
        if not isinstance(prior, IndependentJoint):
            raise NotImplementedError(
                "Analytical KL requires an IndependentJoint prior, "
                f"got {type(prior).__name__}"
            )
        prior_marginals = prior.marginals()
        if len(prior_marginals) != self._nvars:
            raise NotImplementedError(
                f"Prior has {len(prior_marginals)} marginals but family "
                f"has {self._nvars}"
            )
        kl_total = self._bkd.asarray(0.0)
        for q_m, p_m in zip(self._marginals, prior_marginals):
            if not hasattr(q_m, "kl_divergence"):
                raise NotImplementedError(
                    f"Marginal {type(q_m).__name__} has no kl_divergence"
                )
            kl_total = kl_total + q_m.kl_divergence(p_m)
        return kl_total

    def base_distribution(self) -> IndependentJoint:
        """Return the base distribution (heterogeneous independent joint).

        Each row may be drawn from a different base (e.g. N(0,1) for Gaussian
        marginals, U(0,1) for Beta marginals).
        """
        base_marginals = [m.base_distribution() for m in self._marginals]
        return IndependentJoint(base_marginals, self._bkd)

    def __repr__(self) -> str:
        marginal_names = [type(m).__name__ for m in self._marginals]
        return (
            f"IndependentMarginalVariationalFamily("
            f"nvars={self._nvars}, "
            f"marginals={marginal_names}, "
            f"nparams={self._hyp_list.nparams()})"
        )
