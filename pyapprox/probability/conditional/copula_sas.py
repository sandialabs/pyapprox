"""
Conditional copula with SAS normal marginals.

Wraps a GaussianCopula + SASNormalMarginal[] into a conditional
distribution for use with ELBOObjective (variational inference).

The key insight: the Phi/Phi^{-1} cancels in both reparameterize and
logpdf:
- reparameterize: eps -> L*eps -> SAS_j(n_j), no CDF calls needed
- logpdf: Phi^{-1}(F_SAS(z_j)) = x_j (inverse SAS), reused for
  both marginal logpdf and copula density
"""

import math
from typing import Generic, List

import numpy as np

from pyapprox.probability.copula.gaussian import GaussianCopula
from pyapprox.probability.univariate.sas_normal import SASNormalMarginal
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class ConditionalCopulaSAS(Generic[Array]):
    """Gaussian copula with SAS normal marginals as conditional distribution.

    Satisfies ConditionalDistributionProtocol for use with ELBOObjective.
    The conditioning variable x is a dummy (nvars=1) for single-problem VI.

    The copula correlation and SAS marginal parameters are jointly
    optimized through the combined hyp_list.

    Parameters
    ----------
    copula : GaussianCopula[Array]
        Gaussian copula for dependence structure.
    marginals : list of SASNormalMarginal[Array]
        One SAS marginal per dimension.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        copula: GaussianCopula[Array],
        marginals: List[SASNormalMarginal[Array]],
        bkd: Backend[Array],
    ) -> None:
        d = copula.nvars()
        if len(marginals) != d:
            raise ValueError(
                f"Expected {d} marginals, got {len(marginals)}"
            )
        self._copula = copula
        self._marginals = marginals
        self._bkd = bkd
        self._d = d
        self._log_2pi = math.log(2.0 * math.pi)

        # Build combined hyp_list: copula params + all marginal params
        self._hyp_list = copula.hyp_list()
        for m in marginals:
            self._hyp_list = self._hyp_list + m.hyp_list()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """Number of conditioning variables (dummy = 1)."""
        return 1

    def nqoi(self) -> int:
        """Dimension of the latent space."""
        return self._d

    def hyp_list(self) -> HyperParameterList:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._hyp_list.nparams()

    def reparameterize(self, x: Array, base_samples: Array) -> Array:
        """Transform base N(0,I) samples through copula + SAS marginals.

        The chain: eps ~ N(0,I) -> u = L*eps -> z_j = SAS_j(u_j)
        where SAS_j(n) = xi_j + eta_j * sinh((arcsinh(n) + eps_j) / delta_j).

        No CDF/inverse CDF calls needed: the Phi/Phi^{-1} cancel because
        the Gaussian copula sample_transform produces normal variates and
        SAS reparameterize takes normal variates.

        Parameters
        ----------
        x : Array
            Dummy conditioning variable, shape (1, N).
        base_samples : Array
            Standard normal samples, shape (d, N).

        Returns
        -------
        Array
            Reparameterized samples, shape (d, N).
        """
        bkd = self._bkd
        # Apply copula correlation: u = L @ eps
        u = self._copula.correlation_param().sample_transform(base_samples)

        # Apply SAS transform per marginal
        rows = []
        for j in range(self._d):
            u_j = bkd.reshape(u[j], (1, -1))  # (1, N)
            z_j = self._marginals[j].reparameterize(u_j)  # (1, N)
            rows.append(z_j[0])  # (N,)
        return bkd.stack(rows, axis=0)  # (d, N)

    def logpdf(self, x: Array, z: Array) -> Array:
        """Evaluate log q(z) for copula + SAS marginals.

        log q(z) = sum_j log f_j(z_j) + log c(F_1(z_1), ..., F_d(z_d))

        The copula density log c uses x_j = Phi^{-1}(F_SAS_j(z_j)),
        which equals the inverse SAS transform (no CDF/inverse CDF calls).

        Parameters
        ----------
        x : Array
            Dummy conditioning variable, shape (1, N).
        z : Array
            Samples, shape (d, N).

        Returns
        -------
        Array
            Log density, shape (1, N).
        """
        bkd = self._bkd
        d = self._d

        # Compute marginal log-densities and inverse SAS transform x_j
        log_marginal_sum = bkd.zeros((z.shape[1],))
        x_normals = []
        for j in range(d):
            z_j = bkd.reshape(z[j], (1, -1))  # (1, N)
            log_marginal_sum = log_marginal_sum + self._marginals[j].logpdf(
                z_j
            )[0]
            # x_j = sinh(delta * arcsinh((z_j - xi) / eta) - eps)
            # This is Phi^{-1}(F_SAS(z_j)) — the key cancellation
            x_j = self._marginals[j]._sas_inverse(z[j])  # (N,)
            x_normals.append(x_j)

        # Stack x_j values: (d, N)
        x_normal = bkd.stack(x_normals, axis=0)

        # Copula density via correlation parameterization
        # log c = -0.5 * log|Sigma| - 0.5 * x^T (Sigma^{-1} - I) x
        corr_param = self._copula.correlation_param()
        log_det = corr_param.log_det()
        quad = corr_param.quad_form(x_normal)  # (N,)
        log_copula = -0.5 * log_det - 0.5 * quad

        result = log_marginal_sum + log_copula
        return bkd.reshape(result, (1, -1))

    def rvs(self, x: Array) -> Array:
        """Generate random samples.

        Parameters
        ----------
        x : Array
            Dummy conditioning variable, shape (1, N).

        Returns
        -------
        Array
            Random samples, shape (d, N).
        """
        nsamples = x.shape[1]
        base = self._bkd.asarray(
            np.random.randn(self._d, nsamples).astype(np.float64)
        )
        return self.reparameterize(x, base)

    def base_distribution(self):
        """Return the base distribution: N(0, I_d)."""
        from pyapprox.probability.joint.independent import IndependentJoint
        from pyapprox.probability.univariate.gaussian import GaussianMarginal

        marginals = [
            GaussianMarginal(0.0, 1.0, self._bkd) for _ in range(self._d)
        ]
        return IndependentJoint(marginals, self._bkd)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConditionalCopulaSAS(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"nparams={self.nparams()})"
        )
