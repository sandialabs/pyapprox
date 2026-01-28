"""Bayesian conjugate fitter for basis expansions."""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import BasisExpansionProtocol
from pyapprox.typing.inverse.conjugate import DenseGaussianConjugatePosterior
from pyapprox.typing.probability.gaussian import DenseCholeskyMultivariateGaussian


class BayesianConjugateResult(Generic[Array]):
    """Result from Bayesian conjugate fitting.

    Parameters
    ----------
    surrogate : BasisExpansionProtocol[Array]
        Fitted expansion with posterior mean as coefficients.
    posterior : DenseGaussianConjugatePosterior[Array]
        Full posterior object for accessing covariance, sampling, etc.
    """

    def __init__(
        self,
        surrogate: BasisExpansionProtocol[Array],
        posterior: DenseGaussianConjugatePosterior[Array],
    ):
        self._surrogate = surrogate
        self._posterior = posterior

    def surrogate(self) -> BasisExpansionProtocol[Array]:
        """Return fitted expansion (posterior mean as coefficients)."""
        return self._surrogate

    def posterior_mean(self) -> Array:
        """Return posterior mean of coefficients. Shape: (nterms, 1)"""
        return self._posterior.posterior_mean()

    def posterior_covariance(self) -> Array:
        """Return posterior covariance. Shape: (nterms, nterms)"""
        return self._posterior.posterior_covariance()

    def posterior_variable(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """Return posterior as Gaussian distribution for sampling."""
        return self._posterior.posterior_variable()

    def evidence(self) -> float:
        """Return model evidence (marginal likelihood)."""
        return self._posterior.evidence()


class BayesianConjugateFitter(Generic[Array]):
    """Bayesian conjugate fitter for basis expansions.

    Uses Gaussian prior on coefficients and Gaussian noise model.
    Computes exact posterior (no approximation).

    Model: y = Phi @ c + noise
    Prior: c ~ N(prior_mean, prior_covariance)
    Noise: noise ~ N(0, noise_covariance)

    For isotropic prior/noise with prior_var=tau^2 and noise_var=sigma^2,
    the posterior mean equals Ridge regression with alpha = sigma^2/tau^2.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    prior_mean : Array, optional
        Prior mean for coefficients. Shape: (nterms, 1).
        Default: zeros.
    prior_covariance : Array, optional
        Prior covariance. Shape: (nterms, nterms).
        Default: identity (set prior_var for isotropic).
    prior_var : float, optional
        Isotropic prior variance tau^2. Creates prior_cov = tau^2 * I.
        Ignored if prior_covariance is provided.
    noise_covariance : Array, optional
        Noise covariance. Shape: (nsamples, nsamples).
        Default: identity (set noise_var for isotropic).
    noise_var : float, optional
        Isotropic noise variance sigma^2. Creates noise_cov = sigma^2 * I.
        Ignored if noise_covariance is provided.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        prior_mean: Optional[Array] = None,
        prior_covariance: Optional[Array] = None,
        prior_var: Optional[float] = None,
        noise_covariance: Optional[Array] = None,
        noise_var: Optional[float] = None,
    ):
        self._bkd = bkd
        self._prior_mean = prior_mean
        self._prior_covariance = prior_covariance
        self._prior_var = prior_var
        self._noise_covariance = noise_covariance
        self._noise_var = noise_var

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def fit(
        self,
        expansion: BasisExpansionProtocol[Array],
        samples: Array,
        values: Array,
    ) -> BayesianConjugateResult[Array]:
        """Fit via Bayesian conjugate posterior.

        Parameters
        ----------
        expansion : BasisExpansionProtocol
            Must have basis_matrix() and with_params() methods.
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Target values. Shape: (nqoi, nsamples) or (nsamples,).
            Currently only nqoi=1 supported.

        Returns
        -------
        BayesianConjugateResult
            Result with fitted expansion and posterior access.
        """
        bkd = self._bkd
        nterms = expansion.nterms()
        nsamples = samples.shape[1]

        # Handle 1D values
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))
        if values.shape[0] != 1:
            raise ValueError("BayesianConjugateFitter only supports nqoi=1")

        # Get basis matrix: (nsamples, nterms)
        Phi = expansion.basis_matrix(samples)

        # Setup prior
        if self._prior_mean is not None:
            prior_mean = self._prior_mean
        else:
            prior_mean = bkd.zeros((nterms, 1))

        if self._prior_covariance is not None:
            prior_cov = self._prior_covariance
        elif self._prior_var is not None:
            prior_cov = self._prior_var * bkd.eye(nterms)
        else:
            prior_cov = bkd.eye(nterms)

        # Setup noise covariance
        if self._noise_covariance is not None:
            noise_cov = self._noise_covariance
        elif self._noise_var is not None:
            noise_cov = self._noise_var * bkd.eye(nsamples)
        else:
            noise_cov = bkd.eye(nsamples)

        # Create conjugate posterior solver
        posterior = DenseGaussianConjugatePosterior(
            observation_matrix=Phi,
            prior_mean=prior_mean,
            prior_covariance=prior_cov,
            noise_covariance=noise_cov,
            bkd=bkd,
        )

        # Compute posterior from observations
        # DenseGaussianConjugatePosterior expects (nobs, nexperiments)
        # values is (1, nsamples), need (nsamples, 1)
        posterior.compute(values.T)

        # Create fitted expansion with posterior mean
        fitted_expansion = expansion.with_params(posterior.posterior_mean())

        return BayesianConjugateResult(
            surrogate=fitted_expansion,
            posterior=posterior,
        )
