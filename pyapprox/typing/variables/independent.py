from typing import Sequence, Generic, Protocol, runtime_checkable, Union

import numpy as np

from pyapprox.typing.util.backend import Array, validate_backends
from pyapprox.typing.interface.functions.function import validate_samples
from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class MarginalProtocol(Protocol, Generic[Array]):
    _bkd: Backend[Array]

    def __call__(self, samples: Array) -> Array: ...

    def logpdf(self, samples: Array) -> Array: ...

    def is_bounded(self) -> bool: ...

    def interval(self, alpha: float) -> Array: ...

    def rvs(self, nsamples: int) -> Array: ...


@runtime_checkable
class MarginalWithJacobianProtocol(MarginalProtocol[Array], Protocol):
    def jacobian(self, samples: Array) -> Array: ...

    def logpdf_jacobian(self, samples: Array) -> Array: ...


class IndependentRandomVariable(Generic[Array]):
    """
    Represents a random variable with independent marginals.

    Parameters
    ----------
    univariate_marginals : Sequence[MarginalProtocol[Array]]
        A sequence of univariate random variables representing the marginals.
    """

    def __init__(
        self,
        univariate_marginals: Sequence[MarginalProtocol[Array]],
    ):
        self._validate_univariate_marginals(univariate_marginals)
        self._bkd = univariate_marginals[0]._bkd
        self._univariate_marginals = univariate_marginals

    def _validate_univariate_marginals(
        self,
        univariate_marginals: Sequence[MarginalProtocol[Array]],
    ) -> None:
        """
        Validate the univariate marginals.

        Parameters
        ----------
        univariate_marginals : Sequence[MarginalProtocol[Array]]
            The univariate marginals to validate.

        Raises
        ------
        ValueError
            If the marginals are empty or have inconsistent backends.
        TypeError
            If any marginal is not an instance of MarginalProtocol or uses an
            invalid backend.
        """
        if len(univariate_marginals) == 0:
            raise ValueError("Univariate marginals cannot be empty.")

        # Validate that each marginal is an instance of MarginalProtocol
        for marginal in univariate_marginals:
            if not isinstance(marginal, MarginalProtocol):
                raise TypeError(
                    f"Invalid marginal type: expected an instance of "
                    f"MarginalProtocol, got {type(marginal).__name__}."
                )

        # Validate backend consistency
        validate_backends([marginal._bkd for marginal in univariate_marginals])

    def nvars(self) -> int:
        """
        Return the number of variables (dimensions).

        Returns
        -------
        nvars : int
            The number of variables.
        """
        return len(self._univariate_marginals)

    def marginals(self) -> Sequence[MarginalProtocol[Array]]:
        """
        Return the univariate marginals.

        Returns
        -------
        marginals : Sequence[MarginalProtocol[Array]]
            The univariate marginals.
        """
        return self._univariate_marginals

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability distribution function.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        values : Array (1, nsamples)
            The values of the PDF at x.
        """
        validate_samples(self.nvars(), samples)
        marginal_vals = self._bkd.stack(
            [
                marginal(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.prod(marginal_vals, axis=0)[None, :]

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the joint log probability distribution function.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        values : Array (1, nsamples)
            The values of the log PDF at x.
        """
        validate_samples(self.nvars(), samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.logpdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.sum(marginal_vals, axis=0)[None, :]

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        repr : str
            A string representation of the object.
        """
        return (
            f"{self.__class__.__name__}(nvars={self.nvars()}, "
            f"backend={type(self._bkd).__name__})"
        )

    def rvs(self, nsamples: int) -> Array:
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        nsamples : int
            The number of samples to generate.

        Returns
        -------
        samples : Array (nvars, nsamples)
            Independent samples from the target distribution.
        """
        marginal_samples = [
            marginal.rvs(nsamples) for marginal in self.marginals()
        ]
        return self._bkd.stack(marginal_samples, axis=0)

    def domain(self) -> Array:
        """
        Compute the domain of the random variable.

        Returns
        -------
        domain : Array
            The domain of the random variable.
        """
        return self._bkd.stack(
            [
                (
                    marginal.interval(1.0)
                    if marginal.is_bounded()
                    else self._bkd.array([-np.inf, np.inf])
                )
                for marginal in self.marginals()
            ],
            axis=0,
        )


class IndependentMarginalsVariableWithJacobian(Generic[Array]):
    """
    Represents a random variable with independent marginals and Jacobian functionality.

    Parameters
    ----------
    univariate_marginals : Sequence[MarginalWithJacobianProtocol[Array]]
        A sequence of univariate random variables representing the marginals.
    """

    def __init__(
        self,
        univariate_marginals: Sequence[MarginalWithJacobianProtocol[Array]],
    ):
        self._base_random_variable = IndependentRandomVariable(
            univariate_marginals
        )
        self._bkd = self._base_random_variable._bkd

    def marginals(self) -> Sequence[MarginalWithJacobianProtocol[Array]]:
        """
        Return the univariate marginals.

        Returns
        -------
        marginals : Sequence[MarginalWithJacobianProtocol[Array]]
            The univariate marginals.
        """
        return self._base_random_variable.marginals()

    def nvars(self) -> int:
        """
        Return the number of variables (dimensions).

        Returns
        -------
        nvars : int
            The number of variables.
        """
        return self._base_random_variable.nvars()

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the joint probability distribution function.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        values : Array (1, nsamples)
            The values of the PDF at x.
        """
        return self._base_random_variable.pdf(samples)

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the joint log probability distribution function.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        values : Array (1, nsamples)
            The values of the log PDF at x.
        """
        return self._base_random_variable.logpdf(samples)

    def rvs(self, nsamples: int) -> Array:
        """
        Generate samples from a tensor-product probability measure.

        Parameters
        ----------
        nsamples : int
            The number of samples to generate.

        Returns
        -------
        samples : Array (nvars, nsamples)
            Independent samples from the target distribution.
        """
        return self._base_random_variable.rvs(nsamples)

    def domain(self) -> Array:
        """
        Compute the domain of the random variable.

        Returns
        -------
        domain : Array
            The domain of the random variable.
        """
        return self._base_random_variable.domain()

    def pdf_jacobians(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the joint PDF at multiple samples.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        jacobian : Array (nvars, nsamples)
            The Jacobian of the joint PDF.
        """
        validate_samples(self.nvars(), samples)
        pdf_vals = self._bkd.stack(
            [
                marginal(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.stack(
            [
                marginal.pdf_jacobian(samples[ii])
                * self._bkd.prod(pdf_vals[:ii], axis=0)
                * self._bkd.prod(pdf_vals[ii + 1 :], axis=0)
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )

    def logpdf_jacobians(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the joint log PDF at multiple samples.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        jacobian : Array (nvars, nsamples)
            The Jacobian of the joint log PDF.
        """
        validate_samples(self.nvars(), samples)
        return self._bkd.stack(
            [
                marginal.logpdf_jacobian(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=1,
        )
