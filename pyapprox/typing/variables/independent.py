from typing import Sequence, Generic
from pyapprox.typing.util.backend import Array, validate_backends
from pyapprox.typing.variables.pdf import (
    PDFProtocol,
    RandomVariableProtocol,
)
from pyapprox.typing.interface.functions.function import validate_samples


class IndependentMarginalsVariable(Generic[Array]):
    """
    Represents a random variable with independent marginals.

    Parameters
    ----------
    univariate_marginals : Sequence[RandomVariableProtocol]
        A sequence of univariate random variables representing the marginals.
    """

    def __init__(
        self,
        univariate_marginals: Sequence[RandomVariableProtocol],
    ):
        self._validate_univariate_marginals(univariate_marginals)
        self._bkd = univariate_marginals[0]._bkd
        self._univariate_marginals = univariate_marginals

    def _validate_univariate_marginals(
        self, univariate_marginals: Sequence[RandomVariableProtocol]
    ) -> None:
        """
        Validate the univariate marginals.

        Parameters
        ----------
        univariate_marginals : Sequence[RandomVariableProtocol]
            The univariate marginals to validate.

        Raises
        ------
        ValueError
            If the marginals are empty or have inconsistent backends.
        TypeError
            If any marginal is not an instance of PDFProtocol or uses an
            invalid backend.
        """
        if len(univariate_marginals) == 0:
            raise ValueError("Univariate marginals cannot be empty.")

        # Validate that each marginal is an instance of PDFProtocol
        for marginal in univariate_marginals:
            if not isinstance(marginal, PDFProtocol):
                raise TypeError(
                    f"Invalid marginal type: expected an instance of "
                    f" PDFProtocol, got {type(marginal).__name__}."
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

    def marginals(self) -> Sequence[RandomVariableProtocol]:
        """
        Return the univariate marginals.

        Returns
        -------
        marginals : Sequence[RandomVariableProtocol]
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
        values : Array (nsamples, 1)
            The values of the PDF at x.
        """
        validate_samples(self.nvars(), samples)
        marginal_vals = self._bkd.stack(
            [
                marginal.pdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.prod(marginal_vals, axis=0)[:, None]

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the joint log probability distribution function.

        Parameters
        ----------
        samples : Array (nvars, nsamples)
            Values in the domain of the random variable X.

        Returns
        -------
        values : Array (nsamples, 1)
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
        return self._bkd.sum(marginal_vals, axis=0)[:, None]

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
        return self._bkd.stack(
            [marginal.interval(1.0) for marginal in self.marginals()], axis=0
        )


class IndependentMarginalsVariableWithJacobian(IndependentMarginalsVariable):
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
        self._validate_samples(samples)
        pdf_vals = self._bkd.stack(
            [
                marginal.pdf(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ],
            axis=0,
        )
        return self._bkd.hstack(
            [
                marginal.pdf_jacobian(samples[ii])
                * self._bkd.prod(pdf_vals[:ii], axis=0)
                * self._bkd.prod(pdf_vals[ii + 1 :], axis=0)
                for ii, marginal in enumerate(self.marginals())
            ]
        )

    def logpdf_jacobians(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the joint log PDF at mulitple samples.

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
        return self._bkd.hstack(
            [
                marginal.logpdf_jacobian(samples[ii])
                for ii, marginal in enumerate(self.marginals())
            ]
        )
