from typing import Protocol, runtime_checkable, Generic
from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class PDFProtocol(Protocol, Generic[Array]):
    _bkd: Backend[Array]

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (pdf) at the given samples.

        Parameters
        ----------
        samples : Array (nsamples,)
            The points at which to evaluate the pdf.

        Returns
        -------
        vals : Array (nsamples,)
            The evaluated pdf values.
        """
        ...

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log of the probability density function (pdf) at the
        given samples.

        Parameters
        ----------
        samples : Array (nsamples,)
            The points at which to evaluate the log of the pdf.

        Returns
        -------
        vals : Array (nsamples,)
            The evaluated log of the pdf values.
        """
        ...


@runtime_checkable
class PDFWithJacobianProtocol(PDFProtocol[Array], Protocol):
    def jacobian(self, samples: Array) -> Array:
        r"""
        Compute the Jacobian of the log of the PDF of the marginal
        distribution.

        .. math::
            f_j(x) = \frac{\partial}{\partial x_j} \log f(x)

        This function returns the derivative of the log of the PDF with
        respect to the variable.

        Parameters
        ----------
        samples : Array (nsamples,)
            Samples from the marginal distribution of the variable.

        Returns
        -------
        jac : Array (1, nsamples)
            The Jacobian of the log of the PDF of the marginal distribution.
        """
        ...

    def logpdf_jacobian(self, samples: Array) -> Array:
        r"""
        Compute the Jacobian of the log of the PDF of the marginal
        distribution.

        .. math::
            f_j(x) = \frac{\partial}{\partial x_j} \log f(x)

        This function returns the derivative of the log of the PDF with
        respect to the variable.

        Parameters
        ----------
        samples : Array (nsamples,)
            Samples from the marginal distribution of the variable.

        Returns
        -------
        jac : Array (1, nsamples)
            The Jacobian of the log of the PDF of the marginal distribution.
        """
        ...


@runtime_checkable
class RandomVariableProtocol(PDFProtocol[Array], Protocol):
    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the marginal distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array (nsamples,)
            Random samples from the marginal distribution.
        """
        ...
