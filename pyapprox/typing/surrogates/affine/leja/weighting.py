"""Weighting strategies for Leja sequence optimization.

This module provides weighting strategies that determine how candidate
points are weighted during Leja sequence generation. Two main strategies
are provided:

- ChristoffelWeighting: Weight by inverse Christoffel function
- PDFWeighting: Weight by probability density function
"""

from typing import Callable, Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


class ChristoffelWeighting(Generic[Array]):
    """Weight by inverse Christoffel function.

    The Christoffel function is defined as:
        K(x) = sum_i phi_i(x)^2

    where phi_i are the orthonormal basis functions. This weighting uses
    1/K(x) as weights, which emphasizes points where the polynomial
    approximation has lower magnitude.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> weighting = ChristoffelWeighting(bkd)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    >>> weights = weighting(samples, basis_values)
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, samples: Array, basis_values: Array) -> Array:
        """Compute inverse Christoffel function weights.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples) for univariate
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Weights for each sample. Shape: (nsamples, 1)
        """
        nsamples = basis_values.shape[0]
        christoffel = self._bkd.sum(basis_values ** 2, axis=1) / nsamples
        return (1.0 / christoffel)[:, None]

    def jacobian(
        self, samples: Array, basis_values: Array, basis_jacobians: Array
    ) -> Array:
        """Compute Jacobian of weights with respect to samples.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples)
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)
        basis_jacobians : Array
            Jacobians of basis functions. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Weight Jacobians. Shape: (nsamples, 1)
        """
        nsamples = basis_values.shape[0]
        christoffel = self._bkd.sum(basis_values ** 2, axis=1) / nsamples
        christoffel_jac = (
            2.0 / nsamples * self._bkd.sum(basis_values * basis_jacobians, axis=1)
        )
        # d/dx (1/K(x)) = -K'(x) / K(x)^2
        return (-christoffel_jac / christoffel ** 2)[:, None]

    def __repr__(self) -> str:
        return f"ChristoffelWeighting(bkd={self._bkd.__class__.__name__})"


class PDFWeighting(Generic[Array]):
    """Weight by probability density function.

    This weighting uses the PDF value at each sample point as the weight,
    which is appropriate when the Leja sequence should be adapted to
    a specific probability distribution.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    pdf : Callable[[Array], Array]
        Probability density function. Takes samples of shape (nsamples,)
        and returns PDF values of shape (nsamples,).
    pdf_jacobian : Callable[[Array], Array], optional
        Jacobian of PDF. Takes samples of shape (nsamples,) and returns
        Jacobians of shape (nsamples,). Required for gradient-based
        optimization.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> from scipy import stats
    >>> bkd = NumpyBkd()
    >>> rv = stats.norm(0, 1)
    >>> weighting = PDFWeighting(bkd, rv.pdf)
    >>> samples = bkd.asarray([[0.0, 0.5, 1.0]])
    >>> basis_values = bkd.asarray([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    >>> weights = weighting(samples, basis_values)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        pdf: Callable[[Array], Array],
        pdf_jacobian: Optional[Callable[[Array], Array]] = None,
    ):
        self._bkd = bkd
        self._pdf = pdf
        self._pdf_jacobian = pdf_jacobian

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, samples: Array, basis_values: Array) -> Array:
        """Compute PDF weights.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples) for univariate
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)
            Note: basis_values is not used for PDF weighting but is
            included for protocol compatibility.

        Returns
        -------
        Array
            Weights for each sample. Shape: (nsamples, 1)
        """
        # samples shape: (1, nsamples) for univariate
        pdf_vals = self._pdf(samples[0])
        return pdf_vals[:, None]

    def jacobian(
        self, samples: Array, basis_values: Array, basis_jacobians: Array
    ) -> Array:
        """Compute Jacobian of weights with respect to samples.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples)
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)
        basis_jacobians : Array
            Jacobians of basis functions. Shape: (nsamples, nterms)
            Note: Not used for PDF weighting.

        Returns
        -------
        Array
            Weight Jacobians. Shape: (nsamples, 1)
        """
        if self._pdf_jacobian is None:
            raise RuntimeError(
                "PDF Jacobian not provided. Cannot compute weight Jacobian."
            )
        pdf_jac = self._pdf_jacobian(samples[0])
        return pdf_jac[:, None]

    def __repr__(self) -> str:
        return f"PDFWeighting(bkd={self._bkd.__class__.__name__})"


class CompositeWeighting(Generic[Array]):
    """Combine multiple weighting strategies.

    The composite weight is the product of all individual weights.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    *weightings : LejaWeightingProtocol[Array]
        Weighting strategies to combine.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from scipy import stats
    >>> bkd = NumpyBkd()
    >>> rv = stats.norm(0, 1)
    >>> christoffel = ChristoffelWeighting(bkd)
    >>> pdf = PDFWeighting(bkd, rv.pdf)
    >>> composite = CompositeWeighting(bkd, christoffel, pdf)
    """

    def __init__(self, bkd: Backend[Array], *weightings):
        self._bkd = bkd
        self._weightings = weightings

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(self, samples: Array, basis_values: Array) -> Array:
        """Compute composite weights (product of all weights).

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples)
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Composite weights. Shape: (nsamples, 1)
        """
        result = self._bkd.ones((samples.shape[1], 1))
        for weighting in self._weightings:
            result = result * weighting(samples, basis_values)
        return result

    def jacobian(
        self, samples: Array, basis_values: Array, basis_jacobians: Array
    ) -> Array:
        """Compute Jacobian of composite weights using product rule.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples)
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)
        basis_jacobians : Array
            Jacobians of basis functions. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Composite weight Jacobians. Shape: (nsamples, 1)
        """
        # Product rule: d/dx (f*g) = f'*g + f*g'
        n = len(self._weightings)
        if n == 0:
            return self._bkd.zeros((samples.shape[1], 1))

        # Compute all weights and jacobians
        weights = [w(samples, basis_values) for w in self._weightings]
        jacs = [
            w.jacobian(samples, basis_values, basis_jacobians)
            for w in self._weightings
        ]

        # Sum over all terms in product rule
        result = self._bkd.zeros((samples.shape[1], 1))
        for i in range(n):
            term = jacs[i]
            for j in range(n):
                if i != j:
                    term = term * weights[j]
            result = result + term

        return result

    def __repr__(self) -> str:
        weight_names = [w.__class__.__name__ for w in self._weightings]
        return f"CompositeWeighting({', '.join(weight_names)})"
