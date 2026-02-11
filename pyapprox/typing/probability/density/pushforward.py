"""Pushforward density estimation via quadrature projection.

Estimates the density f_Y(y) of a scalar QoI Y = g(xi) by projecting onto a
basis in y-space using pre-computed (y_values, weights) from quadrature in
xi-space.
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.density.protocols import DensityBasisProtocol
from pyapprox.typing.probability.density._fitters import (
    DensityFitterProtocol,
    LinearDensityFitter,
)


class PushforwardDensity(Generic[Array]):
    """Density of Y = g(xi) via quadrature projection.

    Takes pre-computed (y_values, weights) and a DensityBasis, fits via
    a configurable fitter strategy. Satisfies FunctionProtocol (nvars=1,
    nqoi=1).

    Users generate (y_values, weights) externally using existing quadrature
    infrastructure (GaussLagrangeFactory, TensorProductQuadratureRule,
    SobolSampler, etc.), evaluate their function g, and pass the results here.

    Parameters
    ----------
    y_values : Array
        Quadrature points in y-space, shape (1, nquad).
    weights : Array
        Quadrature weights, shape (nquad,).
    basis : DensityBasisProtocol[Array]
        Density basis providing evaluate() and mass_matrix().
    fitter : DensityFitterProtocol[Array], optional
        Fitting strategy. If None, defaults to LinearDensityFitter
        (L2 projection via M*d = b).

    Raises
    ------
    TypeError
        If basis does not satisfy DensityBasisProtocol, or if fitter
        does not satisfy DensityFitterProtocol.
    """

    def __init__(
        self,
        y_values: Array,
        weights: Array,
        basis: DensityBasisProtocol[Array],
        fitter: Optional[DensityFitterProtocol[Array]] = None,
    ) -> None:
        if not isinstance(basis, DensityBasisProtocol):
            raise TypeError(
                f"basis must satisfy DensityBasisProtocol, "
                f"got {type(basis).__name__}"
            )
        self._basis = basis
        self._bkd = basis.bkd()

        if fitter is None:
            fitter = LinearDensityFitter(self._bkd)
        if not isinstance(fitter, DensityFitterProtocol):
            raise TypeError(
                f"fitter must satisfy DensityFitterProtocol, "
                f"got {type(fitter).__name__}"
            )
        self._fitter = fitter

        # Compute basis evaluations and quadrature projection
        bkd = self._bkd
        Phi = basis.evaluate(y_values)  # (nbasis, nquad)
        self._load_vector = bkd.dot(Phi, weights)  # (nbasis,)
        self._mass_matrix = basis.mass_matrix()  # (nbasis, nbasis)
        self._coefficients = fitter.fit(
            self._mass_matrix, self._load_vector,
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables (always 1)."""
        return 1

    def nqoi(self) -> int:
        """Return the number of quantities of interest (always 1)."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the estimated density at given points.

        Parameters
        ----------
        samples : Array
            Query points, shape (1, npts).

        Returns
        -------
        Array
            Density values, shape (1, npts).
        """
        bkd = self._bkd
        Phi = self._basis.evaluate(samples)  # (nbasis, npts)
        d_row = bkd.reshape(self._coefficients, (1, -1))  # (1, nbasis)
        return bkd.dot(d_row, Phi)  # (1, npts)

    def coefficients(self) -> Array:
        """Return the fitted density coefficients, shape (nbasis,)."""
        return self._coefficients

    def basis(self) -> DensityBasisProtocol[Array]:
        """Return the density basis."""
        return self._basis

    def fitter(self) -> DensityFitterProtocol[Array]:
        """Return the fitter used for coefficient computation."""
        return self._fitter

    def ise_score(self) -> Array:
        """Compute relative ISE score: d^T M d - 2 * d^T b.

        Lower values indicate better fit. The constant term
        int f_true^2 dy is omitted since it is the same across
        all fitters for a given dataset.

        Returns
        -------
        Array
            Scalar ISE score.
        """
        bkd = self._bkd
        d = self._coefficients
        M = self._mass_matrix
        b = self._load_vector
        Md = bkd.dot(M, d)
        return bkd.sum(d * Md) - 2.0 * bkd.sum(d * b)


__all__ = ["PushforwardDensity"]
