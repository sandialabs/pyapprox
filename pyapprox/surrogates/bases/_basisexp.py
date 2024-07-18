from abc import ABC, abstractmethod

from pyapprox.surrogates.bases._basis import Basis
from pyapprox.surrogates.bases._linearsystemsolvers import (
    LinearSystemSolver)


class BasisExpansion(ABC):
    """The base class for any linear basis expansion for multiple
       quantities of interest (QoI)."""

    def __init__(self, basis : Basis, solver: LinearSystemSolver,
                 nqoi=1, coef_bounds=None):
        self.basis = basis
        self._solver = solver
        self._nqoi = nqoi
        init_coef = self._la_full((self.basis.nterms()*self.nqoi()), 0.)
        self._coef = self._HyperParameter(
            "coef", self.basis.nterms()*nqoi, init_coef,
            self._parse_coef_bounds(coef_bounds), self._transform)
        self.hyp_list = self._HyperParameterList([self._coef])

    def _parse_coef_bounds(self, coef_bounds):
        if coef_bounds is None:
            return [-self._la_inf(), self._la_inf()]
        return coef_bounds

    def nqoi(self):
        """
        Return the number of quantities of interest (QoI).
        """
        return self._nqoi

    def set_coefficients(self, coef):
        """
        Set the basis coefficients.

        Parameters
        ----------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        if coef.ndim != 2 or coef.shape != (self.basis.nterms(), self.nqoi()):
            raise ValueError(
                "coef shape {0} is must be {1}".format(
                    coef.shape, (self.basis.nterms(), self.nqoi())))
        self._coef.set_values(coef.flatten())

    def get_coefficients(self):
        """
        Get the basis coefficients.

        Returns
        -------
        coef : array (nterms, nqoi)
            The basis coefficients for each quantity of interest (QoI)
        """
        return self._coef.get_values().reshape(
            self.basis.nterms(), self.nqoi())

    def __call__(self, samples):
        """
        Evaluate the expansion at a set of samples.

        ----------
        samples : array (nsamples, nqoi)
            The samples used to evaluate the expansion.

        Returns
        -------
            The values of the expansion for each QoI and sample
        """
        return self.basis(samples) @ self.get_coefficients()

    def __repr__(self):
        return "{0}(basis={1}, nqoi={2})".format(
            self.__class__.__name__, self.basis, self.nqoi())

    def fit(self, samples, values):
        """Fit the expansion by finding the optimal coefficients. """
        if samples.shape[1] != values.shape[0]:
            raise ValueError(
                "Number of cols of samples {0} does not match number of rows of values".format(samples.shape[1], values.shape[0]))
        if values.shape[1] != self.nqoi():
            raise ValueError(
                "Number of cols {0} in values does not match nqoi {1}".format(
                    values.shape[1], self.nqoi()))
        coef = self._solver.solve(self.basis(samples), values)
        self.set_coefficients(coef)
