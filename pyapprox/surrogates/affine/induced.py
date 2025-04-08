from abc import ABC, abstractmethod
import math

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.univariate.orthopoly import HermitePolynomial1D
from pyapprox.variables.joint import IndependentMarginalsVariable


class OrthoPolySampler(ABC):
    def __init__(self, poly: PolynomialChaosExpansion):
        self._bkd = poly._bkd
        self._poly = poly

    @abstractmethod
    def __call__(self, nsamples: int) -> Array:
        raise NotImplementedError


class EquilibriumHermitePolySampler(OrthoPolySampler):
    def __init__(self, poly: PolynomialChaosExpansion):
        for basis in poly.basis()._bases_1d:
            if not isinstance(basis, HermitePolynomial1D):
                raise ValueError(
                    "basis must be an instance of HermitePolynomial1D"
                )
        super().__init__(poly)

    def set_degree(self, degree: int):
        self._degree = degree

    def __call__(self, nsamples: int) -> Array:
        canonical_samples = (
            self._poly.basis()
            ._bases_1d[0]
            ._equilibrium_rvs(self._poly.nvars(), self._degree, nsamples)
        )
        return self._poly.map_from_canonical(canonical_samples)


class EquilibriumBoundedVariablePolySampler(OrthoPolySampler):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        poly: PolynomialChaosExpansion,
    ):
        for marginal in variable.marginals():
            if not marginal.is_bounded():
                raise ValueError("marginal must be bounded")
        super().__init__(poly)

    def set_degree(self, degree: int):
        self._degree = degree

    def __call__(self, nsamples: int) -> Array:
        canonical_samples = self._bkd.cos(
            self._bkd.asarray(
                np.random.uniform(0, math.pi, (self._poly.nvars(), nsamples))
            )
        )
        return self._poly.map_from_canonical(canonical_samples)
