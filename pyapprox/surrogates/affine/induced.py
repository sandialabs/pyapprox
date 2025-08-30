from abc import ABC, abstractmethod
import math
from typing import List
from functools import partial

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.surrogates.affine.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.univariate.orthopoly import (
    HermitePolynomial1D,
    OrthonormalPolynomial1D,
)
from pyapprox.variables.marginals import (
    Marginal,
    GaussianMarginal,
    ContinuousScipyMarginal,
)
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    JointVariable,
)


class OrthoPolySampler(ABC):
    def __init__(
        self, poly: PolynomialChaosExpansion, variable: JointVariable
    ):
        if not isinstance(poly, PolynomialChaosExpansion):
            raise ValueError(
                "poly must be an instance of PolynomialChaosExpansion"
            )
        self._bkd = poly._bkd
        self._poly = poly
        self._variable = variable
        if not self._bkd.bkd_equal(poly._bkd, variable._bkd):
            raise ValueError(
                "poly and variable backends {0}, {1} do not match".format(
                    poly._bkd.__name__, variable._bkd.__name__
                )
            )

    @abstractmethod
    def __call__(self, nsamples: int) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}({1},{2})".format(
            self.__class__.__name__, self._variable, self._poly
        )

    def poly(self) -> PolynomialChaosExpansion:
        return self._poly

    def variable(self) -> JointVariable:
        return self._variable


class EquilibriumHermitePolySampler(OrthoPolySampler):
    def __init__(
        self, poly: PolynomialChaosExpansion, variable: JointVariable
    ):
        for basis in poly.basis()._bases_1d:
            if not isinstance(basis, HermitePolynomial1D):
                raise ValueError(
                    "basis must be an instance of HermitePolynomial1D"
                )
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an instance of IndependentMarginalsVariable"
            )
        for marginal in variable.marginals():
            if not isinstance(marginal, GaussianMarginal) or (
                not isinstance(variable, ContinuousScipyMarginal)
                and variable._name != "norm"
            ):
                super().__init__(poly, variable)

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
        super().__init__(poly, variable)

    def set_degree(self, degree: int):
        self._degree = degree

    def __call__(self, nsamples: int) -> Array:
        canonical_samples = self._bkd.cos(
            self._bkd.asarray(
                np.random.uniform(0, math.pi, (self._poly.nvars(), nsamples))
            )
        )
        return self._poly.map_from_canonical(canonical_samples)


class DiscreteInducedPolySampler(OrthoPolySampler):
    def __init__(
        self,
        poly: PolynomialChaosExpansion,
        variable: IndependentMarginalsVariable,
        quad_samples: List[Array],
        quad_weights: List[Array],
    ):
        if not isinstance(variable, IndependentMarginalsVariable):
            raise ValueError(
                "variable must be an instance of IndependentMarginalsVariable"
            )

        super().__init__(poly, variable)
        self.set_quadrature_rule_tuples(quad_samples, quad_weights)

    def set_quadrature_rule_tuples(
        self, quad_samples: Array, quad_weights: Array
    ):
        if not isinstance(quad_samples, list):
            raise ValueError("quad samples must be a list of arrays")
        if not isinstance(quad_weights, list):
            raise ValueError("quad  must be a list of arrays")
        if len(quad_samples) != self._poly.nvars():
            raise ValueError("must provide quad_samples for each dimension")
        if len(quad_samples) != self._poly.nvars():
            raise ValueError("must provide quad_weights for each dimension")

        for ii in range(self._poly.nvars()):
            if quad_samples[ii].shape[0] != 1:
                raise ValueError("quad_samples has the wrong shape")
            if quad_weights[ii].shape != (quad_samples[ii].shape[1], 1):
                raise ValueError(
                    "quad_weights has the wrong shape, "
                    "was {0} but must be {1}".format(
                        quad_weights[ii].shape, (quad_samples[ii].shape[1], 1)
                    )
                )
            if not self._bkd.allclose(
                self._bkd.sum(quad_weights[ii]),
                self._bkd.ones((1,)),
                atol=1e-15,
            ):
                raise ValueError(
                    "quad_weights must sum to one but summed to {0}".format(
                        self._bkd.sum(quad_weights[ii])
                    )
                )
        self._quad_samples = quad_samples
        self._quad_weights = quad_weights

    def nvars(self) -> int:
        return self._poly.nvars()

    def _univariate_rvs(
        self,
        indices: Array,
        marginal: Marginal,
        basis1d: OrthonormalPolynomial1D,
        quad_samples1d: Array,
        quad_weights1d: Array,
    ) -> Array:
        nsamples = indices.shape[0]
        basis_mat = basis1d(quad_samples1d)
        basis_cdfs = self._bkd.cumsum(quad_weights1d * basis_mat**2, axis=0)
        usamples = np.random.uniform(0.0, 1.0, (nsamples,))
        selected_cdfs = basis_cdfs[:, indices]
        node_idxs = self._bkd.array(
            list(
                map(
                    partial(np.searchsorted, side="right"),
                    self._bkd.to_numpy(selected_cdfs.T),
                    usamples,
                )
            ),
            dtype=int,
        )
        nquad_samples = basis_cdfs.shape[0]
        node_idxs = self._bkd.minimum(
            node_idxs, self._bkd.full((1,), nquad_samples - 1, dtype=int)
        )
        return quad_samples1d[0, node_idxs]

    def christoffel_function(
        self, samples: Array, normalize: bool = True
    ) -> Array:
        vals = self._bkd.sum(self._poly.basis()(samples) ** 2, axis=1)[:, None]
        if not normalize:
            return vals
        return vals / self._poly.basis().nterms()

    def induced_measure(self, samples: Array) -> Array:
        return self._variable.pdf(samples)[:, 0] * self.christoffel_function(
            samples, True
        )

    def __call__(self, nsamples: int) -> Array:
        random_idx = self._bkd.asarray(
            np.random.choice(
                self._poly.basis().nterms(), size=nsamples, replace=True
            ),
            dtype=int,
        )
        random_indices = self._poly.basis().get_indices()[:, random_idx]
        marginals = self._variable.marginals()
        bases_1d = self._poly.basis()._bases_1d
        samples = [
            self._univariate_rvs(
                random_indices[ii],
                marginals[ii],
                bases_1d[ii],
                self._quad_samples[ii],
                self._quad_weights[ii],
            )
            for ii in range(self.nvars())
        ]
        return self._bkd.stack(samples, axis=0)


class GaussQuadratureBasedDiscreteInducedPolySampler(
    DiscreteInducedPolySampler
):

    def __init__(
        self,
        poly: PolynomialChaosExpansion,
        variable: IndependentMarginalsVariable,
        nquad_samples: List[int],
    ):
        if len(nquad_samples) != poly.nvars():
            raise ValueError(
                "must provide quadrature order for each dimension"
            )
        quad_samples, quad_weights = [], []
        for ii in range(poly.nvars()):
            poly.basis()._bases_1d[ii].set_nterms(nquad_samples[ii])
            xx, ww = poly.basis().univariate_quadrature(ii)
            quad_samples.append(xx)
            quad_weights.append(ww)
        super().__init__(poly, variable, quad_samples, quad_weights)
