from abc import ABC, abstractmethod
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.interface.model import SingleSampleModel


class LinearOEDCriterion(SingleSampleModel):
    def __init__(
        self,
        design_factors: Array,
        noise_mult: Array = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._design_factors = design_factors
        self._homos_outer_prods = self.compute_homoscedastic_outer_products()
        ndesign_pts = design_factors.shape[0]
        if noise_mult is not None and (
            noise_mult.ndim != 1 or noise_mult.shape[0] != ndesign_pts
        ):
            raise ValueError("noise multiplier has the wrong shape")
        self._noise_mult = noise_mult
        self._jacobian_implemented = True

    def nqoi(self) -> int:
        return 1

    def compute_homoscedastic_outer_products(self) -> Array:
        r"""
        Compute

        .. math:: f(x_i)f(x_i)^T\quad \forall i=0,\ldots,M

        at a set of design pts :math:`x_i`.

        for the linear model

        .. math::  y(x) = F(x)\theta+\eta(x)\epsilon

        Parameters
        ---------
        factors : np.ndarray (M,N)
            The N factors F of the linear model evaluated at the M design pts

        Returns
        -------
        homoscedastic_outer_products : np.ndarray (N,N,M)
           The outer products of each row of F with itself, i.e.
           :math:`f(x_i)f(x_i)^T`
        """
        ndesign_pts, nfactors = self._design_factors.shape
        return self._bkd.einsum(
            "ij,ik->jki", self._design_factors, self._design_factors
        )

    def is_homoscedastic(self) -> bool:
        return self._noise_mult is None

    def _recall_design_matrices(
        self, design_prob_measure: Array
    ) -> Tuple[Array, Array, Array, Array]:
        # load in matrices if already computed. Useful for jacobian
        # hessian calculations
        design_prob_measure = design_prob_measure[:, 0]

        if not hasattr(
            self, "_stored_design_prob_measure"
        ) or not self._bkd.allclose(
            design_prob_measure,
            self._stored_design_prob_measure,
            atol=1e-15,
            rtol=1e-15,
        ):

            self._stored_design_prob_measure = design_prob_measure
            self._stored_M0, self._stored_M1 = self.design_matrices(
                design_prob_measure
            )
            if self._M0inv_required():
                self._stored_M0inv = self._bkd.inv(self._stored_M0)
            else:
                self._stored_M0inv = None
            self._stored_M1inv = self._bkd.inv(self._stored_M1)

        return (
            self._stored_M0,
            self._stored_M1,
            self._stored_M0inv,
            self._stored_M1inv,
        )

    def _M0inv_required(self) -> bool:
        return False

    def _jacobian(self, design_prob_measure: Array) -> Array:
        if self.is_homoscedastic():
            return self._homoscedastic_jacobian(design_prob_measure)
        return self._hetroscedastic_jacobian(design_prob_measure)

    @abstractmethod
    def _homoscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        raise NotImplementedError


class LinearOEDRegressionMixin(ABC):
    def design_matrices_for_homoscedastic_noise(
        self, design_prob_measure: Array
    ) -> Tuple[Array, Array]:
        M1 = self._homos_outer_prods @ design_prob_measure
        return None, M1

    def design_matrices(
        self, design_prob_measure: Array
    ) -> Tuple[Array, Array]:
        if self.is_homoscedastic():
            return self.design_matrices_for_homoscedastic_noise(
                design_prob_measure
            )
        return self.design_matrices_for_hetroscedastic_noise(
            design_prob_measure
        )


class LstSqLinearOEDRegressionMixin(LinearOEDRegressionMixin):
    def design_matrices_for_hetroscedastic_noise(
        self, design_prob_measure: Array
    ) -> Tuple[Array, Array]:
        M0 = self._homos_outer_prods @ (
            design_prob_measure * self._noise_mult**2
        )
        M1 = self._homos_outer_prods @ design_prob_measure
        return M0, M1


class QuantileLinearOEDRegressionMixin(LinearOEDRegressionMixin):
    def design_matrices_for_hetroscedastic_noise(
        self, design_prob_measure: Array
    ) -> Tuple[Array, Array]:
        M0 = self._homos_outer_prods @ design_prob_measure
        M1 = self._homos_outer_prods @ (design_prob_measure / self._noise_mult)
        return M0, M1


class DOptimalMixin:
    def _M0inv_required(self) -> bool:
        return not self.is_homoscedastic()

    def _evaluate(self, design_prob_measure: Array) -> Array:
        M0, M1, M0_inv, M1_inv = self._recall_design_matrices(
            design_prob_measure
        )
        if self.is_homoscedastic():
            # add zero to make tensor when using torch without effecting
            # gradient computation. Todo make atleast1d and 2d not use
            # clone at use those functions here
            return self._bkd.zeros((1, 1)) + self._bkd.log(
                self._bkd.det(M1_inv)
            )
        gamma = M0.dot(M1_inv)
        return self._bkd.zeros((1, 1)) + self._bkd.log(
            self._bkd.det(M1_inv @ gamma)
        )

    def _homoscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M0, M1, M0_inv, M1_inv = self._recall_design_matrices(
            design_prob_measure
        )
        print(M1_inv.shape, self._design_factors.shape)
        temp = self._design_factors.T * (M1_inv @ self._design_factors.T)
        return -self._bkd.sum(temp, axis=0)[None, :]


class DOptimalLstSqCriterion(
    LstSqLinearOEDRegressionMixin, DOptimalMixin, LinearOEDCriterion
):
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M0, M1, M0_inv, M1_inv = self._recall_design_matrices(
            design_prob_measure
        )
        return (
            -2
            * self._bkd.sum(
                self._design_factors.T * (M1_inv @ self._design_factors.T),
                axis=0,
            )[None, :]
            + self._bkd.sum(
                (self._noise_mult[:, None] * self._design_factors).T
                * (
                    M0_inv
                    @ (self._noise_mult[:, None] * self._design_factors).T
                ),
                axis=0,
            )[None, :]
        )


class DOptimalQuantileCriterion(
    QuantileLinearOEDRegressionMixin, DOptimalMixin, LinearOEDCriterion
):
    def _hetroscedastic_jacobian(self, design_prob_measure: Array) -> Array:
        M0, M1, M0_inv, M1_inv = self._recall_design_matrices(
            design_prob_measure
        )
        return (
            -2
            * self._bkd.sum(
                self._design_factors.T
                * (
                    M1_inv
                    @ (
                        (
                            self._design_factors
                            / self._noise_mult[:, None]
                        ).T
                    )
                ),
                axis=0,
            )[None, :]
            + self._bkd.sum(
                self._design_factors.T * (M0_inv @ self._design_factors.T),
                axis=0,
            )[None, :]
        )


class AOptimalMixin:
    def _evaluate(self, design_prob_measure: Array) -> Array:
        M0, M1, M0_inv, M1_inv = self._recall_design_matrices(
            design_prob_measure
        )
        if self.is_homoscedastic():
            return self._bkd.zeros((1, 1)) + self._bkd.log(
                self._bkd.trace(M1_inv)
            )
        gamma = M0.dot(M1_inv)
        return self._bkd.zeros((1, 1)) + self._bkd.trace(M1_inv @ gamma)

