import warnings
from typing import Tuple, List
from functools import partial
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.surrogates.autogp.exactgp import (
    ExactGaussianProcess,
    GaussianProcessIdentityTransform,
)
from pyapprox.surrogates.kernels.kernels import (
    Kernel,
    MaternKernel,
    ConstantKernel,
    SumKernel,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.basis import (
    FixedTensorProductQuadratureRule,
    GaussQuadratureRule,
)
from pyapprox.util.transforms import Transform


class KernelStatistics(ABC):
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        ctrain_samples: Array,
        nquad_nodes_1d: List[int] = None,
    ):
        self._variable = variable
        self._ctrain_samples = ctrain_samples
        if isinstance(gp._kernel, SumKernel):
            raise NotImplementedError(
                "kernel with white noise is not currently supported"
            )

        if not isinstance(gp._out_trans, GaussianProcessIdentityTransform):
            # TODO There are bugs which I have yet to find
            raise ValueError(
                "Can only compute statistics when not scaling outputs"
            )
        self._gp = gp
        self._bkd = self._gp._bkd
        self._tau_1d, self._cond_P_1d, self._u_1d = None, None, None
        # out trans for GP can be useful for scaling optimization objective
        # when training. However, I see little value of this transformation
        # here. So just use values in user space, which makes computations
        # easier
        self._out_trans = self._gp._out_trans
        # self._out_trans = GaussianProcessIdentityTransform(self._gp._bkd)

    def _get_kernel_length_scale(self):
        found = False
        for hyperparam in self._gp.kernel().hyp_list().hyper_params:
            if hyperparam.name == "lenscale":
                lscale = hyperparam.get_values()
                found = True
        if not found:
            raise RuntimeError(
                "kernel does not have hyperparameter with name lenscale"
            )
        return lscale

    def _get_kernel_variance(self):
        found = False
        for hyperparam in self._gp.kernel().hyp_list().hyper_params:
            if hyperparam.name == "const":
                const = hyperparam.get_values()
                found = True
        if not found:
            # kernel does not have hyperparameter with name const
            const = 1.0
        return const

    @abstractmethod
    def _tau_P(self) -> Tuple[Array, Array]:
        raise NotImplementedError


class MonteCarloKernelStatistics(KernelStatistics):
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        ctrain_samples: Array,
        nquad_samples: int = 10000,
    ):
        super().__init__(gp, variable, ctrain_samples)
        self._set_quadrature_sample_weights(nquad_samples)

    def _set_quadrature_sample_weights(self, nquad_samples: int):
        self._nquad_samples = nquad_samples
        self._cquadx = self._gp._in_trans.map_to_canonical(
            self._variable.rvs(self._nquad_samples)
        )
        self._cquadw = self._bkd.full(
            (self._nquad_samples, 1), 1 / self._nquad_samples
        )

    def _tau_P(self) -> Tuple[Array, Array]:
        Kmat = self._gp.kernel()(self._cquadx, self._ctrain_samples)
        tau = self._bkd.sum(self._cquadw * Kmat, axis=0)
        Pmat = Kmat.T @ (self._cquadw * Kmat)
        return tau, Pmat


class MonteCarloMultiFidelityKernelStatistics(MonteCarloKernelStatistics):
    def _set_quadrature_sample_weights(self, nquad_samples: int):
        self._nquad_samples = nquad_samples
        # must only evaluate highest fidelity kernel at quadrature points
        # assumes last model is the high fidelity model
        empty = self._bkd.zeros((self._gp.nvars(), 0))
        hf_cquadx = self._gp._in_trans.map_to_canonical(
            self._variable.rvs(self._nquad_samples)
        )
        self._cquadx = [
            empty for nn in range(self._gp.kernel().noutputs() - 1)
        ] + [hf_cquadx]
        self._cquadw = self._bkd.full(
            (self._nquad_samples, 1), 1 / self._nquad_samples
        )


class TensorProductQuadratureKernelStatistics(KernelStatistics):
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        ctrain_samples: Array,
        nquad_nodes_1d: List[int] = None,
    ):
        if nquad_nodes_1d is None:
            nquad_nodes_1d = [30] * variable.nvars()
        self._nquad_nodes_1d = nquad_nodes_1d
        super().__init__(gp, variable, ctrain_samples)
        self._set_quadrature_rule()

    def _set_quadrature_rule(self):
        marginal_quad_rules = [
            GaussQuadratureRule(marginal, backend=self._variable._bkd)
            for marginal in self._variable.marginals()
        ]
        marginal_quad_data = [
            quad_rule(nnodes)
            for quad_rule, nnodes in zip(
                marginal_quad_rules, self._nquad_nodes_1d
            )
        ]
        # kernel operates in canonical space so map 1d quadrature rules
        # to canonical space
        self._canonical_quadx_1d = [
            self._gp._in_trans.map_to_canonical_1d(data[0], ii)
            for ii, data in enumerate(marginal_quad_data)
        ]
        # self._canonical_quadx_1d = [
        #     data[0] for ii, data in enumerate(marginal_quad_data)
        # ]
        self._quadw_1d = [data[1] for data in marginal_quad_data]

        # following quadrature rules only needed for gaussian process
        # stats but include here anyway for now

        # Map 2D quadrature rules to canonical space
        self._twodim_quadrules = [
            FixedTensorProductQuadratureRule(
                2,
                [marginal_quad_rules[ii]] * 2,
                [self._nquad_nodes_1d[ii]] * 2,
            )
            for ii in range(self._variable.nvars())
        ]

        # Map 3D quadrature rules to canonical space
        self._threedim_quadrules = [
            FixedTensorProductQuadratureRule(
                3,
                [marginal_quad_rules[ii]] * 3,
                [self._nquad_nodes_1d[ii]] * 3,
            )
            for ii in range(self._variable.nvars())
        ]

    def _two_dim_canonical_quadrature_tuple(
        self, ii: int
    ) -> Tuple[Array, Array]:
        xx_2d, ww_2d = self._twodim_quadrules[ii]()
        return (
            self._bkd.stack(
                [
                    self._gp._in_trans.map_to_canonical_1d(xx_2d[jj], ii)
                    for jj in range(2)
                ],
                axis=0,
            ),
            ww_2d,
        )

    def _three_dim_canonical_quadrature_tuple(
        self, ii: int
    ) -> Tuple[Array, Array]:
        xx_3d, ww_3d = self._threedim_quadrules[ii]()
        return (
            self._bkd.stack(
                [
                    self._gp._in_trans.map_to_canonical_1d(xx_3d[jj], ii)
                    for jj in range(3)
                ],
                axis=0,
            ),
            ww_3d,
        )

    def _integrate_tau_P_1d(
        self, can_xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Tuple[Array, Array]:
        # specific to squared exponential kernel. Move to kernel
        dists_1d_x1_can_xtr = (
            self._bkd.cdist(
                self._canonical_quadx_1d[ii].T / lscale_ii,
                can_xtr_ii.T / lscale_ii,
            )
            ** 2
        )
        K = self._bkd.exp(-0.5 * dists_1d_x1_can_xtr)
        tau = self._quadw_1d[ii][:, 0] @ K
        P = K.T @ (self._quadw_1d[ii] * K)
        return tau, P

    def _univariate_tau_P(self) -> Tuple[Array, Array]:
        if self._tau_1d is not None:
            # store to avoid recomputation
            return self._tau_1d, self._P_1d
        lscale = self._get_kernel_length_scale()
        tau, P = [], []
        for ii in range(self._gp.nvars()):
            tau_ii, P_ii = self._integrate_tau_P_1d(
                self._ctrain_samples[ii : ii + 1, :],
                lscale[ii],
                ii,
            )
            tau.append(tau_ii)
            P.append(P_ii)
        self._tau_1d = self._bkd.stack(tau, axis=0)
        self._P_1d = self._bkd.stack(P, axis=0)
        return self._tau_1d, self._P_1d

    def _integrate_conditional_P_1d(
        self, can_xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Array:
        can_xx_2d, ww_2d = self._two_dim_canonical_quadrature_tuple(ii)
        dists_2d_x2_can_xtr = (
            self._bkd.cdist(
                can_xx_2d[1:2, :].T / lscale_ii, can_xtr_ii.T / lscale_ii
            )
            ** 2
        )
        dists_2d_x1_can_xtr = (
            self._bkd.cdist(
                can_xx_2d[0:1, :].T / lscale_ii, can_xtr_ii.T / lscale_ii
            )
            ** 2
        )
        cond_P = self._bkd.exp(-0.5 * dists_2d_x1_can_xtr).T @ (
            ww_2d * self._bkd.exp(-0.5 * dists_2d_x2_can_xtr)
        )
        return cond_P

    def _univariate_conditional_P(self) -> Array:
        if self._cond_P_1d is not None:
            # store to avoid recomputation
            return self._cond_P_1d
        lscale = self._get_kernel_length_scale()
        cond_P = []
        for ii in range(self._gp.nvars()):
            cond_P_ii = self._integrate_conditional_P_1d(
                self._ctrain_samples[ii : ii + 1, :],
                lscale[ii],
                ii,
            )
            cond_P.append(cond_P_ii)
        self._cond_P_1d = self._bkd.stack(cond_P, axis=0)
        return self._cond_P_1d

    def _tau_P(self) -> Tuple[Array, Array]:
        tau, P = self._univariate_tau_P()
        return self._bkd.prod(tau, axis=0), self._bkd.prod(P, axis=0)

    def _integrate_u_lamda_Pi_nu_1d(
        self, can_xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Tuple[Array, Array]:
        # TODO pass in 1D kernel objects to remove need to pass around lscale
        can_xx_2d, ww_2d = self._two_dim_canonical_quadrature_tuple(ii)
        dists_2d_x1_x2 = (
            can_xx_2d[0, :] / lscale_ii - can_xx_2d[1, :] / lscale_ii
        ) ** 2
        K = self._bkd.exp(-0.5 * dists_2d_x1_x2)
        u = ww_2d[:, 0] @ K
        dists_2d_x1_x2 = (
            can_xx_2d[0:1, :].T / lscale_ii - can_xx_2d[1:2, :].T / lscale_ii
        ) ** 2
        dists_2d_x2_can_xtr = (
            self._bkd.cdist(
                can_xx_2d[1:2, :].T / lscale_ii, can_xtr_ii.T / lscale_ii
            )
            ** 2
        )
        lamda = (
            self._bkd.exp(
                -0.5 * dists_2d_x1_x2.T - 0.5 * dists_2d_x2_can_xtr.T
            )
            @ ww_2d[:, 0]
        )
        dists_2d_x1_can_xtr = (
            self._bkd.cdist(
                can_xx_2d[0:1, :].T / lscale_ii, can_xtr_ii.T / lscale_ii
            )
            ** 2
        )
        w = self._bkd.exp(-0.5 * dists_2d_x1_x2[:, 0]) * ww_2d[:, 0]
        Pi = self._bkd.exp(-0.5 * dists_2d_x1_can_xtr).T @ (
            w[:, None] * self._bkd.exp(-0.5 * dists_2d_x2_can_xtr)
        )
        nu = self._bkd.exp(-dists_2d_x1_x2)[:, 0] @ ww_2d
        return u, lamda, Pi, nu

    def _univariate_u_lamda_Pi_nu(self) -> Tuple[Array, Array, Array, Array]:
        if self._u_1d is not None:
            # store to avoid recomputation
            return self._u_1d, self._lamda_1d, self._Pi_1d, self._nu_1d
        lscale = self._get_kernel_length_scale()
        u, lamda, Pi, nu = [], [], [], []
        for ii in range(self._gp.nvars()):
            u_ii, lamda_ii, Pi_ii, nu_ii = self._integrate_u_lamda_Pi_nu_1d(
                self._ctrain_samples[ii : ii + 1, :],
                lscale[ii],
                ii,
            )
            u.append(u_ii)
            lamda.append(lamda_ii)
            Pi.append(Pi_ii)
            nu.append(nu_ii)

        self._u_1d = self._bkd.stack(u, axis=0)
        self._lamda_1d = self._bkd.stack(lamda, axis=0)
        self._Pi_1d = self._bkd.stack(Pi, axis=0)
        self._nu_1d = self._bkd.stack(nu, axis=0)
        return self._u_1d, self._lamda_1d, self._Pi_1d, self._nu_1d

    def _reset_memory(self):
        self._tau_1d, self._cond_P_1d, self._u_1d = None, None, None

    def _u_lamda_Pi_nu(self) -> Tuple[Array, Array, Array, Array]:
        u, lamda, Pi, nu = self._univariate_u_lamda_Pi_nu()
        return (
            self._bkd.prod(u, axis=0),
            self._bkd.prod(lamda, axis=0),
            self._bkd.prod(Pi, axis=0),
            self._bkd.prod(nu, axis=0),
        )

    def _integrate_xi_1_1d(self, lscale_ii: float, ii: int) -> Array:
        can_xx_3d, ww_3d = self._three_dim_canonical_quadrature_tuple(ii)
        dists_3d_x1_x2 = (
            can_xx_3d[0, :] / lscale_ii - can_xx_3d[1, :] / lscale_ii
        ) ** 2
        dists_3d_x2_x3 = (
            can_xx_3d[1, :] / lscale_ii - can_xx_3d[2, :] / lscale_ii
        ) ** 2
        xi_1 = (
            self._bkd.exp(-0.5 * dists_3d_x1_x2 - 0.5 * dists_3d_x2_x3)
            @ ww_3d[:, 0]
        )
        return xi_1

    def _xi_1(self):
        lscale = self._get_kernel_length_scale()
        xi_1 = []
        for ii in range(self._gp.nvars()):
            xi_1_ii = self._integrate_xi_1_1d(lscale[ii], ii)
            xi_1.append(xi_1_ii)
        return self._bkd.prod(self._bkd.stack(xi_1, axis=0), axis=0)


class GaussianProcessStatistics(TensorProductQuadratureKernelStatistics):
    # GuassianProcessStatistics does depend on function values
    # Only derive from TensorProductQuadratureKernelStatistics
    # as high-accuracy is needed when computing stats of GP

    # Base KernelStatistics does not depend on function values
    # useful for experimental design which does not need super high accuracy
    # and so can be used with MC or gauss quadrature
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        nquad_nodes_1d: List[int] = None,
    ):
        super().__init__(gp, variable, gp._ctrain_samples, nquad_nodes_1d)
        self._set_Ainv()

    def _set_ctrain_samples(self):
        self._ctrain_samples = self._gp._ctrain_samples

    def _set_Ainv(self):
        coef_args = self._gp._factor_training_kernel_matrix()
        Linv = self._gp._inverse_of_cholesky_factor(coef_args[0])
        # store kernel matrix inverse that is not scaled by the
        # kernel variance
        self._Ainv = Linv.T @ Linv * self._get_kernel_variance()

        condition_number = self._bkd.cond(self._Ainv)
        if condition_number > 1e8:
            warnings.warn(
                "\nCondition number of kernel training matrix is "
                f"large {condition_number=}.\n"
                "Accuracy of statistics may be effected (especially "
                "variance).\nIncreasing gp kernel_reg parameter may help."
            )

        # for now assume no trend
        if self._gp._trend is not None:
            raise ValueError("gp._trend must be None")

    def expectation_of_mean(self) -> Array:
        tau = self._tau_P()[0]
        expected_mean = tau @ self._Ainv_y()
        # for now out_trans is the identity
        return self._out_trans.adjust_expectation_of_mean(expected_mean)

    def variance_of_mean(self) -> Array:
        tau = self._tau_P()[0]
        varpi = tau @ self._Ainv @ tau
        u = self._u_lamda_Pi_nu()[0]
        varsigma_sq = u - varpi
        # todo extract kernel variance from kernel
        return self._get_kernel_variance() * varsigma_sq

    def _expectation_of_variance(self, P, tau, v_sq):
        Ainv_y = self._Ainv_y()
        zeta = Ainv_y.T @ P @ Ainv_y
        # reactivate once allow for out_trans to be not None
        zeta = self._out_trans.adjust_zeta(zeta, tau, Ainv_y)
        kernel_var = self._get_kernel_variance()
        expected_mean = self.expectation_of_mean()
        variance_mean = self.variance_of_mean()
        expected_variance = (
            zeta + v_sq * kernel_var - expected_mean**2 - variance_mean
        )
        return expected_variance

    def expectation_of_variance(self):
        tau, P = self._tau_P()
        v_sq = 1.0 - self._bkd.sum(self._Ainv * P)
        return self._expectation_of_variance(P, tau, v_sq)

    def variance_of_variance(self) -> float:
        tau, P = self._tau_P()
        Ainv_P = self._Ainv @ P
        varphi = self._bkd.sum(Ainv_P.T * Ainv_P)
        u, lamda, Pi, nu = self._u_lamda_Pi_nu()
        psi = self._bkd.sum(self._Ainv.T * Pi)
        chi = nu + varphi - 2.0 * psi
        eta = self.expectation_of_mean()
        Ainv_y = self._Ainv_y()
        varrho = lamda @ Ainv_y - tau @ Ainv_P @ Ainv_y
        phi = Ainv_y.T @ Pi @ Ainv_y - self._bkd.multidot(
            (Ainv_y.T, P, Ainv_P, Ainv_y)
        )
        # add back in once out_trans is not None
        # adjust phi with unadjusted varrho
        # phi += 2*y_train_mean*varrho+y_train_mean**2*varsigma_sq
        # now adjust varrho
        # varrho += y_train_mean*varsigma_sq
        varpi = tau @ self._Ainv @ tau
        u = self._u_lamda_Pi_nu()[0]
        varsigma_sq = u - varpi
        phi = self._out_trans.adjust_phi(phi, varrho, varsigma_sq)
        varrho = self._out_trans.adjust_varrho(varrho, varsigma_sq)

        Ainv_tau = self._Ainv @ tau
        xi_1 = self._xi_1()
        xi = xi_1 + tau @ Ainv_P @ Ainv_tau - 2.0 * lamda @ Ainv_tau
        v_sq = 1.0 - self._bkd.sum(self._Ainv * P)
        Ainv_y = self._Ainv_y()
        zeta = Ainv_y.T @ P @ Ainv_y
        zeta = self._out_trans.adjust_zeta(zeta, tau, Ainv_y)
        kernel_var = self._get_kernel_variance()

        # E[I_2^2] (term1)
        term1 = (
            4 * phi * kernel_var
            + 2 * chi * kernel_var**2
            + (zeta + v_sq * kernel_var) ** 2
        )
        # -2E[I_2I^2] (term2)
        term2 = (
            4 * eta * varrho * kernel_var
            + 2 * xi * kernel_var**2
            + zeta * varsigma_sq * kernel_var
            + v_sq * varsigma_sq * kernel_var**2
            + zeta * eta**2
            + eta**2 * v_sq * kernel_var
        )
        # E[I^4] (term 3)
        term3 = (
            3 * varsigma_sq**2 * kernel_var**2
            + 6 * eta**2 * varsigma_sq * kernel_var
            + eta**4
        )
        expected_variance = self.expectation_of_variance()
        variance_of_variance = term1 - 2 * term2 + term3 - expected_variance**2
        return variance_of_variance

    def conditional_variance(self, index: Array) -> float:
        tau_1d, P_1d = self._univariate_tau_P()
        tau = self._bkd.prod(tau_1d, axis=0)
        cond_P_1d = self._univariate_conditional_P()
        u_1d = self._univariate_u_lamda_Pi_nu()[0]
        P_p, U_p = 1, 1
        for ii in range(self._gp.nvars()):
            if index[ii] == 1:
                P_p *= P_1d[ii]
                U_p *= 1
            else:
                P_p *= cond_P_1d[ii]
                U_p *= u_1d[ii]
        trace_A_inv_Pp = self._bkd.sum(self._Ainv * P_p)
        v_sq = U_p - trace_A_inv_Pp
        return self._expectation_of_variance(P_p, tau, v_sq)

    def _Ainv_y(self) -> Array:
        # operator in user space for values
        # this means cannot call self._gp._Kinv_y(self._Ainv)
        # because if operats on _ctrain_values
        return self._Ainv @ self._gp.get_train_values()


class EnsembleGaussianProcessStatistics(GaussianProcessStatistics):
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        nquad_nodes_1d: List[int] = None,
        ninterpolation_samples: int = None,
    ):
        self._ninterpolation_samples = ninterpolation_samples
        super().__init__(gp, variable, nquad_nodes_1d)

    def _set_ctrain_samples(self):
        # set the samples used to interpolate the realizations of
        # the Gaussian process. Cannot use gp._ctrain_samples because
        # only variability in these samples will come from having
        # white noise kernel or from a non-zero kernel_reg value.
        # Samples away from training samples are needed. So generate
        # a well conditioned set using Cholesky sampling.
        # TODO: I can envision an issue with this function if cholesky
        # sampling is used to generate the training samples and then
        # is used to interpolate the gp realizations. Deal with this
        # issue if it becomes a problem. To partially overcome this problem
        # ensure number of realization interpolation samples is larger
        # than number of gp training samples
        if self._ninterpolation_samples is None:
            self._ninterpolation_samples = min(
                2 * self._gp._ctrain_samples.shape[1], 1000
            )
        # import here to avoid circular import. TODO remove circular import
        from pyapprox.surrogates.autogp.activelearning import CholeskySampler

        sampler = CholeskySampler(self._variable, nugget=self._gp._kernel_reg)
        sampler.set_gaussian_process(self._gp)
        self._train_samples = sampler(self._ninterpolation_samples)
        self._ctrain_samples = self._gp._in_trans.map_to_canonical(
            self._train_samples
        )
        self._realization = self._gp(self._train_samples)

    def _set_Ainv(self) -> Array:
        self._set_ctrain_samples()
        A = self._gp.kernel()(self._ctrain_samples)
        A = A + self._bkd.eye(A.shape[0]) * float(self._gp._kernel_reg)
        # store kernel matrix inverse that is not scaled by the
        # kernel variance
        self._Ainv = self._bkd.inv(A) * self._get_kernel_variance()

    def _Ainv_y(self) -> Array:
        return self._Ainv @ self._realization

    def _stats_of_realizations(
        self, stat_function: callable, nrealizations: int
    ) -> Array:
        nrealizations = int(nrealizations)
        if (
            not hasattr(self, "_realizations")
            or nrealizations != self._realizations.shape[1]
        ):
            # reuse same realizations if possible
            self._realizations = self._gp.predict_random_realizations(
                self._train_samples, nrealizations
            )
        means = []
        for ii in range(nrealizations):
            self._realization = self._realizations[:, ii : ii + 1]
            means.append(stat_function())
        return self._bkd.hstack(means)

    def means_of_realizations(self, nrealizations: int) -> Array:
        return self._stats_of_realizations(
            self.expectation_of_mean, nrealizations
        )

    def variances_of_realizations(self, nrealizations: int) -> Array:
        return self._stats_of_realizations(
            self.expectation_of_variance, nrealizations
        )

    def conditional_variances_of_realizations(
        self, index: Array, nrealizations: int
    ) -> Array:
        return self._stats_of_realizations(
            partial(self.conditional_variance, index), nrealizations
        )


class MarginalizedGaussianProcessKernel(Kernel):
    def __init__(self, stat: GaussianProcessStatistics, active_id: int):
        super().__init__(stat._bkd)
        if stat._gp.nvars() == 1:
            raise ValueError("Cannot marginalize a 1D Gaussian Process")
        gp_kernel = stat._gp.kernel()
        # if gp_kernel._nu != np.inf:
        #     raise ValueError("Must be squared exponential kernel")
        # TODO deal with composition kernels
        self._kernel = MaternKernel(
            np.inf,
            lenscale=stat._get_kernel_length_scale()[active_id],
            # bounds are not important becaues variable is fixed
            lenscale_bounds=[0.1, 1],
            nvars=1,
            fixed=True,
            backend=gp_kernel._bkd,
        )
        self._stat = stat
        self._active_id = active_id
        self._marginalize()
        self._hyp_list = self._kernel.hyp_list()

    def _marginalize(self) -> ExactGaussianProcess:
        tau_1d, P_1d = self._stat._univariate_tau_P()
        self._tau = self._bkd.prod(
            tau_1d[: self._active_id], axis=0
        ) * self._bkd.prod(tau_1d[self._active_id + 1 :], axis=0)
        u_1d, lamda_1d, Pi_1d, nu_1d = self._stat._univariate_u_lamda_Pi_nu()
        self._u = self._bkd.prod(
            u_1d[: self._active_id], axis=0
        ) * self._bkd.prod(u_1d[self._active_id + 1 :], axis=0)

    def diag(self, X):
        return self._kernel.diag(X) * self._u

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        return self._kernel(X1, X2) * self._tau


class MarginalizedGaussianProcessInputTransform(Transform):
    def __init__(self, trans: Transform, active_id: int):
        self._trans = trans
        self._active_id = active_id
        super().__init__(trans._bkd)

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        return self._trans.map_from_canonical_1d(
            canonical_samples, self._active_id
        )

    def map_to_canonical(self, user_samples: Array) -> Array:
        return self._trans.map_to_canonical_1d(user_samples, self._active_id)


def marginalize_gaussian_process(
    gp: ExactGaussianProcess,
    variable: IndependentMarginalsVariable,
    active_id: int,
) -> ExactGaussianProcess:
    # TODO: allow marginalization to have more than one active variable
    # not much should change except
    # MarginalizedGaussianProcessKernel._marginalize spliting over more
    # than one variable and gp._ctrain_samples below also having
    # multiple active ids
    stat = GaussianProcessStatistics(gp, variable)
    marginalized_kernel = MarginalizedGaussianProcessKernel(stat, active_id)
    kernel_var = stat._get_kernel_variance()
    constant_kernel = ConstantKernel(
        kernel_var,
        (1e-3, 1e1),  # bounds do not matter because params fixed
        fixed=True,
        backend=gp._bkd,
    )
    marginalized_kernel = constant_kernel * marginalized_kernel
    marginalized_gp = ExactGaussianProcess(
        1,
        marginalized_kernel,
        trend=gp.trend(),
        kernel_reg=1e-7,
    )
    marginalized_gp.set_output_transform(gp._out_trans)
    marginalized_gp.set_input_transform(
        MarginalizedGaussianProcessInputTransform(gp._in_trans, active_id)
    )
    marginalized_gp._ctrain_samples = gp._ctrain_samples[
        active_id : active_id + 1
    ]
    marginalized_gp._ctrain_values = gp._ctrain_values
    marginalized_gp._coef_args = gp._coef_args
    marginalized_gp._coef = gp._coef
    return marginalized_gp
