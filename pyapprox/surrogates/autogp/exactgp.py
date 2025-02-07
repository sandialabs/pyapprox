from abc import abstractmethod
from typing import Tuple, List
import warnings

import numpy as np

from pyapprox.surrogates.regressor import OptimizedRegressor
from pyapprox.surrogates.autogp.mokernels import MultiPeerKernel
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.kernels.kernels import Kernel
from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    MultiStartOptimizer,
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.basis import (
    FixedTensorProductQuadratureRule,
    GaussQuadratureRule,
)
from pyapprox.util.transforms import (
    IdentityTransform,
    StandardDeviationTransform,
)


class GaussianProcessTransform:
    @abstractmethod
    def map_stdev_from_canonical(self, canonical_stdevs: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def map_covariance_from_canonical(self, canonical_cov: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def adjust_expectation_of_mean(self, expected_mean: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def adjust_zeta(self, zeta: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def adjust_phi(
        self, phi: float, varrho: float, varsigma_sq: float
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def adjust_varrho(self, varrho: float, varsigma_sq: float) -> float:
        raise NotImplementedError


class GaussianProcessIdentityTransform(
    IdentityTransform, GaussianProcessTransform
):
    def map_stdev_from_canonical(self, canonical_stdevs: Array) -> Array:
        return canonical_stdevs

    def map_covariance_from_canonical(self, canonical_cov: Array) -> Array:
        return canonical_cov

    def adjust_expectation_of_mean(self, expected_mean: float) -> float:
        return expected_mean

    def adjust_zeta(self, zeta: float, tau: Array, Ainv_y: Array) -> float:
        return zeta

    def adjust_phi(
        self, phi: float, varrho: float, varsigma_sq: float
    ) -> float:
        return phi

    def adjust_varrho(self, varrho: float, varsigma_sq: float) -> float:
        return varrho


class GaussianProcessStandardDeviationTransform(
    StandardDeviationTransform, GaussianProcessTransform
):
    def map_stdev_from_canonical(self, canonical_stdevs: Array) -> Array:
        return canonical_stdevs * self._stdevs

    def map_covariance_from_canonical(self, canonical_cov: Array) -> Array:
        return canonical_cov * self._stdevs**2

    def adjust_expectation_of_mean(self, expected_mean: float) -> float:
        # accounting for trend may just requred + gp.trend.mean()
        return expected_mean * self._stdevs + self._means

    def adjust_zeta(self, zeta: float, tau: Array, Ainv_y: Array) -> float:
        # Need to determine how to account for trend
        return (
            zeta * self._stdevs**2
            + 2 * tau @ Ainv_y * self._means * self._stdevs
            + self._means**2
        )

    def adjust_phi(
        self, phi: float, varrho: float, varsigma_sq: float
    ) -> float:
        raise NotImplementedError("Tests do not pass")
        return (
            phi * self._stdevs**2
            + 2 * varrho * self._means * self._stdevs
            + self._means**2 * varsigma_sq
        )

    def adjust_varrho(self, varrho: float, varsigma_sq: float) -> float:
        raise NotImplementedError("Tests do not pass")
        return varrho * self._stdevs + self._means * varsigma_sq


class GPNegLogLikelihoodLoss(LossFunction):
    def _loss_values(self, active_opt_params):
        vals = self._model._neg_log_like(active_opt_params[:, 0])[:, None]
        return vals

    def _check_model(self, model):
        if not isinstance(ExactGaussianProcess):
            raise ValueError(
                "model must be an instance of ExactGaussianProcess"
            )
        super()._check_model(model)

    def _jacobian(self, active_opt_params: Array) -> Array:
        if self._model._analytical_neg_log_like_jacobian_implemented:
            return self._model._jacobian_neg_log_like_with_hyperparam_trend(
                active_opt_params[:, 0]
            )
        return super()._jacobian(active_opt_params)


class ExactGaussianProcess(OptimizedRegressor):
    def __init__(
        self,
        nvars: int,
        kernel: Kernel,
        trend: BasisExpansion = None,
        kernel_reg: float = 0,
    ):
        super().__init__(kernel._bkd)
        self._kernel = kernel
        self._trend = trend
        self._kernel_reg = kernel_reg

        self._coef = None
        self._coef_args = None

        self._hyp_list = self._kernel.hyp_list()
        if trend is not None:
            self._hyp_list += self._trend.hyp_list()
        self.set_optimizer()
        self._analytical_neg_log_like_jacobian_implemented = True

    def _set_default_transforms(self):
        self.set_input_transform(IdentityTransform())
        self.set_output_transform(GaussianProcessIdentityTransform())

    def set_output_transform(self, out_trans: GaussianProcessTransform):
        if not isinstance(out_trans, GaussianProcessTransform):
            raise ValueError(
                f"out_trans {out_trans} must be an instance of "
                "GaussianProcessTransform"
            )
        self._out_trans = out_trans

    def kernel(self) -> Kernel:
        return self._kernel

    def trend(self) -> BasisExpansion:
        return self._trend

    def _default_iterator_gen(self):
        iterate_gen = RandomUniformOptimzerIterateGenerator(
            self._hyp_list.nactive_vars(), backend=self._bkd
        )
        iterate_gen.set_bounds(
            self._bkd.to_numpy(self._hyp_list.get_active_opt_bounds())
        )
        return iterate_gen

    def set_optimizer(
        self,
        ncandidates: int = 1,
        verbosity: int = 0,
        iterate_gen: OptimizerIterateGenerator = None,
    ):
        local_optimizer = ScipyConstrainedOptimizer()
        local_optimizer.set_options(
            gtol=1e-8,
            # ftol=1e-12,
            maxiter=1000,
            # method="L-BFGS-B",
            method="trust-constr",
        )
        local_optimizer.set_verbosity(verbosity - 1)
        ms_optimizer = MultiStartOptimizer(
            local_optimizer, ncandidates=ncandidates
        )
        if iterate_gen is None:
            iterate_gen = self._default_iterator_gen()
        if not isinstance(iterate_gen, OptimizerIterateGenerator):
            raise ValueError(
                "iterate_gen must be an instance of OptimizerIterateGenerator"
            )
        ms_optimizer.set_initial_iterate_generator(iterate_gen)
        ms_optimizer.set_verbosity(verbosity)
        super().set_optimizer(ms_optimizer)
        self.set_loss(GPNegLogLikelihoodLoss())
        self._loss._hessian_implemented = False
        self._loss._apply_hessian_implemented = False

    def nqoi(self):
        return 1

    def get_train_samples(self):
        return self._in_trans.map_from_canonical(self._ctrain_samples)

    def get_train_values(self):
        return self._out_trans.map_from_canonical(self._ctrain_values.T).T

    def _training_kernel_matrix(self) -> Tuple:
        # must only pass in X and not Y to kernel otherwise if noise kernel
        # is present it will not be evaluted correctly.
        kmat = self._kernel(self._ctrain_samples)
        # Below is an inplace operation that will not work with autograd
        # kmat[np.diag_indices_from(kmat)] += self._kernel_reg
        # This also does not work
        # kmat += diag(full((kmat.shape[0], 1), float(self._kernel_reg)))
        kmat = kmat + self._bkd.eye(kmat.shape[0]) * float(self._kernel_reg)
        return kmat

    def _training_kernel_jacobian(self):
        return self._kernel.jacobian(self._ctrain_samples)

    def _factor_training_kernel_matrix(self):
        # can be specialized
        kmat = self._training_kernel_matrix()
        try:
            return (self._bkd.cholesky(kmat),)
        except Exception:
            return (None, kmat)

    def _solve_coefficients(self, *args) -> Tuple:
        # can be specialized when _factor_training_kernel_matrix is specialized
        diff = self._ctrain_values - self._canonical_trend(
            self._ctrain_samples
        )
        return self._bkd.cholesky_solve(args[0], diff)

    def _Linv_y(self, *args) -> Array:
        diff = self._ctrain_values - self._canonical_trend(
            self._ctrain_samples
        )
        return self._bkd.solve_triangular(args[0], diff)

    def _Kinv_y(self, Kinv: Array) -> Array:
        diff = self._ctrain_values - self._canonical_trend(
            self._ctrain_samples
        )
        return Kinv @ diff

    def _log_determinant(self, coef_res: Tuple) -> float:
        # can be specialized when _factor_training_kernel_matrix is specialized
        chol_factor = coef_res[0]
        return 2 * self._bkd.log(self._bkd.get_diagonal(chol_factor)).sum()

    def _canonical_posterior_pointwise_variance(
        self, canonical_samples: Array, kmat_pred: Array
    ):
        # can be specialized when _factor_training_kernel_matrix is specialized
        tmp = self._bkd.solve_triangular(self._coef_args[0], kmat_pred.T)
        update = self._bkd.einsum("ji,ji->i", tmp, tmp)
        return (self._kernel.diag(canonical_samples) - update)[:, None]

    def _canonical_trend(self, canonical_samples: Array):
        if self._trend is None:
            return self._bkd.full((canonical_samples.shape[1], 1), 0.0)
        return self._trend(canonical_samples)

    def _neg_log_like_with_hyperparam_trend(self) -> float:
        # this can also be used if treating the trend as hyper_params
        # but cannot be used if assuming a prior on the coefficients
        coef_args = self._factor_training_kernel_matrix()
        if coef_args[0] is None:
            # cholesky factorization failed
            return coef_args[1][0, 0] * 0 + self._bkd.atleast1d(np.inf)
        Linv_y = self._Linv_y(*coef_args)
        nsamples = self._ctrain_values.shape[0]
        return 0.5 * (
            self._bkd.multidot((Linv_y.T, Linv_y))
            + self._log_determinant(coef_args)
            + nsamples * np.log(2 * np.pi)
        ).sum(axis=1)

    def _cholesky_inverse(self, Lmat):
        rhs = self._bkd.eye(Lmat.shape[0])
        Linv = self._bkd.solve_triangular(Lmat, rhs, lower=True)
        return Linv

    def _jacobian_neg_log_like_with_hyperparam_trend(
        self, active_opt_params: Array
    ) -> Array:
        self._hyp_list.set_active_opt_params(active_opt_params)
        # First compute jacobian with respect to kernel parameters
        # TODO this recomputes cholesky factorization
        coef_args = self._factor_training_kernel_matrix()
        if coef_args[0] is None:
            # cholesky factorization failed
            return self._bkd.full((1, active_opt_params.shape[0]), np.inf)
        Linv = self._cholesky_inverse(coef_args[0])
        Kinv = Linv.T @ Linv
        Kinv_y = self._Kinv_y(Kinv)
        Mat = Kinv_y @ Kinv_y.T - Kinv
        Kjac = self._training_kernel_jacobian()
        kernel_jac = -0.5 * self._bkd.einsum("ij,jik->k", Mat, Kjac)
        kernel_jac = kernel_jac[
            ..., self._kernel.hyp_list().get_active_indices()
        ]
        if self._trend is None:
            return kernel_jac[None, :]
        # Second compute jacobian with respect to trend parameters
        # __init__ adds trend parameters to hyperlist last so
        # just concatenate trend jacobian at the end of kernel jacobian
        trend_jac = (
            -Kinv_y.T
            @ self._trend.basis(self._ctrain_samples)[
                ..., self._trend.hyp_list().get_active_indices()
            ]
        )
        jac = self._bkd.hstack((kernel_jac[None, :], trend_jac))
        return jac

    def _neg_log_like_with_uncertain_trend(self) -> float:
        # See Equation 2.45 in Rasmussen's Book
        # trend cannot be passed as a HyperParameter but is estimated
        # probabilistically.
        raise NotImplementedError

    def _posterior_variance_with_uncertain_trend(self) -> float:
        # See Equation 2.42 in Rasmussen's Book
        # Because the coeficients of the trend are uncertain the posterior
        # trend and variance formulas change
        # These formulas are derived in the limit of uniformative prior
        # variance on the trend coefficients. Thus the prior trend variance
        # cannot be calculated exactly. So just pretend prior trend is fixed
        # i.e. the prior uncertainty does not add to prior variance
        raise NotImplementedError

    def _neg_log_like(self, active_opt_params: Array):
        self._hyp_list.set_active_opt_params(active_opt_params)
        return self._neg_log_like_with_hyperparam_trend()

    def _evaluate_canonical_prior(self, samples: Array, return_std: bool):
        canonical_trend = self._canonical_trend(
            self._in_trans.map_to_canonical(samples)
        )
        if not return_std:
            return canonical_trend, None
        canonical_std = self._bkd.sqrt(self._kernel.diag(samples))
        return canonical_trend, canonical_std

    def _evaluate_canonical_posterior(self, samples: Array, return_std: bool):
        if self._coef is None:
            self._coef_args = self._factor_training_kernel_matrix()
            if self._coef_args[0] is None:
                raise RuntimeError(
                    "Cholesky Factorization failed. "
                    "Look at Optimization history, likely failed"
                )
            self._coef = self._solve_coefficients(*self._coef_args)

        canonical_samples = self._in_trans.map_to_canonical(samples)
        kmat_pred = self._kernel(canonical_samples, self._ctrain_samples)
        canonical_trend = self._canonical_trend(
            canonical_samples
        ) + self._bkd.multidot((kmat_pred, self._coef))
        if not return_std:
            return canonical_trend, None

        canonical_pointwise_variance = (
            self._canonical_posterior_pointwise_variance(
                canonical_samples, kmat_pred
            )
        )
        if canonical_pointwise_variance.min() < 0:
            msg = "Some pointwise variances were negative. The largest "
            msg += "magnitude of the negative values was {0}".format(
                canonical_pointwise_variance.min()
            )
            warnings.warn(msg, UserWarning)
        canonical_pointwise_variance[canonical_pointwise_variance < 0] = 0
        canonical_pointwise_stdev = self._bkd.sqrt(
            canonical_pointwise_variance.T
        ).T
        assert canonical_pointwise_stdev.shape == canonical_trend.shape
        return canonical_trend, canonical_pointwise_stdev

    def _canonical_evaluate(self, samples: Array, return_std: bool):
        if self._ctrain_samples is None:
            return self._evaluate_canonical_prior(samples, return_std)
        return self._evaluate_canonical_posterior(samples, return_std)

    def covariance(self, samples: Array):
        canonical_samples = self._in_trans.map_to_canonical(samples)
        kmat_pred = self._kernel(canonical_samples, self._ctrain_samples)
        tmp = self._bkd.solve_triangular(self._coef_args[0], kmat_pred.T)
        canonical_cov = self._kernel(canonical_samples) - tmp.T @ tmp
        return self._out_trans.map_covariance_from_canonical(canonical_cov)

    def evaluate(self, samples: Array, return_std: bool):
        """
        Use when standard deviation of GP is needed.
        Otherwise use __call__
        """
        can_vals, can_std = self._canonical_evaluate(samples, return_std)
        if return_std:
            return (
                self._out_trans.map_from_canonical(can_vals),
                self._out_trans.map_stdev_from_canonical(can_std),
            )
        return self._out_trans.map_from_canonical(can_vals)

    def _values(self, samples):
        # regressor expects values in the canonical domain
        return self._canonical_evaluate(samples, False)[0]

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self._hyp_list._short_repr()
        )

    def _plot_1d(
        self,
        ax,
        test_samples,
        gp_trend,
        gp_std,
        nstdevs,
        fill_kwargs,
        plt_kwargs,
        plot_samples,
    ):
        if plot_samples is None:
            plot_samples = test_samples[0, :]
        im0 = ax.plot(plot_samples, gp_trend, **plt_kwargs)
        color = im0[-1].get_color()
        if "color" not in fill_kwargs:
            fill_kwargs["color"] = color
            added_fill_color = True
        else:
            added_fill_color = False
        im1 = ax.fill_between(
            plot_samples,
            gp_trend - nstdevs * gp_std,
            gp_trend + nstdevs * gp_std,
            **fill_kwargs,
        )
        if added_fill_color:
            del fill_kwargs["color"]
        return [im0, im1]

    def plot_1d(
        self,
        ax,
        bounds,
        npts_1d=101,
        nstdevs=2,
        plt_kwargs={},
        fill_kwargs={"alpha": 0.3},
        prior_kwargs=None,
        plot_samples=None,
    ):
        test_samples = np.linspace(bounds[0], bounds[1], npts_1d)[None, :]
        gp_trend, gp_std = self.evaluate(test_samples, return_std=True)
        ims = self._plot_1d(
            ax,
            test_samples,
            gp_trend[:, 0],
            gp_std[:, 0],
            nstdevs,
            fill_kwargs,
            plt_kwargs,
            plot_samples,
        )
        if prior_kwargs is None:
            return ims
        ims += self._plot_1d(
            ax, test_samples, gp_trend, gp_std, nstdevs, **prior_kwargs
        )
        return ims

    def plot(self, ax, bounds: Array, **kwargs):
        if len(bounds) % 2 != 0:
            raise ValueError(
                "Lower and upper bounds must be provied for each dimension"
            )
        nvars = len(bounds) // 2
        if nvars > 1:
            raise ValueError("plot was called but gp is not 1D")
            return
        if self._ctrain_samples.shape[0] != nvars:
            raise ValueError("nvars is inconsistent with training data")
        return self.plot_1d(ax, bounds, **kwargs)

    def predict_random_realizations(
        self, samples: Array, nrealizations: int
    ) -> Array:
        mean = self(samples)
        cov = self.covariance(samples)
        U, S, V = self._bkd.svd(cov)
        L = U * np.sqrt(S)
        # create nsamples x nvars then transpose so same samples
        # are produced if this function is called repeatedly with nsamples=1
        rand_noise = self._bkd.asarray(
            np.random.normal(0, 1, (int(nrealizations), mean.shape[0])).T
        )
        vals = mean + L @ rand_noise
        return vals


class MOExactGaussianProcess(ExactGaussianProcess):
    def _set_training_data(self, train_samples: list, train_values: list):
        # For now analytical neg log like from exact gaussian process
        # does not pass tests when used here so turn off.
        self._analytical_neg_log_like_jacobian_implemented = False
        self._ctrain_samples = [
            s for s in self._map_samples_to_canonical(train_samples)
        ]
        self._ctrain_values = self._bkd.vstack(
            [self._out_trans.map_to_canonical(v.T).T for v in train_values]
        )

    def _map_samples_to_canonical(self, samples):
        return [self._in_trans.map_to_canonical(s) for s in samples]

    def _canonical_trend(self, canonical_samples):
        if self._trend is not None:
            raise ValueError("Non-zero trend not supported for mulitoutput")
        return self._bkd.full(
            (sum([s.shape[1] for s in canonical_samples]), 1), 0.0
        )

    def get_train_samples(self):
        return [
            self._in_trans.map_to_canonical(s) for s in self._ctrain_samples
        ]

    def get_train_values(self):
        cnt = 0
        train_values = []
        for s in self._ctrain_samples:
            train_values.append(
                self._out_trans.map_from_canonical(
                    self._ctrain_values[cnt : cnt + s.shape[1]].T
                ).T
            )
            cnt += s.shape[1]
        return train_values

    def plot_1d(
        self,
        ax,
        bounds,
        output_id,
        npts_1d=101,
        nstdevs=2,
        plt_kwargs={},
        fill_kwargs={"alpha": 0.3},
        prior_kwargs=None,
        plot_samples=None,
    ):
        test_samples_base = self._bkd.linspace(bounds[0], bounds[1], npts_1d)[
            None, :
        ]
        noutputs = len(self._ctrain_samples)
        test_samples = [np.array([[]]) for ii in range(noutputs)]
        test_samples[output_id] = test_samples_base
        gp_trend, gp_std = self.evaluate(test_samples, return_std=True)
        ims = self._plot_1d(
            ax,
            test_samples[output_id],
            gp_trend[:, 0],
            gp_std[:, 0],
            nstdevs,
            fill_kwargs,
            plt_kwargs,
            plot_samples,
        )
        if prior_kwargs is None:
            return ims
        ims += self._plot_1d(
            ax, test_samples, gp_trend, gp_std, nstdevs, **prior_kwargs
        )
        return ims

    def plot(self, ax, bounds, output_id=-1, **kwargs):
        if len(bounds) % 2 != 0:
            raise ValueError(
                "Lower and upper bounds must be provied for each dimension"
            )
        nvars = len(bounds) // 2
        if nvars > 1:
            raise ValueError("plot was called but gp is not 1D")
            return
        if self._ctrain_samples[0].shape[0] != nvars:
            raise ValueError("nvars is inconsistent with training data")
        return self.plot_1d(ax, bounds, output_id, **kwargs)


class MOPeerExactGaussianProcess(MOExactGaussianProcess):
    def _solve_coefficients(self, *args) -> Tuple:
        # can be specialized when _factor_training_kernel_matrix is specialized
        diff = self._ctrain_values - self._canonical_trend(
            self._ctrain_samples
        )
        return MultiPeerKernel._cholesky_solve(*args, diff, self._bkd)

    def _log_determinant(self, coef_res: Tuple) -> float:
        # can be specialized when _factor_training_kernel_matrix is specialized
        return MultiPeerKernel._logdet(*coef_res, self._bkd)

    def _training_kernel_matrix(self) -> Tuple:
        # must only pass in X and not Y to kernel otherwise if noise kernel
        # is present it will not be evaluted correctly.
        blocks = self._kernel(self._ctrain_samples, block_format=True)
        for ii in range(len(blocks)):
            blocks[ii][ii] = blocks[ii][ii] + self._bkd.eye(
                blocks[ii][ii].shape[0]
            ) * float(self._kernel_reg)
        return blocks

    def _factor_training_kernel_matrix(self):
        blocks = self._training_kernel_matrix()
        return MultiPeerKernel._cholesky(
            len(blocks[0]), blocks, self._bkd, block_format=True
        )

    def _Linv_y(self, *args):
        diff = self._ctrain_values - self._canonical_trend(
            self._ctrain_samples
        )
        return MultiPeerKernel._lower_solve_triangular(*args, diff, self._bkd)

    def _canonical_posterior_pointwise_variance(
        self, canonical_samples, kmat_pred
    ):
        # can be specialized when _factor_training_kernel_matrix is specialized
        tmp = MultiPeerKernel._lower_solve_triangular(
            *self._coef_args, kmat_pred.T, self._bkd
        )
        update = self._bkd.einsum("ji,ji->i", tmp, tmp)
        return (self._kernel.diag(canonical_samples) - update)[:, None]


class GaussianProcessStatistics:
    def __init__(
        self,
        gp: ExactGaussianProcess,
        variable: IndependentMarginalsVariable,
        nquad_nodes_1d: List[int] = None,
    ):
        self._gp = gp
        self._bkd = self._gp._bkd
        self.set_quadrature_rule(variable, nquad_nodes_1d)

        # todo consider storing this in gp once it is trained
        coef_args = self._gp._factor_training_kernel_matrix()
        Linv = self._gp._cholesky_inverse(coef_args[0])
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

        # for now assume out_trans is the identity
        # if not isinstance(
        #         self._gp._out_trans, GaussianProcessIdentityTransform
        # ):
        #     raise ValueError(
        #         "gp._out_trans must be an instance of "
        #         "GaussianProcessIdentityTransform"
        #     )

        # for now assume no trend
        if self._gp._trend is not None:
            raise ValueError("gp._trend must be None")

        # store train samples in user space for ease of reference
        self._train_samples = self._gp._in_trans.map_from_canonical(
            self._gp._ctrain_samples
        )
        self._train_values = self._gp._out_trans.map_from_canonical(
            self._gp._ctrain_values
        )

    def set_quadrature_rule(
        self,
        variable: IndependentMarginalsVariable,
        nquad_nodes_1d: List[int] = None,
    ):
        if nquad_nodes_1d is None:
            nquad_nodes_1d = [30] * variable.num_vars()
        marginal_quad_rules = [
            GaussQuadratureRule(marginal, backend=variable._bkd)
            for marginal in variable.marginals()
        ]
        marginal_quad_data = [
            quad_rule(nnodes)
            for quad_rule, nnodes in zip(marginal_quad_rules, nquad_nodes_1d)
        ]
        self._quadx_1d = [data[0] for data in marginal_quad_data]
        self._quadw_1d = [data[1] for data in marginal_quad_data]

        self._twodim_quadrules = [
            FixedTensorProductQuadratureRule(
                2, [marginal_quad_rules[ii]] * 2, [nquad_nodes_1d[ii]] * 2
            )
            for ii in range(variable.num_vars())
        ]

        self._threedim_quadrules = [
            FixedTensorProductQuadratureRule(
                3, [marginal_quad_rules[ii]] * 3, [nquad_nodes_1d[ii]] * 3
            )
            for ii in range(variable.num_vars())
        ]

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

    def _integrate_tau_P_1d(
        self, xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Tuple[Array, Array]:
        # specific to squared exponential kernel. Move to kernel
        dists_1d_x1_xtr = (
            self._bkd.cdist(
                self._quadx_1d[ii].T / lscale_ii,
                xtr_ii.T / lscale_ii,
            )
            ** 2
        )
        K = self._bkd.exp(-0.5 * dists_1d_x1_xtr)
        tau = self._quadw_1d[ii][:, 0] @ K
        P = K.T @ (self._quadw_1d[ii] * K)
        return tau, P

    def _tau_P(self) -> Tuple[Array, Array]:
        lscale = self._get_kernel_length_scale()
        tau, P = [], []
        for ii in range(self._gp.nvars()):
            tau_ii, P_ii = self._integrate_tau_P_1d(
                self._train_samples[ii : ii + 1, :], lscale[ii], ii
            )
            tau.append(tau_ii)
            P.append(P_ii)
        return (
            self._bkd.prod(self._bkd.stack(tau, axis=0), axis=0),
            self._bkd.prod(self._bkd.stack(P, axis=0), axis=0),
        )

    def _integrate_u_lamda_Pi_nu_1d(
        self, xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Tuple[Array, Array]:
        # TODO pass in 1D kernel objects to remove need to pass around lscale
        xx_2d, ww_2d = self._twodim_quadrules[ii]()
        dists_2d_x1_x2 = (
            xx_2d[0, :] / lscale_ii - xx_2d[1, :] / lscale_ii
        ) ** 2
        K = self._bkd.exp(-0.5 * dists_2d_x1_x2)
        u = ww_2d[:, 0] @ K
        dists_2d_x1_x2 = (
            xx_2d[0:1, :].T / lscale_ii - xx_2d[1:2, :].T / lscale_ii
        ) ** 2
        dists_2d_x2_xtr = (
            self._bkd.cdist(xx_2d[1:2, :].T / lscale_ii, xtr_ii.T / lscale_ii)
            ** 2
        )
        lamda = (
            self._bkd.exp(-0.5 * dists_2d_x1_x2.T - 0.5 * dists_2d_x2_xtr.T)
            @ ww_2d[:, 0]
        )
        dists_2d_x1_xtr = (
            self._bkd.cdist(xx_2d[0:1, :].T / lscale_ii, xtr_ii.T / lscale_ii)
            ** 2
        )
        w = self._bkd.exp(-0.5 * dists_2d_x1_x2[:, 0]) * ww_2d[:, 0]
        Pi = self._bkd.exp(-0.5 * dists_2d_x1_xtr).T @ (
            w[:, None] * self._bkd.exp(-0.5 * dists_2d_x2_xtr)
        )
        nu = self._bkd.exp(-dists_2d_x1_x2)[:, 0] @ ww_2d
        return u, lamda, Pi, nu

    def _u_lamda_Pi_nu(self) -> Tuple[Array, Array, Array, Array]:
        lscale = self._get_kernel_length_scale()
        u, lamda, Pi, nu = [], [], [], []
        for ii in range(self._gp.nvars()):
            u_ii, lamda_ii, Pi_ii, nu_ii = self._integrate_u_lamda_Pi_nu_1d(
                self._train_samples[ii : ii + 1, :], lscale[ii], ii
            )
            u.append(u_ii)
            lamda.append(lamda_ii)
            Pi.append(Pi_ii)
            nu.append(nu_ii)
        return (
            self._bkd.prod(self._bkd.stack(u, axis=0), axis=0),
            self._bkd.prod(self._bkd.stack(lamda, axis=0), axis=0),
            self._bkd.prod(self._bkd.stack(Pi, axis=0), axis=0),
            self._bkd.prod(self._bkd.stack(nu, axis=0), axis=0),
        )

    def expectation_of_mean(self) -> Array:
        tau = self._tau_P()[0]
        expected_mean = tau @ self._gp._Kinv_y(self._Ainv)
        # for now out_trans is the identity
        return self._gp._out_trans.adjust_expectation_of_mean(expected_mean)

    def variance_of_mean(self) -> Array:
        tau = self._tau_P()[0]
        varpi = tau @ self._Ainv @ tau
        u = self._u_lamda_Pi_nu()[0]
        varsigma_sq = u - varpi
        # todo extract kernel variance from kernel
        return self._get_kernel_variance() * varsigma_sq

    def expectation_of_variance(self):
        tau, P = self._tau_P()
        v_sq = 1.0 - self._bkd.sum(self._Ainv * P)
        Ainv_y = self._gp._Kinv_y(self._Ainv)
        zeta = Ainv_y.T @ P @ Ainv_y
        # reactivate once allow for out_trans to be not None
        zeta = self._gp._out_trans.adjust_zeta(zeta, tau, Ainv_y)

        kernel_var = self._get_kernel_variance()
        expected_mean = self.expectation_of_mean()
        variance_mean = self.variance_of_mean()
        expected_variance = (
            zeta + v_sq * kernel_var - expected_mean**2 - variance_mean
        )
        return expected_variance

    def _integrate_xi_1_1d(
        self, xtr_ii: Array, lscale_ii: float, ii: int
    ) -> Array:
        xx_3d, ww_3d = self._threedim_quadrules[ii]()
        dists_3d_x1_x2 = (
            xx_3d[0, :] / lscale_ii - xx_3d[1, :] / lscale_ii
        ) ** 2
        dists_3d_x2_x3 = (
            xx_3d[1, :] / lscale_ii - xx_3d[2, :] / lscale_ii
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
            xi_1_ii = self._integrate_xi_1_1d(
                self._train_samples[ii : ii + 1, :], lscale[ii], ii
            )
            xi_1.append(xi_1_ii)
        return self._bkd.prod(self._bkd.stack(xi_1, axis=0), axis=0)

    def variance_of_variance(self):
        tau, P = self._tau_P()
        Ainv_P = self._Ainv @ P
        varphi = self._bkd.sum(Ainv_P.T * Ainv_P)
        u, lamda, Pi, nu = self._u_lamda_Pi_nu()
        psi = self._bkd.sum(self._Ainv.T * Pi)
        chi = nu + varphi - 2.0 * psi
        eta = self.expectation_of_mean()
        Ainv_y = self._gp._Kinv_y(self._Ainv)
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
        phi = self._gp._out_trans.adjust_phi(phi, varrho, varsigma_sq)
        varrho = self._gp._out_trans.adjust_varrho(varrho, varsigma_sq)

        Ainv_tau = self._Ainv @ tau
        xi_1 = self._xi_1()
        xi = xi_1 + tau @ Ainv_P @ Ainv_tau - 2.0 * lamda @ Ainv_tau
        v_sq = 1.0 - self._bkd.sum(self._Ainv * P)
        Ainv_y = self._gp._Kinv_y(self._Ainv)
        zeta = Ainv_y.T @ P @ Ainv_y
        zeta = self._gp._out_trans.adjust_zeta(zeta, tau, Ainv_y)
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
