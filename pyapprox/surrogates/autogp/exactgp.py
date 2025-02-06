from typing import Tuple
import warnings

import numpy as np
import torch
import scipy

from pyapprox.surrogates.regressor import OptimizedRegressor
from pyapprox.surrogates.autogp.mokernels import MultiPeerKernel
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.kernels.kernels import Kernel
from pyapprox.util.transforms import Transform, IdentityTransform
from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.surrogates.loss import LossFunction
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    MultiStartOptimizer,
    OptimizerIterateGenerator,
    RandomUniformOptimzerIterateGenerator,
)
from pyapprox.interface.model import ScipyModelWrapper, Model


class GPNegLogLikelihoodLoss(LossFunction):
    def _loss_values(self, active_opt_params):
        vals = self._model._neg_log_likelihood(
            active_opt_params[:, 0]
        )[:, None]
        return vals

    def _check_model(self, model):
        if not isinstance(ExactGaussianProcess):
            raise ValueError(
                "model must be an instance of ExactGaussianProcess"
            )
        super()._check_model(model)

    def _jacobian(self, active_opt_params: Array) -> Array:
        if self._model._analytical_neg_log_likelihood_jacobian_implemented:
            return self._model._jacobian_neg_log_likelihood_with_hyperparameter_trend(
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
        self._analytical_neg_log_likelihood_jacobian_implemented = True

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
            iterate_gen: OptimizerIterateGenerator = None
    ):
        optimizer = ScipyConstrainedOptimizer()
        optimizer.set_options(
            gtol=1e-8, ftol=1e-12, maxiter=1000, method="L-BFGS-B",
        )
        optimizer.set_verbosity(0)
        ms_optimizer = MultiStartOptimizer(optimizer, ncandidates=ncandidates)
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

    def _neg_log_likelihood_with_hyperparameter_trend(self) -> float:
        # this can also be used if treating the trend as hyper_params
        # but cannot be used if assuming a prior on the coefficients
        coef_args = self._factor_training_kernel_matrix()
        if coef_args[0] is None:
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

    def _jacobian_neg_log_likelihood_with_hyperparameter_trend(
            self, active_opt_params: Array
    ) -> Array:
        self._hyp_list.set_active_opt_params(active_opt_params)
        # First compute jacobian with respect to kernel parameters
        # TODO this recomputes cholesky factorization
        coef_args = self._factor_training_kernel_matrix()
        Linv = self._cholesky_inverse(coef_args[0])
        Kinv = Linv.T @ Linv
        Kinv_y = self._Kinv_y(Kinv)
        Mat = (Kinv_y @ Kinv_y.T - Kinv)
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
        trend_jac = -Kinv_y.T @ self._trend.basis(
            self._ctrain_samples
        )[
            ..., self._trend.hyp_list().get_active_indices()
        ]
        jac = self._bkd.hstack((kernel_jac[None, :],  trend_jac))
        return jac

    def _neg_log_likelihood_with_uncertain_trend(self) -> float:
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

    def _neg_log_likelihood(self, active_opt_params: Array):
        self._hyp_list.set_active_opt_params(active_opt_params)
        return self._neg_log_likelihood_with_hyperparameter_trend()

    def _evaluate_canonical_prior(self, samples: Array, return_std: bool):
        canonical_trend = self._canonical_trend(
            self._in_trans.map_to_canonical(samples).T
        ).T
        if not return_std:
            return canonical_trend, None
        canonical_std = self._bkd.sqrt(self._kernel.diag(samples))
        return canonical_trend, canonical_std

    def _evaluate_canonical_posterior(self, samples: Array, return_std: bool):
        if self._coef is None:
            self._coef_args = self._factor_training_kernel_matrix()
            self._coef = self._solve_coefficients(*self._coef_args)

        canonical_samples = self._in_trans.map_to_canonical(samples)
        kmat_pred = self._kernel(
            canonical_samples, self._ctrain_samples
        )
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
        canonical_pointwise_stdev = np.sqrt(canonical_pointwise_variance.T).T
        assert canonical_pointwise_stdev.shape == canonical_trend.shape
        return canonical_trend, canonical_pointwise_stdev

    def _canonical_evaluate(self, samples: Array, return_std: bool):
        if self._ctrain_samples is None:
            return self._evaluate_canonical_prior(
                samples, return_std
            )
        return self._evaluate_canonical_posterior(samples, return_std)

    def evaluate(self, samples: Array, return_std: bool):
        """
        Use when standard deviation of GP is needed.
        Otherwise use __call__
        """
        can_vals, can_std = self._canonical_evaluate(samples, return_std)
        if return_std:
            return (
                self._out_trans.map_from_canonical(can_vals),
                self._out_trans.map_from_canonical(can_std)
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

    def plot(self, ax, bounds, **kwargs):
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


class MOExactGaussianProcess(ExactGaussianProcess):
    def _set_training_data(self, train_samples: list, train_values: list):
        # For now analytical neg log like from exact gaussian process
        # does not pass tests when used here so turn off.
        self._analytical_neg_log_likelihood_jacobian_implemented = False

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
                    self._ctrain_values[cnt:cnt+s.shape[1]].T
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
        test_samples_base = self._bkd.linspace(bounds[0], bounds[1], npts_1d)[None, :]
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
    def __init__(self, gp):
        self._gp = gp
        self._bkd = self._gp._bkd

        # todo consider storing this in gp once it is trained
        coef_args = self._gp._factor_training_kernel_matrix()
        Linv = self._gp._self._cholesky_inverse(coef_args[0])
        self._Kinv = Linv.T @ Linv

        # for now assume out_trans is the identity
        if not isinstance(gp._out_trans, IdentityTransform):
            raise ValueError("gp._out_trans must be the identity")

        # for now assume no trend
        if gp._trend is None:
            raise ValueError("gp._out_trans must be the identity")

        # store train samples in user space for ease of reference
        self._train_samples = self._gp._trans.map_from_canonical(
            self._gp._ctrain_samples
        )

    def _integrate_tau_P_1d(self, xtr_ii, lscale_ii) -> Tuple[float, float]:
        # specific to squared exponential kernel. Move to kernel
        dists_1d_x1_xtr = self._bkd.cdist(
            self._quadx_1d[:, None]/lscale_ii,
            xtr_ii.T/lscale_ii,
            metric='sqeuclidean'
        )
        K = self._bkd.exp(-.5*dists_1d_x1_xtr)
        tau = self._quadx @ (K)
        P = K.T @ (self._quadw_1d[:, None] * K)
        return tau, P

    def _tau_P(self) -> Tuple[float, float]:
        lscale = self.kernel.length_scale
        tau, P = [], []
        for ii in range(self._gp.nvars()):
            tau_ii, P_ii = self._integrate_tau_P_1d(
                self._train_samples[ii:ii+1, :], lscale[ii]
            )
            tau.append(tau_ii)
            P.append(P_ii)
        return self._bkd.asarray(tau), self._bkd.asarray(P)

    def expected_mean(self) -> Array:
        tau = self._tau_P()
        expected_random_mean = tau.dot(self._gp.Kinv_y(self._Kinv))
        # for now out_trans is the identity
        # expected_random_mean += y_train_mean
        return expected_random_mean
