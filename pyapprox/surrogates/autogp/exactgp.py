from abc import ABC, abstractmethod
from typing import Tuple
import warnings

import numpy as np
import torch
import scipy

from pyapprox.interface.model import Model
from pyapprox.surrogates.autogp.mokernels import MultiPeerKernel
from pyapprox.surrogates.bases.basisexp import BasisExpansion
from pyapprox.surrogates.kernels.kernels import Kernel
from pyapprox.util.transforms import Transform, IdentityTransform


class ExactGaussianProcess(Model):
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 trend: BasisExpansion = None,
                 kernel_reg : float = 0):
        super().__init__()
        self._bkd = kernel._bkd
        self.kernel = kernel
        self.trend = trend
        self.kernel_reg = kernel_reg
        if var_trans is None:
            var_trans = IdentityTransform(backend=kernel._bkd)
        self.var_trans = var_trans
        if values_trans is None:
            values_trans = IdentityTransform(backend=kernel._bkd)
        self.values_trans = values_trans

        self._coef = None
        self._coef_args = None
        self.train_samples = None
        self.train_values = None
        self.canonical_train_samples = None
        self.canonical_train_values = None

        self.hyp_list = self.kernel.hyp_list
        if trend is not None:
            self.hyp_list += self.trend.hyp_list

    def _training_kernel_matrix(self) -> Tuple:
        # must only pass in X and not Y to kernel otherwise if noise kernel
        # is present it will not be evaluted correctly.
        kmat = self.kernel(self.canonical_train_samples)
        # Below is an inplace operation that will not work with autograd
        # kmat[np.diag_indices_from(kmat)] += self.kernel_reg
        # This also does not work
        # kmat += diag(full((kmat.shape[0], 1), float(self.kernel_reg)))
        kmat = kmat + self._bkd._la_eye(kmat.shape[0])*float(self.kernel_reg)
        return kmat

    def _factor_training_kernel_matrix(self):
        # can be specialized
        kmat = self._training_kernel_matrix()
        try:
            return (self._bkd._la_cholesky(kmat), )
        except:
            return None, kmat

    def _solve_coefficients(self, *args) -> Tuple:
        # can be specialized when _factor_training_kernel_matrix is specialized
        diff = (self.canonical_train_values -
                self._canonical_trend(self.canonical_train_samples))
        return self._bkd._la_cholesky_solve(args[0], diff)

    def _Linv_y(self, *args):
        diff = (self.canonical_train_values -
                self._canonical_trend(self.canonical_train_samples))
        return self._bkd._la_solve_triangular(args[0], diff)

    def _log_determinant(self, coef_res: Tuple) -> float:
        # can be specialized when _factor_training_kernel_matrix is specialized
        chol_factor = coef_res[0]
        return 2*self._bkd._la_log(self._bkd._la_get_diagonal(chol_factor)).sum()

    def _canonical_posterior_pointwise_variance(
            self, canonical_samples, kmat_pred):
        # can be specialized when _factor_training_kernel_matrix is specialized
        tmp = self._bkd._la_solve_triangular(self._coef_args[0], kmat_pred.T)
        update = self._bkd._la_einsum("ji,ji->i", tmp, tmp)
        return (self.kernel.diag(canonical_samples) - update)[:, None]

    def _canonical_trend(self, canonical_samples):
        if self.trend is None:
            return self._bkd._la_full((canonical_samples.shape[1], 1), 0.)
        return self.trend(canonical_samples)

    def _neg_log_likelihood_with_hyperparameter_trend(self) -> float:
        # this can also be used if treating the trend as hyper_params
        # but cannot be used if assuming a prior on the coefficients
        coef_args = self._factor_training_kernel_matrix()
        if coef_args[0] is None:
            return coef_args[1][0, 0]*0+np.inf
        Linv_y = self._Linv_y(*coef_args)
        nsamples = self.canonical_train_values.shape[0]
        return 0.5 * (
            self._bkd._la_multidot((Linv_y.T, Linv_y)) +
            self._log_determinant(coef_args) +
            nsamples*np.log(2*np.pi)
        ).sum(axis=1)

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

    def _neg_log_likelihood(self, active_opt_params):
        self.hyp_list.set_active_opt_params(active_opt_params)
        return self._neg_log_likelihood_with_hyperparameter_trend()
        # return self._neg_log_likelihood_with_uncertain_trend()

    def _fit_objective(self, active_opt_params_np):
        val, grad = self._bkd._la_grad(
            self._neg_log_likelihood, self._bkd._la_array(active_opt_params_np)
        )
        for hyp in self.hyp_list.hyper_params:
            self._bkd._la_detach(hyp)
        return (
            self._bkd._la_to_numpy(self._bkd._la_detach(val)),
            self._bkd._la_to_numpy(self._bkd._la_detach(grad))
        )

    def _local_optimize(self, init_active_opt_params_np, bounds):
        method = "L-BFGS-B"
        res = scipy.optimize.minimize(
            self._fit_objective, init_active_opt_params_np, method=method,
            jac=True, bounds=bounds, options={"iprint": -1})
        return res

    def _get_random_optimizer_initial_guess(self, bounds):
        return np.random.uniform(bounds[:, 0], bounds[:, 1])

    def _global_optimize(self, max_nglobal_opt_iters=1):
        bounds = self._bkd._la_to_numpy(self.hyp_list.get_active_opt_bounds())
        if len(bounds) == 0:
            return
        results = []
        # Start first optimizer with values set in hyperparams
        init_active_opt_params_np = self.hyp_list.get_active_opt_params()
        results = [self._local_optimize(init_active_opt_params_np, bounds)]
        best_idx, best_obj = 0, results[-1].fun
        for ii in range(1, max_nglobal_opt_iters):
            # active bounds are in the transfomed space so just sample
            # uniformly
            init_active_opt_params_np = (
                self._get_random_optimizer_initial_guess(bounds))
            results.append(
                self._local_optimize(init_active_opt_params_np, bounds))
            if results[-1].fun < best_obj:
                best_idx = ii
                best_obj = results[-1].fun
        self.hyp_list.set_active_opt_params(
            self._bkd._la_atleast1d(results[best_idx].x))

    def set_training_data(self, train_samples, train_values):
        self.train_samples = train_samples
        self.train_values = train_values
        self.canonical_train_samples = (
            self._map_samples_to_canonical(train_samples))
        self.canonical_train_values = (
            self.values_trans.map_to_canonical(train_values))

    def fit(self, train_samples, train_values, **kwargs):
        self.set_training_data(train_samples, train_values)
        self._global_optimize(**kwargs)

    def _evaluate_prior(self, samples, return_std):
        trend = self.values_trans.map_from_canonical(
            self._canonical_trend(self.var_trans.map_to_canonical(samples)))
        if not return_std:
            return trend
        return trend, self.values_trans.map_stdev_from_canonical(
            self._bkd._la_sqrt(self.kernel.diag(samples)))

    def _map_samples_to_canonical(self, samples):
        return self.var_trans.map_to_canonical(samples)

    def _evaluate_posterior(self, samples, return_std):
        # import warnings
        # warnings.filterwarnings("error")
        if self._coef is None:
            self._coef_args = self._factor_training_kernel_matrix()
            self._coef = self._solve_coefficients(*self._coef_args)

        canonical_samples = self._map_samples_to_canonical(samples)
        kmat_pred = self.kernel(
            canonical_samples, self.canonical_train_samples)
        canonical_trend = (self._canonical_trend(canonical_samples) +
                           self._bkd._la_multidot((kmat_pred, self._coef)))
        trend = self.values_trans.map_from_canonical(canonical_trend)
        if not return_std:
            return trend

        canonical_pointwise_variance = (
            self._canonical_posterior_pointwise_variance(
                canonical_samples, kmat_pred))
        if canonical_pointwise_variance.min() < 0:
            msg = "Some pointwise variances were negative. The largest "
            msg += "magnitude of the negative values was {0}".format(
                canonical_pointwise_variance.min())
            warnings.warn(msg, UserWarning)
        canonical_pointwise_variance[canonical_pointwise_variance < 0] = 0
        pointwise_stdev = self.values_trans.map_stdev_from_canonical(
            np.sqrt(canonical_pointwise_variance))
        assert pointwise_stdev.shape == trend.shape
        return trend, pointwise_stdev
        # return trend, canonical_pointwise_variance[:, None]

    def __call__(self, samples, return_std=False, return_grad=False):
        if return_grad and return_std:
            msg = "if return_grad is True then return_std must be False"
            raise ValueError(msg)

        if self.canonical_train_samples is None:
            if return_grad:
                msg = "return_grad must be False when evaluating GP prior"
                raise ValueError(msg)
            return self._evaluate_prior(samples, return_std)

        return self._evaluate_posterior(samples, return_std)

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self.hyp_list._short_repr())

    def _plot_1d(self, ax, test_samples, gp_trend, gp_std, nstdevs,
                 fill_kwargs, plt_kwargs, plot_samples):
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
            plot_samples, gp_trend-nstdevs*gp_std,
            gp_trend+nstdevs*gp_std, **fill_kwargs)
        if added_fill_color:
            del fill_kwargs["color"]
        return [im0, im1]

    def plot_1d(self, ax, bounds, npts_1d=101, nstdevs=2, plt_kwargs={},
                fill_kwargs={'alpha': 0.3}, prior_kwargs=None,
                plot_samples=None):
        test_samples = np.linspace(
            bounds[0], bounds[1], npts_1d)[None, :]
        gp_trend, gp_std = self(test_samples, return_std=True)
        ims = self._plot_1d(
            ax, test_samples, gp_trend[:, 0], gp_std[:, 0], nstdevs,
            fill_kwargs, plt_kwargs, plot_samples)
        if prior_kwargs is None:
            return ims
        ims += self._plot_1d(
            ax, test_samples, gp_trend, gp_std, nstdevs, **prior_kwargs)
        return ims

    def plot(self, ax, bounds, **kwargs):
        if len(bounds) % 2 != 0:
            raise ValueError(
                "Lower and upper bounds must be provied for each dimension")
        nvars = len(bounds)//2
        if nvars > 1:
            raise ValueError("plot was called but gp is not 1D")
            return
        if self.canonical_train_samples.shape[0] != nvars:
            raise ValueError("nvars is inconsistent with training data")
        return self.plot_1d(ax, bounds, **kwargs)


class MOExactGaussianProcess(ExactGaussianProcess):
    def set_training_data(self, train_samples: list, train_values: list):
        self.train_samples = train_samples
        self.train_values = train_values
        self.canonical_train_samples = [
            s for s in self._map_samples_to_canonical(train_samples)]
        self.canonical_train_values = self._bkd._la_vstack(
            [self.values_trans.map_to_canonical(v) for v in train_values])

    def _map_samples_to_canonical(self, samples):
        return [self.var_trans.map_to_canonical(s) for s in samples]

    def _canonical_trend(self, canonical_samples):
        if self.trend is not None:
            raise ValueError("Non-zero trend not supported for mulitoutput")
        return self._bkd._la_full(
            (sum([s.shape[1] for s in canonical_samples]), 1), 0.)

    def plot_1d(self, ax, bounds, output_id, npts_1d=101, nstdevs=2,
                plt_kwargs={}, fill_kwargs={'alpha': 0.3}, prior_kwargs=None,
                plot_samples=None):
        test_samples_base = np.linspace(
            bounds[0], bounds[1], npts_1d)[None, :]
        noutputs = len(self.canonical_train_samples)
        test_samples = [np.array([[]]) for ii in range(noutputs)]
        test_samples[output_id] = test_samples_base
        gp_trend, gp_std = self(test_samples, return_std=True)
        ims = self._plot_1d(
            ax, test_samples[output_id], gp_trend[:, 0], gp_std[:, 0],
            nstdevs, fill_kwargs, plt_kwargs, plot_samples)
        if prior_kwargs is None:
            return ims
        ims += self._plot_1d(
            ax, test_samples, gp_trend, gp_std, nstdevs, **prior_kwargs)
        return ims

    def plot(self, ax, bounds, output_id=-1, **kwargs):
        if len(bounds) % 2 != 0:
            raise ValueError(
                "Lower and upper bounds must be provied for each dimension")
        nvars = len(bounds)//2
        if nvars > 1:
            raise ValueError("plot was called but gp is not 1D")
            return
        if self.canonical_train_samples[0].shape[0] != nvars:
            raise ValueError("nvars is inconsistent with training data")
        return self.plot_1d(ax, bounds, output_id, **kwargs)


class MOPeerExactGaussianProcess(MOExactGaussianProcess):
    def _solve_coefficients(self, *args) -> Tuple:
        # can be specialized when _factor_training_kernel_matrix is specialized
        diff = (self.canonical_train_values -
                self._canonical_trend(self.canonical_train_samples))
        return MultiPeerKernel._cholesky_solve(*args, diff, self._bkd)

    def _log_determinant(self, coef_res: Tuple) -> float:
        # can be specialized when _factor_training_kernel_matrix is specialized
        return MultiPeerKernel._logdet(*coef_res, self._bkd)

    def _training_kernel_matrix(self) -> Tuple:
        # must only pass in X and not Y to kernel otherwise if noise kernel
        # is present it will not be evaluted correctly.
        blocks = self.kernel(self.canonical_train_samples, block_format=True)
        for ii in range(len(blocks)):
            blocks[ii][ii] = (
                blocks[ii][ii] +
                self._bkd._la_eye(blocks[ii][ii].shape[0])*float(self.kernel_reg))
        return blocks

    def _factor_training_kernel_matrix(self):
        blocks = self._training_kernel_matrix()
        return MultiPeerKernel._cholesky(
                len(blocks[0]), blocks, self._bkd, block_format=True)
        # try:
        #     return MultiPeerKernel._cholesky(
        #         len(blocks[0]), blocks, self._bkd, block_format=True)
        # except:
        #     return None, blocks[0][0][0]

    def _Linv_y(self, *args):
        diff = (self.canonical_train_values -
                self._canonical_trend(self.canonical_train_samples))
        return MultiPeerKernel._lower_solve_triangular(*args, diff, self._bkd)

    def _canonical_posterior_pointwise_variance(
            self, canonical_samples, kmat_pred):
        # can be specialized when _factor_training_kernel_matrix is specialized
        tmp = MultiPeerKernel._lower_solve_triangular(
            *self._coef_args, kmat_pred.T, self._bkd)
        update = self._bkd._la_einsum("ji,ji->i", tmp, tmp)
        return (self.kernel.diag(canonical_samples) - update)[:, None]


class MOICMPeerExactGaussianProcess(MOExactGaussianProcess):
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 output_kernel: Kernel,
                 var_trans: Transform = None,
                 values_trans: Transform = None,
                 kernel_reg: float = 0):
        super().__init__(
            nvars, kernel, var_trans, values_trans, None, kernel_reg)
        self.output_kernel = output_kernel

    @staticmethod
    def _constraint_fun(active_opt_params_np, *args):
        ii, jj, gp, okernel = args
        active_opt_params = torch.tensor(
            active_opt_params_np, dtype=torch.double, requires_grad=False)
        gp.hyp_list.set_active_opt_params(active_opt_params)
        val = okernel(ii, jj).item()
        # val = log(okernel(ii, jj)).item()-np.log(1e-16)
        return val

    @staticmethod
    def _constraint_jac(active_opt_params_np, *args):
        ii, jj, gp, okernel = args
        active_opt_params = torch.tensor(
            active_opt_params_np, dtype=torch.double, requires_grad=True)
        gp.hyp_list.set_active_opt_params(active_opt_params)
        val = okernel(ii, jj)
        # val = log(okernel(ii, jj))-np.log(1e-16)
        val.backward()
        grad = active_opt_params.grad.detach().numpy().copy()
        active_opt_params.grad.zero_()
        for hyp in gp.hyp_list.hyper_params:
            hyp.detach()
        return grad

    def _get_constraints(self, noutputs):
        icm_cons = []
        for ii in range(2, noutputs):
            for jj in range(1, ii):
                con = {'type': 'eq',
                       'fun': self._constraint_fun,
                       'jac': self._constraint_jac,
                       'args': (ii, jj, self, self.output_kernel)}
                icm_cons.append(con)
        return icm_cons

    def _local_optimize(self, init_active_opt_params_np, bounds):
        # TODO use new optimization classes
        method = "trust-constr"
        # method = "slsqp"
        if method == "trust-constr":
            optim_options = {'disp': True, 'gtol': 1e-8,
                             'maxiter': 1000, "verbose": 0}
        if method == "slsqp":
            optim_options = {'disp': True, 'ftol': 1e-10,
                             'maxiter': 1000, "iprint": 0}
        res = scipy.optimize.minimize(
            self._fit_objective, init_active_opt_params_np, method=method,
            jac=True, bounds=bounds,
            constraints=self._get_constraints(len(self.train_values)),
            options=optim_options)
        return res
