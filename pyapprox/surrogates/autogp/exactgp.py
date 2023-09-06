from typing import Tuple
import numpy as np
import torch
import scipy

from pyapprox.variables.transforms import IdentityTransformation
from pyapprox.surrogates.autogp._torch_wrappers import (
    diag, full, cholesky, cholesky_solve, log, solve_triangular, einsum,
    multidot, array, asarray, sqrt)
from pyapprox.surrogates.autogp.kernels import Kernel
from pyapprox.surrogates.autogp.transforms import (
    StandardDeviationTransformation)


class ExactGaussianProcess():
    def __init__(self,
                 nvars: int,
                 kernel: Kernel,
                 kernel_reg: float = 0,
                 var_trans=None,
                 values_trans=None):
        self.kernel = kernel
        self.kernel_reg = kernel_reg
        if var_trans is None:
            self.var_trans = IdentityTransformation(nvars)
        else:
            self.var_trans = var_trans
        if self.var_trans.num_vars() != nvars:
            raise ValueError("var_trans and nvars are inconsistent")
        if values_trans is None:
            self.values_trans = StandardDeviationTransformation()
        else:
            self.values_trans = values_trans

        self._coef = None
        self._coef_args = None
        self.canonical_train_samples = None
        self.canonical_train_values = None

    def _training_kernel_matrix(self) -> Tuple:
        kmat = self.kernel(
            self.canonical_train_samples, self.canonical_train_samples)
        # Below is an inplace operation that will not work with autograd
        # kmat[np.diag_indices_from(kmat)] += self.kernel_reg
        # This also does not work
        # kmat += diag(full((kmat.shape[0], 1), float(self.kernel_reg)))
        kmat = kmat + diag(full((kmat.shape[0], 1), float(self.kernel_reg)))
        return kmat

    def _factor_training_kernel_matrix(self):
        # can be specialized
        return (cholesky(self._training_kernel_matrix()), )

    def _solve_coefficients(self, *args) -> Tuple:
        # can be specialized when _factor_training_kernel_matrix is specialized
        diff = (self.canonical_train_values -
                self._canonical_mean(self.canonical_train_samples))
        return cholesky_solve(args[0], diff)

    def _log_determinant(self, coef_res: Tuple) -> float:
        # can be specialized when _factor_training_kernel_matrix is specialized
        chol_factor = coef_res
        return log(diag(chol_factor)).sum()

    def _canonical_posterior_pointwise_variance(
            self, canonical_samples, kmat_pred):
        # can be specialized when _factor_training_kernel_matrix is specialized
        tmp = solve_triangular(self._coef_args[0], kmat_pred.T)
        update = einsum("ji,ji->i", tmp, tmp)
        return self.kernel.diag(canonical_samples) - update

    def _canonical_mean(self, canonical_samples):
        # todo allow for polynomial mean functions
        return full((canonical_samples.shape[1], 1), 0.)

    def _neg_log_likelihood(self, active_opt_params: array) -> float:
        self.kernel.hyp_list.set_active_opt_params(active_opt_params)
        coef_args = self._factor_training_kernel_matrix()
        coef = self._solve_coefficients(*coef_args)
        nsamples = self.canonical_train_values.shape[0]
        diff = (self.canonical_train_values -
                self._canonical_mean(self.canonical_train_samples))
        return 0.5 * (
            multidot((diff.T, coef)) +
            self._log_determinant(*coef_args) +
            nsamples*np.log(2*np.pi)
        ).sum(axis=1)

    def _fit_objective(self, active_opt_params_np):
        # this is only pplace where torch should be called explicitly
        # as we are using its functionality to compute the gradient of their
        # negative log likelihood. We could replace this with a grad
        # computed analytically
        active_opt_params = torch.tensor(
            active_opt_params_np, dtype=torch.double, requires_grad=True)
        nll = self._neg_log_likelihood(active_opt_params)
        nll.backward()
        val = nll.item()
        # copy is needed because zero_ is called
        nll_grad = active_opt_params.grad.detach().numpy().copy()
        active_opt_params.grad.zero_()
        # must set requires grad to False after gradient is computed
        # otherwise when evaluate_posterior will fail because it will
        # still think the hyper_params require grad. Extra copies coould be
        # avoided by doing this after fit is complete. However then fit
        # needs to know when torch is being used
        for hyp in self.kernel.hyp_list.hyper_params:
            hyp._values = hyp._values.clone().detach()
        return val, nll_grad

    def _local_optimize(self, init_active_opt_params_np, bounds):
        res = scipy.optimize.minimize(
            self._fit_objective, init_active_opt_params_np, method="L-BFGS-B",
            jac=True, bounds=bounds)
        return res

    def _global_optimize(self, max_nglobal_opt_iters=1):
        bounds = self.kernel.hyp_list.get_active_opt_bounds().numpy()
        results = []
        best_idx, best_obj = None, np.inf
        for ii in range(max_nglobal_opt_iters):
            # active bounds are in the transfomed space so just sample
            # uniformly
            init_active_opt_params_np = np.random.uniform(
                bounds[:, 0], bounds[:, 1])
            results.append(
                self._local_optimize(init_active_opt_params_np, bounds))
            if results[-1].fun < best_obj:
                best_idx = ii
                best_obj = results[-1].fun
        self.kernel.hyp_list.set_active_opt_params(
            asarray(results[best_idx].x))

    def set_training_data(self, train_samples: array, train_values: array):
        self.canonical_train_samples = asarray(
            self.var_trans.map_to_canonical(train_samples))
        self.canonical_train_values = asarray(
            self.values_trans.map_to_canonical(train_values.T).T)

    def fit(self, train_samples: array, train_values: array, **kwargs):
        self.set_training_data(train_samples, train_values)
        self._global_optimize(**kwargs)

    def _evalaute_prior(self, samples, return_std):
        mean = self.values_trans.map_from_canonical(
            self._canonical_mean(self.var_trans.map_to_canonical(samples)))
        if not return_std:
            return mean
        return mean, self.values_trans.map_stdev_from_canonical(
            sqrt(self.kernel.diag(samples)).T).T

    def _evaluate_posterior(self, samples, return_std):
        if self._coef is None:
            self._coef_args = self._factor_training_kernel_matrix()
            self._coef = self._solve_coefficients(*self._coef_args)

        canonical_samples = self.var_trans.map_to_canonical(samples)
        kmat_pred = self.kernel(
            canonical_samples, self.canonical_train_samples)
        canonical_mean = self._canonical_mean(canonical_samples) + multidot((
            kmat_pred, self._coef))
        mean = self.values_trans.map_from_canonical(canonical_mean.T).T

        if not return_std:
            return mean

        canonical_pointwise_variance = (
            self._canonical_posterior_pointwise_variance(
                canonical_samples, kmat_pred))
        pointwise_stdev = np.sqrt(self.values_trans.map_stdev_from_canonical(
            canonical_pointwise_variance))
        return mean, pointwise_stdev

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
