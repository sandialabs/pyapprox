import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from pyapprox.surrogates.gaussianprocess.kernels import (
    MultilevelKernel, SequentialMultilevelKernel,
    MultifidelityPeerKernel)
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GaussianProcess, extract_gaussian_process_attributes_for_integration)


class MultifidelityGaussianProcess(GaussianProcessRegressor):
    def __init__(self, kernel, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None, normalize_y=False):
        if not isinstance(kernel, (MultilevelKernel, MultifidelityPeerKernel)):
            raise ValueError("Multilevel Kernel must be provided")
        super().__init__(
            kernel=kernel, alpha=alpha,
            optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train,
            random_state=random_state)
        self._samples = None
        self._values = None

    def set_data(self, samples, values):
        self.nmodels = len(samples)
        assert len(values) == self.nmodels
        self._samples = samples
        self._values = values
        assert samples[0].ndim == 2
        assert self._values[0].ndim == 2
        assert self._values[0].shape[1] == 1
        self.nvars = samples[0].shape[0]

        if hasattr(self, "kernel"):
            self.kernel.set_nsamples_per_model(np.asarray([
                s.shape[1] for s in samples]))

        for ii in range(1, self.nmodels):
            assert samples[ii].ndim == 2
            assert samples[ii].shape[0] == self.nvars
            assert self._values[ii].ndim == 2
            assert self._values[ii].shape[1] == 1
            assert self._values[ii].shape[0] == samples[ii].shape[1]

    def fit(self):
        XX_train = np.hstack(self._samples).T
        YY_train = np.vstack(self._values)
        super().fit(XX_train, YY_train)

    def __call__(self, XX_test, return_std=False, return_cov=False,
                 model_eval_id=None):
        if model_eval_id is None:
            model_eval_id = self.nmodels-1
        self.kernel_.model_eval_id = model_eval_id
        result = self.predict(XX_test.T, return_std, return_cov)
        if type(result) != tuple:
            return result
        # when returning prior stdev covariance then must reshape vals
        if result[0].ndim == 1:
            result = [result[0][:, None]] + [r for r in result[1:]]
            result = tuple(result)
        return result

    def plot_1d(self, num_XX_test, bounds,
                ax=None, num_stdev=2, plt_kwargs={},
                fill_kwargs={"alpha": 0.3},
                prior_fill_kwargs=None, model_eval_id=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[None, :]
        # return_std=True does not work for gradient enhanced krigging
        # gp_mean, gp_std = predict(XX_test,return_std=True)
        gp_mean, gp_std = self(
            XX_test, return_std=True, model_eval_id=model_eval_id)
        gp_mean = gp_mean[:, 0]
        if prior_fill_kwargs is not None:
            if model_eval_id is not None:
                self.kernel_.model_eval_id = model_eval_id
            prior_std = np.sqrt(self.kernel_.diag(XX_test.T))
            ax.fill_between(
                XX_test[0, :], self._y_train_mean-num_stdev*prior_std,
                self._y_train_mean+num_stdev*prior_std, **prior_fill_kwargs)
        ax.fill_between(
           XX_test[0, :], gp_mean-num_stdev*gp_std, gp_mean+num_stdev*gp_std,
           **fill_kwargs)
        ax.plot(XX_test[0, :], gp_mean, **plt_kwargs)
        return ax

    # def integrate(self, variable, nquad_samples):
    #     (X_train, y_train, K_inv, kernel_length_scale, kernel_var,
    #      transform_quad_rules) = (
    #          extract_gaussian_process_attributes_for_integration(self))
    #     print(transform_quad_rules)
    #     tau, P = self.kernel_._integrate_tau_P(
    #         variable, nquad_samples, X_train, transform_quad_rules)
    #     A_inv = K_inv*kernel_var
    #     # No kernel_var because it cancels out because it appears in K (1/s^2)
    #     # and t (s^2)
    #     A_inv_y = A_inv.dot(y_train)
    #     expected_random_mean = tau.dot(A_inv_y)
    #     expected_random_mean += self._y_train_mean
    #     return expected_random_mean, P


class SequentialMultifidelityGaussianProcess(MultifidelityGaussianProcess):
    def __init__(self, kernels, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None,
                 default_rho=1.0, rho_bounds=(1e-1, 1)):
        self._raw_kernels = kernels
        self._alpha = alpha
        self._n_restarts_optimizer = n_restarts_optimizer
        self._copy_X_train = copy_X_train
        self._random_state = random_state

        self._gps = None
        self.rho = None
        self.default_rho = default_rho

    def _get_kernel(self, kk):
        for ii in range(kk):
            self._raw_kernels[ii].length_scale_bounds = "fixed"
        if kk > 0:
            if kk > 1:
                fixed_rho = self.rho[:kk-1]
            else:
                fixed_rho = []
            length_scale = np.atleast_1d(self._raw_kernels[kk].length_scale)
            kernel = SequentialMultilevelKernel(
                self._raw_kernels[:kk+1], fixed_rho, length_scale,
                self._raw_kernels[kk].length_scale_bounds, self.default_rho)
        else:
            kernel = self._raw_kernels[kk]
        return kernel

    def fit(self):
        nmodels = len(self._samples)
        self.rho = [1 for ii in range(nmodels-1)]
        self._gps = []
        lf_values = None
        for ii in range(nmodels):
            if ii > 1:
                lf_values += self.rho[ii-1]*lf_values + self._gps[ii-1](
                    self._samples[ii])
            elif ii == 1:
                lf_values = self._gps[ii-1](self._samples[ii])
            gp = SequentialGaussianProcess(
                lf_values,
                kernel=self._get_kernel(ii),
                n_restarts_optimizer=self._n_restarts_optimizer,
                alpha=self._alpha, random_state=self._random_state,
                copy_X_train=self._copy_X_train)
            gp.fit(self._samples[ii], self._values[ii])
            if ii > 0:
                self.rho[ii-1] = gp.kernel_.rho
            self._gps.append(gp)

    def __call__(self, samples, return_std=False, model_eval_id=None):
        nmodels = len(self._gps)
        means, variances, stds = [], [], []
        if model_eval_id is None:
            model_eval_id = [nmodels-1]
        ml_mean, ml_var = 0, 0
        for ii in range(np.max(model_eval_id)+1):
            discrepancy, std = self._gps[ii](samples, return_std=True)
            if ii > 0:
                ml_mean = self.rho[ii-1]*ml_mean + discrepancy
                ml_var = self.rho[ii-1]**2*ml_var + std**2
            else:
                ml_mean = discrepancy
                ml_var = std**2
            stds.append(std)
            means.append(ml_mean)
            variances.append(ml_var)
        if len(model_eval_id) == 1:
            if return_std:
                # return (means[model_eval_id[0]],
                #         np.sqrt(variances[model_eval_id[0]]).squeeze())
                return (means[model_eval_id[0]], std[model_eval_id[0]])
            return means[model_eval_id[0]]
        if return_std:
            #return ([means[idx] for idx in model_eval_id],
            #        [np.sqrt(variances[idx]).squeeze() for idx in model_eval_id]
            return ([means[idx] for idx in model_eval_id],
                    [stds[idx].squeeze() for idx in model_eval_id])
        return [means[idx] for idx in model_eval_id]

    def __repr__(self):
        return "{0}(rho={1})".format(
            self.__class__.__name__, self.rho)


class SequentialGaussianProcess(GaussianProcess):
    def __init__(self, lf_values, **kwargs):
        super().__init__(**kwargs)
        self.lf_values = lf_values

    def _shift_data(self, y_train, theta):
        if self.lf_values is not None:
            rho = np.exp(theta[-1])
            shift = rho*self.lf_values
        else:
            shift = 0
        return y_train-shift

    def _log_marginal_likelihood(self, theta):
        # make sure to keep copy of y_train
        y_train = self.y_train_.copy()
        # adjust y_train by rho*lf_valus
        self.y_train_ = self._shift_data(self.y_train_, theta)
        val = super().log_marginal_likelihood(theta, clone_kernel=False)
        # reset y_train
        self.y_train_ = y_train
        return val

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        val = self._log_marginal_likelihood(theta)
        if not eval_gradient:
            return val
        from scipy.optimize import approx_fprime
        grad = approx_fprime(
            theta, self._log_marginal_likelihood, 1e-10)
        return val, grad

    def __str__(self):
        msg = f"SequentialGaussianProcess(kernel={self.kernel_})"
        return msg

    def __repr__(self):
        return self.__str__()

    def fit(self, X, y):
        # super().fit does not know how to include shift of y_train when
        # computing self.alpha_ used for predicting with the GP so adjust
        # here
        super().fit(X, y)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        from scipy.linalg import cho_solve
        self.alpha_ = cho_solve(
            (self.L_, True),
            self._shift_data(self.y_train_.copy(), self.kernel_.theta),
            check_finite=False)
        return self


from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GreedyIntegratedVarianceSampler)
class GreedyMultifidelityIntegratedVarianceSampler(
        GreedyIntegratedVarianceSampler):

    def __init__(self, nmodels, num_vars, nquad_samples,
                 ncandidate_samples_per_model, generate_random_samples,
                 variable=None, econ=True,
                 compute_cond_nums=False, nugget=0, model_costs=None,
                 candidate_samples=None, quadrature_rule=None):
        # if econ:
        #     raise NotImplementedError()
        self.nmodels = nmodels
        self.model_costs = model_costs
        self.ncandidate_samples_per_model = ncandidate_samples_per_model
        self.nsamples_per_model = np.zeros(self.nmodels, dtype=int)
        self.training_sample_costs = []
        self.ivar_delta = 0.0

        # todo currently 1D quadrature with seperable kernels is not suported
        use_gauss_quadrature = False
        super().__init__(num_vars, nquad_samples,
                         ncandidate_samples_per_model, generate_random_samples,
                         variable, use_gauss_quadrature, econ,
                         compute_cond_nums, nugget, candidate_samples,
                         quadrature_rule)
        # super assumes one model when checking size of candidate samples
        # so pass in for one model then override
        if candidate_samples is not None:
            self.candidate_samples = np.hstack(
                [self.candidate_samples]*self.nmodels)

    def _generate_candidate_samples(
            self, ncandidate_samples_per_model):
        # for now assume ncandidate_samples_per_model is integer used for
        # all models
        single_model_candidate_samples = super()._generate_candidate_samples(
            ncandidate_samples_per_model)
        candidate_samples = np.hstack(
            [single_model_candidate_samples for kk in range(self.nmodels)])
        return candidate_samples

    def set_kernel(self,
                   kernel, integration_method=None, variable=None, **kwargs):
        if kernel.nmodels != self.nmodels:
            raise ValueError("kernel is not consistent with self.nmodels")
        self.kernel = kernel
        self.kernel.nsamples_per_model = np.full(
            (self.kernel.nmodels,), self.ncandidate_samples_per_model)

        from pyapprox.surrogates.integrate import integrate
        if integration_method is not None:
            assert variable is not None
            self.pred_samples, ww = integrate(
                integration_method, variable, **kwargs)
            ww = ww[:, 0]
        else:
            xx, ww = self._quadrature_rule()
            self.pred_samples = xx

        K = self.kernel(self.pred_samples.T, self.candidate_samples.T)
        self.P = K.T.dot(ww[:, np.newaxis]*K)
        self.compute_A()
        self.add_nugget()
        self.kernel.nsamples_per_model = self.nsamples_per_model

    def _model_id(self, new_sample_index):
        return new_sample_index//self.ncandidate_samples_per_model

    def _ivar_delta(self, new_sample_index, pivots):
        indices = np.concatenate(
            [pivots, [new_sample_index]]).astype(int)
        model_id = self._model_id(new_sample_index)
        self.kernel.nsamples_per_model[model_id] += 1
        A = self.A[np.ix_(indices, indices)]
        A_inv = np.linalg.inv(A)
        P = self.P[np.ix_(indices, indices)]
        self.kernel.nsamples_per_model[model_id] -= 1
        ivar_delta = np.trace(A_inv.dot(P))
        return ivar_delta

    def objective(self, new_sample_index, return_ivar_delta=False):
        model_id = self._model_id(new_sample_index)
        ivar_delta = self._ivar_delta(new_sample_index, self.pivots)
        obj_val = (self.ivar_delta-ivar_delta)/self.model_costs[model_id]
        if not return_ivar_delta:
            return obj_val
        else:
            return obj_val, ivar_delta

    def objective_econ(self, new_sample_index, return_ivar_delta=False):
        model_id = self._model_id(new_sample_index)
        self.kernel.nsamples_per_model[model_id] += 1
        if len(self.best_obj_vals) > 0:
            best_obj_val = self.best_obj_vals[-1]
            # base class assumes objective is -ivar_delta
            # but here it is something different so update
            # so ivar is updated correctly using super().objective_econ
            self.best_obj_vals[-1] = -self.ivar_delta
        ivar_delta = -super().objective_econ(new_sample_index)
        if len(self.best_obj_vals) > 0:
            self.best_obj_vals[-1] = best_obj_val
        self.kernel.nsamples_per_model[model_id] -= 1
        obj_val = (self.ivar_delta-ivar_delta)/self.model_costs[model_id]
        if not return_ivar_delta:
            return obj_val
        else:
            return obj_val, ivar_delta

    def vectorized_objective_vals_econ(self):
        # TODO vectorizing requires isolating all candidates for
        # a specific model
        obj_vals = np.inf*np.ones(self.candidate_samples.shape[1])
        for mm in range(self.candidate_samples.shape[1]):
            if mm not in self.pivots:
                obj_vals[mm] = self.objective_econ(mm)
        return obj_vals

    def update_training_samples(self, pivot):
        # todo avoid recomputing
        self.ivar_delta = self.objective(pivot, True)[1]
        self.pivots.append(pivot)
        # multilevel kernel assumes that points for each model
        # are concatenated after one another.
        model_id = self._model_id(pivot)
        index = int(self.nsamples_per_model[:model_id+1].sum())
        self.training_samples = np.insert(
            self.training_samples, index,
            self.candidate_samples[:, pivot], axis=1)
        self.nsamples_per_model[model_id] += 1
        self.training_sample_costs.append(self.model_costs[model_id])

    def samples_per_model(self, pivots):
        samples_per_model = [[] for ii in range(self.nmodels)]
        for ii in range(len(pivots)):
            model_id = self._model_id(self.pivots[ii])
            sample = self.candidate_samples[:, pivots[ii]][:, None]
            samples_per_model[model_id].append(sample)
        for ii in range(self.nmodels):
            if len(samples_per_model[ii]) > 0:
                samples_per_model[ii] = np.hstack(samples_per_model[ii])
            else:
                samples_per_model[ii] = np.empty(
                    (self.candidate_samples.shape[0], 0))
        return samples_per_model
