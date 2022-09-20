import pymc3 as pm
import numpy as np
import theano.tensor as tt
from scipy.optimize import approx_fprime

from pyapprox.variables.marginals import get_distribution_info
from pyapprox.variables.joint import JointVariable


class GaussianLogLike(object):
    r"""
    A Gaussian log-likelihood function for a model with parameters given in
    sample
    """

    def __init__(self, model, data, noise_covar):
        r"""
        Initialise the Op with various things that our log-likelihood
        function requires.

        Parameters
        ----------
        model : callable
            The model relating the data and noise

        data : np.ndarray (nobs)
            The "observed" data

        noise_covar : float, np.ndarray (nobs), np.ndarray (nobs,nobs)
            The noise covariance
        """
        self.model = model
        self.data = data
        assert self.data.ndim == 1
        self.ndata = data.shape[0]
        self.noise_covar_inv, self.log_noise_covar_det = (
            self.noise_covariance_inverse(noise_covar))

    def noise_covariance_inverse(self, noise_covar):
        if np.isscalar(noise_covar):
            return 1/noise_covar, np.log(noise_covar)
        if noise_covar.ndim == 1:
            assert noise_covar.shape[0] == self.data.shape[0]
            return 1/noise_covar, np.log(np.prod(noise_covar))
        elif noise_covar.ndim == 2:
            assert noise_covar.shape == (self.ndata, self.ndata)
            return np.linalg.inv(noise_covar), np.log(
                np.linalg.det(noise_covar))
        raise ValueError("noise_covar has the wrong shape")

    # def noise_covariance_determinant(self, noise_covar):
    #     r"""The determinant is only necessary in log likelihood if the noise
    #     covariance has a hyper-parameter which is being inferred which is
    #     not currently supported"""
    #     if np.isscalar(noise_covar):
    #         determinant = noise_covar**self.ndata
    #     elif noise_covar.ndim==1:
    #         determinant = np.prod(noise_covar)
    #     else:
    #         determinant = np.linalg.det(noise_covar)
    #     return determinant

    def __call__(self, samples):
        model_vals = self.model(samples)
        assert model_vals.ndim == 2
        assert model_vals.shape[1] == self.ndata
        vals = np.empty((model_vals.shape[0], 1))
        for ii in range(model_vals.shape[0]):
            residual = self.data - model_vals[ii, :]
            if (np.isscalar(self.noise_covar_inv) or
                    self.noise_covar_inv.ndim == 1):
                vals[ii] = (residual.T*self.noise_covar_inv).dot(residual)
            else:
                vals[ii] = residual.T.dot(self.noise_covar_inv).dot(residual)
        vals += self.ndata*np.log(2*np.pi) + self.log_noise_covar_det
        vals *= -0.5
        return vals


class LogLike(tt.Op):
    r"""
    Specify what type of object will be passed and returned to the Op
    when it is called. In our case we will be passing it a vector of
    values (the parameters that define our model) and returning a
    single "scalar" value (the log-likelihood)
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        samples, = inputs  # important
        # call the log-likelihood function
        if samples.ndim == 1:
            samples = samples[:, None]
        logl = self.likelihood(samples).squeeze()
        outputs[0][0] = np.array(logl)  # output the log-likelihood


class LogLikeWithGrad(LogLike):

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, loglike_grad=None):
        r"""
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined

        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        super().__init__(loglike)

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, loglike_grad)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        samples, = inputs  # important
        return [g[0]*self.logpgrad(samples)]


class LogLikeGrad(tt.Op):

    r"""
    This Op will be called with a vector of values and also return a
    vector of values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, loglike_grad=None):
        r"""
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        self.likelihood = loglike
        self.likelihood_grad = loglike_grad

    def perform(self, node, inputs, outputs):
        samples, = inputs
        # calculate gradients
        if self.likelihood_grad is None:
            # define version of likelihood function to pass to
            # derivative function
            def lnlike(ss):
                if ss.ndim == 1:
                    ss = ss[:, None]
                return self.likelihood(ss).squeeze()
            grads = approx_fprime(
                samples, lnlike, 2*np.sqrt(np.finfo(float).eps))
        else:
            if samples.ndim == 1:
                samples = samples[:, None]
            if self.likelihood_grad == True:
                grads = self.likelihood(samples, jac=True)[1]
            else:
                grads = self.likelihood_grad(samples)
            if grads.ndim == 2:
                grads = grads[:, 0]
            assert grads.ndim == 1
        outputs[0][0] = grads


def extract_mcmc_chain_from_pymc3_trace(
        trace, var_names, nsamples, nburn, njobs):
    nvars = len(var_names)
    samples = np.empty((nvars, (nsamples-nburn)*njobs))
    effective_sample_size = -np.ones(nvars)
    for ii in range(nvars):
        samples[ii, :] = trace.get_values(
            var_names[ii], burn=nburn, chains=np.arange(njobs))
        try:
            effective_sample_size[ii] = pm.ess(trace)[var_names[ii]].values
        except:
            print('could not compute ess. likely issue with theano')

    return samples, effective_sample_size


def extract_map_sample_from_pymc3_dict(map_sample_dict, var_names):
    nvars = len(var_names)
    map_sample = np.empty((nvars, 1))
    for ii in range(nvars):
        map_sample[ii] = map_sample_dict[var_names[ii]]
    return map_sample


def get_pymc_variables(variables, pymc_var_names=None):
    nvars = len(variables)
    if pymc_var_names is None:
        pymc_var_names = ['z_%d' % ii for ii in range(nvars)]
    assert len(pymc_var_names) == nvars
    pymc_vars = []
    for ii in range(nvars):
        pymc_vars.append(
            get_pymc_variable(variables[ii], pymc_var_names[ii]))
    return pymc_vars, pymc_var_names


def get_pymc_variable(rv, pymc_var_name):
    name, scales, shapes = get_distribution_info(rv)
    if rv.dist.name == 'norm':
        return pm.Normal(
            pymc_var_name, mu=scales['loc'], sigma=scales['scale'])
    if rv.dist.name == 'uniform':
        return pm.Uniform(pymc_var_name, lower=scales['loc'],
                          upper=scales['loc']+scales['scale'])
    msg = f'Variable type: {name} not supported'
    raise Exception(msg)


class MCMCVariable(JointVariable):
    def __init__(self, variable, loglike, algorithm, loglike_grad=None,
                 burn_fraction=0.1, njobs=1):
        self._variable = variable
        self._loglike = loglike
        self._loglike_grad = loglike_grad
        self._njobs = njobs
        self._algorithm = algorithm
        self._burn_fraction = burn_fraction
        self._njobs = njobs

    def _sample(self, nsamples):
        nburn = int(nsamples*self._burn_fraction)
        return run_bayesian_inference_gaussian_error_model(
            self._loglike, self._variable, nsamples, nburn, self._njobs,
            algorithm=self._algorithm, get_map=False, print_summary=False,
            loglike_grad=self._loglike_grad, seed=None)[0]

    def maximum_aposteriori_point(self):
        """
        Find the point of maximum aposteriori probability (MAP)

        Returns
        -------
        map_sample : np.ndarray (nvars, 1)
            the MAP point
        """
        return run_bayesian_inference_gaussian_error_model(
            self._loglike, self._variable, 0, 0, self._njobs,
            algorithm=self._algorithm, get_map=True, print_summary=False,
            loglike_grad=self._loglike_grad, seed=None)[2]

    def rvs(self, num_samples):
        """
        Generate samples from a random variable.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        Returns
        -------
        samples : np.ndarray (num_vars, num_samples)
            Independent samples from the target distribution
        """
        return self._sample(num_samples)

    def __str__(self):
        string = "MCMCVariable with prior:\n"
        string += self._variable.__str__()
        return string

    def unnormalized_pdf(self, samples):
        pdf_vals = self._variable.pdf(samples).squeeze()
        nll_vals = self._loglike(samples).squeeze()
        # vals = np.exp(nll_vals)*pdf_vals
        vals = np.exp(nll_vals+np.log(pdf_vals))
        # make posterior 1 at map point
        # vals /= (self._variable.pdf(map_sample)*np.exp(
        #     -self._negloglike(map_sample)))
        return vals[:, None]

    def _unnormalized_pdf_for_marginalization(self, sub_indices, samples):
        marginal_pdf_vals = self._variable.evaluate("pdf", samples)
        sub_pdf_vals = marginal_pdf_vals[sub_indices, :].prod(axis=0)
        nll_vals = self._loglike(samples).squeeze()
        # only use sub_pdf_vals. The other vals will be accounted for
        # with quadrature rule used to marginalize
        return np.exp(nll_vals+np.log(sub_pdf_vals))[:, None]

    def marginalize_unnormalized_pdf(self, sub_indices, sub_samples,
                                     quad_degrees):
        from pyapprox.surrogates.polychaos.gpc import _marginalize_function_nd
        from functools import partial
        return _marginalize_function_nd(
            partial(self.unnormalized_pdf, sub_indices),
            self._variable, quad_degrees, sub_indices, sub_samples)

    def plot_2d_marginals(self, nsamples_1d=100, variable_pairs=None,
                          subplot_tuple=None, qoi=0, num_contour_levels=20,
                          plot_samples=None):
        from pyapprox.variables.joint import get_truncated_range
        from pyapprox.surrogates.interp.indexing import (
            compute_anova_level_indices)
        from pyapprox.util.configure_plots import plt
        from pyapprox.util.visualization import get_meshgrid_samples
        from functools import partial

        if variable_pairs is None:
            variable_pairs = np.array(
                compute_anova_level_indices(self._variable.num_vars(), 2))
            # make first column values vary fastest so we plot lower triangular
            # matrix of subplots
            variable_pairs[:, 0], variable_pairs[:, 1] = \
                variable_pairs[:, 1].copy(), variable_pairs[:, 0].copy()

        if variable_pairs.shape[1] != 2:
            raise ValueError("Variable pairs has the wrong shape")

        if subplot_tuple is None:
            nfig_rows, nfig_cols = (
                self._variable.num_vars(), self._variable.num_vars())
        else:
            nfig_rows, nfig_cols = subplot_tuple

        if nfig_rows*nfig_cols < len(variable_pairs):
            raise ValueError("Number of subplots is insufficient")

        fig, axs = plt.subplots(
            nfig_rows, nfig_cols, figsize=(nfig_cols*8, nfig_rows*6))
        all_variables = self._variable.marginals()

        # if plot_samples is not None and type(plot_samples) == np.ndarray:
        #     plot_samples = [
        #         [plot_samples, {"c": "k", "marker": "o", "alpha": 0.4}]]

        for ii, var in enumerate(all_variables):
            lb, ub = get_truncated_range(var, unbounded_alpha=0.995)
            quad_degrees = np.array([20]*(self._variable.num_vars()-1))
            samples_ii = np.linspace(lb, ub, nsamples_1d)
            from pyapprox.surrogates.polychaos.gpc import (
                _marginalize_function_1d, _marginalize_function_nd)
            values = _marginalize_function_1d(
                partial(self._unnormalized_pdf_for_marginalization,
                        np.array([ii])),
                self._variable, quad_degrees, ii, samples_ii, qoi=0)
            axs[ii][ii].plot(samples_ii, values)
            if plot_samples is not None:
                for s in plot_samples:
                    axs[ii][ii].scatter(s[0][ii, :], s[0][ii, :]*0, **s[1])

        for ii, pair in enumerate(variable_pairs):
            # use pair[1] for x and pair[0] for y because we reverse
            # pairs above
            var1, var2 = all_variables[pair[1]], all_variables[pair[0]]
            axs[pair[1], pair[0]].axis("off")
            lb1, ub1 = get_truncated_range(var1, unbounded_alpha=0.995)
            lb2, ub2 = get_truncated_range(var2, unbounded_alpha=0.995)
            X, Y, samples_2d = get_meshgrid_samples(
                [lb1, ub1, lb2, ub2], nsamples_1d)
            quad_degrees = np.array([10]*(self._variable.num_vars()-2))
            values = _marginalize_function_nd(
                partial(self._unnormalized_pdf_for_marginalization,
                        np.array([pair[1], pair[0]])),
                self._variable, quad_degrees, np.array([pair[1], pair[0]]),
                samples_2d, qoi=qoi)
            Z = np.reshape(values, (X.shape[0], X.shape[1]))
            ax = axs[pair[0]][pair[1]]
            # place a text box in upper left in axes coords
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            ax.text(0.05, 0.95, r"$(\mathrm{%d, %d})$" % (pair[1], pair[0]),
                    transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            ax.contourf(
                X, Y, Z, levels=np.linspace(Z.min(), Z.max(),
                                            num_contour_levels),
                cmap='jet')
            if plot_samples is not None:
                for s in plot_samples:
                    # use pair[1] for x and pair[0] for y because we reverse
                    # pairs above
                    axs[pair[0]][pair[1]].scatter(
                        s[0][pair[1], :], s[0][pair[0], :], **s[1])

        return fig, axs


def run_bayesian_inference_gaussian_error_model(
        loglike, variable, nsamples, nburn, njobs,
        algorithm='nuts', get_map=False, print_summary=False,
        loglike_grad=None, seed=None):
    r"""
    Draw samples from the posterior distribution using Markov Chain Monte
    Carlo for data that satisfies

    .. math:: y=f(z)+\epsilon

    where :math:`y` is a vector of observations, :math:`z` are the
    parameters of a function which are to be inferred, and :math:`\epsilon`
    is Gaussian noise.

    Parameters
    ----------
    loglike : pyapprox.bayes.markov_chain_monte_carlo.GaussianLogLike
        A log-likelihood function associated with a Gaussian error model

    variable : pya.IndependentMarginalsVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    nsamples : integer
        The number of posterior samples

    nburn : integer
        The number of samples to discard during initialization

    njobs : integer
        The number of prallel chains

    algorithm : string
        The MCMC algorithm should be one of

        - 'nuts'
        - 'metropolis'
        - 'smc'

    get_map : boolean
        If true return the MAP

    print_summary : boolean
        If true print summary statistics about the posterior samples

    loglike_grad : callable
        Function with signature

       ``loglikegrad(z) -> np.ndarray (nvars)``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples

    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one. PyMC3 does not
        produce consistent results by setting numpy.random.seed instead
        seed must be passed in
    """

    if loglike_grad is None:
        logl = LogLike(loglike)
    elif loglike_grad == "FD":
        logl = LogLikeWithGrad(loglike)
    else:
        logl = LogLikeWithGrad(loglike, loglike_grad)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model():
        # must be defined inside with pm.Model() block
        pymc_variables, pymc_var_names = get_pymc_variables(
            variable.marginals())

        # convert m and c to a tensor vector
        theta = tt.as_tensor_variable(pymc_variables)

        # use a DensityDist (use a lamdba function to "call" the Op)
        # pm.DensityDist(
        #    'likelihood', lambda v: logl(v), observed={'v': theta})
        pm.Potential('likelihood', logl(theta))

        if get_map:
            map_sample_dict = pm.find_MAP(progressbar=False)
            map_sample = extract_map_sample_from_pymc3_dict(
                map_sample_dict, pymc_var_names)
        else:
            map_sample = None

        if nsamples == 0:
            return None, None, map_sample

        if algorithm == 'smc':
            assert njobs == 1  # njobs is always 1 when using smc
            trace = pm.sample_smc(nsamples)
        else:
            if algorithm == 'metropolis':
                step = pm.Metropolis(pymc_variables)
            elif algorithm == 'nuts':
                step = pm.NUTS(pymc_variables)
            else:
                msg = f"Algorithm {algorithm} not supported"
                raise ValueError(msg)

            trace = pm.sample(
                nsamples, tune=nburn, discard_tuned_samples=True,
                start=None, cores=njobs, step=step,
                compute_convergence_checks=False, random_seed=seed,
                progressbar=False)
                # return_inferencedata=False)
            # compute_convergence_checks=False avoids bugs in theano

        if print_summary:
            try:
                print(pm.summary(trace))
            except:
                print('could not print summary. likely issue with theano')

        samples, effective_sample_size = extract_mcmc_chain_from_pymc3_trace(
            trace, pymc_var_names, nsamples, nburn, njobs)

    return samples, effective_sample_size, map_sample


class PYMC3LogLikeWrapper():
    r"""
    Turn pyapprox model in to one which can be used by PYMC3.
    Main difference is that PYMC3 often passes 1d arrays where as
    Pyapprox assumes 2d arrays.
    """

    def __init__(self, loglike, loglike_grad=None):
        self.loglike = loglike
        self.loglike_grad = loglike_grad

    def __call__(self, x, jac=False):
        if x.ndim == 1:
            xr = x[:, np.newaxis]
        else:
            xr = x
        if jac is False:
            vals = self.loglike(xr)
            return vals.squeeze()
        if self.loglike_grad == True:
            return self.loglike(xr, jac=True)
        return self.loglike(xr).squeeze(), self.gradient(xr)

    def gradient(self, x):
        if self.loglike_grad is None:
            msg = "Cannot compute gradient. loglikegrad is set to None"
            raise ValueError(msg)
        if x.ndim == 1:
            xr = x[:, np.newaxis]
        else:
            xr = x
        return self.loglike_grad(xr).squeeze()


def loglike_from_negloglike(negloglike, samples, jac=False):
    if not jac:
        return -negloglike(samples)
    vals, grads = negloglike(samples, jac=jac)
    return -vals, -grads
