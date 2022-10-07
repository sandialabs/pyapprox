import numpy as np
import os
from pyapprox.util.randomized_svd import randomized_svd
from scipy.linalg import eigh as generalized_eigevalue_decomp


class PriorConditionedHessianMatVecOperator(object):
    r"""
    Compute the action of prior conditioned misfit Hessian on a vector.

    E.g. for a arbitrary vector w, the Cholesky factor L of the prior
    and the misfit Hessian H compute
        L*H*L'*w
    """
    def __init__(self, prior_covariance_sqrt_operator,
                 misfit_hessian_operator):
        self.prior_covariance_sqrt_operator = prior_covariance_sqrt_operator
        self.misfit_hessian_operator = misfit_hessian_operator

    def apply(self, vectors, transpose=None):
        r"""
        Compute L'*H*L*w.

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        transpose : boolean (default=True)
            The prior-conditioned Hessian is Symmetric so transpose does
            not matter. But randomized svd  assumes operator has a function
            apply(x, transpose)

        Returns
        -------
        z : (num_dims,num_vectors) matrix
            The matrix vector products: L'*H*L*w
        """
        x = self.prior_covariance_sqrt_operator.apply(vectors, transpose=False)
        assert x.shape[1] == vectors.shape[1], \
            'prior_covariance_sqrt_operator is returning incorrect values'
        y = self.misfit_hessian_operator.apply(x)
        assert y.shape[1] == x.shape[1], 'misfit_hessian_operator is returning incorrect values'
        z = self.prior_covariance_sqrt_operator.apply(y, transpose=True)
        return z

    def num_rows(self):
        return self.prior_covariance_sqrt_operator.num_vars()

    def num_cols(self):
        return self.prior_covariance_sqrt_operator.num_vars()


class LaplaceSqrtMatVecOperator(object):
    r"""
    Compute the action of the sqrt of the covariance of a Laplace
    approximation of the posterior on a vector.

    E.g. for a arbtirary vector w, the Cholesky factor L of the prior,
    the low rank eigenvalues e_r and eigenvectors V_r of the misfit hessian,
    and the misfit Hessian H compute
        L*(V*D*V'+I)*w
    where D = diag(np.sqrt(1./(e_r+1.))-1)
    """

    def __init__(self, prior_covariance_sqrt_operator, e_r=None, V_r=None,
                 M=None, filename=None):
        r"""
        Parameters
        ----------
        e_r : (rank,1) vector
            The r largest eigenvalues of the prior conditioned misfit hessian

        V_r : (num_dims,rank) matrix
            The eigenvectors corresponding to the r-largest eigenvalues

        M : (num_dims) vector (default=None)
            Weights defineing a weighted inner product

        filename : string
            The name of the file that contains data to initialize object.
            e_r, V_r, and M will be ignored.
        """
        self.prior_covariance_sqrt_operator = prior_covariance_sqrt_operator
        if filename is not None:
            assert V_r is None and e_r is None and M is None
            self.load(filename)
        else:
            assert V_r is not None and e_r is not None
            self.V_r = V_r
            self.M = M
            self.set_eigenvalues(e_r)

    def num_vars(self):
        return self.prior_covariance_sqrt_operator.num_vars()

    def set_eigenvalues(self, e_r):
        self.diagonal = np.sqrt(1./(e_r+1.))-1
        self.e_r = e_r

    def save(self, filename):
        if self.M is not None:
            np.savez(filename, e_r=self.e_r, V_r=self.V_r, M=self.M)
        else:
            # savez cannot save python None
            np.savez(filename, e_r=self.e_r, V_r=self.V_r)

    def load(self, filename):
        if not os.path.exists(filename):
            raise Exception('file %s does not exist' % filename)
        data = np.load(filename)
        self.V_r = data['V_r']
        if 'M' in list(data.keys()):
            self.M = data['M']
        else:
            self.M = None
        self.set_eigenvalues(data['e_r'])

    def apply_mass_weighted_eigvec_adjoint(self, vectors):
        r"""
        Apply the mass weighted adjoint of the eigenvectors V_r to a set
        of vectors w. I.e. compute
           x = V_r^T*M*w

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        Returns
        -------
        x : (rank,num_vectors) matrix
            The matrix vector products: V'M*w
        """
        if self.M is not None:
            print((self.M, 'a', type(self.M)))
            assert self.M.ndim == 1 and self.M.shape[0] == vectors.shape[0]
            for i in range(vectors.shape[0]):
                vectors[i, :] *= self.M[i]
        # else: M is the identity so do nothing
        return np.dot(self.V_r.T, vectors)

    def apply(self, vectors, transpose=False):
        r"""
        Compute L*(V*D*V'+I)*w = L(V*D*V'*w+w)

        Parameters
        ----------
        vectors : (num_dims,num_vectors) matrix
            A set or arbitrary vectors w.

        transpose : boolean
            True - apply L.T
            False - apply L

        Returns
        -------
        z : (num_dims,num_vectors) matrix
            The matrix vector products: L*(V*D*V'+I)*w
        """
        if transpose:
            vectors = self.prior_covariance_sqrt_operator.apply(
                vectors, transpose=True)
        # x = V'*vectors
        x = self.apply_mass_weighted_eigvec_adjoint(vectors)
        # y = D*x
        y = x*self.diagonal[:, np.newaxis]
        # z = V*y
        z = np.dot(self.V_r, y)
        z += vectors
        if not transpose:
            z = self.prior_covariance_sqrt_operator.apply(z, transpose=False)
        return z

    def __call__(self, vectors, transpose=False):
        return self.apply(vectors, transpose)


def get_laplace_covariance_sqrt_operator(
        prior_covariance_sqrt_operator, misfit_hessian_operator, svd_opts,
        weights=None, min_singular_value=0.1):
    r"""
    Get the operator representing the action of the cholesky factorization
    of the Laplace posterior approximation on a vector.

    Parameters
    ----------
    prior_covariance_sqrt_operator : Matrix vector multiplication operator
        The operator representing the action of the sqrt of the prior
        covariance on a vector. This is often but not always the cholesky
        factorization oft the prior covariance.

    misfit_hessian_operator : Matrix vector multiplication operator
        The operator representing the action of the misfit hessian on a vector

    svd_opts : dictionary
       The options to the SVD algorithm. See documentation of randomized_svd().

    weights : (num_dims) vector (default=None)
        Weights defineing a weighted inner product

    min_singular_value : double (default=0.1)
       The minimum singular value to retain in SVD. Note
       This can be different from the entry 'min_singular_value' in svd_opts

    Returns
    -------
    covariance_sqrt_operator :  Matrix vector multiplication operator
        The action of the sqrt of the covariance on a vector
    """
    e_r, V_r = get_low_rank_prior_conditioned_misfit_hessian(
        prior_covariance_sqrt_operator, misfit_hessian_operator, svd_opts,
        min_singular_value)

    operator = LaplaceSqrtMatVecOperator(
        prior_covariance_sqrt_operator, e_r, V_r, weights)

    return operator


def get_low_rank_prior_conditioned_misfit_hessian(
        prior_covariance_sqrt_operator, misfit_hessian_operator, svd_opts,
        min_singular_value=0.1):
    r"""
    Get the low rank approximation of the prior conditioned misfit hessian
    using only matrix vector multiplication operators.

    Parameters
    ----------
    prior_covariance_sqrt_operator : Matrix vector multiplication operator
        The operator representing the action of the sqrt of the prior
        covariance on a vector. This is often but not always the cholesky
        factorization oft the prior covariance.

    misfit_hessian_operator : Matrix vector multiplication operator
        The operator representing the action of the misfit hessian on a vector

    min_singular_value : double (default=0.1)
       The minimum singular value to retain in SVD. Note
       This can be different from the entry 'min_singular_value' in svd_opts

    svd_opts : dictionary
       The options to the SVD algorithm. See documentation of randomized_svd().


    Returns
    -------
    e_r : (rank,1) vector
        The r largest eigenvalues of the prior conditioned misfit hessian
    V_r : (num_dims,rank) matrix
        The eigenvectors corresponding to the r-largest eigenvalues
    """

    operator = PriorConditionedHessianMatVecOperator(
        prior_covariance_sqrt_operator, misfit_hessian_operator)

    svd_opts['single_pass'] = True
    U, S, V = randomized_svd(operator, svd_opts)
    I = np.where(S >= min_singular_value)[0]
    e_r = S[I]
    V_r = U[:, I]
    return e_r, V_r


def find_map_point(objective, initial_guess, opts=None):
    r"""
    Find the maximum of the log posterior of Bayes rule.

    Parameters
    ----------
    objective : Model object
        The log of the posterior using Bayes rule. Does not need to be
        normalized e.g, can simply be
           misfit(x) + log(prior(x)),
        Objective must implement with .evaluate()
        and .gradient() functions

    initial guess : (num_dims,1) vector
        The initial point to start the local optimization

    opts : dictionary (default=None)
        Options for the optimizer
        If None opts is set to opts = {'maxiter':1000,'gtol':1e-10}

    Returns
    -------
    map_point : (num_dims,1) vector
        The coordinates of the maximum of the log posterior

    obj_max : float
        The maximum of the log posterior
    """
    if opts is None:
        opts = {'maxiter': 1000, 'gtol': 1e-10}

    def obj_func(x): return - \
        objective(x[:, np.newaxis], {'eval_type': 'value'})[0, :]

    def obj_grad(x): return - \
        objective(x[:, np.newaxis], {'eval_type': 'grad'})[0, :]
    from scipy.optimize import fmin_bfgs
    out = fmin_bfgs(obj_func, fprime=obj_grad,
                    x0=initial_guess, gtol=opts['gtol'],
                    maxiter=opts['maxiter'], disp=False, full_output=True)
    map_point = out[0]
    obj_max = out[1]
    return map_point, obj_max


def laplace_posterior_approximation_for_linear_models(
        linear_matrix, prior_mean, prior_hessian, noise_covariance_inv, obs,
        bvec=None):
    r"""
    Compute the mean and covariance of the Laplace posterior of a linear model
    with a Gaussian prior

    Given some data d and a linear forward model, A(x) = Ax+b,
    and a Gaussian likelihood and a Gaussian prior, the resulting posterior
    is always Gaussian.

    Parameters
    ----------
    linear_matrix : (num_qoi, num_dims) matrix
        The matrix reprsenting the linear forward model.

    prior_mean : (num_dims, 1) vector
        The mean of the Gaussian prior

    prior_hessian: (num_dims, num_dims) matrix
        The Hessian (inverse of the covariance) of the Gaussian prior

    noise_covariance_inv : (num_qoi, num_qoi) matrix
        The inverse of the covariance of the osbervational noise

    obs : (num_qoi, 1) vector
        The observations

    bvec : np.ndarray(num_qoi)
        The deterministic shift of the linear model

    Returns
    -------
    posterior_mean : (num_dims, 1) vector
        The mean of the Gaussian posterior

    posterior_covariance: (num_dims, num_dims) matrix
        The covariance of the Gaussian posterior
    """
    if prior_mean.ndim == 1:
        prior_mean = prior_mean[:, np.newaxis]
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]
    assert prior_mean.ndim == 2 and prior_mean.shape[1] == 1
    assert obs.ndim == 2 and obs.shape[1] == 1
    assert linear_matrix.shape[0] == obs.shape[0]
    assert prior_mean.shape[0] == linear_matrix.shape[1]
    assert noise_covariance_inv.shape[0] == obs.shape[0]
    misfit_hessian = np.dot(
        np.dot(linear_matrix.T, noise_covariance_inv), linear_matrix)
    posterior_covariance = np.linalg.inv(misfit_hessian + prior_hessian)
    residual = obs-np.dot(linear_matrix, prior_mean)
    if bvec is not None:
        residual -= bvec
    temp = linear_matrix.T.dot(noise_covariance_inv.dot(residual))
    posterior_mean = np.dot(posterior_covariance, temp)+prior_mean
    return posterior_mean, posterior_covariance


def push_forward_gaussian_though_linear_model(A, b, mean, covariance):
    r"""
    Find the mean and covariance of a gaussian distribution when it
    is push forward through a linear model. A linear transformation
    applied to a Gaussian is still a Gaussian.

    Original Gaussian with mean x and covariance \Sigma
    z~N(x,\Sigma)

    Transformation with b is a constant vector, e.g has no variance
    y = Az + b

    Distribution of resulting gaussian
    y~N(Ax+b,A\Sigma A^T)
    """

    y_mean = np.dot(A, mean)+b
    y_covariance = np.dot(np.dot(A, covariance), A.T)

    return y_mean, y_covariance


class MisfitHessianVecOperator(object):
    r"""
    Operator which computes the Hessian vector product. The Hessian
    is the Hessian of a misfit function and if not available
    the action of the Hessian is computed using finite differences of
    gradients of the misfit of from function evaluations.
    """

    def __init__(self, model, map_point,
                 fd_eps=2*np.sqrt(np.finfo(float).eps)):
        r"""
        Initialize the MisfitHessianVecOperator

        Parameters
        ----------
        model : Model object
            A model which must allow model.gradient_set()
            and possess model.rv_trans object

        map_point : (num_dims) vector
           The point x that maximizes likelihood(x)*prior(x)

        fd_eps : float (default=2*mach_eps)
            The finite difference step size. If not None
            Then action of hessian will be computed with finite
            difference even if model has a hessian attribute
        """
        self.model = model
        self.map_point = map_point
        self.fd_eps = fd_eps

        self.map_point_misfit_gradient = None

        if not hasattr(self.model, 'hessian') or fd_eps is not None:
            assert fd_eps is not None
            assert fd_eps >= 2*np.sqrt(np.finfo(float).eps)
            if hasattr(self.model, 'gradient_set'):
                self.map_point_misfit_gradient = self.model.gradient_set(
                    map_point[:, np.newaxis])[:, 0]
                assert (self.map_point_misfit_gradient.shape[0] ==
                        self.map_point.shape[0])
            else:
                msg = 'model does not have member function called gradient'
                raise Exception(msg)

    def num_rows(self):
        return self.map_point.shape[0]

    def num_cols(self):
        return self.num_rows()

    def apply(self, vectors, transpose=None):
        r"""
        Compute action of hessian on a vector

        If self.model has no function hessian() then
        use first-order finite difference of gradient to compute action
        of Hessian on a vector, e.g

        H(x)v = (g(x+v*h)-g(x))/h

        Notes
        -----
        Laplace posterior is only defined for Gaussian prior
        so we do not have to worry about exceeding bounds
        with finite difference, so always use forward finite difference.

        Parameters
        ----------
        vectors : (num_dimx,num_vectors) matrix
            Vectors to which the action of the Hessian will be applied

        transpose : boolean
            Hessian is symmetric so transpose is a needless parameter
            But randomized svd  assumes operator has a function
            apply(x, transpose)

        Returns
        -------
        hessian_vector_products : (num_dims,num_vectors) matrix
            The Hessian vector products
        """
        if hasattr(self.model, 'hessian') and self.fd_eps is None:
            print('TODO replace by opearator hess_vec_prod = model.hess.apply(map_point,vectors). first arg says where to evaluate hessian opearator')
            H = self.model.hessian(self.map_point)
            hessian_vector_products = np.dot(H, vectors)
        elif hasattr(self.model, 'gradient_set'):
            def grad_func(x): return self.model.gradient_set(x).T
            # function passed to directional_derivatives function must return
            # np.ndarray with shape (num_samples,num_vars)
            # each gradient entry is considered a qoi of a function
            # directional_derivatives function also returns np.ndarray of shape
            # (num_vectors,num_dims) so must transpose result
            hessian_vector_products = directional_derivatives(
                grad_func, self.map_point,
                self.map_point_misfit_gradient, vectors, self.fd_eps).T
        else:
            msg = 'To implement action of hessian you need to specify hessian function or gradient_set function'
            raise Exception(msg)
        return hessian_vector_products


def directional_derivatives(function, sample, value_at_sample, vectors, fd_eps,
                            normalize_vectors=False,
                            use_central_finite_difference=False):
    r"""
    Compute the first-order forward difference directional derivative of a
    vector valued function.

    Parameters
    ----------
    function : callable_function
        Vector valued function of interest. If function returns a gradient.
        then function must return matrix (num_samples, num_dims)

    sample : (num_dims,1) vector
        The sample at which the directional derivative needs to be computed

    value_at_sample : (1,num_qoi) vector
        The function value(s) at sample

    vectors : (num_dims,num_vectors) matrix
        The direction vectors of the directional finite differences

    fd_eps : float
        The finite difference step size

    normalize_vectors : bool
        Normalize the directional derivatives by the l2 norms of the
        direction vectors

    use_central_finite_difference : bool
        True: use central finite difference (value_at_sample is ignored)
        False: use forward finite difference (using value_at_sample)

    Returns
    -------
    directional_derivatives : (num_dims,num_vectors) matrix
        The directional derivatives in the direction of the vectors
    """
    if sample.ndim == 1:
        sample = sample[:, np.newaxis]
    else:
        assert sample.shape[1] == 1
    assert vectors.ndim == 2

    if not use_central_finite_difference:
        if value_at_sample.ndim == 1:
            value_at_sample = value_at_sample[np.newaxis, :]
        else:
            assert value_at_sample.shape[0] == 1
        num_perturbed_samples = vectors.shape[1]
        perturbed_samples = np.tile(
            sample, (1, num_perturbed_samples))
        perturbed_samples += fd_eps*vectors
        perturbed_values = function(perturbed_samples)
        assert perturbed_values.shape[0] == vectors.shape[1]
        directional_derivatives = (
            perturbed_values-value_at_sample)/(fd_eps)
        assert directional_derivatives.shape[1] == value_at_sample.shape[1]
    else:
        num_perturbed_samples = 2*vectors.shape[1]
        perturbed_samples = np.tile(
            sample, (1, num_perturbed_samples))
        perturbed_samples[:, :num_perturbed_samples/2] += fd_eps*vectors
        perturbed_samples[:, num_perturbed_samples/2:] -= fd_eps*vectors
        perturbed_values = function(perturbed_samples)
        assert perturbed_values.shape[0] == 2*vectors.shape[1]
        directional_derivatives = (perturbed_values[:num_perturbed_samples/2, :] -
                                   perturbed_values[num_perturbed_samples/2:, :])/(2*fd_eps)

    if normalize_vectors:
        directional_derivatives /= np.linalg.norm(vectors, axis=0)
    assert directional_derivatives.shape[0] == vectors.shape[1]
    return directional_derivatives


def sample_from_laplace_posterior(laplace_mean, laplace_covariance_sqrt,
                                  num_dims, num_samples, weights=None):
    r"""
    Parameters
    -------
    laplace_mean : vector (num_dims)
        The mean of the Laplace posterior distribution

    laplace_covariance_sqrt :  Matrix vector multiplication operator
        The action of the sqrt of the covariance on a vector

    num_dims : integer
       The number of random variables

    num_samples : integer
       The desired number of posterior samples

    weights : vector (num_dims) (default=None)
       weights defining a weighted inner product

    Returns
    -------
    posterior_samples : matrix (num_dims,num_samples)
        Samples from the posterior
    """
    assert laplace_mean.ndim == 2 and laplace_mean.shape[1] == 1
    std_normal_samples = np.random.normal(0., 1., (num_dims, num_samples))
    if weights is not None:
        assert weights.ndim == 1 and weights.shape[0] == num_dims
        std_normal_samples /= np.sqrt(weights)

    posterior_samples = \
        laplace_covariance_sqrt.apply(std_normal_samples, transpose=False) +\
        laplace_mean
    return posterior_samples


def get_pointwise_laplace_variance(prior, laplace_covariance_sqrt):
    prior_pointwise_variance = prior.pointwise_variance()
    return get_pointwise_laplace_variance_using_prior_variance(
        prior, laplace_covariance_sqrt, prior_pointwise_variance)


def get_pointwise_laplace_variance_using_prior_variance(
        prior, laplace_covariance_sqrt, prior_pointwise_variance):
    # compute L*V_r
    tmp1 = prior.apply_covariance_sqrt(laplace_covariance_sqrt.V_r, False)
    # compute D*(L*V_r)**2
    tmp2 = laplace_covariance_sqrt.e_r/(1.+laplace_covariance_sqrt.e_r)
    tmp3 = np.sum(tmp1**2*tmp2, axis=1)
    return prior_pointwise_variance-tmp3, prior_pointwise_variance


def generate_and_save_laplace_posterior(
        prior, misfit_model, num_singular_values,
        svd_history_filename='svd-history.npz',
        Lpost_op_filename='laplace_sqrt_operator.npz',
        num_extra_svd_samples=10,
        fd_eps=2*np.sqrt(np.finfo(float).eps)):

    if os.path.exists(svd_history_filename):
        raise Exception(
            'File %s already exists. Exiting so as not to overwrite' %
            svd_history_filename)
    if os.path.exists(Lpost_op_filename):
        raise Exception(
            'File %s already exists. Exiting so as not to overwrite' %
            Lpost_op_filename)

    sample = misfit_model.map_point()
    misfit_hessian_operator = MisfitHessianVecOperator(
        misfit_model, sample, fd_eps=fd_eps)
    standard_svd_opts = {
        'num_singular_values': num_singular_values,
        'num_extra_samples': num_extra_svd_samples}
    svd_opts = {'single_pass': True, 'standard_opts': standard_svd_opts,
                'history_filename': svd_history_filename}
    L_post_op = get_laplace_covariance_sqrt_operator(
        prior.sqrt_covariance_operator, misfit_hessian_operator,
        svd_opts, weights=None, min_singular_value=0.0)

    L_post_op.save(Lpost_op_filename)
    return L_post_op


def generate_and_save_pointwise_variance(
        prior, L_post_op,
        prior_variance_filename='prior_pointwise-variance.npz',
        posterior_variance_filename='posterior_pointwise-variance.npz'):
    if not os.path.exists(prior_variance_filename):
        posterior_pointwise_variance, prior_pointwise_variance =\
            get_pointwise_laplace_variance(prior, L_post_op)
        np.savez(
            prior_variance_filename, prior_pointwise_variance=prior_pointwise_variance)
        np.savez(posterior_variance_filename,
                 posterior_pointwise_variance=posterior_pointwise_variance)
    else:
        print(('File %s already exists. Loading data' % prior_variance_filename))
        prior_pointwise_variance = np.load(prior_variance_filename)[
            'prior_pointwise_variance']
        if not os.path.exists(posterior_variance_filename):
            posterior_pointwise_variance, prior_pointwise_variance = \
                get_pointwise_laplace_variance_using_prior_variance(
                    prior, L_post_op, prior_pointwise_variance)
            np.savez(posterior_variance_filename,
                     posterior_pointwise_variance=posterior_pointwise_variance)
        else:
            posterior_pointwise_variance = np.load(posterior_variance_filename)[
                'posterior_pointwise_variance']
    return prior_pointwise_variance, posterior_pointwise_variance


def compute_posterior_mean_covar_optimal_for_prediction(
        obs, obs_matrix, prior_mean, prior_covar, obs_noise_covar,
        pred_matrix, economical=False):

    assert pred_matrix.shape[0] <= prior_mean.shape[0]

    # step 1
    OP = np.dot(pred_matrix, prior_covar)
    # step 2
    C = np.dot(OP, obs_matrix.T)
    # step 3
    Pz = np.dot(OP, pred_matrix.T)
    # step 4
    Pz_inv = np.linalg.inv(Pz)
    # step 5
    A = np.dot(C.T, np.dot(Pz_inv, C))
    # step 6
    data_covar = np.dot(np.dot(obs_matrix, prior_covar), obs_matrix.T) +\
        obs_noise_covar
    # step 7
    # print 'TODO replace generalized_eigevalue_decomp by my subspace iteration'
    evals, evecs = generalized_eigevalue_decomp(A, data_covar)
    evecs = evecs[:, ::-1]
    evals = evals[::-1]
    rank = min(pred_matrix.shape[0], obs_matrix.shape[0])
    evecs = evecs[:, :rank]
    evals = evals[:rank]
    # step 8
    ppf_covar_evecs = np.dot(C, evecs)

    residual = obs - np.dot(obs_matrix, prior_mean)
    opt_pf_covar = Pz - np.dot(ppf_covar_evecs, ppf_covar_evecs.T)
    opt_pf_mean = np.dot(ppf_covar_evecs, np.dot(evecs.T, residual))+np.dot(
        pred_matrix, prior_mean)

    if economical:
        return opt_pf_mean, opt_pf_covar
    else:
        posterior_evec = np.dot(np.dot(OP.T, Pz_inv), ppf_covar_evecs)
        posterior_covar = prior_covar-np.dot(posterior_evec, posterior_evec.T)
        posterior_mean = np.dot(np.dot(posterior_evec, evecs.T), residual) +\
            prior_mean

        return opt_pf_mean, opt_pf_covar, posterior_mean, posterior_covar


def laplace_evidence(likelihood_fun, prior_pdf, post_covariance, map_point):
    """
    References
    ----------
    Ryan, K. (2003). Estimating Expected Information Gains for Experimental
    Designs with Application to the Random Fatigue-Limit Model. Journal of
    Computational and Graphical Statistics, 12(3), 585-603.
    http://www.jstor.org/stable/1391040

    Friel, N. and Wyse, J. (2012), Estimating the evidence – a review.
    Statistica Neerlandica, 66: 288-308.
    https://doi.org/10.1111/j.1467-9574.2011.00515.x
    """
    assert map_point.ndim == 2
    nvars = post_covariance.shape[0]
    lval = likelihood_fun(map_point)
    prior_val = prior_pdf(map_point)
    assert lval.ndim == 1
    assert prior_val.ndim == 2
    evidence = (2*np.pi)**(nvars/2)*np.sqrt(np.linalg.det(post_covariance))
    evidence *= lval[0]*prior_val[0, 0]
    return evidence
