import numpy
def get_low_rank_prior_conditioned_misfit_hessian_deprecated(
        prior_chol_factor, misfit_hessian, min_singular_value, verbosity):
    r"""
    Get the low rank approximation of the prior conditioned misfit hessian
    using dense matrices.

    This is now deprecated because the operator based function is more general.
    This function was kept for testing purposes only.

    Parameters
    ----------
    prior_covariance_chol_factor : (num_dims x num_dims) matrix
        The lower-triangular cholesky factor of the prior covariance

    misfit_hessian : (num_dims x num_dims ) matrix
        The Hessian of the misfit function (at an arbitrary point in parameter
        space)

    min_singular_value : float
        The tolerance used to generate a low rank approximation of the
        posterior. All eigvalues less than or equal to min_singular_value
        will be discarded

    verbosity : integer (default=0)
        0 - no output
        1 - minimal output
        2 - debug output

    Returns
    -------
    e_r : (rank x 1) vector
        The r largest eigenvalues of the prior conditioned misfit hessian
    V_r : (num_dims x rank) matrix
        The eigenvectors corresponding to the r-largest eigenvalues
    """

    L = prior_chol_factor
    H = misfit_hessian
    LHL = numpy.dot( L.T, numpy.dot( H, L ) )
    eigvals, eigvecs = numpy.linalg.eig( LHL )
    indices = numpy.argsort( eigvals )[::-1]
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:,indices]

    rank = numpy.count_nonzero(eigvals > min_singular_value )
    e_r = eigvals[:rank]
    V_r = eigvecs[:,:rank]

    if verbosity > 1:
        print('prior conditioned misfit proposal covariance')
        print('\tnum vars:', H.shape[0])
        print('\trank:', rank)
        print('\tmin eigval', eigvals[-1])
    return e_r, V_r


def get_prior_conditioned_misfit_covariance_chol_factor(
        prior_covariance_chol_factor, misfit_hessian, min_singular_value,
        verbosity=0):
    """
    Get the cholesky factorization of the Laplace posterior approximation.

    Parameters
    ----------
    prior_covariance_chol_factor : (num_dims x num_dims) matrix
        The lower-triangular cholesky factor of the prior covariance

    misfit_hessian : (num_dims x num_dims ) matrix
        The Hessian of the misfit function (at an arbitrary point in parameter
        space)

    min_singular_value : float
        The tolerance used to generate a low rank approximation of the
        posterior. All eigvalues less than or equal to min_singular_value
        will be discarded

    verbosity : integer (default=0)
        0 - no output
        1 - minimal output
        2 - debug output

    Returns
    -------
    covariance_chol_factor : (num_dims x num_dims) matrix
        The cholesky factor of covariance of the posterior
    """
    e_r, V_r = get_low_rank_prior_conditioned_misfit_hessian_deprecated(
        prior_covariance_chol_factor, misfit_hessian, min_singular_value,
        verbosity)

    #diagonal = diag(1./(e_r+1.)-1)
    #prior_conditioned_hessian_inv = \
    #    numpy.dot(numpy.dot(L,numpy.dot(numpy.dot(V_r, diagonal), V_r.T ) + I),
    #                 L.T )

    #covariance_chol_factor = \
    #    numpy.dot( L, numpy.dot( numpy.dot( V_r, diagonal ), V_r.T ) + I )
    L = prior_covariance_chol_factor
    diagonal = numpy.sqrt(1./(e_r+1.))-1
    Y = V_r.T*diagonal[:,numpy.newaxis]
    Z = numpy.dot(V_r,Y)
    for i in range(Z.shape[0]):
        Z[i,i] += 1.
    covariance_chol_factor = numpy.dot(L,Z)

    return covariance_chol_factor#, prior_conditioned_hessian_inv

def get_laplace_covariance_chol_factor(
        sample,misfit_hessian,prior_chol_factor,min_singular_value,verbosity=0):
    r"""
    Given a Gaussian prior with mean $\bar{z}$ and covariance $\Sigma$

    Computes the Laplace approximation at a sample z.
    That is evaluate the hessian of the negative log likelihood
    f(z) = 1/2r(z)^T\Sigma_\text{noise}^{-1}r(z) + 1/2(z-\bar{z})^T\Sigma_\text{prior}^{-1}(z-\bar{z})
    and compute the corresponding Gaussian centered at the sample z
    """
    prior_hessian = numpy.linalg.inv(
        numpy.dot(prior_chol_factor,prior_chol_factor.T))
    return get_prior_conditioned_misfit_covariance_chol_factor(
          prior_chol_factor, misfit_hessian, min_singular_value, verbosity)

def get_stochastic_newton_proposal_mean(sample, misfit_grad, prior_mean,
                                        prior_hessian,
                                        laplace_covariance_chol_factor):
    """
    Get the mean of the stochastic Newton MCMC proposal distribution.

    mean = x + H_m^{-1}*g

    where

    g = g_m + g_p

    and the gradient of the prior is

    g_p = H_p*(x-prior_mean)

    Parameters
    ----------
    sample : (num_dims x 1) vector
        The current sample x in the MCMC chain

    misfit_grad : (num_dims x 1) vector
        The gradient of the misfit function at the sample x

    prior_hessian : (num_dims x num_dims) matrix
        The Hessian H_p of the prior distribution at the sample x

    prior_mean : (num_dims x 1) vector
        The mean of the prior distribution

    laplace_covariance_chol_factor : (num_dims x num_dims) matrix
        Cholesky factor of the covariance of the Laplace approximation of
        the posterior

    Returns
    -------
    mean : (num_dims x 1) vector
        The mean of the stochastic Newton proposal density
    """
    proposal_covariance = numpy.dot(laplace_covariance_chol_factor,
                              laplace_covariance_chol_factor.T)

    prior_grad = numpy.dot(prior_hessian,sample-prior_mean).squeeze()
    grad = misfit_grad + prior_grad
    mean_shift = -numpy.dot( proposal_covariance, grad ).squeeze()
    mean = sample.squeeze() + mean_shift.squeeze()
    return mean

def get_stochastic_newton_proposal_mean_and_covar_chol_factor(
        sample, misfit_grad, misfit_hessian, prior_mean,
        prior_chol_factor, min_singular_value=0., verbosity=0):
    """
    Compute the stochastic Newton proposal density as per
    Petra, Martin, Stadler, Ghattas 2014.

    Parameters
    ----------
    sample : (num_dims x 1) vector
        The current sample x in the MCMC chain

    misfit_grad : (num_dims x 1) vector
        The gradient of the misfit function at the sample x

    prior_hessian : (num_dims x num_dims) matrix
        The Hessian H_p of the prior distribution at the sample x

    prior_mean : (num_dims x 1) vector
        The mean of the prior distribution

    min_singular_value : float
        The tolerance used to generate a low rank approximation of the
        posterior. All eigvalues less than or equal to min_singular_value will
        be discarded

    verbosity : integer (default=0)
        0 - no output
        1 - minimal output
        2 - debug output

    Returns
    -------
    mean : (num_dims x 1) vector
        The mean of the stochastic Newton proposal density

    covariance_chol_factor : (num_dims x num_dims) matrix
        Cholesky factor of the covariance of the Laplace approximation of
        the posterior
    """

    covariance_chol_factor = get_laplace_covariance_chol_factor(
        sample,misfit_hessian,prior_chol_factor,min_singular_value,verbosity)

    prior_hessian = numpy.linalg.inv(
        numpy.dot(prior_chol_factor,prior_chol_factor.T))
    mean = get_stochastic_newton_proposal_mean(
        sample, misfit_grad, prior_mean, prior_hessian, covariance_chol_factor)

    return mean, covariance_chol_factor

def test_laplace_approximation_for_linear_models(self):
    num_dims = 2; num_qoi = 3
    self.help_test_laplace_approximation_for_linear_models(
        num_dims,num_qoi)

    num_dims = 2; num_qoi = 2
    self.help_test_laplace_approximation_for_linear_models(
        num_dims,num_qoi)

    num_dims = 3; num_qoi = 2
    self.help_test_laplace_approximation_for_linear_models(
        num_dims,num_qoi)

def help_test_laplace_approximation_for_linear_models(
        self,num_dims,num_qoi):
    """
    Test laplace approximation is exact for linear model f(z) = A*z
    """
    linear_matrix = np.random.normal(0.,1.,(num_qoi,num_dims))
    prior_mean = np.zeros((num_dims),float)
    prior_chol_factor = np.eye(num_dims)
    prior_hessian = np.eye(num_dims)
    noise_covariance_inv = np.eye(num_qoi)
    truth_sample = np.random.normal(0.,1.,num_dims)
    obs = numpy.dot(linear_matrix,truth_sample)+\
      np.random.normal(0.,1.,num_qoi)
    laplace_mean_1, laplace_covariance_1 = \
      laplace_posterior_approximation_for_linear_models(
          linear_matrix,prior_mean,prior_hessian,noise_covariance_inv,obs)
    # for a linear model sample will not matter as misfit hessian
    # and gradient are the same throughout parameter space
    sample = np.random.normal(0.,1.,num_dims)

    misfit_hessian=numpy.dot(
        numpy.dot(linear_matrix.T,noise_covariance_inv),linear_matrix)
    temp = numpy.dot(numpy.dot(linear_matrix,sample)-obs,noise_covariance_inv)
    misfit_grad = numpy.dot( temp, linear_matrix)

    min_singular_value =0.
    laplace_chol_factor_2 = get_laplace_covariance_chol_factor(
        sample, misfit_hessian, prior_chol_factor, min_singular_value )
    laplace_covariance_2 = numpy.dot(
        laplace_chol_factor_2,laplace_chol_factor_2.T)
    assert np.allclose(laplace_covariance_1,laplace_covariance_2)
    laplace_mean_2 =  get_stochastic_newton_proposal_mean(
        sample, misfit_grad, prior_mean, prior_hessian,
        laplace_chol_factor_2)
    assert np.allclose(laplace_mean_1,laplace_mean_2)

    laplace_mean_3, laplace_chol_factor_3 = \
      get_stochastic_newton_proposal_mean_and_covar_chol_factor(
          sample, misfit_grad, misfit_hessian, prior_mean,
          prior_chol_factor, min_singular_value=0.)
    laplace_covariance_3 = numpy.dot(
        laplace_chol_factor_3,laplace_chol_factor_3.T)
    assert np.allclose(laplace_mean_1,laplace_mean_3)
    assert np.allclose(laplace_covariance_1,laplace_covariance_3)


def get_stochastic_newton_proposal_mean_and_covar_chol_factor_from_function(
        gradient,hessian,sample,prior_mean,prior_chol_factor,
        min_singular_value=0.,verbosity=0):
    """
    assumes PCE is an approximation negative log likelihood
         1/2*r.T*inv(sigma_noise)*r
    """

    num_dims = prior_mean.shape[0]
    misfit_grad = gradient( sample ).squeeze()
    misfit_hessian = hessian(sample)
    return get_stochastic_newton_proposal_mean_and_covar_chol_factor(
        sample, misfit_grad, misfit_hessian, prior_mean,
        prior_chol_factor, min_singular_value, verbosity )
