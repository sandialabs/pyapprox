import numpy
from laplace_deprecated import get_stochastic_newton_proposal_mean_and_covar_chol_factor
from utilities.compressed_sensing_based_pce import \
     initialize_and_build_pce_using_compressed_sensing
from model_cpp import define_homogeneous_input_space
from compressed_sensing_cpp import OMPSolver, LARSSolver
def build_hermite_pce(num_dims,degree,build_samples,build_values,
                      test_data=None,use_cross_validation=True,verbosity=0,
                      solver_type='omp'):
    assert build_samples.shape[0] == num_dims
    assert build_samples.shape[1] == build_values.shape[0]

    rv_trans = define_homogeneous_input_space( 'gaussian', num_dims,
                                                mean = 0., std_dev=1. )

    pce, cv_error = initialize_and_build_pce_using_compressed_sensing(
        rv_trans, degree, build_samples, build_values,
        test_data=test_data, use_cross_validation=use_cross_validation,
        verbosity=0, solver_type=solver_type, precond_type='none',
        use_derivatives=False,total_degree=None, cv_opts=None)

    return pce

def build_hermite_pce_on_active_subspace_from_function(
        function,num_dims,num_samples,degree,num_test_samples,W1=None):
    """
    Build a Hermite PCE on an active subspace

    function: callable function
       a function that accepts a set of samples (num_dims x num_samples)
       and returns a matrix (num_samples x num_qoi) of values

    num_dims: integer
       dimension of function

    num_samples:
       the number of samples to evaluate function at and use to build the PCE

    degree: integer
       the degree of the total-degree PCE

    num_test_samples: integer
       the number of additional samples which are used in a cross validation
       procedure to choose the best step in the compressed sensing algorithm
       use to aproximate the PCE coefficients

    W1: matrix (num_dims x num_active_dims)
       The rotation matrix associated with the active subspace. Takes points
       with dimension num_dims and transforms them into poinst with dimension
       num_active_dims

    Additonal Notes:
       assumes hermite defined with respect to standard normal gaussian

       assumes function does not return gradients so gradient data,
       of build_hermite_pce function, is None
    """
    build_samples = numpy.random.normal(0.,1.,(num_dims,num_samples))
    build_values = function(build_samples)

    test_samples = numpy.random.normal(0.,1.,(num_dims,num_samples))
    test_values = function(test_samples)

    if W1 is None:
        W1 = numpy.eye(num_dims)

    assert W1.shape[0] == num_dims
    num_active_dims = W1.shape[1]
    active_build_samples = numpy.dot(W1.T,build_samples)
    active_test_samples = numpy.dot(W1.T,test_samples)

    test_data = (active_test_samples, test_values)

    asub_pce = build_hermite_pce(
        num_active_dims, degree, active_build_samples,build_values,
        test_data,use_cross_validation='exact-data')
    return asub_pce

def build_laplace_posterior_from_quadratic_function_on_active_subspace(
        num_dims,num_active_dims, W,gradient,hessian,
        return_rotated_data=False):
    """
    Build a gaussian posterior from a quadratic function defined on an
    active subspace, assuming a standard normal Gaussian prior on the
    full space

    num_dims: integer
       the number of variables in the original full space

    num_active_dims: integer [1,num_dims]
       the number of variables in the active subspace

    W: matrix (num_dims x num_active_dims)
       The rotation matrix associated with the rotated inactive and active
       subspaces.

    gradient: callable function
       the gradient of the quadratic function in the active variables
       must accept a matrix (num_active_dims x num_samples)
       must return a vector (num_active_dims)

    hessian: callable function
       the hessian of the quadratic function in the active variables
       must accept a vector (num_active_dims)
       must return a matrix (num_active_dims x num_active_dims)

    return_rotated_data: bool
      flag specifying (if True) to return the mean and covariance of
      the distribution in the full rotated coordinates in addition to
      default return values
    """
    num_inactive_dims = num_dims-num_active_dims

    # Define the prior over the inactive dimensions
    # Because we assume standard normal priors in the full space
    # the prior will be standard normal in the active dimensions
    asub_prior_mean = numpy.zeros(num_active_dims)
    asub_prior_chol_factor = numpy.eye(num_active_dims)

    # Choose sample in the active subspace at which to compute the hessian
    # of the quadratic function.
    # Since the function is quadratic the sample at which each hessian is
    # computed does not matter, so just choose zeros
    asub_sample = numpy.zeros((num_active_dims))

    asub_posterior_mean, asub_posterior_covariance_chol_factor = \
      get_stochastic_newton_proposal_mean_and_covar_chol_factor_from_function(
          gradient,hessian,asub_sample,asub_prior_mean,asub_prior_chol_factor,
        min_singular_value=0.,verbosity=0)
    asub_posterior_covariance = numpy.dot(
        asub_posterior_covariance_chol_factor,
        asub_posterior_covariance_chol_factor.T)

    if num_inactive_dims > 0:
        # Define the prior over the inactive dimensions
        # Because we assume standard normal priors in the full space
        # the prior will be standard normal in the inactive dimensions
        asub_prior_mean = numpy.zeros(num_active_dims)
        asub_prior_chol_factor = numpy.eye(num_active_dims)
        isub_prior_mean = numpy.zeros(num_inactive_dims)
        isub_prior_chol_factor = numpy.eye(num_inactive_dims)
        isub_prior_covariance = numpy.dot(
            isub_prior_chol_factor,isub_prior_chol_factor.T)

        # When the priors are standard gaussian the posterior is the
        # the product of the posterior in the active dimensions p_{post}(y)
        # and the prior in the inactive dimensions p_{pr}(z), i.e.
        # p_{post}(x) = p_{post}(y)p_{pr}(z). This means covariance
        # is block diagonal
        from scipy.linalg import block_diag
        rotated_posterior_mean = numpy.hstack(
            (asub_posterior_mean.squeeze(),isub_prior_mean.squeeze()))
        assert rotated_posterior_mean.shape[0] == num_dims
        rotated_posterior_covariance = block_diag(
            asub_posterior_covariance,isub_prior_covariance)
    else:
        # The posterior is just in a rotated space with
        # num_active_dims==num_dims
        rotated_posterior_mean = asub_posterior_mean
        rotated_posterior_covariance = asub_posterior_covariance

    # for debugging
    #rotated_posterior_density = NormalDensity(
    #    rotated_posterior_mean,covariance=rotated_posterior_covariance)

    # Return the mean and covariance of the posterior in the original space
    # Use W here (and not W1) because we have already produced the rotated
    # posterior in the rotated space with dimension num_dims
    posterior_mean = numpy.dot(W,rotated_posterior_mean)
    posterior_covariance = numpy.dot(W,numpy.dot(rotated_posterior_covariance,W.T))

    if return_rotated_data:
        return posterior_mean, posterior_covariance, rotated_posterior_mean, rotated_posterior_covariance
    else:
        return posterior_mean, posterior_covariance
