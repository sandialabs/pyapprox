from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from numpy.linalg import cholesky
from pyapprox.density import NormalDensity, \
    map_gaussian_samples_to_canonical_domain, map_from_canonical_gaussian,\
    plot_gaussian_contours
from pyapprox.bayesian_inference.laplace import \
    get_laplace_covariance_sqrt_operator
from pyapprox.models.algebraic_models import LogUnormalizedPosterior
import copy
from pyapprox.bayesian_inference.deprecated_laplace import get_stochastic_newton_proposal_mean_and_covar_chol_factor_from_function
from pyapprox.variable_transformations import \
    define_iid_random_variable_transformation
from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
from pyapprox.polynomial_sampling import get_oli_leja_samples, \
    total_degree_basis_generator
from functools import partial
from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
from pyapprox.orthogonal_least_interpolation import LeastInterpolationSolver
import matplotlib.pyplot as plt
def get_default_surrogate_opts():
    opts = {'precondition_type': 1,
            'num_candidate_samples': 1000,
            'condition_number_tol': 1e8,
            'surrogate_method': 'single',
            'use_all_samples': True}
    return opts

def evaluate_cross_validated_acceptance_ratios(
        proposal_sample_pce_numerator_vals,
        prev_sample_pce_numerator_vals):
    """
    Parameters
    ----------
    proposal_sample_pce_numerator_vals : vector (num_folds)
       The value of likelihood(x)*prior(x), for the proposal sample x,
       for each cross validation fold

    prev_sample_pce_numerator_vals : vector (num_folds)
       The value of likelihood(y)*prior(y), for the prev MCMC sample y,
       for each cross validation fold

    Returns
    -------
    accept_sample : integer [-1,0,1]
        -1 - update the surrogate
         0 - reject the sample
         1 - accept the sample
    """

    assert prev_sample_pce_numerator_vals.shape[0]==\
      proposal_sample_pce_numerator_vals.shape[0]
    num_folds = proposal_sample_pce_numerator_vals.shape[0]

    acceptance_val = np.random.uniform(0., 1)
    refine_surrogate = False
    for i in range(num_folds):
        acceptance_ratio = \
          proposal_sample_pce_numerator_vals[i]/prev_sample_pce_numerator_vals[i]
        accept_sample = int(accept_proposal_sample_using_ratio(
            acceptance_ratio, acceptance_val))
        # if acceptance flags disagree then surrogate is
        # to innaccurate and needs to be udpated
        if i>0 and accept_sample!=prev_accept_sample:
            accept_sample = -1
            break
        prev_accept_sample = accept_sample

    return accept_sample

def define_polynomial_chaos_expansion_for_mcmc(num_vars):
    """
    Initialize a canonical Hermite PCE.

    Parameters
    ----------
    num_vars : integer
        The number of random variables

    Returns
    -------
    pce : PolynomialChaosExpansion object
        A Hermite pce defined on the canonical Gaussian domain, i.e.
        for a standard Normal
    """
    var_trans = define_iid_random_variable_transformation(
            'gaussian',num_vars,{'mean':0,'variance':1}) 
    pce = PolynomialChaosExpansion()
    pce.configure({'poly_type':'hermite','var_trans':var_trans})
    return pce

def select_build_data(density, build_samples, build_values,
                      precondition_type, condition_number_tol):
    """
    density : Density object
        The density used to weight the build_samples.

    build_samples : matrix (num_vars x num_build_samples)
        All samples used to build surrogates thus far

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    precondition_type : integer [0,1,2] [optional]
        The type of preconditioning employed when building selecting
        the new build sample
        0 - No preconditioning
        1 - Precondition using weight function of orthogonal basis
        2 - Precondition using the Christoffel function

    condition_number_tol : double
        When this number is exceeded then the least interpolation
        will gracefully exit. The samples interpolated will
        unlikely be as large as max_num_samples.
    """

    num_vars = build_samples.shape[0]
    pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)

    # map the build_samples to the canonical domain
    num_samples = build_samples.shape[1]
    canonical_build_samples = map_gaussian_samples_to_canonical_domain(
        build_samples, density.mean, density.chol_factor)

    num_build_samples = build_samples.shape[1]
    oli_solver, permuted_samples = build_orthogonal_least_interpolant(
        pce, None, canonical_build_samples,
        max_num_samples=num_build_samples,
        precondition_type=precondition_type)

    permuted_samples = \
      restrict_least_interpolation_samples_using_condition_number(
          oli_solver, permuted_samples, condition_number_tol)
    num_permuted_samples = permuted_samples.shape[1]

    P = oli_solver.get_current_permutation()[:num_permuted_samples]
    permuted_values = build_values[P]

    return permuted_samples, permuted_values


def refine_mcmc_surrogate(forward_proposal_density, backward_proposal_density,
    negloglikelihood, build_samples, build_values, surrogate_opts, ranges):
    """
    Refine a MCMC surrogate, by adding one sample that keeps build samples well
    conditioned.

    Parameters
    ----------
    forward_proposal_density : Density object
        The density used to draw proposal sample.
        The forward_proposal quantifies the probability of
        transitioning from prev_sample to proposal_sample.

    backward_proposal_density : Density object
        The backward_proposal_density quantifies the probability of
        transitioning from proposal_sample to prev_sample.

    build_samples : matrix (num_vars x num_build_samples)
        The samples used to build the current surrogate

    negloglikelihood : Model object
        The negative loglikelihood function.
        Must have attributes evaluate, evaluate_set
        Must have call structure negloglikelihood.evaluate_set(eval_samples)

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    surrogate_opts : dict()
        keys - 'precondition_type','num_candidate_samples',
               'condition_number_tol', 'surrogate_method', 'use_all_samples'

    Returns
    -------
    build_samples : matrix (num_vars x num_build_samples+num_new_build_samples)
        The samples used to build the current surrogate + the new build samples

    build_values : vector (num_build_samples+num_new_build_samples)
        The values of the negative log likelihood at each of the build
        samples previous and new
    """

    num_candidate_samples = surrogate_opts['num_candidate_samples']
    precondition_type = surrogate_opts['precondition_type']

    # Randomly choose to refine from forward or backward proposal densities
    tmp = np.random.uniform(0.,1.)
    if tmp > 0.5: proposal_density = forward_proposal_density
    else: proposal_density = backward_proposal_density

    # Select new sample that keeps build_samples well conditioned
    # TODO: store oli_solver and only update in this loop
    new_build_sample = select_new_build_sample_for_posterior_surrogate(
        build_samples, proposal_density, ranges,
        num_candidate_samples=num_candidate_samples,
        precondition_type=precondition_type)

    # evaluate the simulation model
    new_build_value = negloglikelihood.evaluate(new_build_sample)
    new_build_value = np.asscalar(new_build_value)

    # update the build samples and values
    num_vars = build_samples.shape[0]
    build_samples = np.hstack(
        (build_samples,new_build_sample.reshape(num_vars,1)))
    build_values = np.append(build_values,new_build_value)
    assert ( (build_values.ndim==1) and
            (build_values.shape[0]==build_samples.shape[1]) )

    return build_samples, build_values

def accept_proposal_sample_using_surrogate(
    forward_proposal_density, backward_proposal_density,
    prior_density, negloglikelihood, build_samples, build_values,
    proposal_sample, prev_sample, surrogate_opts, ranges, force_refinement):
    """
    Determine acceptance of a MCMC proposal sample using surrogates
    accurate over a proposal distribution.

    Refine surrogate until accuracy is good enough to accurately determine
    acceptance.

    Parameters
    ----------
    forward_proposal_density : Density object
        The density used to draw proposal sample.
        The forward_proposal quantifies the probability of
        transitioning from prev_sample to proposal_sample.

    backward_proposal_density : Density object
        The backward_proposal_density quantifies the probability of
        transitioning from proposal_sample to prev_sample.

    build_samples : matrix (num_vars x num_build_samples)
        The samples used to build the current surrogate

    prior_density : Density object
        The prior density

    negloglikelihood : Model object
        The negative loglikelihood function.
        Must have attributes evaluate, evaluate_set
        Must have call structure negloglikelihood.evaluate_set(eval_samples)

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    prev_sample : vector (num_vars)
        The last accepted sample in the MCMC chain

    proposal_sample : vector (num_vars)
        A sample from the proposal distribution associated with
        prev_sample

    surrogate_opts : dict()
        keys - 'precondition_type','num_candidate_samples',
               'condition_number_tol', 'surrogate_method', 'use_all_samples'

    force_refinement : boolean
        False - use cross validation to determine if refinement is necessary
        True  - skip cross validation and refine the surrogate

    ranges : vector (2*num_vars)
        Lower and upper bounds stored as [ln1,ub1,lb2,ub2,...]

    Returns
    -------
    proposal_accepted : bool
       False - reject the proposal point
       True  - accept the proposal point

    build_samples : matrix (num_vars x num_build_samples+num_new_build_samples)
        The samples used to build the current surrogate + the new build samples

    build_values : vector (num_build_samples+num_new_build_samples)
        The values of the negative log likelihood at each of the build
        samples previous and new

    pce : PolynomialChaosExpansion object
        A surrogate built over the best conditioned set of build samples
        (previous and new)
    """

    num_vars = proposal_sample.shape[0]
    # Refine surrogate until cross validation estimates of
    # acceptance ratio all agree
    cntr = 0
    max_iters = 10
    if force_refinement:
        max_iters = 1
    for i in range(max_iters):
        if not force_refinement:
            accept_sample = check_acceptance_using_cross_validation(
                forward_proposal_density, backward_proposal_density,
                prior_density, negloglikelihood, build_samples, build_values,
                proposal_sample, prev_sample, surrogate_opts)
            if accept_sample>=0:
                break

        build_samples, build_values = refine_mcmc_surrogate(
            forward_proposal_density, backward_proposal_density,
            negloglikelihood, build_samples, build_values,
            surrogate_opts, ranges)

        # allow this to be striclty enforced or just to limit number of
        # samples added per MCMC iteration
        #assert i<max_iters, 'Surrogate is still to inaccurate'

    # Build the pce on all the build samples
    pce = build_mcmc_surrogate(
        build_samples, build_values, forward_proposal_density,
        surrogate_opts['precondition_type'],
        surrogate_opts['condition_number_tol'])

    # Compute acceptance. Even if acceptance has been determined
    # by cross validation, recompute here using all data. This will
    # allow a different error metric to be returned by
    # check_acceptance_using_cross_validation, e.g. instead of return consitent
    #acceptance of rejection may return error metric like in Conrad 2016.
    eval_samples = np.hstack(
        (proposal_sample.reshape(num_vars,1),
             prev_sample.reshape(num_vars,1)))
    # When evaluating the Bayes rule numerator
    # evaluate unnormalized likelihood because we are taking ratio
    # normalization factors will cancel
    ratio_vals = np.exp(
        -pce.evaluate_set(eval_samples)[:,0])*prior_density.pdf(eval_samples)
    acceptance_ratio = ratio_vals[0]/ratio_vals[1]
    accept_proposal = accept_proposal_sample_using_ratio(acceptance_ratio)

    return accept_proposal, build_samples, build_values, pce

def check_acceptance_using_cross_validation(
        forward_proposal_density, backward_proposal_density,
        prior_density, negloglikelihood,build_samples, build_values,
        proposal_sample, prev_sample, surrogate_opts):
    """
    Determine acceptance of a MCMC proposal sample using surrogates
    accurate over a proposal distribution.

    Cross validation is used to check accuracy of Bayes rule numerator values.
    If all cross validation values agree to accept or reject then that
    value is returned, otherwise samples of the negloglikelihood are added
    until all cross validation values do agree.

    Two surrogate methods are provided.
        'single' - Build a single surrogate centered at the prev_sample
                   and use this surrogate to evaluate the Bayes rule numerator
                   at both prev_sample and proposal_sample
        'local' - Build two surrogates, one centered at the prev_sample
                  and the other at proposal_sample. Use these surrogates
                  to evaluate the Bayes rule numerator at the prev_sample
                  and proposal_sample respectively

    Each method attempts to use as many build samples as possible until
    the condition number of the least interpolant basis exceeds the specified
    tolerance. When building two local surrogates all samples may be selected.
    In this situation use_all_samples is set to True and
    only a single surrogate is built.

    The user can specify on entry to the function to set use_all_samples=True
    so that if they no all samples will be selected then the selection
    procedure does not have to be peformed to save computational time.
    For example, if at the last iteration the number of surrogate build samples
    was equal to the total number of global build samples
    and the number of global build samples has not increased
    then no need to apply selection procedure just use all the global
    samples again.

    Parameters
    ----------
    forward_proposal_density : Density object
        The density used to draw proposal sample.
        The forward_proposal quantifies the probability of
        transitioning from prev_sample to proposal_sample.

    backward_proposal_density : Density object
        The backward_proposal_density quantifies the probability of
        transitioning from proposal_sample to prev_sample.

    build_samples : matrix (num_vars x num_build_samples)
        The samples used to build the current surrogate

    prior_density : Density object
        The prior density

    negloglikelihood : callable function
        The negative loglikelihood function.
        Must have call structure negloglikelihood(eval_samples)

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    prev_sample : vector (num_vars)
        The last accepted sample in the MCMC chain

    proposal_sample : vector (num_vars)
        A sample from the proposal distribution associated with
        prev_sample

    surrogate_opts : dict()
        keys - 'precondition_type','num_candidate_samples',
               'condition_number_tol', 'surrogate_method', 'use_all_samples'

    Returns
    -------
    accept_sample : integer [-1,0,1]
        -1 - update the surrogate
         0 - reject the sample
         1 - accept the sample
    """

    precondition_type = surrogate_opts['precondition_type']
    condition_number_tol = surrogate_opts['condition_number_tol']
    surrogate_method = surrogate_opts['surrogate_method']
    use_all_samples = surrogate_opts['use_all_samples']

    num_vars = proposal_sample.shape[0]
    eval_samples = np.hstack(
        (proposal_sample.reshape(num_vars,1),prev_sample.reshape(num_vars,1)))

    if not use_all_samples:
        proposal_sample_pce_build_samples, \
          proposal_sample_pce_build_values = select_build_data(
              backward_proposal_density, build_samples, build_values,
              precondition_type, condition_number_tol)
        #The user might think all samples will not be used but check
        # they are right
        if proposal_sample_pce_build_values.shape[1]==\
          build_samples.shape[1]: use_all_samples=True

    if not surrogate_method=='single' and not use_all_samples:
        prev_sample_pce_build_samples, prev_sample_pce_build_values = \
          select_build_data(
              forward_proposal_density, build_samples, build_values,
              precondition_type, condition_number_tol)
    else:
        proposal_sample_pce_build_samples = build_samples
        proposal_sample_pce_build_values = build_values
    # Build pce based on samples near proposal_sample.
    # Evaluate pce at both proposal_sample and prev_samples
    # as both these values may be needed
    proposal_sample_pce_numerator_vals = \
      generate_bayes_numerator_vals_using_cross_validation(
          proposal_sample_pce_build_samples, proposal_sample_pce_build_values,
          prior_density, negloglikelihood, eval_samples, surrogate_opts)

    if surrogate_method=='single' or use_all_samples:
        prev_sample_pce_numerator_vals = \
          proposal_sample_pce_numerator_vals[:,1]
    else:
        # Build pce based on samples near prev_sample. For clean code
        # evaluate pce at both proposal_sample and prev_samples
        # but we need only value prev_sample
        prev_sample_pce_numerator_vals = \
          generate_bayes_numerator_vals_using_cross_validation(
              prev_sample_pce_build_samples, prev_sample_pce_build_values,
              prior_density, negloglikelihood, eval_samples,
              surrogate_opts)[:,1]

    proposal_sample_pce_numerator_vals = \
      proposal_sample_pce_numerator_vals[:,0]

    accept_sample = evaluate_cross_validated_acceptance_ratios(
        proposal_sample_pce_numerator_vals,
        prev_sample_pce_numerator_vals)

    return accept_sample

def select_new_build_sample_for_posterior_surrogate_from_candidates(
    build_samples, proposal_density, canonical_candidate_samples,
    precondition_type=None):
    """
    Generate a well conditioned sample to be used to update a posterior
    adapted surrogate.

    The sample is selected by generating a Leja sequence using
    all the previous build samples and candidate samples from the
    proposal distribution.

    Parameters
    ----------
    build_samples : matrix (num_vars x num_samples)
        The samples used to build the current surrogate

    proposal_density :  ProposalDensity object
       Density used to generate candidate samples

    canonical_candidate_samples : matrix (num_vars x num_candidate_samples)
       The candidate samples used to generate the Leja sequence
       candidates must be in the canonical Gaussian domain, i.e. standard
       normal

    precondition_type : integer [0,1,2] [optional]
        The type of preconditioning employed when building selecting
        the new build sample
        0 - No preconditioning
        1 - Precondition using weight function of orthogonal basis
        2 - Precondition using the Christoffel function

    Returns
    -------
    new_build_sample : vector (num_vars)
       A well conditioned sample to be used to update a posterior
       adapted surrogate

    TODO: store oli_solver that interpolates current build samples
    and update this factorization
    """

    assert build_samples.shape[0] == proposal_density.mean.shape[0]

    num_vars, num_build_samples = build_samples.shape
    pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)

    # map the build_samples to the canonical domain
    num_samples = build_samples.shape[1]
    canonical_build_samples = map_gaussian_samples_to_canonical_domain(
        build_samples, proposal_density.mean,
        proposal_density.chol_factor)

    # Find the next point in the Leja sequence
    oli_solver, permuted_samples = build_orthogonal_least_interpolant(
        pce, canonical_build_samples, canonical_candidate_samples,
        max_num_samples=num_build_samples+1,
        precondition_type=precondition_type)
    
    if  oli_solver.get_initial_points_degenerate():
        # Some times build points will be degenerate. In such situations
        # take the first candidate sample chosen as the new build point
        P = oli_solver.get_current_permutation()
        I = np.where(P>num_build_samples)[0][0]
        new_build_sample = permuted_samples[:,I]
    else:
        # if build points are not degenerate then the last permuted sample
        # will be the first selected canidate sample
        new_build_sample = permuted_samples[:,-1]

    # map new_build_sample back to original sample domain
    new_build_sample = map_from_canonical_gaussian(
        new_build_sample.reshape(num_vars,1), proposal_density.mean,
        proposal_density.chol_factor)[:,0]
    return new_build_sample


def select_new_build_sample_for_posterior_surrogate(
    build_samples, proposal_density, ranges,
    num_candidate_samples=1000, precondition_type=None):
    """
    Generate a well conditioned sample to be used to update a posterior
    adapted surrogate.

    The sample is selected by generating a Leja sequence using
    all the previous build samples and candidate samples from the
    proposal distribution.

    Parameters
    ----------
    build_samples : matrix (num_vars x num_samples)
        The samples used to build the current surrogate

    proposal_density :  ProposalDensity object
       Density used to generate candidate samples

    num_candidate_samples : integer [1000] [optional]
       The number of candidate samples used to generate the
       Leja sequence

    precondition_type : integer [0,1,2] [optional]
        The type of preconditioning employed when building selecting
        the new build sample
        0 - No preconditioning
        1 - Precondition using weight function of orthogonal basis
        2 - Precondition using the Christoffel function

    ranges : vector (2*num_vars)
        Lower and upper bounds of the parameters,
        stored as [ln1,ub1,lb2,ub2,...]

    Returns
    -------
    new_build_sample : vector (num_vars)
       A well conditioned sample to be used to update a posterior
       adapted surrogate
    """

    # generate candidate samples in the canonical domain
    candidate_samples, canonical_candidate_samples = \
      proposal_density.generate_samples(num_candidate_samples,True)

    # reject any samples that are outside bounds
    I = admissible_samples(ranges, candidate_samples)
    canonical_candidate_samples = canonical_candidate_samples[:,I]

    return select_new_build_sample_for_posterior_surrogate_from_candidates(
        build_samples, proposal_density, canonical_candidate_samples,
        precondition_type=precondition_type)

def admissible_samples(ranges, samples):
    """
    Check which samples satisfy a set of linear constraints

    Parameters
    ----------
    ranges : vector (2*num_vars)
        Lower and upper bounds of the parameters,
        stored as [ln1,ub1,lb2,ub2,...]

    samples : matrix (num_vars x num_samples)
        The samples

    Returns
    -------
    idx : vector (num_admissible_samples)
        The indices of the admissible samples in the original samples array
    """
    num_vars, num_samples = samples.shape
    idx = np.empty((num_samples),int)
    cnt=0
    for i in range(num_samples):
        if np.all((samples[:,i]>=ranges[::2])&(samples[:,i]<=ranges[1::2])):
            idx[cnt]=i; cnt+=1
    idx = idx[:cnt]
    return idx

def accept_proposal_sample_using_ratio(
        acceptance_ratio, acceptance_val=None):
    """
    Determine whether to accept an MCMC sample based upon
    a pre-computed acceptance ratio.

    This function is deprecated as we base all decisions on the logarithm
    of the bayes numerators and proposal densities. See
    accept_proposal_sample_using_log_ratio.

    Parameters
    ----------
    acceptance_ratio : double
        The ratio
          lihood(x)*prior(x)/lihood(y)*prior(y)*back_proposal(y)/forw_proposal(x)
        for a MCMC proposal sample x and the prev MCMC sample y

    acceptance_val: double [optional]
        A value in [0,1] used to determine acceptance

    Returns
    -------
    proposal_accepted : bool
       False - reject the proposal point
       True  - accept the proposal point
    """
    if acceptance_val is None:
        acceptance_val = np.random.uniform(0., 1)
    if ( acceptance_ratio >= acceptance_val ):
        proposal_accepted = True
    else:
        proposal_accepted = False
    return proposal_accepted

def accept_proposal_sample_using_log_ratio(
        log_acceptance_ratio, log_acceptance_val=None):
    """
    Determine whether to accept an MCMC sample based upon the logarithm
    of a pre-computed acceptance ratio.

    Parameters
    ----------
    log_acceptance_ratio : double
        The logarithm of the ratio
          lihood(x)*prior(x)/lihood(y)*prior(y)*back_proposal(y)/forw_proposal(x)
        for an MCMC proposal sample x and the prev MCMC sample y

    log_acceptance_val: double [optional]
        A value in (-inf,0] used to determine acceptance

    Returns
    -------
    proposal_accepted : bool
       False - reject the proposal point
       True  - accept the proposal point
    """
    if log_acceptance_val is None:
        log_acceptance_val = np.log(np.random.uniform(0., 1))
    if ( log_acceptance_ratio >= log_acceptance_val ):
        proposal_accepted = True
    else:
        proposal_accepted = False
    return proposal_accepted

def build_mcmc_surrogate(build_samples, build_values, density,
                         precondition_type, condition_number_tol):
    """
    Parameters
    ----------
    build_samples : matrix (num_vars x num_build_samples)
        The samples used to build the current surrogate

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    density : Density object
        The density used to weight the build_samples.

    precondition_type : integer [0,1,2] [optional]
        The type of preconditioning employed when building selecting
        the new build sample
        0 - No preconditioning
        1 - Precondition using weight function of orthogonal basis
        2 - Precondition using the Christoffel function

    condition_number_tol : double
        When this number is exceeded then the least interpolation
        will gracefully exit. The samples interpolated will
        unlikely be as large as max_num_samples.

    Returns
    -------
    current_pce : PolynomialChaosExpansion object
        The pce that interpolates the build samples and values
    """
    # build interpolant
    num_vars, num_build_samples = build_samples.shape
    proposal_pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)

    # from scipy.stats import multivariate_normal
    # proposal_pce_precond_func = lambda samples: multivariate_normal.pdf(
    #     samples.T, mean=np.zeros(num_vars), cov=np.eye(num_vars))
    # leja_opts = {'proposal_pce_precond_func':proposal_pce_precond_func,
    #              'num_candidate_samples':1000}
    # precond_func = leja_opts.get('proposal_pce_precond_func')
    # num_candidate_samples = leja_opts['num_candidate_samples']
    # # replace following with samples from induced distribution
    # var_trans = define_iid_random_variable_transformation(
    #     'gaussian',num_vars,{'mean':0,'variance':1}) 
    # generate_candidate_samples = partial(
    #     generate_independent_random_samples,var_trans)
    # samples, data_structures = get_oli_leja_samples(
    #         proposal_pce, generate_candidate_samples,
    #         num_candidate_samples,num_build_samples,
    #         preconditioning_function=precond_func)

    oli_solver, permuted_samples = build_orthogonal_least_interpolant(
        proposal_pce, None, build_samples, max_num_samples=num_build_samples,
        precondition_type=precondition_type)
    
    P = oli_solver.get_current_permutation()
    permuted_values = build_values[P]
    current_pce = oli_solver.get_current_interpolant(
        permuted_samples, permuted_values)
    return current_pce

def generate_bayes_numerator_vals_using_cross_validation(
    build_samples, build_values, prior_density, negloglikelihood,
    eval_samples, surrogate_opts):
    """Compute Bayes numerator, e.g. likelihood(y)*prior(y) at a set of samples
    on a set of cross validation folds

    Parameters
    ----------
    build_samples : matrix (num_vars x num_build_samples)
        Samples used to build the surrogate

    build_values : vector (num_build_samples)
        The values of the negative log likelihood at each of the build
        samples

    prior_density : Density object
        The prior density

    negloglikelihood : callable function
        The negative loglikelihood function.
        Must have call structure misfit = negloglikelihood(eval_samples)
        misfit : matrix (num_residuals)

    eval_samples: matrix (num_vars x num_eval_samples)
        The samples at which the Bayes numerator must be evaluated

    Returns
    -------
    bayes_numerator_values : vector (num_eval_samples)
        The value of the Bayes rule numerator at eval_samples
    """
    if eval_samples.ndim==1:
        eval_samples = eval_samples.reshape(eval_samples.shape[0],1)

    num_vars, num_build_samples = build_samples.shape
    num_folds = min(10,num_build_samples)

    from polynomial_chaos_cpp import CrossValidationIterator
    cv_iterator = CrossValidationIterator()
    cv_iterator.set_seed( np.random.randint(1) )
    cv_iterator.set_num_folds( num_folds )
    cv_iterator.set_num_points( num_build_samples )
    cv_iterator.set_num_equations_per_point( 1 )

    canonical_build_samples = map_gaussian_samples_to_canonical_domain(
        build_samples, prior_density.mean, prior_density.chol_factor)

    num_eval_samples = eval_samples.shape[1]
    numerator_vals = np.empty((cv_iterator.num_folds(),num_eval_samples),
                                float)
    for it in range( cv_iterator.num_folds() ):
        training_indices, validation_indices = \
          cv_iterator.get_fold_indices( it )
        training_samples = canonical_build_samples[:,training_indices]
        training_values = build_values[training_indices]

        pce = build_mcmc_surrogate(
            training_samples,training_values,prior_density,
            surrogate_opts['precondition_type'],
            surrogate_opts['condition_number_tol'])

        # When evaluating the Bayes rule numerator
        # exvaluate unnormalized likelihood because we are taking ratio
        # normalization factors will cancel
        numerator_vals[it,:] = np.exp(
            -pce(eval_samples)[:,0])*prior_density.pdf(eval_samples)[:,0]

    return numerator_vals

def restrict_least_interpolation_samples_using_condition_number(
        oli_solver, samples, condition_number_tol):
    """Find the maximum number of samples such that the condition number
    of the least orthogonal basis is below the specfied tolerance.

    WARNING: condition numbers will unlikely grow monotonically, thus
    we cannot guarentee that the index found is the first or last time that
    the condition number transitions from below the tolerance to above
    To guarentee that the condition number of the set of samples
    returned will be less than the tolerance, we return the largest index
    found during the search that has a condition number less than tolerance.
    This may not be the largest set of samples that satifsy this condition
    however the number samples returned will be only one away from
    the maximum number for which the condition number is below the tolerance
    (local to the returned index).

    Parameters
    ----------
    oli_solver : LeastInterpolationSolver object
        Stores the state of the least interpolation after finding the basis
        that interpolates samples

    samples : matrix (num_vars, num_samples)
        The samples that have been interpolated

    condition_number_tol : double
        The maximum condition number allowed

    Returns
    -------
    restricted_samples : matrix (num_vars x num_well_conditioned_samples)
        The set of samples that when interpolated produced a least interpolation
        basis less than condition_number_tol.
    """
    L,U,H = oli_solver.get_current_LUH_factors()
    num_samples = L.shape[0]
    assert num_samples == samples.shape[1]

    n = num_samples
    cond = np.linalg.cond(np.dot(L[:n,:n],U[:n,:n]))
    if (cond<=condition_number_tol): return samples

    low_idx = 0
    high_idx = num_samples-1
    max_low_idx = low_idx
    min_high_idx = high_idx
    while high_idx-low_idx>1:
        n = (low_idx+high_idx)//2
        cond = np.linalg.cond(np.dot(L[:n,:n],U[:n,:n]))
        if (cond>condition_number_tol):
            min_high_idx = n
            cond_min_high_idx = cond
            high_idx = n
        else:
            max_low_idx = n
            cond_max_low_idx = cond
            low_idx = n+1
    return samples[:,:max_low_idx]

def build_orthogonal_least_interpolant(
        pce, init_samples, samples, max_num_samples=None,
        precondition_type=None):
    """
    Return the least orthogonal interpolant that interpolates
    a set of samples.

    Parameters
    ----------
    pce : PolynomialChaosExpansion object
        Defines the pce based used to interpolate the data. If
        basis_indices has been set then only those indices will
        be used to interpolate the data. If they have not been set
        then the least interpolation algorithm will determine
        the total-degree indices needed to interpolate the samples.

    init_samples : matrix (num_vars x num_init_samples)
        The samples that must be used first to build the interpolant

    samples : matrix (num_vars x num_init_samples)
        Samples that may be selected once all init_samples have been
        interpolated

    max_num_samples : integer [optional]
        The number of samples to interpolate.
        num_samples + num_init_samples >= max_num_samples
        max_num_samples >= num_init_samples

    precondition_type : integer [0,1,2] [optional]
        The type of preconditioning employed when building the
        interpolant
        0 - No preconditioning
        1 - Precondition using weight function of orthogonal basis
        2 - Precondition using the Christoffel function

    Returns
    -------
    oli_solver : LeastInterpolationSolver object
        The least basis that interpolates all init_samples
        and the specified number of samples

    permuted_samples : matrix (num_vars x max_num_samples)
        The samples interpolated by the least interpolation basis.
        The samples are in the order that they were selected by oli_solver.
        init_samples are chosen first, then the next set of samples selected
        greedily maximize the determinant of the least basis interpolation
        matrix
    """

    oli_opts = {'enforce_all_initial_points_used':True}
    oli_solver = LeastInterpolationSolver()
    oli_solver.configure(oli_opts)
    oli_solver.set_pce(pce)
    if precondition_type == 1:
        from scipy.stats import multivariate_normal
        precond_function = lambda samples: multivariate_normal.pdf(
            samples.T, mean=np.zeros(pce.num_vars()), cov=np.eye(pce.num_vars()))
    elif precondition_type == 2:
        raise Exception('christoffel not implemented')
    #else: do nothing

    
        oli_solver.set_preconditioning_function(precond_function)
    oli_solver.set_basis_generator(total_degree_basis_generator)
    if max_num_samples is None:
        max_num_samples=samples.shape[1]
    oli_solver.factorize(
        samples, init_samples, num_selected_pts=max_num_samples)
    permuted_samples = oli_solver.get_current_points()
    
    return oli_solver, permuted_samples

class MetropolisHastings(object):
    def __init__(self):
        self.ax = None
        self.build_samples = None
        self.build_values = None

        self.surrogate_opts = get_default_surrogate_opts()

    def build_initial_surrogate(self):
        # build Leja sequence for linear interpolation
        # drawing candidate samples from induced distribution of the prior
        
        precond_func = self.surrogate_opts['prior_pce_precond_func']
        prior_pce = self.surrogate_opts['prior_pce']
        generate_candidate_samples = \
            self.surrogate_opts['generate_candidate_samples_for_prior_pce']
        num_candidate_samples = self.surrogate_opts['num_candidate_samples']
        
        self.build_samples, data_structures = get_oli_leja_samples(
            prior_pce,generate_candidate_samples,
            num_candidate_samples,num_leja_samples,
            preconditioning_function=precond_func)

        oli_solver=data_structures[0]

        self.build_values = self.negloglikelihood(
            self.build_samples)

        self.surrogate = oli_solver.get_current_least_interpolant(
            self.build_samples, self.build_values)


    def initialize_proposal_densities(self,sample,covariance_chol_factor):
        forward_density = NormalDensity()
        backward_density = NormalDensity()
        if self.mcmc_method == 'MH':
            forward_density.set_mean(sample)
            forward_density.set_covariance(
                covariance_chol_factor=covariance_chol_factor)
        elif ( self.mcmc_method == 'SN' or self.mcmc_method=='SNAMP' or
                   self.mcmc_method == 'ISMAP' or self.mcmc_method=='SymSN'):
            self.update_proposal_density(sample,forward_density,True,True)
        else:
            raise Exception('Incorrect mcmc algorithm specified')
        if self.mcmc_method !='SN':
            backward_density = copy.deepcopy(forward_density)
        return forward_density, backward_density

    def update_proposal_density(self,sample,density,use_sn_proposal,
                                use_sn_mean):
        if use_sn_proposal:
            if self.use_surrogate:
                gradient = self.surrogate.gradient
                hessian = self.surrogate.hessian
            else:
                gradient = self.negloglikelihood.gradient
                hessian = self.negloglikelihood.hessian

            mean, covariance_chol_factor = \
              get_stochastic_newton_proposal_mean_and_covar_chol_factor_from_function(
                  gradient,hessian,sample,
                  self.prior_density.mean,
                  self.prior_density.chol_factor,
                  self.hessian_rank_trunc_tol,self.verbosity)
            # stochastic newton requires update of both mean covariance
            if use_sn_mean: density.set_mean(mean)
            else: density.set_mean(sample)
            density.set_covariance(
                covariance_chol_factor=covariance_chol_factor)
        else:
            # other methods only require update of mean
            density.set_mean(sample)

        return density

    def parse_options(self,opts):

        self.initial_sample = opts['initial_sample']
        self.num_vars = self.initial_sample.shape[0]
        self.num_samples = opts['num_samples']
        self.prior_density = opts['prior_density']
        self.negloglikelihood = opts['nll']
        self.log_posterior = LogUnormalizedPosterior(
            self.negloglikelihood,self.negloglikelihood.gradient_set,
            self.prior_density.pdf,self.prior_density.log_pdf,
            self.prior_density.log_pdf_gradient)
        if 'random_refinement_interval' in opts:
            self.random_refinement_interval=opts['random_refinement_interval']
        if 'use_surrogate' in opts:
            self.use_surrogate = opts['use_surrogate']
            if self.use_surrogate:
                self.prior_rv_trans = opts['prior_rv_trans']
        else: self.use_surrogate = False

        if 'mcmc_method' in opts:
            self.mcmc_method = opts['mcmc_method']
        else:
            self.mcmc_method == 'MH'

        if 'max_iterations' in opts:
            self.max_iterations = opts['max_iterations']
        else:
            self.max_iterations = 5*self.num_samples

        if 'init_proposal_covariance' in opts:
            self.init_proposal_covariance = opts['init_proposal_covariance']
            self.init_proposal_covariance.shape[0] == self.num_vars
        else:
            self.init_proposal_covariance = np.eye(self.num_vars)

        if 'bounds' in opts:
            bounds = opts['bounds']
            self.lower_bounds = bounds[::2]
            self.upper_bounds = bounds[1::2]
            self.ranges = bounds
        else:
            self.lower_bounds = [-np.inf]*self.num_vars
            self.upper_bounds = [np.inf]*self.num_vars
            self.ranges = np.inf*np.ones((2*self.num_vars),float)
            self.ranges[::2] = -np.inf

        if self.mcmc_method == 'SN':
            assert hasattr( self.negloglikelihood, 'gradient' )
            assert hasattr( self.negloglikelihood, 'hessian' )

        if 'plot' in opts:
            self.plot=opts['plot']
        else:
            self.plot = False

        if 'verbosity' in opts:
            self.verbosity = opts['verbosity']
        else:
            self.verbosity = 0

        if 'hessian_rank_trunc_tol' in opts:
            self.hessian_rank_trunc_tol = opts['hessian_rank_trunc_tol']
        else:
            self.hessian_rank_trunc_tol = 0.

        self.num_refinements = None
        if 'num_refinements' in opts:
            self.num_refinements = opts['num_refinements']

        self.plot_backward_proposal = opts.get('plot_backward_proposal',False)

    def force_refinement(self,iteration_count):
        """
        Determine whether to enforce refinement of surrogate.

        To prove MCMC on surrogate will converge assymptotically
        we need some amount of random refinement.

        iteration_count : integer
           The current iteration number in the MCMC alogrithm
           This helps trigger refinement of surrogate if it is being used
        """
        if self.random_refinement_interval is None: return False
        if iteration_count%self.random_refinement_interval==0: return True
        else: return False

    def accept_proposal_sample(
            self, prev_sample, proposal_sample, forward_proposal,
            backward_proposal, prev_log_posterior_val, iteration_count):
        """
        Determine whether to accept a proposal sample.

        If self.use_surrogate is True then the acceptance criteria
        is determined using a surrogate. In this case we enforce
        that the proposal distribution is symmetric so that
        the value of the proposal density (centered at prev_sample)
        at the proposal sample is equal to the value of the proposal
        density (centered at proposal_sample). This means that
        Stochastic Newton cannot be used with surrogate based MCMC

        Paramters
        ---------
        prev_sample : vector (num_vars)
            The last accepted sample in the MCMC chain

        proposal_sample : vector (num_vars)
            A sample from the proposal distribution associated with
            prev_sample

        forward_proposal_density : Density object
            The density used to draw proposal sample.
            The forward_proposal quantifies the probability of
            transitioning from prev_sample to proposal_sample.

        backward_proposal_density : Density object
            The backward_proposal_density quantifies the probability of
            transitioning from proposal_sample to prev_sample.

        prev_log_posterior_val : double
            The logarithm of the unnormalized posterior at prev_sample

        iteration_count : integer
           The current iteration number in the MCMC alogrithm
           This helps trigger refinement of surrogate if it is being used

        Returns
        -------
        accept_proposal : bool
            False - reject the proposal sample
            True - accept the proposal sample

        proposal_log_posterior_val : double
            The logarithm of the unnormalized posterior at proposal_sample
        """

        if self.use_surrogate:
            force_refinement_flag = self.force_refinement(iteration_count)
            accept_proposal,self.build_samples,self.build_values,\
                self.surrogate=\
                accept_proposal_sample_using_surrogate(
                forward_proposal, backward_proposal,
                self.prior_density, self.negloglikelihood,
                self.build_samples, self.build_values, proposal_sample,
                prev_sample, self.surrogate_opts, self.ranges,
                force_refinement_flag)
            proposal_log_posterior_val = None
        else:
            accept_proposal, proposal_log_posterior_val = \
              self.accept_proposal_using_simulation_model(
                proposal_sample, prev_sample, forward_proposal,
                backward_proposal, prev_log_posterior_val)

        return accept_proposal, proposal_log_posterior_val

    def accept_proposal_using_simulation_model(
        self, proposal_sample, prev_sample, forward_proposal_density,
        backward_proposal_density, prev_log_posterior_val):
        """
        Determine whether to accept a proposal sample using the simulation model.

        Paramters
        ---------
        prev_sample : vector (num_vars)
            The last accepted sample in the MCMC chain

        proposal_sample : vector (num_vars)
            A sample from the proposal distribution associated with
            prev_sample

        forward_proposal_density : Density object
            The density used to draw proposal sample.
            The forward_proposal quantifies the probability of
            transitioning from prev_sample to proposal_sample.

        backward_proposal_density : Density object
            The backward_proposal_density quantifies the probability of
            transitioning from proposal_sample to prev_sample.

        prev_log_posterior_val : double
            The logarithm of the unnormalized posterior at prev_sample

        Returns
        -------
        accept_proposal : bool
            False - reject the proposal sample
            True - accept the proposal sample

        proposal_log_posterior_val : double
            The logarithm of the unnormalized posterior at proposal_sample
        """

        proposal_log_posterior_val = self.log_posterior(
            proposal_sample[:,np.newaxis])[0,:]

        log_posterior_ratio = proposal_log_posterior_val-prev_log_posterior_val

        log_proposals_ratio = self.compute_log_proposals_ratio(
            forward_proposal_density, backward_proposal_density, prev_sample,
            proposal_sample)

        log_acceptance_ratio = min(0., log_posterior_ratio+log_proposals_ratio)

        accept_proposal = accept_proposal_sample_using_log_ratio(
            log_acceptance_ratio)

        return accept_proposal, proposal_log_posterior_val

    def compute_log_proposals_ratio(self, forward_proposal_density,
                                    backward_proposal_density, prev_sample,
                                    proposal_sample):
        """
        Compute the logarithm of the ratio of the forward and backward
        proposal densities, i.e.
            backward_proposal(y)/forward_proposal(x)
        for a MCMC proposal sample x and the prev MCMC sample y

        The ratio of proposals will be one and the logarithm of the rario zero
        if the proposal is symmetric.

        Notes
        -----
        The normalizing constants of the proposal distributions matter
        here as they will be different (different covariances) for
        asymmetric proposal distributions.

        For now proposal_dnesity.neg_log_pdf() is assumed only to be
        propotional to the pdf up to a constant (e.g. the normalizing
        constant of the gaussian distribution. Thus here compute the pdf
        values at x and y and then take logarithm, instead of simply
        computing
          -(backward_proposal.neg_log_pdf(y)-forward_proposal.neg_log_pdf(x))

        Paramters
        ---------
        forward_proposal_density : Density object
            The density used to draw proposal sample.
            The forward_proposal quantifies the probability of
            transitioning from prev_sample to proposal_sample.

        backward_proposal_density : Density object
            The backward_proposal_density quantifies the probability of
            transitioning from proposal_sample to prev_sample.

        prev_sample : vector (num_vars)
            The last accepted sample in the MCMC chain

        proposal_sample : vector (num_vars)
            A sample from the proposal distribution associated with
            prev_sample

        Returns
        -------
        log_proposals_ratio : float
            The logarithm of the ratio the forward and backward
            proposal densities.
        """
        forward_proposal_val=forward_proposal_density.pdf(proposal_sample)[0]
        backward_proposal_val=backward_proposal_density.pdf(prev_sample)[0]

        proposals_ratio = backward_proposal_val/forward_proposal_val
        log_proposals_ratio = np.log(proposals_ratio)
        return log_proposals_ratio

    def run(self,opts):
        """
        Run MCMC algorithm

        opts : dict()
            See parse_options() doc string
        """
        self.result = dict()
        self.parse_options(opts)

        if self.use_surrogate:
            self.build_initial_surrogate()

        # ---------------- #
        # Initialize chain #
        # ---------------- #
        prev_sample = self.initial_sample
        prev_log_posterior_val = self.log_posterior(
            self.initial_sample[:,np.newaxis])[0,:]
        num_accepted_proposals = 1

        sample_chain = np.empty((self.num_vars,self.max_iterations),
                                    np.double)
        sample_chain[:,0]= self.initial_sample

        proposal_covariance_chol_factor=cholesky(
            self.init_proposal_covariance)

        forward_proposal, backward_proposal = \
          self.initialize_proposal_densities(
            self.initial_sample, proposal_covariance_chol_factor)

        # ---------------- #
        # Run MCMC
        # ---------------- #

        for it in range( 1, self.max_iterations ):

            # Draw proposal samples until a sample is found that satisfies the
            # bounds on the random variables
            bounds_violated = True
            while bounds_violated:
                # Draw proposal sample
                proposal_sample=\
                  forward_proposal.generate_samples(1)[:,0]

                # check if proposal point is feasiable
                if (np.any( proposal_sample<self.lower_bounds) or
                    (np.any(proposal_sample>self.upper_bounds))):
                    bounds_violated = True
                else:
                    bounds_violated = False

            # update bacwkards proposal density to reflect new proposal
            # if SN update covariance and mean of backward proposal,
            # otherwise only update mean
            self.update_proposal_density(
                proposal_sample, backward_proposal,
                self.mcmc_method=='SN',self.mcmc_method=='SN')

            # Determine acceptance
            proposal_accepted, proposal_log_posterior_val = \
              self.accept_proposal_sample(
                  prev_sample, proposal_sample, forward_proposal,
                  backward_proposal, prev_log_posterior_val, it)

            self.plot_proposals(
                    proposal_sample, prev_sample,
                    forward_proposal, backward_proposal, it, proposal_accepted)


            if proposal_accepted:

                prev_sample = proposal_sample
                prev_log_posterior_val=proposal_log_posterior_val
                num_accepted_proposals += 1
                sample_chain[:,it] = proposal_sample.copy()

                # Udpate forward proposal density. If using a symmetric adaptive
                # MCMC algorithm udpate covariance of backwards proposal
                # to that of forward_proposal
                if self.mcmc_method != 'SymSN' and 'AM' not in self.mcmc_method:
                    # For SN and any method that does not adapt the proposal
                    # covariance the forward proposal is just the backward
                    # proposal from the previous step
                    forward_proposal = copy.deepcopy(backward_proposal)
                else:
                    # for symmetric adaptive MCMC methods update covariance
                    # of forward and backward proposal
                    self.update_proposal_density(
                        proposal_sample, forward_proposal,
                        self.mcmc_method=='SymSN',False)
                    # copy covariance of forward proposal
                    backward_proposal = copy.deepcopy(forward_proposal)
            else:
                sample_chain[:,it] = prev_sample.copy()

            if num_accepted_proposals == self.num_samples:
                break

        acceptance_ratio = \
          float(num_accepted_proposals)/float(it+1)
        self.result['acceptance_ratio'] = acceptance_ratio
        self.result['num_iter'] = it+1
        return sample_chain[:,:it+1]

    def plot_trace(self,sample_chain,show=False):
        num_vars, num_samples = sample_chain.shape
        indices = np.arange(1,num_samples+1,dtype=int)
        plt.clf()
        num_rows=max(1,num_vars//3)
        num_cols=min(num_vars,3)
        f,axs=plt.subplots(num_rows,num_cols,sharey=False,
                           figsize=(num_cols*8,6))
        for ii in range(num_vars):
            axs[ii].plot(indices,sample_chain[ii,:])
        if show:
            plt.show()

    def plot_chain(self,sample_chain,ranges=None,show=False):

        if sample_chain.shape[0] == 1:
            plt.plot(sample_chain[0,:],sample_chain[0,:]*0,'ok')
        elif sample_chain.shape[0] == 2:
            plt.plot(sample_chain[0,:],sample_chain[1,:],'ok')
        else:
            raise Exception('cannot plot chains with more than 2 dimensions')

        if ranges is not None:
            plt.xlim(ranges[0],ranges[1])
            plt.ylim(ranges[2],ranges[3])
        if show:
            plt.show()

    def plot_proposals(self,proposal_sample, prev_sample,
                       forward_proposal, backward_proposal, it,
                       proposal_accepted):

        if not self.plot or self.num_vars!=2:
            return
        if self.ax is None:
            f = plt.gcf()
            self.ax = plt.gca()
        color1 = 'gray' # forward proposal and proposal_sample if accepted
        color2 = 'black'# backward proposal and prev_sample
        color3 = 'red'  # forward proposal and proposal_sample if rejected
        if proposal_accepted:
            color=color1
        else:
            color=color3
        line1,=self.ax.plot([proposal_sample[0]],[proposal_sample[1]],
                            'o',color=color)


        ellips1 = plot_gaussian_contours(
            forward_proposal.mean,forward_proposal.chol_factor,ax=self.ax,
            color=color1)

        line2,=self.ax.plot(
            [prev_sample[0]],[prev_sample[1]],'o',color=color2)
        if self.plot_backward_proposal:
            ellips2 = plot_gaussian_contours(
                backward_proposal.mean,backward_proposal.chol_factor,
                ax=self.ax,color=color2)

        if self.use_surrogate:
            self.ax.plot(self.build_samples[0,:],self.build_samples[1,:],
                        's',color='red',ms=10)

        plt.draw()
        plt.savefig('mcmc-sampling-%d.png'%it)
        #plt.pause(.1)
        for ellip in ellips1:
            ellip.remove()
        if self.plot_backward_proposal:
            for ellip in ellips2:
                ellip.remove()
        line1.remove()
        line2.remove()

"""
FOR DRAM
def update_chain_covariance( samples, weights, samples_covar_prev,
                             samples_mean_prev, weights_sum_prev ):
    num_vars, num_samples = samples.shape

    if num_samples == 0:
        # nothing to do
        return samples_covar_prev, samples_mean_prev, weights_sum_prev

    if np.isscalar( weights ):
        weights = weights * np.ones( num_samples )

    samples_covar = np.zeros( ( num_vars, num_vars ) )
    if samples_covar_prev is None:
        weights_sum = np.sum( weights )
        samples_mean = np.sum(samples * weights, axis=1) / weights_sum
        assert samples_mean.shape[0] == num_vars
        if ( weights_sum > 1 ):
            for i in range( num_vars ):
                for j in range( i+1 ):
                    samples_i=samples[i,:].copy().reshape(
                        samples.shape[1],1) - samples_mean[i]
                    samples_j=samples[j,:].copy().reshape(
                        samples.shape[1],1) - samples_mean[j]
                    samples_j=(samples_j.squeeze()*weights).reshape(
                        samples.shape[1],1)
                    samples_covar[i,j] = \
                        dot((samples_i).T,(samples_j)) / (weights_sum-1.)
                    samples_covar[j,i] = samples_covar[i,j]
    else:
        for i in range( num_samples ):
            samples_i = samples[:,i]
            weights_sum = weights[i]
            samples_mean = samples_mean_prev + \
                weights_sum/(weights_sum+weights_sum_prev)*\
                (samples_i-samples_mean_prev)
            temp=(samples_i.squeeze()-samples_mean_prev.squeeze()).reshape(
                1,samples.shape[0])
            samples_covar = samples_covar_prev + \
                weights_sum / (weights_sum+weights_sum_prev-1.) * \
                (weights_sum_prev/(weights_sum+weights_sum_prev)*
                 dot(temp.T,temp)-samples_covar_prev)
            weights_sum += weights_sum_prev
            samples_mean_prev = samples_mean
            samples_covar_prev = samples_covar
            weights_sum_prev = weights_sum

    return samples_covar, samples_mean, weights_sum
"""

#DTRCON Estimates the reciprocal of the condition number of a triangular matrix, in either the 1-norm or the infinity-norm.

#DGECON Estimates the reciprocal of the condition number of a general matrix, in either the 1-norm or the infinity-norm, using the LU factorization computed by DGETRF. assumes L has unit diagonal

# USE autocorrelation to estimate effective number of samples and select these
# number of samples evenly from the chain after burnin in
