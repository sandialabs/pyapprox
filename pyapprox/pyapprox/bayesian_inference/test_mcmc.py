import unittest
from pyapprox.bayesian_inference.mcmc import *
from pyapprox.bayesian_inference.laplace import *
from pyapprox.density import *
import numpy as np
from pyapprox.bayesian_inference.diagnostics import *
from pyapprox.models.algebraic_models import *

def set_mcmc_options(negloglikelihood, prior_density, num_samples,
                     initial_sample, mcmc_method, plot, verbosity,
                     use_surrogate, random_refinement_interval,
                     max_iterations, init_proposal_covariance,
                     bounds, num_refinements):
    mcmc_opts = dict()
    mcmc_opts['nll']=negloglikelihood
    mcmc_opts['prior_density'] = prior_density
    mcmc_opts['num_samples'] = num_samples
    mcmc_opts['initial_sample'] = initial_sample
    mcmc_opts['mcmc_method'] = mcmc_method
    mcmc_opts['plot'] = plot
    mcmc_opts['verbosity'] = verbosity
    if random_refinement_interval is not None:
        mcmc_opts['random_refinement_interval']=random_refinement_interval
    mcmc_opts['use_surrogate']=use_surrogate
    if use_surrogate:
        mcmc_opts['prior_rv_trans'] = prior_density.rv_trans
    if max_iterations is not None:
        mcmc_opts['max_iterations'] = max_iterations
    if init_proposal_covariance is not None:
        mcmc_opts['init_proposal_covariance'] = init_proposal_covariance
    if bounds is not None:
        mcmc_opts['bounds'] = bounds
    if num_refinements is not None:
        mcmc_opts['num_refinements'] = num_refinements
    return mcmc_opts

def help_mcmc(
        negloglikelihood, prior_density, mcmc_method, num_samples,
        use_surrogate=True, plot=False, verbosity=0,
        random_refinement_interval=None, max_iterations=None,
        init_proposal_covariance=None,bounds=None, num_refinements=None,
        optimizer_initial_sample=None):

    # set initial sample to be map point
    if optimizer_initial_sample is None:
        optimizer_initial_sample = prior_density.mean
    log_unnormalized_posterior = LogUnormalizedPosterior(
        negloglikelihood, negloglikelihood.gradient_set, prior_density.pdf, 
        prior_density.log_pdf, prior_density.log_pdf_gradient)
    initial_sample, obj_min = find_map_point(
        log_unnormalized_posterior, optimizer_initial_sample)
    print(initial_sample, obj_min)

    # set mcmc options
    mcmc_opts = set_mcmc_options(
        negloglikelihood, prior_density, num_samples, initial_sample,
        mcmc_method, plot, verbosity, use_surrogate, random_refinement_interval,
        max_iterations, init_proposal_covariance, bounds,
        num_refinements)

    mcmc_sampler = MetropolisHastings()

    # Overlay plot of unnormalized posterior_density if requested
    if plot and prior_density.mean.shape[0]==2:
        function=lambda x: np.exp(log_unnormalized_posterior(x))
        plot_limits=prior_density.plot_limits
        #from utilities.visualisation import imshow_from_function
        #imshow_from_function(function,plot_limits,show=False)
        import matplotlib.pyplot as plt
        import matplotlib
        from pyapprox.visualization import get_meshgrid_function_data
        X,Y,Z = get_meshgrid_function_data(function, plot_limits, 100, qoi=0)
        fig = plt.figure()
        ax = fig.gca()
        ax.contourf(
            X, Y, Z, levels=np.linspace(Z.min(),Z.max(),30),
            cmap=matplotlib.cm.coolwarm)
        print(np.log(Z).min(), np.log(Z).max())

        mcmc_sampler.ax=ax

    # run mcmc
    mcmc_chain = mcmc_sampler.run(mcmc_opts)
    return mcmc_sampler, mcmc_chain

skiptest = unittest.skip(
    "these tests are being kept around in case needed in the future")

@skiptest
class TestMCMC(unittest.TestCase):

    def test_exponential_quartic(self):
        negloglikelihood = ExponentialQuarticLogLikelihoodModel(modify=True)
        from pyapprox.bayesian_inference.laplace import \
            directional_derivatives
        e_vectors = np.eye(2)
        sample = np.ones((2,1))

        value_at_sample = negloglikelihood(sample)[0,:]
        derivs = directional_derivatives(
            negloglikelihood, sample, value_at_sample, e_vectors, fd_eps=1e-7)
        vectors = np.random.normal(0.,1.,(2,1))
        dir_derivs = directional_derivatives(
            negloglikelihood, sample, value_at_sample, vectors, fd_eps=1e-7)
        assert np.allclose(np.dot(derivs.T,vectors),dir_derivs)

        def gradient_set(x):
            grads = np.empty_like(x)
            for ii in range(x.shape[1]):
                grads[:,ii]=negloglikelihood.gradient(x[:,ii])
            return grads.T
        gradient_at_sample = gradient_set(sample)[0,:]

        derivs = directional_derivatives(
            gradient_set, sample, gradient_at_sample, e_vectors, fd_eps=1e-7)
        vectors = np.random.normal(0.,1.,(2,1))
        dir_derivs = directional_derivatives(
            gradient_set, sample, gradient_at_sample, vectors, fd_eps=1e-7)

        assert np.allclose(np.dot(derivs.T,vectors),dir_derivs.T)

    def setUp( self ):
        np.random.seed()
        #np.seterr(all='raise')

    def help_mcmc_gaussian_posterior(
            self, num_vars, linear_model_rank, num_qoi, noise_covariance,
            prior_mean, prior_covariance, init_proposal_covariance,
            mcmc_method):
        obs = np.random.normal(0.,1.,(num_qoi))
        noise_covariance_inv = np.linalg.inv(noise_covariance)
        negloglikelihood = QuadraticMisfitModel(
            num_vars,linear_model_rank,num_qoi,obs,
            noise_covariance=noise_covariance)

        prior_hessian = np.linalg.inv(prior_covariance)
        prior_density = NormalDensity(
            prior_mean, covariance=prior_covariance)

        posterior_mean, posterior_covariance = \
          laplace_posterior_approximation_for_linear_models(
              negloglikelihood.Amatrix,prior_mean,prior_hessian,
              noise_covariance_inv,obs)

        use_surrogate=False; plot=False
        covariance_errors = []
        num_trials = 3
        for num_samples in [1000,10000]:#,100000]:
            for ii in range(num_trials):
                mcmc_sampler, mcmc_chain = help_mcmc(
                    negloglikelihood, prior_density, mcmc_method, num_samples,
                    use_surrogate=use_surrogate, plot=plot, verbosity=1,
                    init_proposal_covariance=init_proposal_covariance,
                    random_refinement_interval=20,
                    max_iterations=10*num_samples, bounds=None,
                    num_refinements=1)
                plot_auto_correlation(mcmc_chain)
                eff_sample_size = effective_sample_size(mcmc_chain)
                print('effective_sample_size', eff_sample_size)
                
                #mcmc_sampler.plot_trace(mcmc_chain)
                filtered_chain = mcmc_chain[:,min(10000,num_samples//10):]
                mcmc_samples_covariance = np.cov(filtered_chain)
                mean_error=np.linalg.norm(
                    posterior_mean-filtered_chain.mean(axis=1))
                covariance_error = np.linalg.norm(
                    posterior_covariance-mcmc_samples_covariance)
                print('mean and covariance errors',mean_error,covariance_error)
                covariance_errors.append(covariance_error)
            traces = get_random_traces(mcmc_chain,4,10)
            reshaped_traces = np.empty(
                (traces.shape[0],traces.shape[2],traces.shape[1]),float)
            for i in range(traces.shape[0]):
                reshaped_traces[i,:,:]=traces[i,:,:].T
            print('gelman_rubin',gelman_rubin(traces))
            print('acceptance_ratio',mcmc_sampler.result['acceptance_ratio'])
            print('num_iter',mcmc_sampler.result['num_iter'])

    def test_mcmc_gaussian_posterior_stochastic_newton(self):
        num_vars = 2; linear_model_rank = 2; num_qoi = 2
        noise_covariance = np.eye(num_qoi)
        prior_mean = np.ones((num_vars),float)*0.
        variance = 1.
        prior_covariance = np.eye(num_vars)*variance
        init_proposal_covariance = prior_covariance
        mcmc_method = 'SN';

        self.help_mcmc_gaussian_posterior(
            num_vars, linear_model_rank, num_qoi, noise_covariance,
            prior_mean, prior_covariance, init_proposal_covariance, mcmc_method)

    def test_mcmc_gaussian_posterior_metropolis_hastings(self):
        num_vars = 2; linear_model_rank = 2; num_qoi = 2
        noise_covariance = np.eye(num_qoi)
        prior_mean = np.ones((num_vars),float)*0.
        variance = 1.
        prior_covariance = np.eye(num_vars)*variance
        init_proposal_covariance = prior_covariance
        mcmc_method = 'MH';

        self.help_mcmc_gaussian_posterior(
            num_vars, linear_model_rank, num_qoi, noise_covariance,
            prior_mean, prior_covariance, init_proposal_covariance, mcmc_method)

    def test_mcmc_rosenbrock(self):
        num_vars = 2
        negloglikelihood = ExponentialQuarticLogLikelihoodModel(modify=False)
        #negloglikelihood = RosenbrockLogLikelihoodModel()
        prior_var_trans = define_iid_random_variable_transformation(
            'uniform',num_vars,{'range':[-1,1]}) 
        ranges=prior_var_trans.get_variable_ranges()

        prior_mean = 0.5*(ranges[::2]+ranges[1::2])
        variance = (ranges[1]-ranges[0])**2/12.
        prior_covariance = np.eye(num_vars)*variance
        prior_density = UniformDensity(ranges)
        prior_density.rv_trans = prior_var_trans

        mcmc_method = 'SN'; use_surrogate=False;
        plot=True
        #plot = False
        num_samples = 4
        mcmc_sampler, mcmc_chain = help_mcmc(
            negloglikelihood, prior_density, mcmc_method, num_samples,
            use_surrogate=use_surrogate, plot=plot, verbosity=0,
            init_proposal_covariance=np.eye(num_vars)*.5,
            random_refinement_interval=20,
            max_iterations=10*num_samples, bounds=ranges, num_refinements=1,
            optimizer_initial_sample=[1,1])


        mcmc_sampler.plot_chain(mcmc_chain,prior_density.plot_limits)
        mcmc_sampler.plot_trace(mcmc_chain)

        import glob
        filenames = glob.glob('mcmc-sampling*.png')
        for filename in filenames:
            os.remove(filename)

    def test_admissible_constraints(self):
        num_vars = 2
        samples = np.array([[4,1],[0,0],[-2,-2,],[-2,3]]).T
        prior_var_trans = define_iid_random_variable_transformation(
            'uniform',num_vars,{'range':[-3,3]}) 
        ranges=prior_var_trans.get_variable_ranges()
        idx = admissible_samples(ranges, samples)
        assert np.allclose(idx,[1,2,3])
        
@skiptest
class TestSurrogateBasedMCMCFunctions(unittest.TestCase):
    def setUp( self ):
        np.random.seed(2)

    def test_restrict_least_interpolation_samples_using_condition_number(self):

        num_vars = 3
        num_samples = 101
        pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)
        init_samples = None
        build_samples = np.random.normal(0.,1.,(num_vars,num_samples))
        oli_solver, permuted_samples = build_orthogonal_least_interpolant(
            pce, init_samples, build_samples, max_num_samples=None,
            precondition_type=None)

        L,U,H = oli_solver.get_current_LUH_factors()

        for condition_number_tol in [100,1000,2000,300,np.finfo(float).max]:
            restricted_samples = \
            restrict_least_interpolation_samples_using_condition_number(
                oli_solver, permuted_samples, condition_number_tol)

            n = restricted_samples.shape[1]
            if condition_number_tol == np.finfo(float).max:
                assert n==build_samples.shape[1]
            else:
                cond1 = np.linalg.cond(np.dot(L[:n,:n],U[:n,:n]))
                assert cond1 <= condition_number_tol
                cond2 = np.linalg.cond(np.dot(L[:n+1,:n+1],U[:n+1,:n+1]))
                if (cond2 <= condition_number_tol):
                    cond3 = np.linalg.cond(
                        np.dot(L[:n+2,:n+2],U[:n+2,:n+2]))
                    assert cond3 > condition_number_tol

    def test_select_build_data(self):
        num_vars = 3;
        num_samples = 100
        mean = np.ones((num_vars),float)
        A = np.random.normal(0.,1.,(num_vars,num_vars))
        covariance = np.dot(A.T,A)
        density = NormalDensity(mean=mean,covariance=covariance)

        # generated correlated samples
        canonical_build_samples = np.random.normal(
            0.,1.,(num_vars,num_samples))
        build_samples = map_from_canonical_gaussian(
            canonical_build_samples, density.mean, density.chol_factor)
        build_values = np.arange(num_samples,dtype=float)

        precondition_type = None # pce weight function
        condition_number_tol=1/np.sqrt(np.finfo(float).eps)
        selected_build_samples, selected_build_values=select_build_data(
            density, build_samples, build_values,
            precondition_type, condition_number_tol)

        pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)
        oli_solver, permuted_samples = build_orthogonal_least_interpolant(
            pce, None, selected_build_samples,
            max_num_samples=None,precondition_type=precondition_type)
        P = oli_solver.get_current_permutation()
        assert np.allclose(P,np.arange(P.shape[0]))
        L,U,H = oli_solver.get_current_LUH_factors()
        cond = np.linalg.cond(np.dot(L,U))
        assert cond<=condition_number_tol


    def test_evaluate_cross_validated_acceptance_ratios(self):
        n = 20
        proposal_sample_pce_numerator_vals = np.ones(n) * 0.5
        prev_sample_pce_numerator_vals = np.ones(n)

        accept = evaluate_cross_validated_acceptance_ratios(
            proposal_sample_pce_numerator_vals,
            prev_sample_pce_numerator_vals)
        assert accept >= 0

        proposal_sample_pce_numerator_vals = np.linspace(0.,1.,n)
        prev_sample_pce_numerator_vals = np.ones(n)

        accept = evaluate_cross_validated_acceptance_ratios(
            proposal_sample_pce_numerator_vals,
            prev_sample_pce_numerator_vals)
        assert accept < 0


    def test_select_new_build_sample_for_posterior_surrogate(self):
        """
        Given a set of 'good' samples select from a set of points all
        clustered next to a sample in the 'good' set except one sample.
        The outlier should be chosen.
        """
        # Generate a good set of samples
        num_vars = 3;
        num_samples = 100
        mean = np.ones((num_vars),float)
        A = np.random.normal(0.,1.,(num_vars,num_vars))
        covariance = np.dot(A.T,A)
        density = NormalDensity(mean=mean,covariance=covariance)
        canonical_build_samples = np.random.normal(
            0.,1.,(num_vars,num_samples))
        candidate_build_samples = map_from_canonical_gaussian(
            canonical_build_samples, density.mean, density.chol_factor)

        precondition_type = 1
        pce = define_polynomial_chaos_expansion_for_mcmc(num_vars)
        oli_solver, permuted_samples = build_orthogonal_least_interpolant(
            pce, None, canonical_build_samples, max_num_samples=10,
            precondition_type=precondition_type)
        P = oli_solver.get_current_permutation()
        build_samples = candidate_build_samples[:,P[:-1]]

        # Generate a poor set of candidates, except an outlier which is 'good'
        candidate_samples = np.tile(
            build_samples[:,:1],(1,10))+np.random.normal(
                0.,0.001,(num_vars,10))
        candidate_samples[:,-1] = canonical_build_samples[:,P[-1]]

        proposal_density = density
        new_build_sample = \
          select_new_build_sample_for_posterior_surrogate_from_candidates(
            build_samples, proposal_density, candidate_samples,
              precondition_type=precondition_type)

        # Check the outlier is selected
        assert np.allclose(new_build_sample,candidate_build_samples[:,P[-1]])


    def test_accept_proposal_sample_using_surrogate(self):
        assert False, 'test not implemented'

    def test_generate_bayes_numerator_vals_using_cross_validation(self):
        num_vars = 2; num_samples = 19;
        noise_covariance_det = 1.
        prior_density = NormalDensity(mean=np.zeros(num_vars),covariance=1.)
        canonical_build_samples = np.random.normal(
            0.,1.,(num_vars,num_samples))
        build_samples = map_from_canonical_gaussian(
            canonical_build_samples, prior_density.mean,
            prior_density.chol_factor)

        negloglikelihood_model = RosenbrockLogLikelihoodModel()
        negloglikelihood = negloglikelihood_model
        build_values = negloglikelihood(build_samples)

        surrogate_opts = get_default_surrogate_opts()
        eval_samples = np.array([[1.,1.],[0.95,1.]]).T
        numerator_vals = generate_bayes_numerator_vals_using_cross_validation(
            build_samples, build_values, prior_density, negloglikelihood,
            eval_samples, surrogate_opts)

        assert numerator_vals.shape[0] == min(num_samples,10)
        # The misfit is a 4th degree polynomial so using a large number of
        # samples ~> 18
        # the pce will be exact on all folds so check that numerator values
        # are the same on all folds
        if num_samples > 18:
            assert np.all(
                np.absolute(numerator_vals-numerator_vals[0])<1e-4)
        else:
            assert not np.all(
                np.absolute(numerator_vals-numerator_vals[0])<1e-4)

if __name__ == '__main__':
    #tests=unittest.makeSuite(TestSurrogateBasedMCMCFunctions, 'test' )
    # tests=unittest.makeSuite(TestMCMC, 'test' )
    # tests=unittest.makeSuite(TestMCMC, 'test_mcmc_rosenbrock' )
    #tests=unittest.makeSuite(TestSurrogateBasedMCMCFunctions, 'test_generate')
    #all_tests = unittest.TestSuite([tests])
    #runner = unittest.TextTestRunner()
    #runner.run( all_tests )

    mcmc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestMCMC)
    unittest.TextTestRunner(verbosity=2).run(mcmc_test_suite)


#pymc located at
#~/miniconda2/pkgs/pymc-2.3.6-np111py27_2/lib/python2.7/site-packages/pymc/tests/
