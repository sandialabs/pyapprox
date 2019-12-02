import unittest
from pyapprox.iterative_hard_thresholding import *
from pyapprox.function_train import *
from pyapprox.orthonormal_polynomials_1d import jacobi_recurrence
from functools import partial
import copy

class TestIHT(unittest.TestCase):

    def test_gaussian_matrix(self):
        num_samples=30
        sparsity=3
        num_terms=30
        Amatrix = np.random.normal(
            0.,1.,(num_samples,num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        I = np.random.permutation(num_terms)[:sparsity]
        true_sol[I]=np.random.normal(0.,1.,(sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix,true_sol)

        approx_eval = lambda x: np.dot(Amatrix,x)
        apply_approx_adjoint_jacobian = lambda x,y: -np.dot(Amatrix.T,y)
        project = partial(s_sparse_projection,sparsity=sparsity)

        initial_guess = np.zeros_like(true_sol)
        tol = 1e-5
        max_iter=100
        result = iterative_hard_thresholding(
            approx_eval, apply_approx_adjoint_jacobian, project,
            obs, initial_guess, tol, max_iter)
        sol = result[0]
        assert np.allclose(true_sol,sol,atol=10*tol)

    def test_random_function_train(self):
        np.random.seed(5)
        num_vars = 2
        degree = 5
        rank = 2

        sparsity_ratio = 0.2
        sample_ratio   = .9
        
        ranks = rank*np.ones(num_vars+1,dtype=np.uint64)
        ranks[0]=1; ranks[-1]=1

        alpha=0; beta=0; 
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha,beta=beta,probability=True)

        ft_data = generate_random_sparse_function_train(
            num_vars,rank,degree+1,sparsity_ratio)
        true_sol = ft_data[1]

        num_ft_params = true_sol.shape[0]
        num_samples = int(sample_ratio*num_ft_params)
        samples = np.random.uniform(-1.,1.,(num_vars,num_samples))

        function = lambda samples: evaluate_function_train(
            samples,ft_data,recursion_coeffs)
        
        values = function(samples)

        assert np.linalg.norm(values)>0, (np.linalg.norm(values))

        num_validation_samples=100
        validation_samples = np.random.uniform(
            -1.,1.,(num_vars,num_validation_samples))
        validation_values = function(validation_samples)

        zero_ft_data = copy.deepcopy(ft_data)
        zero_ft_data[1]=np.zeros_like(zero_ft_data[1])
        # DO NOT use ft_data in following two functions.
        # These function only overwrites parameters associated with the
        # active indices the rest of the parameters are taken from ft_data.
        # If ft_data is used some of the true data will be kept and give
        # an unrealisticaly accurate answer
        approx_eval = partial(modify_and_evaluate_function_train,samples,
                              zero_ft_data,recursion_coeffs,None)

        apply_approx_adjoint_jacobian = partial(
            apply_function_train_adjoint_jacobian,samples,zero_ft_data,
            recursion_coeffs,1e-3)
        
        sparsity = np.where(true_sol!=0)[0].shape[0]
        print(('sparsity',sparsity,'num_samples',num_samples))
        
        # sparse
        project = partial(s_sparse_projection,sparsity=sparsity)
        # non-linear least squres
        #project = partial(s_sparse_projection,sparsity=num_ft_params)

        # use uninormative initial guess
        #initial_guess = np.zeros_like(true_sol)

        # use linear approximation as initial guess
        linear_ft_data =  ft_linear_least_squares_regression(
            samples,values,degree,perturb=None)
        initial_guess = linear_ft_data[1]
        
        # use initial guess that is close to true solution
        # num_samples required to obtain accruate answer decreases signficantly
        # over linear or uniformative guesses. As size of perturbation from
        # truth increases num_samples must increase
        initial_guess = true_sol.copy()+np.random.normal(0.,.1,(num_ft_params))
        
        tol = 5e-3
        max_iter=1000
        result = iterative_hard_thresholding(
            approx_eval, apply_approx_adjoint_jacobian, project,
            values[:,0], initial_guess, tol, max_iter, verbosity=1)
        sol = result[0]
        residnorm = result[1]

        recovered_ft_data=copy.deepcopy(ft_data)
        recovered_ft_data[1]=sol
        ft_validation_values = evaluate_function_train(
            validation_samples,recovered_ft_data,recursion_coeffs)

        validation_error = np.linalg.norm(
            validation_values-ft_validation_values)
        rel_validation_error=validation_error/np.linalg.norm(validation_values)
        # compare relative error because exit condition is based upon
        # relative residual
        assert rel_validation_error<10*tol, rel_validation_error

        # interestingly enough the error in the function can be low
        # but the error in the ft parameters can be large
        #assert np.allclose(true_sol,sol,atol=10*tol)

class TestOMP(unittest.TestCase):
    def test_gaussian_matrix(self):
        num_samples=30
        sparsity=5
        num_terms=30
        Amatrix = np.random.normal(
            0.,1.,(num_samples,num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        I = np.random.permutation(num_terms)[:sparsity]
        true_sol[I]=np.random.normal(0.,1.,(sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix,true_sol)

        approx_eval = lambda x: np.dot(Amatrix,x)
        apply_approx_adjoint_jacobian = lambda x,y: -np.dot(Amatrix.T,y)
        least_squares_regression = \
            lambda indices, initial_guess: np.linalg.lstsq(
                Amatrix[:,indices],obs,rcond=None)[0]

        initial_guess = np.zeros_like(true_sol)
        tol = 1e-5
        active_indices = None
        
        result = orthogonal_matching_pursuit(
            approx_eval, apply_approx_adjoint_jacobian,
            least_squares_regression,
            obs, active_indices, num_terms, tol, sparsity)
        sol = result[0]
        assert np.allclose(true_sol,sol,atol=10*tol)

    def test_gaussian_matrix_with_initial_active_indices(self):
        num_samples=30
        sparsity=5
        num_terms=30
        Amatrix = np.random.normal(
            0.,1.,(num_samples,num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        I = np.random.permutation(num_terms)[:sparsity]
        true_sol[I]=np.random.normal(0.,1.,(sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix,true_sol)

        approx_eval = lambda x: np.dot(Amatrix,x)
        apply_approx_adjoint_jacobian = lambda x,y: -np.dot(Amatrix.T,y)
        least_squares_regression = \
            lambda indices, initial_guess: np.linalg.lstsq(
                Amatrix[:,indices],obs,rcond=None)[0]

        initial_guess = np.zeros_like(true_sol)
        tol = 1e-5
        # use first three sparse terms
        active_indices = I[:3]
        
        result = orthogonal_matching_pursuit(
            approx_eval, apply_approx_adjoint_jacobian,
            least_squares_regression,
            obs, active_indices, num_terms, tol, sparsity)
        sol = result[0]
        assert np.allclose(true_sol,sol,atol=10*tol)

    def test_sparse_function_train(self):
        np.random.seed(5)
        num_vars = 2
        degree = 5
        rank = 2

        tol = 1e-5
        
        sparsity_ratio = 0.2
        sample_ratio   = 0.6
        
        ranks = rank*np.ones(num_vars+1,dtype=np.uint64)
        ranks[0]=1; ranks[-1]=1

        alpha=0; beta=0; 
        recursion_coeffs = jacobi_recurrence(
            degree+1, alpha=alpha,beta=beta,probability=True)

        ft_data = generate_random_sparse_function_train(
            num_vars,rank,degree+1,sparsity_ratio)
        true_sol = ft_data[1]

        num_ft_params = true_sol.shape[0]
        num_samples = int(sample_ratio*num_ft_params)
        samples = np.random.uniform(-1.,1.,(num_vars,num_samples))

        
        function = lambda samples: evaluate_function_train(
            samples,ft_data,recursion_coeffs)
        #function = lambda samples: np.cos(samples.sum(axis=0))[:,np.newaxis]

        values = function(samples)
        print(values.shape)

        assert np.linalg.norm(values)>0, (np.linalg.norm(values))

        num_validation_samples=100
        validation_samples = np.random.uniform(
            -1.,1.,(num_vars,num_validation_samples))
        validation_values = function(validation_samples)

        zero_ft_data = copy.deepcopy(ft_data)
        zero_ft_data[1]=np.zeros_like(zero_ft_data[1])
        # DO NOT use ft_data in following two functions.
        # These function only overwrites parameters associated with the
        # active indices the rest of the parameters are taken from ft_data.
        # If ft_data is used some of the true data will be kept and give
        # an unrealisticaly accurate answer
        approx_eval = partial(modify_and_evaluate_function_train,samples,
                              zero_ft_data,recursion_coeffs,None)

        apply_approx_adjoint_jacobian = partial(
            apply_function_train_adjoint_jacobian,samples,zero_ft_data,
            recursion_coeffs,1e-3)

        def least_squares_regression(indices,initial_guess):
            #if initial_guess is None:
            st0 = np.random.get_state()
            np.random.seed(1)
            initial_guess = np.random.normal(0.,.01,indices.shape[0])
            np.random.set_state(st0)            
            result = ft_non_linear_least_squares_regression(
                samples, values, ft_data, recursion_coeffs, initial_guess,
                indices,{'gtol':tol,'ftol':tol,'xtol':tol,'verbosity':0})
            return result[indices]

        sparsity = np.where(true_sol!=0)[0].shape[0]
        print(('sparsity',sparsity,'num_samples',num_samples,
               'num_ft_params',num_ft_params))

        print(true_sol)
        active_indices = None

        use_omp = True
        #use_omp = False
        if not use_omp:
            sol = least_squares_regression(np.arange(num_ft_params),None)
        else:
            result = orthogonal_matching_pursuit(
                approx_eval, apply_approx_adjoint_jacobian,
                least_squares_regression, values[:,0], active_indices,
                num_ft_params, tol, min(num_samples,num_ft_params), verbosity=1)
            sol = result[0]
            residnorm = result[1]

        
        recovered_ft_data=copy.deepcopy(ft_data)
        recovered_ft_data[1]=sol
        ft_validation_values = evaluate_function_train(
            validation_samples,recovered_ft_data,recursion_coeffs)


        validation_error = np.linalg.norm(
            validation_values-ft_validation_values)
        rel_validation_error=validation_error/np.linalg.norm(validation_values)
        # compare relative error because exit condition is based upon
        # relative residual
        print(rel_validation_error)
        assert rel_validation_error<100*tol, rel_validation_error

        # interestingly enough the error in the function can be low
        # but the error in the ft parameters can be large
        #print np.where(true_sol!=0)[0]
        #print np.where(sol!=0)[0]
        #assert np.allclose(true_sol,sol,atol=100*tol)


if __name__== "__main__":    
    iht_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestIHT)
    #unittest.TextTestRunner(verbosity=2).run(iht_test_suite)
    omp_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestOMP)
    #unittest.TextTestRunner(verbosity=2).run(omp_test_suite)
    unittest.main()
    


