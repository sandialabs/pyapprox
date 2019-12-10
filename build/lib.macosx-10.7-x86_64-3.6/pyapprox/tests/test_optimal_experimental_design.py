import unittest
from pyapprox.optimal_experimental_design import *
from pyapprox.monomial import univariate_monomial_basis_matrix
from functools import partial

def check_derivative(function,num_design_pts):
    design_prob_measure = np.random.uniform(0,1,(num_design_pts,1))
    direction = np.random.uniform(0,1,(num_design_pts,1))
    t     = 1
    dt    = 0.1
    f,g = function(design_prob_measure,return_grad=True)
    dold  = g.T.dot(direction)
    #print('\n\n')
    print(dold.shape,g.shape)
    diff = []
    for i in range(1,13):
        fleft = function(design_prob_measure-t*direction,return_grad=False)
        fright = function(design_prob_measure+t*direction,return_grad=False)
        dnew = (fright-fleft)/(2*t)
        #print(fnew)
        #print(t,dold,dnew,abs(dold-dnew))
        t    = t*dt
        diff.append(abs(dold-dnew))
    #print('\n\n')
    return np.array(diff)

class TestOptimalExperimentalDesign(unittest.TestCase):
    def test_homoscedastic_ioptimality_criterion(self):
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        pred_samples = np.random.uniform(-1,1,51)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        error_hessians = compute_error_hessians(design_factors)
        ioptimality_criterion_wrapper = partial(
            ioptimality_criterion,error_hessians,design_factors,pred_factors)
        diffs = check_derivative(ioptimality_criterion_wrapper,num_design_pts)
        assert diffs.min()<6e-7, diffs

    def test_hetroscedastic_ioptimality_criterion(self):
        """
        Test homoscedastic and hetroscedastic API produce same value
        when noise is homoscedastic
        """
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = 0*design_samples**2+1#homoscedastic noise
        pred_samples = np.random.uniform(-1,1,51)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        error_hessians = compute_error_hessians(design_factors)
        subgradient_covariances = compute_subgradient_covariances(
            design_factors,noise_multiplier)
        ioptimality_criterion_wrapper = partial(
            ioptimality_criterion,error_hessians,design_factors,pred_factors,
            subgradient_covariances=subgradient_covariances,
            noise_multiplier=noise_multiplier)
        
        # Test homoscedastic and hetroscedastic API produce same value
        # when noise is homoscedastic
        pp=np.random.uniform(0,1,(num_design_pts,1))
        assert np.allclose(
            ioptimality_criterion_wrapper(pp,return_grad=False),
            ioptimality_criterion(error_hessians,design_factors,pred_factors,
                                  pp,return_grad=False))
        
        # Test hetroscedastic API gradients are correct        
        diffs = check_derivative(ioptimality_criterion_wrapper,num_design_pts)
        assert diffs.min()<6e-7,diffs

    def test_hetroscedastic_coptimality_criterion(self):
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = design_samples**2
        pred_samples = np.random.uniform(-1,1,51)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        subgradient_covariances = compute_subgradient_covariances(
            design_factors,noise_multiplier)
        error_hessians = compute_error_hessians(design_factors)
        coptimality_criterion_wrapper = partial(
            coptimality_criterion,error_hessians,design_factors,pred_factors,
            subgradient_covariances=subgradient_covariances,
            noise_multiplier=noise_multiplier)
        diffs = check_derivative(coptimality_criterion_wrapper,num_design_pts)
        #print (diffs)
        assert diffs.min()<4e-7,diffs

    def test_homoscedastic_coptimality_criterion(self):
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = design_samples**2
        pred_samples = np.random.uniform(-1,1,51)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        error_hessians = compute_error_hessians(design_factors)
        coptimality_criterion_wrapper = partial(
            coptimality_criterion,error_hessians,design_factors,pred_factors)
        diffs = check_derivative(coptimality_criterion_wrapper,num_design_pts)
        #print (diffs)
        assert diffs.min()<4e-7,diffs


if __name__== "__main__":    
    oed_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestOptimalExperimentalDesign)
    unittest.TextTestRunner(verbosity=2).run(oed_test_suite)
