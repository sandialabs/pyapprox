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
    #print(dold.shape,g.shape)
    print('eps','dfun','dfd','error')
    diff = []
    for i in range(1,13):
        fleft = function(design_prob_measure-t*direction,return_grad=False)
        fright = function(design_prob_measure+t*direction,return_grad=False)
        dnew = (fright-fleft)/(2*t)
        print(t,dold,dnew,abs(dold-dnew))
        t    = t*dt
        diff.append(abs(dold-dnew))
    #print('\n\n')
    return np.array(diff)

class TestOptimalExperimentalDesign(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
    
    def test_homoscedastic_ioptimality_criterion(self):
        poly_degree = 10;
        num_design_pts = 101
        num_pred_pts = 51
        pred_samples = np.random.uniform(-1,1,num_pred_pts)
        # TODO check if design factors may have to be a subset of pred_factors
        #pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        #assert num_design_pts<=pred_factors.shape[0]
        #design_factors = pred_factors[:num_design_pts,:]
        design_samples = np.linspace(-1,1,num_design_pts)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        homog_outer_prods = compute_homoscedastic_outer_products(design_factors)
        ioptimality_criterion_wrapper = partial(
            ioptimality_criterion,homog_outer_prods,design_factors,pred_factors)
        diffs = check_derivative(ioptimality_criterion_wrapper,num_design_pts)
        assert diffs.min()<6e-7, diffs

    def test_hetroscedastic_ioptimality_criterion(self):
        """
        Test homoscedastic and hetroscedastic API produce same value
        when noise is homoscedastic

        WARING current test is just homoscedastic noise but it is still wront
        """
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = design_samples**2+1
        pred_samples = np.random.uniform(-1,1,51)
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        homog_outer_prods = compute_homoscedastic_outer_products(design_factors)
        hetero_outer_prods = compute_heteroscedastic_outer_products(
            design_factors,noise_multiplier)
        ioptimality_criterion_wrapper = partial(
            ioptimality_criterion,homog_outer_prods,design_factors,pred_factors,
            hetero_outer_prods=hetero_outer_prods,
            noise_multiplier=noise_multiplier)
        
        # Test homoscedastic and hetroscedastic API produce same value
        # when noise is homoscedastic
        pp=np.random.uniform(0,1,(num_design_pts,1))
        assert np.allclose(
            ioptimality_criterion_wrapper(pp,return_grad=False),
            ioptimality_criterion(homog_outer_prods,design_factors,pred_factors,
                                  pp,return_grad=False))
        
        # Test hetroscedastic API gradients are correct        
        diffs = check_derivative(ioptimality_criterion_wrapper,num_design_pts)
        assert diffs.min()<6e-7,diffs

    def test_hetroscedastic_coptimality_criterion(self):
        poly_degree = 5
        num_design_pts = 11#101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = design_samples**2
        pred_samples = np.random.uniform(-1,1,5)#51
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=univariate_monomial_basis_matrix(poly_degree,pred_samples)
        hetero_outer_prods = compute_heteroscedastic_outer_products(
            design_factors,noise_multiplier)
        homog_outer_prods = compute_homoscedastic_outer_products(design_factors)
        coptimality_criterion_wrapper = partial(
            coptimality_criterion,homog_outer_prods,design_factors,pred_factors,
            hetero_outer_prods=hetero_outer_prods,
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
        homog_outer_prods = compute_homoscedastic_outer_products(design_factors)
        coptimality_criterion_wrapper = partial(
            coptimality_criterion,homog_outer_prods,design_factors,pred_factors)
        diffs = check_derivative(coptimality_criterion_wrapper,num_design_pts)
        #print (diffs)
        assert diffs.min()<4e-7,diffs

    def test_homoscedastic_doptimality_criterion(self):
        poly_degree = 10;
        num_design_pts = 101
        design_samples = np.linspace(-1,1,num_design_pts)
        noise_multiplier = None
        design_factors = univariate_monomial_basis_matrix(
            poly_degree,design_samples)
        pred_factors=None
        homog_outer_prods = compute_homoscedastic_outer_products(design_factors)
        doptimality_criterion_wrapper = partial(
            doptimality_criterion,homog_outer_prods,design_factors,pred_factors)
        diffs = check_derivative(doptimality_criterion_wrapper,num_design_pts)
        #print (diffs)
        assert diffs.min()<5e-7,diffs

    def test_gradient_log_determinant(self):
        """
        Test the identities 
        -log (det(Y)) = log(det(inv(Y)))
        d/dw_i X.T.dot(diag(w)).dot(X)=X.T.dot(diag(e_i)).dot(X)
        where e_i is unit vector with 1 in ith entry
        d/dw_i log(Y) = trace(inv(Y)*dY/dw_i) 
        """
        X = np.random.normal(0,1,(3,3))
        w = np.arange(1,4,dtype=float)[:,np.newaxis]
        homog_outer_prods = compute_homoscedastic_outer_products(X)
        get_Y = lambda w: homog_outer_prods.dot(w)[:,:,0]
        Y = get_Y(w)

        assert np.allclose(
            -np.log(np.linalg.det(Y)),np.log(np.linalg.det(np.linalg.inv(Y))))
        
        log_det = np.log(np.linalg.det(Y))
        eps=1e-7
        grad_Y    = np.zeros((3,Y.shape[0],Y.shape[1]))
        fd_grad_Y = np.zeros((3,Y.shape[0],Y.shape[1]))
        for ii in range(3):
            w_eps = w.copy(); w_eps[ii]+=eps
            Y_eps = get_Y(w_eps)
            fd_grad_Y[ii] = (Y_eps-Y)/eps
            dw = np.zeros((3,1)); dw[ii]=1
            grad_Y[ii] = get_Y(dw)
            assert np.allclose(grad_Y[ii],homog_outer_prods[:,:,ii])
        assert np.allclose(fd_grad_Y,grad_Y)

        eps=1e-7
        grad_log_det = np.zeros(3)
        fd_grad_log_det = np.zeros(3)
        Y_inv = np.linalg.inv(Y)
        for ii in range(3):
            grad_log_det[ii] = np.trace(Y_inv.dot(grad_Y[ii]))
            w_eps = w.copy(); w_eps[ii]+=eps
            Y_eps = get_Y(w_eps)
            log_det_eps = np.log(np.linalg.det(Y_eps))
            fd_grad_log_det[ii] = (log_det_eps-log_det)/eps

        assert np.allclose(grad_log_det,fd_grad_log_det)


if __name__== "__main__":    
    oed_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestOptimalExperimentalDesign)
    unittest.TextTestRunner(verbosity=2).run(oed_test_suite)
