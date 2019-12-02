import unittest
from pyapprox.manipulate_polynomials import *
from pyapprox.indexing import compute_hyperbolic_indices, \
    argsort_indices_leixographically
from scipy.special import binom
from pyapprox.monomial import monomial_mean_uniform_variables, \
    monomial_basis_matrix
from pyapprox.multivariate_polynomials import jacobi_recurrence, \
    evaluate_multivariate_orthonormal_polynomial
from pyapprox.orthonormal_polynomials_1d import \
    evaluate_orthonormal_polynomial_1d

class TestManipulatePolynomials(unittest.TestCase):

    def test_multiply_multivariate_polynomials(self):
        num_vars = 2
        degree1 = 1
        degree2 = 2

        indices1 = compute_hyperbolic_indices(num_vars,degree1,1.0)
        coeffs1 = np.ones((indices1.shape[1]),dtype=float)
        indices2 = compute_hyperbolic_indices(num_vars,degree2,1.0)
        coeffs2 = 2.0*np.ones((indices2.shape[1]),dtype=float)
        
        indices,coeffs = multiply_multivariate_polynomials(
            indices1,coeffs1,indices2,coeffs2)
        indices = indices[:,argsort_indices_leixographically(indices)]

        true_indices = compute_hyperbolic_indices(num_vars,degree1+degree2,1.0)
        true_indices = \
            true_indices[:,argsort_indices_leixographically(true_indices)]
        #print (true_indices,'\n',indices)
        assert np.allclose(true_indices,indices)
        true_coeffs = np.array([2,4,4,4,4,6,2,4,4,2])
        assert np.allclose(true_coeffs,coeffs)

    def test_multinomial_coefficients(self):
        num_vars = 2; degree = 5
        coeffs, indices = multinomial_coeffs_of_power_of_nd_linear_polynomial(
            num_vars,degree)

        true_coeffs = np.empty(coeffs.shape[0],float)
        for i in range(0,degree+1):
            true_coeffs[i] = binom(degree,i)
        assert true_coeffs.shape[0]==coeffs.shape[0]
        coeffs=coeffs[argsort_indices_leixographically(indices)]
        assert np.allclose(coeffs, true_coeffs)

        num_vars=3; degree = 3
        coeffs, indices = multinomial_coeffs_of_power_of_nd_linear_polynomial(
            num_vars,degree)
        coeffs = multinomial_coefficients(indices)
        coeffs = coeffs[argsort_indices_leixographically(indices)]

        true_coeffs = np.array([1,3,3,1,3,6,3,3,3,1])
        assert np.allclose(coeffs, true_coeffs)

    def test_coeffs_of_power_of_nd_linear_polynomial(self):
        num_vars = 3; degree = 2
        linear_coeffs = [2.,3.,4]
        coeffs, indices = coeffs_of_power_of_nd_linear_polynomial(
            num_vars,degree,linear_coeffs)
        sorted_idx = argsort_indices_leixographically(indices)
        true_coeffs = [linear_coeffs[2]**2,2*linear_coeffs[1]*linear_coeffs[2],
                       linear_coeffs[1]**2,2*linear_coeffs[0]*linear_coeffs[2],
                       2*linear_coeffs[0]*linear_coeffs[1],linear_coeffs[0]**2]
        assert np.allclose(true_coeffs,coeffs[sorted_idx])

    def test_group_like_terms(self):
        num_vars = 2; degree = 2

        # define two set of indices that have a non-empty intersection
        indices1 = compute_hyperbolic_indices(num_vars, degree, 1.0)
        indices2 = compute_hyperbolic_indices(num_vars, degree-1, 1.0)
        num_indices1 = indices1.shape[1]
        coeffs = np.arange(num_indices1+indices2.shape[1])
        indices1 = np.hstack((indices1,indices2))

        # make it so coefficients increase by 1 with lexiographical order of
        # combined indices
        indices = indices1[:,argsort_indices_leixographically(indices1)]
        coeffs, indices = group_like_terms(coeffs, indices)

        # Check that only unique indices remain
        assert indices.shape[1]==num_indices1
        #print_indices(indices,num_vars)
        true_indices = np.asarray([[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices,indices[:,sorted_idx])

        # check that the coefficients of the unique indices are the sum of
        # all original common indices
        true_coeffs = [1,5,9,6,7,8]
        assert np.allclose(coeffs[sorted_idx][:,0],true_coeffs)

    def test_add_polynomials(self):
        num_vars = 2; degree = 2

        # define two set of indices that have a non-empty intersection
        indices1 = compute_hyperbolic_indices(num_vars, degree, 1.0)
        indices1 = indices1[:,argsort_indices_leixographically(indices1)]
        coeffs1 = np.arange(indices1.shape[1])[:,np.newaxis]
        indices2 = compute_hyperbolic_indices(num_vars, degree-1, 1.0)
        indices2 = indices2[:,argsort_indices_leixographically(indices2)] 
        coeffs2 = np.arange(indices2.shape[1])[:,np.newaxis]

        indices, coeffs = add_polynomials([indices2,indices1],[coeffs2,coeffs1])

        # check that the coefficients of the new polynomial are the union
        # of the original polynomials
        true_indices = np.asarray([[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices,indices[:,sorted_idx])
        
        # check that the coefficients of the new polynomials are the sum of
        # all original polynomials
        true_coeffs = np.asarray([[0,2,4,3,4,5]]).T
        assert np.allclose(coeffs[sorted_idx],true_coeffs)


        num_vars = 2; degree = 2

        # define two set of indices that have a non-empty intersection
        indices3 = compute_hyperbolic_indices(num_vars, degree+1, 1.0)
        indices3 = indices3[:,argsort_indices_leixographically(indices3)]
        coeffs3 = np.arange(indices3.shape[1])[:,np.newaxis]

        indices, coeffs = add_polynomials(
            [indices2,indices1,indices3],[coeffs2,coeffs1,coeffs3])

        # check that the coefficients of the new polynomial are the union
        # of the original polynomials
        true_indices = np.asarray(
            [[0,0],[0,1],[1,0],[0,2],[1,1],[2,0],[0,3],[1,2],[2,1],[3,0]]).T
        sorted_idx = argsort_indices_leixographically(indices)
        assert np.allclose(true_indices,indices[:,sorted_idx])
        
        # check that the coefficients of the new polynomials are the sum of
        # all original polynomials
        true_coeffs = np.asarray([[0,3,6,6,8,10,6,7,8,9]]).T
        assert np.allclose(coeffs[sorted_idx],true_coeffs)
                              
    def test_coeffs_of_power_of_polynomial(self):
        num_vars,degree,power=1,2,3
        indices = compute_hyperbolic_indices(num_vars,degree,1.0)
        coeffs = np.ones((indices.shape[1],1))
        new_coeffs, new_indices = coeffs_of_power_of_polynomial(
            indices, coeffs, power)
        assert new_indices.shape[1]==10
        print(np.hstack((new_indices.T,new_coeffs)))
        true_indices = np.asarray([[0],[1],[2],[2],[3],[4],[3],[4],[5],[6]]).T
        true_coeffs  = np.asarray([[1,3,3,3,6,3,1,3,3,1]]).T
        # must include coefficient in sort because when there are mulitple
        # entries with same index, argsort can return different orders
        # if initial orders of indices is different.
        #new_sorted_idx = argsort_indices_leixographically(
        #    np.hstack([new_indices.T,new_coeffs]).T)
        #true_sorted_idx = argsort_indices_leixographically(
        #    np.hstack([true_indices.T,true_coeffs]).T)
        # alternatively just group like terms before sort
        new_coeffs, new_indices  = group_like_terms(new_coeffs,new_indices)
        true_coeffs,true_indices = group_like_terms(true_coeffs,true_indices)
        new_sorted_idx = argsort_indices_leixographically(new_indices)
        true_sorted_idx = argsort_indices_leixographically(true_indices)
        
        assert np.allclose(
            true_indices[:,true_sorted_idx],new_indices[:,new_sorted_idx])
        assert np.allclose(
            true_coeffs[true_sorted_idx],new_coeffs[new_sorted_idx])

        num_vars,degree,power=2,1,2
        indices = np.array([[0,0],[1,0],[0,1],[1,1]]).T
        coeffs = np.ones((indices.shape[1],1))
        new_coeffs, new_indices = coeffs_of_power_of_polynomial(
            indices, coeffs, power)
        true_indices = np.asarray(
            [[0,0],[1,0],[0,1],[1,1],[2,0],[1,1],[2,1],[0,2],[1,2],[2,2]]).T
        true_coeffs  = np.asarray([[1,2,2,2,1,2,2,1,2,1]]).T
        new_coeffs, new_indices  = group_like_terms(new_coeffs,new_indices)
        true_coeffs,true_indices = group_like_terms(true_coeffs,true_indices)
        new_sorted_idx = argsort_indices_leixographically(new_indices)
        true_sorted_idx = argsort_indices_leixographically(true_indices)
        assert np.allclose(
            true_indices[:,true_sorted_idx],new_indices[:,new_sorted_idx])
        assert np.allclose(
            true_coeffs[true_sorted_idx],new_coeffs[new_sorted_idx])

    def test_substitute_polynomial_for_variables_in_single_basis_term(self):
        """
        Substitute 
          y1 = (1+x1+x2+x1*x2)
          y2 = (2+2*x1+2*x1*x3)
        into 
          y3 = y1**2*x4**3*y2
             = (1+x1+x2+x1*x2)**2*x4*(2+2*x1+2*x1*x3)
             = (1+2x1+2*x2+2*(x1*x2)+x1**2+2*x1*x2+2*x1*(x1*x2)+x2**2+2*x2*(x1*x2)
               +(x1*x2)**2)*x4*(2+2*x1+2*x1*x3)

        Global ordering of variables in y3
        [y1,x4,y2] = [x1,x2,x4,x1,x3]
        Only want unique variables so reduce to
        [x1,x2,x4,x3]

        TODO
        Need example where
          y1 = (1+x1+x2+x1*x2)
          y2 = (2+2*x1+0*x2+2*x1*x2)
          y3 = (3+3*x1+3*x3)
        where y1 and y2 but not y3
        are functions of the same variables. The first example above
        arises when two separate scalar output models are inputs to a 
        downstream model C, e.g. 

        A(x1,x2) \
                  C
        A(x1,x2) /

        This second model arises when two outputs of the same vector output 
        model A are inputs to a downstream model C as is one output of another 
        upstream scalar output model B, e.g.

        A(x1,x2) \
                  C
        B(x1,x3) /

        In later case set 
        indices_in = [np.array([[0,0],[1,0],[0,1],[1,1]]).T,
                      np.array([[0,0],[1,0],[0,1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1],2)),
                     np.ones((indices_in[1].shape[1],1))]
        coeffs_in[0][:,1]=2; coeffs_in[0][2,1]=0

        Actually may be better to treat all inputs as if from three seperate 
        models
        indices_in = [np.array([[0,0],[1,0],[0,1],[1,1]]).T,
                      np.array([[0,0],[1,0],[0,1],[1,1]]).T
                      np.array([[0,0],[1,0],[0,1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1],1)),
                     np.ones((indices_in[0].shape[1],1)),
                     np.ones((indices_in[1].shape[1],1))]
        coeffs_in[1][:,0]=2; coeffs_in[1][2,0]=0
        
        
        """
        num_global_vars = 4
        global_var_idx = [[0,1],[0,2]]
        indices_in = [np.array([[0,0],[1,0],[0,1],[1,1]]).T,
                      np.array([[0,0],[1,0],[1,1]]).T]
        coeffs_in = [np.ones((indices_in[0].shape[1],1)),
                     2*np.ones((indices_in[1].shape[1],1))]
        basis_index = np.array([[2,1,3]]).T
        basis_coeff = np.array([[1]])
        var_idx = np.array([0,2])
        new_coeffs, new_indices = \
            substitute_polynomial_for_variables_in_single_basis_term(
                indices_in,coeffs_in,basis_index,basis_coeff,var_idx,
                global_var_idx,num_global_vars)
        print(new_coeffs,new_indices)
        

if __name__== "__main__":    
    manipulate_polynomials_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestManipulatePolynomials)
    unittest.TextTestRunner(verbosity=2).run(manipulate_polynomials_test_suite)
