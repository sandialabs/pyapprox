import unittest
from pyapprox.barycentric_interpolation import *
from pyapprox.utilities import cartesian_product
from pyapprox.univariate_quadrature import clenshaw_curtis_in_polynomial_order
import copy

from pyapprox.multivariate_polynomials import *
from pyapprox.variable_transformations import *
from scipy.stats import norm
from pyapprox.univariate_quadrature import gauss_hermite_pts_wts_1D
def preconditioned_barycentric_weights():
    num_samples = 2**8+1
    samples = gauss_hermite_pts_wts_1D(num_samples)[0]
    poly = PolynomialChaosExpansion()
    var_trans = define_iid_random_variable_transformation(
        norm(),1) 
    poly_opts = define_poly_options_from_variable_transformation(var_trans)
    poly.configure(poly_opts)
    poly.set_indices(np.arange(num_samples))

    precond_weights = np.sqrt(
        (poly.basis_matrix(samples[np.newaxis,:])**2).mean(axis=1))
    #precond_weights = np.ones(num_samples)
    
    bary_weights = compute_barycentric_weights_1d(
            samples, interval_length=samples.max()-samples.min())

    def barysum(x,y,w,f):
        x = x[:,np.newaxis]
        y = y[np.newaxis,:]
        temp = w*f/(x-y)
        return np.sum(temp,axis=1)

    function = lambda x: np.cos(2*np.pi*x)
    
    y = samples
    w = precond_weights*bary_weights
    x = np.linspace(-3,3,301)
    f = function(y)/precond_weights
    
    r1 = barysum(x,y,w,f)
    r2 = barysum(x,y,w,1/precond_weights)
    interp_vals = r1/r2
    import matplotlib.pyplot as plt
    plt.plot(x,interp_vals,'k')
    plt.plot(x,function(x),'r--')
    #plt.plot(samples,function(samples),'ro')
    print(num_samples)
    print(np.linalg.norm(interp_vals-function(x)))
    plt.show()

    

class TestBarycentricInterpolation(unittest.TestCase):

    def test_barycentric_weights_1d(self):
        eps = 1e-12
        
        # test barycentric weights for uniform points using direct calculation
        abscissa = np.linspace( -1, 1., 5 )
        weights = compute_barycentric_weights_1d(
            abscissa,normalize_weights=False)
        n = abscissa.shape[0]-1
        h = 2. / n
        true_weights = np.empty( ( n+1 ), np.double )
        for j in range( n+1 ):
            true_weights[j] = (-1.)**(n-j)*nchoosek(n,j)/( h**n*factorial(n) )
        assert np.allclose( true_weights, weights, eps )

        # test barycentric weights for uniform points using analytical formula
        # and with scaling on
        weights = compute_barycentric_weights_1d(
            abscissa, interval_length=1,normalize_weights=False)
        weights_analytical = equidistant_barycentric_weights( 5 )
        ratio = weights / weights_analytical;
        # assert the two weights array differ by only a constant factor
        assert np.allclose( np.min( ratio ), np.max( ratio ) )

        # test barycentric weights for clenshaw curtis points
        level = 7
        abscissa, tmp = clenshaw_curtis_pts_wts_1D( level )
        n = abscissa.shape[0]
        weights =  compute_barycentric_weights_1d(
            abscissa,normalize_weights=False,interval_length=2)
        true_weights = np.empty( ( n ), np.double )
        true_weights[0] = true_weights[n-1] = 0.5
        true_weights[1:n-1]=[(-1)**ii for ii in range(1,n-1)]
        factor = true_weights[1]/weights[1]
        assert np.allclose( true_weights/factor, weights, atol=eps )

        # check barycentric weights are correctly computed regardless of
        # order of points. Eventually ordering can effect numerical stability
        # but not until very high level
        abscissa, tmp = clenshaw_curtis_in_polynomial_order(level)
        I = np.argsort(abscissa)
        n = abscissa.shape[0]
        weights =  compute_barycentric_weights_1d(
            abscissa,normalize_weights=False,
            interval_length=abscissa.max()-abscissa.min())
        true_weights = np.empty( ( n ), np.double )
        true_weights[0] = true_weights[n-1] = 0.5
        true_weights[1:n-1]=[(-1)**ii for ii in range(1,n-1)]
        factor = true_weights[1]/weights[I][1]
        assert np.allclose( true_weights/factor, weights[I], eps )

        from pyapprox.univariate_quadrature import gauss_hermite_pts_wts_1D
        num_samples = 65
        abscissa, tmp = gauss_hermite_pts_wts_1D(num_samples)
        weights =  compute_barycentric_weights_1d(
            abscissa,normalize_weights=False,
            interval_length=abscissa.max()-abscissa.min())
        print(weights)
        print(np.absolute(weights).max(),np.absolute(weights).min())
        print(np.absolute(weights).max()/np.absolute(weights).min())
        

    def test_multivariate_barycentric_lagrange_interpolation(self):
        f = lambda x: np.sum( x**2, axis = 0 )
        eps = 1e-14

        # test 1d barycentric lagrange interpolation
        level = 5
        #abscissa, __ = clenshaw_curtis_pts_wts_1D( level )
        #barycentric_weights_1d = [clenshaw_curtis_barycentric_weights(level)]
        from pyapprox.univariate_quadrature import \
            clenshaw_curtis_in_polynomial_order
        abscissa, __ = clenshaw_curtis_in_polynomial_order(level,False)
        abscissa_1d = [abscissa]
        barycentric_weights_1d=[compute_barycentric_weights_1d(abscissa_1d[0])]
        fn_vals=f(np.array( abscissa ).reshape(
            1,abscissa.shape[0]))[:,np.newaxis]
        pts = np.linspace( -1., 1., 3 ).reshape(1,3)
        poly_vals = multivariate_barycentric_lagrange_interpolation(
            pts,abscissa_1d,barycentric_weights_1d,fn_vals,np.array([0]))
        
        #import pylab
        #print poly_vals.squeeze().shape
        #pylab.plot(pts[0,:],poly_vals.squeeze())
        #pylab.plot(abscissa_1d[0],fn_vals.squeeze(),'ro')
        #print np.linalg.norm( poly_vals - f( pts ) )
        #pylab.show()
        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        # test 2d barycentric lagrange interpolation
        # with the same abscissa in each dimension
        a = -3.0; b = 3.0
        x = np.linspace( a, b, 21 )
        [X,Y] = np.meshgrid( x, x )
        pts = np.vstack( ( X.reshape( ( 1, X.shape[0]*X.shape[1] ) ), 
           Y.reshape( ( 1, Y.shape[0]*Y.shape[1] ) ) ) )

        num_abscissa = [10,10]
        abscissa_1d = [np.linspace(a,b,num_abscissa[0]),
                           np.linspace(a,b,num_abscissa[1])]

        abscissa = cartesian_product( abscissa_1d, 1 )
        fn_vals = f( abscissa )
        barycentric_weights_1d = [
            compute_barycentric_weights_1d( abscissa_1d[0] ),
            compute_barycentric_weights_1d( abscissa_1d[1] )]

        poly_vals = multivariate_barycentric_lagrange_interpolation( 
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([0,1]))

        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        #test 2d barycentric lagrange interpolation
        # with different abscissa in each dimension
        a = -1.0; b = 1.0
        x = np.linspace( a, b, 21 )
        [X,Y] = np.meshgrid( x, x )
        pts = np.vstack( ( X.reshape( ( 1, X.shape[0]*X.shape[1] ) ), 
                                  Y.reshape( ( 1, Y.shape[0]*Y.shape[1] ) ) ) )

        level = [1,2]
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        nodes_1, tmp = clenshaw_curtis_pts_wts_1D( level[1] )
        abscissa_1d = [nodes_0,nodes_1]
        barycentric_weights_1d=[clenshaw_curtis_barycentric_weights(level[0]),
                                clenshaw_curtis_barycentric_weights(level[1])]
        abscissa = cartesian_product( abscissa_1d, 1 )
        fn_vals = f( abscissa )

        poly_vals = multivariate_barycentric_lagrange_interpolation( 
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([0,1]))

        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )


        # test 3d barycentric lagrange interpolation
        # with different abscissa in each dimension
        num_dims = 3
        a = -1.0; b = 1.0
        pts = np.random.uniform(-1.,1.,(num_dims,10))

        level = [1,1,1]
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        nodes_1, tmp = clenshaw_curtis_pts_wts_1D( level[1] )
        nodes_2, tmp = clenshaw_curtis_pts_wts_1D( level[2] )
        abscissa_1d = [nodes_0,nodes_1,nodes_2]
        barycentric_weights_1d=[clenshaw_curtis_barycentric_weights(level[0]),
                                clenshaw_curtis_barycentric_weights(level[1]),
                                clenshaw_curtis_barycentric_weights(level[2])]
        abscissa = cartesian_product( abscissa_1d, 1 )
        fn_vals = f( abscissa )

        poly_vals = multivariate_barycentric_lagrange_interpolation( 
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([0,1,2]))
        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        # test 3d barycentric lagrange interpolation
        # with different abscissa in each dimension
        # and only two active dimensions (0 and 2)
        num_dims = 3
        a = -1.0; b = 1.0
        pts = np.random.uniform(-1.,1.,(num_dims,5))


        level = [2,0,1]
        # to get fn_vals we must specify abscissa for all three dimensions
        # but only the abscissa of the active dimensions should get passed
        # to the interpolation function
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        nodes_1, tmp = clenshaw_curtis_pts_wts_1D( level[1] )
        nodes_2, tmp = clenshaw_curtis_pts_wts_1D( level[2] )
        abscissa_1d = [nodes_0,
                           nodes_1,
                           nodes_2]
        abscissa = cartesian_product( abscissa_1d, 1 )
        abscissa_1d = [nodes_0,
                           nodes_2]
        barycentric_weights_1d=[clenshaw_curtis_barycentric_weights(level[0]),
                                clenshaw_curtis_barycentric_weights(level[2])]
        fn_vals = f( abscissa )

        poly_vals = multivariate_barycentric_lagrange_interpolation(
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([0,2]))
        pts[1,:] = 0.
        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        # test 3d barycentric lagrange interpolation
        # with different abscissa in each dimension
        # and only two active dimensions (0 and 1)
        num_dims = 3
        a = -1.0; b = 1.0
        pts = np.random.uniform(-1.,1.,(num_dims,5))

        level = [2,3,0]
        # to get fn_vals we must specify abscissa for all three dimensions
        # but only the abscissa of the active dimensions should get passed
        # to the interpolation function
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        nodes_1, tmp = clenshaw_curtis_pts_wts_1D( level[1] )
        nodes_2, tmp = clenshaw_curtis_pts_wts_1D( level[2] )
        abscissa_1d = [nodes_0,
                        nodes_1,
                             nodes_2]
        abscissa = cartesian_product( abscissa_1d, 1 )
        abscissa_1d = [nodes_0,
                           nodes_1]
        barycentric_weights_1d=[clenshaw_curtis_barycentric_weights(level[0]),
                                clenshaw_curtis_barycentric_weights(level[1])]
        fn_vals = f( abscissa )

        poly_vals = multivariate_barycentric_lagrange_interpolation(
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([0,1]))
        # The interpolant will only be correct on the plane involving
        # the active dimensions so we must set the coordinate of the inactive
        # dimension to the abscissa coordinate of the inactive dimension.
        # The interpoolation algorithm is efficient in the sense that it 
        # ignores all dimensions involving only one point because the 
        # interpolant will be a constant in that direction
        pts[2,:] = 0.
        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        # test 3d barycentric lagrange interpolation
        # with different abscissa in each dimension
        # and only two active dimensions (1 and 2)
        num_dims = 3
        a = -1.0; b = 1.0
        pts = np.random.uniform(-1.,1.,(num_dims,5))

        level = [0,2,4]
        # to get fn_vals we must specify abscissa for all three dimensions
        # but only the abscissa of the active dimensions should get passed
        # to the interpolation function
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        nodes_1, tmp = clenshaw_curtis_pts_wts_1D( level[1] )
        nodes_2, tmp = clenshaw_curtis_pts_wts_1D( level[2] )
        abscissa_1d = [nodes_0,
                       nodes_1,
                           nodes_2]
        abscissa = cartesian_product( abscissa_1d, 1 )
        abscissa_1d = [nodes_1,
                           nodes_2]
        barycentric_weights_1d=[clenshaw_curtis_barycentric_weights(level[1]),
                                clenshaw_curtis_barycentric_weights(level[2])]
        fn_vals = f( abscissa )

        poly_vals = multivariate_barycentric_lagrange_interpolation( 
            pts, abscissa_1d, barycentric_weights_1d, fn_vals[:,np.newaxis],
            np.array([1,2]))
        pts[0,:] = 0.
        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

        # test 2d barycentric lagrange interpolation
        # with different abscissa in each dimension and only some of
        # the coefficients of the basis terms being non-zero. This situation
        # arises in hierarchical interpolation. In these cases we need
        # to construct the basis functions on all abscissa but we only
        # need to add the basis functions that are one at the hierachical
        # nodes
        a = -1.0; b = 1.0
        #x = np.linspace( a, b, 21 )
        x = np.linspace( a, b, 5 )
        [X,Y] = np.meshgrid( x, x )
        pts = np.vstack( ( X.reshape( ( 1, X.shape[0]*X.shape[1] ) ), 
                                  Y.reshape( ( 1, Y.shape[0]*Y.shape[1] ) ) ) )

        poly_vals = np.ones((pts.shape[1],1), np.double) * \
          f( np.array( [[0.0,0.0]] ).T )[:,np.newaxis]

        level = [1]
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        abscissa_1d = [nodes_0]
        barycentric_weights_1d=[compute_barycentric_weights_1d(abscissa_1d[0])]
        sets = copy.copy( abscissa_1d )
        sets.append( np.array( [0.0] ) )
        abscissa = cartesian_product( sets, 1 )
        hier_indices = np.array( [[0,2]], np.int32 )
        abscissa = abscissa[:,hier_indices[0]]
        fn_vals = f( abscissa )
        poly_vals_increment = \
          multivariate_hierarchical_barycentric_lagrange_interpolation( 
              pts,abscissa_1d,barycentric_weights_1d,
              ( fn_vals - np.ones( ( abscissa.shape[1] ), np.double ) * f(
                  np.array( [0.0,0.0] ) ) )[:,np.newaxis],
              np.array([0]), hier_indices)
        poly_vals += poly_vals_increment

        level = [1]
        nodes_0, tmp = clenshaw_curtis_pts_wts_1D( level[0] )
        abscissa_1d = [nodes_0]
        #barycentric_weights_1d = [barycentric_weights( np.array( [0.0] ) ),
        #                          barycentric_weights( abscissa_1d[0] )]
        barycentric_weights_1d=[compute_barycentric_weights_1d(abscissa_1d[0])]

        sets = [np.array( [0.0] )]
        sets.append( nodes_0 )
        abscissa = cartesian_product( sets, 1 )
        hier_indices = np.array( [[0,2]], np.int32 )
        abscissa = abscissa[:,hier_indices[0]]
        fn_vals = f( abscissa )
        poly_vals += \
          multivariate_hierarchical_barycentric_lagrange_interpolation( 
              pts, abscissa_1d, barycentric_weights_1d,
              ( fn_vals - np.ones( ( abscissa.shape[1] ), np.double ) * f(
                  np.array( [[0.0,0.0]] ).T ) )[:,np.newaxis],
              np.array([1]),hier_indices)

        assert np.allclose( poly_vals, f( pts )[:,np.newaxis], eps )

    def test_interpolation_gaussian_leja_sequence(self):
        f = lambda x: np.exp(-np.sum(x**2,axis=0))
        
        from pyapprox.univariate_quadrature import \
            gaussian_leja_quadrature_rule, gauss_hermite_pts_wts_1D
        level = 30
        abscissa_leja,__ = gaussian_leja_quadrature_rule(
            level,return_weights_for_all_levels=False)
        abscissa_gauss = gauss_hermite_pts_wts_1D(level+1)[0]
        abscissa = abscissa_leja
        #abscissa = abscissa_gauss
        #print(abscissa_leja.shape,abscissa_gauss.shape)
        
        abscissa_1d = [abscissa]
        barycentric_weights_1d=[compute_barycentric_weights_1d(abscissa_1d[0])]
        #print(barycentric_weights_1d[0])
        barycentric_weights_1d[0] /= barycentric_weights_1d[0].max()
        #print(barycentric_weights_1d[0])
        fn_vals=f(np.array(abscissa).reshape(1,abscissa.shape[0]))[:,np.newaxis]
        #print(fn_vals.shape)
        
        samples = np.random.normal(0,1,(1,1000))
        poly_vals = multivariate_barycentric_lagrange_interpolation(
            samples,abscissa_1d,barycentric_weights_1d,fn_vals,
            np.array([0]))[:,0]
        l2_error=np.linalg.norm(poly_vals-f(samples))/np.sqrt(samples.shape[1])
        #print('l2_error',l2_error)

        # pts = np.linspace(abscissa.min(),abscissa.max(),101).reshape(1,101)
        # poly_vals = multivariate_barycentric_lagrange_interpolation(
        #     pts,abscissa_1d,barycentric_weights_1d,fn_vals,np.array([0]))
        # import matplotlib.pyplot as plt
        # plt.plot(pts[0,:],poly_vals.squeeze())
        # plt.plot(abscissa_1d[0],fn_vals.squeeze(),'r*')
        # plt.plot(abscissa_leja,abscissa_leja*0,'ro')
        # plt.plot(abscissa_gauss,abscissa_gauss*0,'ks',ms=3)
        # plt.ylim(-1,2)
        # plt.show()

        assert l2_error<1e-2

if __name__== "__main__":
    #preconditioned_barycentric_weights()
    barycentric_interpolation_test_suite = \
     unittest.TestLoader().loadTestsFromTestCase(
        TestBarycentricInterpolation)
    unittest.TextTestRunner(verbosity=2).run(
        barycentric_interpolation_test_suite)
