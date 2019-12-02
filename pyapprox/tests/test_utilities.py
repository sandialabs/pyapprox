import unittest
from pyapprox.utilities import *
from pyapprox.univariate_quadrature import gauss_jacobi_pts_wts_1D
from scipy.linalg import lu_factor, lu as scipy_lu

class TestUtilities(unittest.TestCase):

    def test_cartesian_product(self):
        # test when num elems = 1
        s1 = np.arange( 0, 3 )
        s2 = np.arange( 3, 5 )

        sets = np.array( [[0,3], [1,3], [2,3], [0,4],
                                [1,4], [2,4]], np.int )
        output_sets = cartesian_product( [s1,s2], 1 )
        assert np.array_equal( output_sets.T, sets )

        # # test when num elems > 1
        # s1 = np.arange( 0, 6 )
        # s2 = np.arange( 6, 10 )

        # sets = np.array( [[ 0, 1, 6, 7], [ 2, 3, 6, 7],
        #                   [ 4, 5, 6, 7], [ 0, 1, 8, 9],
        #                   [ 2, 3, 8, 9], [ 4, 5, 8, 9]], np.int )
        # output_sets = cartesian_product( [s1,s2], 2 )
        # assert np.array_equal( output_sets.T, sets )

    def test_outer_product(self):
        s1 = np.arange( 0, 3 )
        s2 = np.arange( 3, 5 )

        test_vals = np.array( [0.,3.,6.,0.,4.,8.])
        output = outer_product( [s1,s2] )
        assert np.allclose( test_vals, output )

        output = outer_product( [s1] )
        assert np.allclose( output, s1 )

        
    def test_truncated_pivoted_lu_factorization(self):
        np.random.seed(2)
        # test truncated_pivoted lu factorization
        A = np.random.normal( 0, 1, (4,4) )
        scipy_LU,scipy_p  = lu_factor(A)
        scipy_pivots = get_final_pivots_from_sequential_pivots(scipy_p)
        num_pivots = 3
        L,U,pivots = truncated_pivoted_lu_factorization(A, num_pivots)
        assert np.allclose( pivots, scipy_pivots[:num_pivots] )
        assert np.allclose(A[pivots,:num_pivots], np.dot(L, U))
        P = get_pivot_matrix_from_vector(pivots,A.shape[0])
        assert np.allclose(P.dot(A[:,:num_pivots]), np.dot(L, U))

        # test truncated_pivoted lu factorization which enforces first
        # n rows to be chosen in exact order
        # mess up array so that if pivots are not enforced correctly a different
        # pivot order would be returne, Put best pivot in last place in matrix
        # and worst in first row, then enforce first and second rows to be chosen
        # first.
        tmp = A[pivots[0],:].copy()
        A[pivots[0],:] = A[pivots[-1],:].copy()
        A[pivots[-1],:] = tmp
        num_pivots = 3
        num_initial_rows = np.array([0,1])
        L,U,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, num_initial_rows )
        assert np.allclose( A[pivots,:num_pivots], np.dot( L, U ) )
        assert np.allclose( pivots, [0,1,3] )

        # test truncated_pivoted lu factorization which enforces first
        # n rows to be chosen in any order
        tmp = A[pivots[0],:].copy()
        A[pivots[0],:] = A[0,:].copy()
        A[0,:] = tmp
        num_pivots = 3
        num_initial_rows = 1
        L,U,pivots = truncated_pivoted_lu_factorization( A, num_pivots, 
                                                        num_initial_rows )
        assert np.allclose( A[pivots,:num_pivots], np.dot( L, U ) )
        assert np.allclose( pivots, [0,3,1] )

        # Modify the above test to first factorize 4,3 A then factorize
        # B = [A; C] where C is 2*3 and if B was factorized without enforcing
        # A then the factors would be different. Then check that first 
        # 4 rows of LU factors of B are the same as when A was factored.

    def test_tensor_product_quadrature(self):
        num_vars = 2
        alpha_poly=1
        beta_poly=2
        def univariate_quadrature_rule(n):
            x,w = gauss_jacobi_pts_wts_1D(n,alpha_poly,beta_poly)
            x=(x+1)/2.
            return x,w
        
        x,w = get_tensor_product_quadrature_rule(
            100,num_vars,univariate_quadrature_rule)
        function = lambda x: np.sum(x**2,axis=0)
        assert np.allclose(np.dot(function(x),w),0.8)

        #samples = np.random.beta(beta_poly+1,alpha_poly+1,(num_vars,10000))
        #print function(samples).mean()

    def test_canonical_piecewise_quadratic_interpolation(self):
        num_mesh_points=101
        mesh = np.linspace(0.,1.,3)
        mesh_vals = mesh**2
        #do not compare at right boundary because it will be zero
        interp_mesh = np.linspace(0.,1.,num_mesh_points)[:-1]
        interp_vals=canonical_piecewise_quadratic_interpolation(
            interp_mesh,mesh_vals)
        assert np.allclose(interp_vals,interp_mesh**2)

    def test_piecewise_quadratic_interpolation(self):
        def function(x):
            return (x-0.5)**3
        num_mesh_points = 301
        mesh = np.linspace(0.,1.,num_mesh_points)
        mesh_vals = function(mesh)
        #interp_mesh = np.random.uniform(0.,1.,101)
        interp_mesh = np.linspace(0.,1.,1001)
        ranges = [0,1]
        interp_vals=piecewise_quadratic_interpolation(
            interp_mesh,mesh,mesh_vals,ranges)
        # print np.linalg.norm(interp_vals-function(interp_mesh))
        # import pylab as plt
        # I= np.argsort(interp_mesh)
        # plt.plot(interp_mesh[I],interp_vals[I],'k-')
        # plt.plot(mesh,mesh_vals,'o')
        # plt.show()
        assert np.linalg.norm(interp_vals-function(interp_mesh))<1e-6

    def test_add_columns_to_pivoted_lu_factorization(self):
        """
        Let 
        A  = [1 2 4]
             [2 1 3]
             [3 2 4]

        Recursive Algorithm
        -------------------
        The following Permutation swaps the thrid and first rows
        P1 = [0 0 1]
             [0 1 0]
             [1 0 0]

        Gives
        P1*A  = [3 2 4]
                [2 1 3]
                [1 2 4]

        Conceptually partition matrix into block matrix
        P1*A = [A11 A12]
               [A21 A22]

             = [1    0 ][u11 U12]
               [L21 L22][ 0  U22]
             = [u11           U12      ]
               [u11*L21 L21*U12+L22*U22]

        Then 
        u11 = a11
        L21 = 1/a11 A21
        U12 = A12

        e.g. 
        a11 = 3  L21 = [2/3]  U12 = [2 4]  u11 = 3
                       [1/3]

        Because A22 = L21*U12+L22*U22
        L22*U22 = A22-L21*U12
        We also know L22=I

        LU sublock after 1 step is 
        S1 = L22*U22 = A22-L21*U12

           = [1 3]-[4/3 8/3] = [-1/3 1/3]
             [2 4] [2/3 4/3]   [ 4/3 8/3]

        LU after 1 step is
        LU1 = [u11 U12]
              [L21 S1 ]

              [3     2   4  ]
            = [1/3 -1/3 1/3 ]
              [2/3  4/3 8/32]

        The following Permutation swaps the first and second rows of S1
        P2 = [0 1]
             [1 0]

        Conceptually partition matrix into block matrix
        P2*S1 = [ 4/3 8/3] = [A11 A12]
                [-1/3 1/3] = [A21 A22] 

        L21 = 1/a11 A21
        U12 = A12

        e.g. 
        a11 = 4/3   L21 = [-1/4]  U12 = [8/3] u11 = 4/3

        LU sublock after 1 step is 
        S2 = A22-L21*U12
           = 1/3 + 1/4*8/3 = 1

        LU after 2 step is
        LU2 = [ 3    2   4 ]
              [1/3  u11 U12]
              [2/3  L21 S2 ]

            = [ 3    2   4 ]
              [1/3  4/3 8/3]
              [2/3 -1/4 S2 ]
    

        Matrix multiplication algorithm
        -------------------------------
        The following Permutation swaps the thrid and first rows
        P1 = [0 0 1]
             [0 1 0]
             [1 0 0]

        Gives
        P1*A  = [3 2 4]
                [2 1 3]
                [1 2 4]

        Use Matrix M1 to eliminate entries in second and third row of column 1
             [  1  0 1]
        M1 = [-2/3 1 0]
             [-1/3 0 1]

        So U factor after step 1 is
        U1  = M1*P1*A

              [3   2   4  ]
            = [0 -1/3 1/3 ]
              [0  4/3 8/32]

        The following Permutation swaps the third and second rows
        P2 = [1 0 0]
             [0 0 1]
             [0 1 0]

        M2 = [1  0  0]
             [0  1  0]
             [0 1/4 1]

        U factor after step 2 is
        U2  = M2*P2*M1*P1*A

              [3  2   4  ]
            = [0 4/3 8/3 ]
              [0  0   1  ]

        L2 = (M2P2M1P1)^{-1}
           = [ 1    0  0]
             [1/3   1  0]
             [2/3 -1/4 1]

        P*A = P2*P1*A = L2U2   
        """
        A = np.random.normal( 0, 1, (6,6) )
            
        num_pivots = 6
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)

        
        LU_factor_init,pivots_init = \
          truncated_pivoted_lu_factorization(
            A[:,:num_pivots], num_pivots, truncate_L_factor=False)

        new_cols = A[:,LU_factor_init.shape[1]:].copy()

        LU_factor_final=add_columns_to_pivoted_lu_factorization(
            LU_factor_init,new_cols,pivots_init[:num_pivots])
        assert np.allclose(LU_factor_final,LU_factor)

        A = np.random.normal( 0, 1, (6,6) )
            
        num_pivots = 2
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)

        
        LU_factor_init,pivots_init = \
          truncated_pivoted_lu_factorization(
            A[:,:num_pivots], num_pivots, truncate_L_factor=False)

        new_cols = A[:,LU_factor_init.shape[1]:].copy()

        LU_factor_final=add_columns_to_pivoted_lu_factorization(
            LU_factor_init,new_cols,pivots_init[:num_pivots])
        assert np.allclose(LU_factor_final,LU_factor)


    def test_split_lu_factorization_matrix(self):
        A = np.random.normal( 0, 1, (4,4) )
        num_pivots = A.shape[0]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)
        L_factor,U_factor = split_lu_factorization_matrix(LU_factor)
        assert np.allclose(L_factor.dot(U_factor),pivot_rows(pivots,A,False))

        A = np.random.normal( 0, 1, (4,4) )
        num_pivots = 2
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)

        L_factor,U_factor = split_lu_factorization_matrix(LU_factor,num_pivots)
        assert np.allclose(L_factor.dot(U_factor),pivot_rows(pivots,A,False))

    def test_add_rows_to_pivoted_lu_factorization(self):

        np.random.seed(3)
        A = np.random.normal( 0, 1, (10,3) )
           
        num_pivots = A.shape[1]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)

        # create matrix for which pivots do not matter
        A = pivot_rows(pivots,A,False)
        # check no pivoting is necessary
        L,U,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=True)
        assert np.allclose(pivots,np.arange(num_pivots))

        LU_factor_init,pivots_init = \
          truncated_pivoted_lu_factorization(
            A[:num_pivots,:], num_pivots, truncate_L_factor=False)
         
        new_rows = A[num_pivots:,:].copy()

        LU_factor_final=add_rows_to_pivoted_lu_factorization(
            LU_factor_init,new_rows,num_pivots)
        assert np.allclose(LU_factor_final,LU_factor)

        #######
        # only pivot some of the rows
        
        A = np.random.normal( 0, 1, (10,5) )
           
        num_pivots = 3
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False)

        # create matrix for which pivots do not matter
        A = pivot_rows(pivots,A,False)
        print(A.shape)
        # check no pivoting is necessary
        L,U,pivots = truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=True)
        assert np.allclose(pivots,np.arange(num_pivots))

        LU_factor_init,pivots_init = \
          truncated_pivoted_lu_factorization(
            A[:num_pivots,:], num_pivots, truncate_L_factor=False)
         
        new_rows = A[num_pivots:,:].copy()

        LU_factor_final=add_rows_to_pivoted_lu_factorization(
            LU_factor_init,new_rows,num_pivots)
        assert np.allclose(LU_factor_final,LU_factor)

    def test_unprecondition_LU_factor(self):
        A = np.random.normal( 0, 1, (4,4) )
        num_pivots = A.shape[0]
        precond_weights = 1/np.linalg.norm(A,axis=1)[:,np.newaxis]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A*precond_weights, num_pivots, truncate_L_factor=False)

        unprecond_LU_factor,unprecond_pivots=truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False,
            num_initial_rows=pivots)
        L_unprecond,U_unprecond = split_lu_factorization_matrix(
            unprecond_LU_factor)
        assert np.allclose(unprecond_pivots,pivots)
        assert np.allclose(
            L_unprecond.dot(U_unprecond),pivot_rows(unprecond_pivots,A,False))

        precond_weights = pivot_rows(pivots,precond_weights,False)

        L,U = split_lu_factorization_matrix(LU_factor)
        W = np.diag(precond_weights[:,0])
        Wi = np.linalg.inv(W)
        assert np.allclose(Wi.dot(L).dot(U),pivot_rows(pivots,A,False))
        assert np.allclose(
            (L/precond_weights).dot(U),pivot_rows(pivots,A,False))
        # inv(W)*L*W*inv(W)*U
        L = L/precond_weights*precond_weights.T
        U = U/precond_weights
        assert np.allclose(L.dot(U),pivot_rows(pivots,A,False))
        assert np.allclose(L,L_unprecond)
        assert np.allclose(U,U_unprecond)
        
        LU_factor = unprecondition_LU_factor(LU_factor,precond_weights)
        assert np.allclose(LU_factor,unprecond_LU_factor)
       
        A = np.random.normal( 0, 1, (4,4) )
        num_pivots = 2
        precond_weights = 1/np.linalg.norm(A,axis=1)[:,np.newaxis]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A*precond_weights, num_pivots, truncate_L_factor=False)
        L,U = split_lu_factorization_matrix(LU_factor,num_pivots)
        assert np.allclose(
            L.dot(U),pivot_rows(pivots[:num_pivots],A*precond_weights,False))

        unprecond_LU_factor,unprecond_pivots=truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False,
            num_initial_rows=pivots)
        L_unprecond,U_unprecond = split_lu_factorization_matrix(
            unprecond_LU_factor,num_pivots)
        assert np.allclose(unprecond_pivots,pivots)
        assert np.allclose(
            L_unprecond.dot(U_unprecond),
            pivot_rows(unprecond_pivots[:num_pivots],A,False))

        precond_weights = pivot_rows(pivots,precond_weights,False)
        LU_factor = unprecondition_LU_factor(
            LU_factor,precond_weights,num_pivots)
        assert np.allclose(LU_factor,unprecond_LU_factor)

        A = np.random.normal( 0, 1, (5,4) )
        num_pivots = 3
        precond_weights = 1/np.linalg.norm(A,axis=1)[:,np.newaxis]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A*precond_weights, num_pivots, truncate_L_factor=False)
        L,U = split_lu_factorization_matrix(LU_factor,num_pivots)
        assert np.allclose(
            L.dot(U),pivot_rows(pivots[:num_pivots],A*precond_weights,False))

        unprecond_LU_factor,unprecond_pivots=truncated_pivoted_lu_factorization(
            A, num_pivots, truncate_L_factor=False,
            num_initial_rows=pivots)
        L_unprecond,U_unprecond = split_lu_factorization_matrix(
            unprecond_LU_factor,num_pivots)
        assert np.allclose(unprecond_pivots,pivots)
        assert np.allclose(
            L_unprecond.dot(U_unprecond),
            pivot_rows(unprecond_pivots[:num_pivots],A,False))

        precond_weights = pivot_rows(pivots,precond_weights,False)
        LU_factor = unprecondition_LU_factor(
            LU_factor,precond_weights,num_pivots)
        assert np.allclose(LU_factor,unprecond_LU_factor)

    def check_LU_factor(self,LU_factor,pivots,num_pivots,A):
        L,U = split_lu_factorization_matrix(LU_factor,num_pivots)
        return np.allclose(L.dot(U),pivot_rows(pivots,A,False))


    def test_update_christoffel_preconditioned_lu_factorization(self):
        np.random.seed(3)
        A = np.random.normal( 0, 1, (4,4) )

        precond_weights = 1/np.linalg.norm(A,axis=1)[:,np.newaxis]
           
        num_pivots = A.shape[1]
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A*precond_weights, num_pivots, truncate_L_factor=False)

        # create matrix for which pivots do not matter
        A_precond = pivot_rows(pivots,A*precond_weights,False)
        # check no pivoting is necessary
        L,U,pivots = truncated_pivoted_lu_factorization(
            A_precond, num_pivots, truncate_L_factor=True)
        assert np.allclose(pivots,np.arange(num_pivots))

        ii=1
        A_sub = A[:,:ii].copy()
        precond_weights = 1/np.linalg.norm(A_sub,axis=1)[:,np.newaxis]
        A_sub *= precond_weights
        LU_factor,pivots = truncated_pivoted_lu_factorization(
            A_sub, num_pivots, truncate_L_factor=False)
        for ii in range(2,A.shape[1]):
            A_sub = A[:,:ii].copy()
            precond_weights_prev = precond_weights.copy()
            precond_weights = 1/np.linalg.norm(A_sub,axis=1)[:,np.newaxis]
            pivots_prev = pivots.copy()
            pivoted_precond_weights_prev = pivot_rows(
                pivots_prev,precond_weights_prev,False)
            pivoted_precond_weights = pivot_rows(pivots,precond_weights,False)
            
            # what is factorization using old precond weights but with
            # extra column
            true_LU_factor_extra_cols,p= truncated_pivoted_lu_factorization(
                A_sub*precond_weights_prev, ii-1, truncate_L_factor=False,
                num_initial_rows=pivots_prev)
            assert np.allclose(p,pivots_prev)
            assert self.check_LU_factor(
                true_LU_factor_extra_cols,pivots_prev,ii-1,
                A_sub*precond_weights_prev)
            new_cols = A_sub[:,ii-1:ii].copy()
            new_cols*=precond_weights_prev
            LU_factor = add_columns_to_pivoted_lu_factorization(
                LU_factor.copy(),new_cols,pivots_prev[:ii-1])
            assert np.allclose(LU_factor,true_LU_factor_extra_cols)
            assert self.check_LU_factor(
                LU_factor,pivots_prev,ii-1,A_sub*precond_weights_prev)

            # what is factorization with extra column but no preconditioning
            true_LU_factor_extra_cols_unprecond,p = \
                truncated_pivoted_lu_factorization(
                    A_sub, ii-1, truncate_L_factor=False,
                    num_initial_rows=pivots_prev)
            assert np.allclose(p,pivots_prev)
            assert self.check_LU_factor(
                true_LU_factor_extra_cols_unprecond,pivots_prev,ii-1,A_sub)
            LU_factor_unprecond = unprecondition_LU_factor(
                LU_factor,pivoted_precond_weights_prev,ii-1)
            assert self.check_LU_factor(
                LU_factor_unprecond,pivots_prev,ii-1,A_sub)            
            assert np.allclose(
                LU_factor_unprecond,true_LU_factor_extra_cols_unprecond)

            # what is factorization using new precond weights and
            # extra column
            true_LU_factor_extra_cols,_= truncated_pivoted_lu_factorization(
                A_sub*precond_weights, ii-1, truncate_L_factor=False,
                num_initial_rows=pivots_prev)
            LU_factor = unprecondition_LU_factor(
                LU_factor,pivoted_precond_weights_prev/pivoted_precond_weights,
                ii-1)
            assert np.allclose(LU_factor,true_LU_factor_extra_cols)

            max_iters = A_sub.shape[1]
            LU_factor,pivots,it = continue_pivoted_lu_factorization(
                LU_factor.copy(),pivots_prev,ii-1,max_iters,num_initial_rows=0)

            true_LU_factor,_= truncated_pivoted_lu_factorization(
                A_sub*precond_weights, num_pivots, truncate_L_factor=False,
                num_initial_rows=pivots)
            assert np.allclose(LU_factor,true_LU_factor)

    def test_cholesky_decomposition(self):
        nrows = 4
        A = np.random.normal(0.,1.,(nrows,nrows))
        A = A.T.dot(A)
        L_np = np.linalg.cholesky(A)
        L = cholesky_decomposition(A)

    def test_pivoted_cholesky_decomposition(self):
        nrows, npivots = 4, 4
        A = np.random.normal(0.,1.,(nrows,nrows))
        A = A.T.dot(A)
        L, pivots, error = pivoted_cholesky_decomposition(A,npivots)
        assert np.allclose(L.dot(L.T),A)

        nrows, npivots = 4, 2
        A = np.random.normal(0.,1.,(npivots,nrows))
        A = A.T.dot(A)
        L, pivots, error = pivoted_cholesky_decomposition(A,npivots)
        assert L.shape == (nrows,npivots)
        assert pivots.shape[0]==npivots
        assert np.allclose(L.dot(L.T),A)

        # check init_pivots are enforced
        nrows, npivots = 4, 2
        A = np.random.normal(0.,1.,(npivots+1,nrows))
        A = A.T.dot(A)
        L, pivots, error = pivoted_cholesky_decomposition(A,npivots+1)
        L, new_pivots, error = pivoted_cholesky_decomposition(
            A,npivots+1,init_pivots=pivots[1:2])
        assert np.allclose(new_pivots[:npivots+1],pivots[[1,0,2]])

        L = L[pivots,:]
        assert np.allclose(A[pivots,:][:,pivots],L.dot(L.T))

        P = get_pivot_matrix_from_vector(pivots,nrows)
        assert np.allclose(P.dot(A).dot(P.T),L.dot(L.T))

        A = np.array([[4,12,-16],[12,37,-43],[-16,-43,98.]])
        L, pivots, error = pivoted_cholesky_decomposition(A,A.shape[0])

        # reorder entries of A so that cholesky requires pivoting
        true_pivots = np.array([2,1,0])
        A_no_pivots = A[true_pivots,:][:,true_pivots]
        L_np = np.linalg.cholesky(A_no_pivots)
        assert np.allclose(L[pivots,:],L_np)

        # Create A with which needs cholesky with certain pivots
        A = np.array([[4,12,-16],[12,37,-43],[-16,-43,98.]])
        true_pivots = np.array([1,0,2])
        A = A[true_pivots,:][:,true_pivots]
        L, pivots, error = pivoted_cholesky_decomposition(A,A.shape[0])
        assert np.allclose(L[pivots,:],L_np)

    def test_beta_pdf_on_ab(self):
        from scipy.stats import beta as beta_rv
        alpha_stat,beta_stat = 5,2
        lb,ub=-2,1
        xx = np.linspace(lb,ub,100)
        vals = beta_pdf_on_ab(alpha_stat,beta_stat,lb,ub,xx)
        true_vals = beta_rv.pdf((xx-lb)/(ub-lb),alpha_stat,beta_stat)/(ub-lb)
        #true_vals = beta_rv.pdf(xx,alpha_stat,beta_stat,loc=lb,scale=ub-lb)
        assert np.allclose(vals,true_vals)

        import sympy as sp
        x = sp.Symbol('x')
        assert np.allclose(1,
            float(sp.integrate(beta_pdf_on_ab(alpha_stat,beta_stat,lb,ub,x),
                         (x,[lb,ub]))))

        alpha_stat,beta_stat = 5,2
        lb,ub=0,1
        xx = np.linspace(lb,ub,100)
        vals = beta_pdf_on_ab(alpha_stat,beta_stat,lb,ub,xx)
        true_vals = beta_rv.pdf((xx-lb)/(ub-lb),alpha_stat,beta_stat)/(ub-lb)
        assert np.allclose(vals,true_vals)

        import sympy as sp
        x = sp.Symbol('x')
        assert np.allclose(1,
            float(sp.integrate(beta_pdf_on_ab(alpha_stat,beta_stat,lb,ub,x),
                         (x,[lb,ub]))))

        eps=1e-7
        x = 0.5
        deriv = beta_pdf_derivative(alpha_stat,beta_stat,x)
        fd_deriv = (beta_pdf_on_ab(alpha_stat,beta_stat,0,1,x)-
                    beta_pdf_on_ab(alpha_stat,beta_stat,0,1,x-eps))/eps
        assert np.allclose(deriv,fd_deriv)

        eps=1e-7
        x = np.array([0.5,0,-0.25])
        from functools import partial
        pdf_deriv = partial(beta_pdf_derivative,alpha_stat,beta_stat)
        deriv = pdf_derivative_under_affine_map(
            pdf_deriv,-1,2,x)
        fd_deriv = (beta_pdf_on_ab(alpha_stat,beta_stat,-1,1,x)-
                    beta_pdf_on_ab(alpha_stat,beta_stat,-1,1,x-eps))/eps
        assert np.allclose(deriv,fd_deriv)

        

if __name__== "__main__":    
    utilities_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestUtilities)
    unittest.TextTestRunner(verbosity=2).run(utilities_test_suite)

