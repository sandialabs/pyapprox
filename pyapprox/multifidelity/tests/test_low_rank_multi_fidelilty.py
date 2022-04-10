import unittest
import numpy as np
from scipy.stats import uniform
from scipy.special import jv as bessel_function

from pyapprox.multifidelity.low_rank_multifidelity import (
    compute_mean_l2_error, BiFidelityModel,
    select_nodes, select_nodes_cholesky
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.util.linalg import get_pivot_matrix_from_vector


class OscillatoryPolyLowFidelityModel(object):
    def __init__(self, mesh_dof=100, num_terms=35):
        self.mesh = np.linspace(-1., 1., mesh_dof)
        self.num_terms = num_terms

        variable = [uniform(-1, 2)]
        var_trans = AffineTransform(variable)
        self.poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
        self.poly.configure(poly_opts)
        self.poly.set_indices(compute_hyperbolic_indices(
            1, self.num_terms-1))

    def basis_matrix(self):
        # compute vandermonde matrix, i.e. all legendre polynomials up
        # at most degree self.num_terms
        basis_matrix = self.poly.basis_matrix(
            self.mesh.reshape(1, self.mesh.shape[0]))
        return basis_matrix

    def compute_abs_z(self, z):
        abs_z = np.absolute(z)
        return abs_z

    def __call__(self, samples):
        z = samples[0, :]
        # z in [0,10*pi]

        basis_matrix = self.basis_matrix()

        coeffs = np.zeros((self.num_terms, samples.shape[1]), float)
        abs_z = self.compute_abs_z(z)
        for k in range(self.num_terms):
            ck = np.exp(np.sign(z)*1j)*1j**k
            ck = ck.real
            gk = ck * np.sqrt(np.pi*(2.*k+1.) / abs_z) *\
                bessel_function(k+.5, abs_z)
            # gk not defined for z=0
            coeffs[k, :] = gk
            # must divide by sqrt(2), due to using orthonormal basis with
            # respect to w=1/2, but needing orthonormal basis with respect
            # to w=1
            coeffs[k, :] /= np.sqrt(2)

        result = np.dot(basis_matrix, coeffs).T
        return result

    def generate_samples(self, num_samples):
        num_vars = 1
        return np.random.uniform(0, 10.*np.pi, (num_vars, num_samples))


class OscillatoryHighFidelityModel(OscillatoryPolyLowFidelityModel):
    def __init__(self, mesh_dof=100, num_terms=35, eps=1e-3):
        super().__init__(mesh_dof, num_terms)
        self.eps = eps

    def compute_abs_z(self, z):
        abs_z = np.absolute(z+self.eps*z**2)
        return abs_z


class OscillatorySinLowFidelityModel(OscillatoryPolyLowFidelityModel):
    def __init__(self, mesh_dof=100, num_terms=35, eps=1e-3):
        super().__init__(mesh_dof, num_terms)
        self.eps = eps

    def basis_matrix(self):
        kk = np.arange(self.num_terms)[np.newaxis, :]
        basis_matrix = np.sin(np.pi*(kk+1)*self.mesh[:, np.newaxis])
        return basis_matrix


class TestLowRankMultiFidelity(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_select_nodes(self):
        A = np.array([[1., 1., 1], [1., 2., 5.5], [1., 3., 13.]])
        A = np.random.normal(0, 1, (3, 3))
        G = np.dot(A.T, A)
        pivots, L = select_nodes(A.copy(), A.shape[1])
        numpy_L = np.linalg.cholesky(G)
        P = np.eye(pivots.shape[0])[pivots, :]
        assert np.allclose(np.dot(P, np.dot(G, P.T)),
                           np.dot(L, L.T))
        assert np.allclose(
            np.dot(P.T, np.dot(np.dot(L, L.T), P)), G)

        A = np.random.normal(0., 1., (4, 3))
        G = np.dot(A.T, A)
        pivots, L = select_nodes(A.copy(), A.shape[1])
        P = np.eye(pivots.shape[0])[pivots, :]
        assert np.allclose(np.dot(P, np.dot(G, P.T)),
                           np.dot(L, L.T))
        assert np.allclose(
            np.dot(P.T, np.dot(np.dot(L, L.T), P)), G)

    def test_select_nodes_cholesky(self):
        A = np.array([[1., 1., 1], [1., 2., 5.5], [1., 3., 13.]])
        A = np.random.normal(0, 1, (3, 3))
        G = np.dot(A.T, A)
        pivots, L = select_nodes_cholesky(A, A.shape[1])
        numpy_L = np.linalg.cholesky(G)
        P = get_pivot_matrix_from_vector(pivots, G.shape[0])
        assert np.allclose(P.dot(G).dot(P.T), L.dot(L.T))

        A = np.random.normal(0., 1., (4, 3))
        G = np.dot(A.T, A)
        pivots, L = select_nodes_cholesky(A, A.shape[1])
        numpy_L = np.linalg.cholesky(G)
        P = get_pivot_matrix_from_vector(pivots, G.shape[0])
        assert np.allclose(P.dot(G).dot(P.T), L.dot(L.T))

    def test_select_nodes_update(self):
        A = np.random.normal(0., 1., (5, 4))
        G = np.dot(A.T, A)
        pivots, L = select_nodes(A.copy(), A.shape[1], order=[1, 3, 0])
        assert np.allclose(pivots, [1, 3, 0, 2])
        P = np.eye(pivots.shape[0])[pivots, :]
        assert np.allclose(np.dot(P, np.dot(G, P.T)),
                           np.dot(L, L.T))
        assert np.allclose(
            np.dot(P.T, np.dot(np.dot(L, L.T), P)), G)

    def test_oscillatory_model(self):
        eps = 1.e-3
        mesh_dof = 100
        K = 35
        lf_model2 = OscillatorySinLowFidelityModel(mesh_dof, K)
        hf_model = OscillatoryHighFidelityModel(mesh_dof, 100, eps)
        lf_model = lf_model2

        # for tutorial
        # samples = np.array([[5]])
        # import matplotlib.pyplot as plt
        # fig,axs=plt.subplots(1,2,figsize=(2*8,6))
        # hf_model.eps=1e-2
        # axs[0].plot(hf_model.mesh,hf_model(samples)[0,:],label='$u_0$')
        # hf_model.eps=1e-3
        # axs[0].plot(hf_model.mesh,lf_model1(samples)[0,:],label='$u_1$')
        # axs[0].plot(hf_model.mesh,lf_model2(samples)[0,:],label='$u_2$')
        # axs[0].legend()

        # samples = np.linspace(0.01,np.pi*10-0.1,101)[np.newaxis,:]
        # hf_model.eps=1e-2
        # axs[1].plot(samples[0,:],hf_model(samples)[:,50],label='$u_0$')
        # hf_model.eps=1e-3
        # axs[1].plot(samples[0,:],lf_model1(samples)[:,50],label='$u_1$')
        # axs[1].plot(samples[0,:],lf_model2(samples)[:,50],label='$u_2$')

        # plt.show()
        # assert False

        # number of quantities of interest/outputs
        num_QOI = mesh_dof
        # number of random paramters/inputs
        num_dims = 1
        # number of initial candidates/snapshots for low-fidelity model
        num_lf_candidates = int(1e4)
        # number of interpolations nodes/high-fidelity runs
        num_hf_runs = 20

        num_test_samples = int(1e3)
        test_samples = hf_model.generate_samples(num_test_samples)
        hf_test_values = hf_model(test_samples)
        lf_test_values = lf_model(test_samples)

        mf_model = BiFidelityModel(lf_model, hf_model)
        mf_model.build(num_hf_runs, hf_model.generate_samples,
                       num_lf_candidates)

        # regression test. To difficult to compute a unit test
        mf_test_values = mf_model(test_samples)

        error_mf = compute_mean_l2_error(hf_test_values,
                                         mf_test_values)[1]
        assert error_mf < 1e-4  # 3.0401959914364483e-05)

        return
        # for tutorial
        hf_runs = [i*2 for i in range(1, 11)]
        error_mf = np.empty((len(hf_runs)))
        error_lf = np.empty((len(hf_runs)))
        # error_nodes = np.empty((len(hf_runs)))
        # condition = np.empty((len(hf_runs)))

        for j in range(len(hf_runs)):
            num_hf_runs = hf_runs[j]

            mf_model = BiFidelityModel(lf_model, hf_model)
            mf_model.build(num_hf_runs, hf_model.generate_samples,
                           num_lf_candidates)

            mf_test_values = mf_model(test_samples)

            error_mf[j] = compute_mean_l2_error(hf_test_values,
                                                mf_test_values)[1]
            error_lf[j] = compute_mean_l2_error(hf_test_values,
                                                lf_test_values)[1]
            print("|hf-lf|", error_lf[j])
            print("|hf-mf|", error_mf[j])

        # plt.semilogy(hf_runs,error_mf,label=f'$K={K}$')
        # plt.show()


if __name__ == "__main__":
    low_rank_mf_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLowRankMultiFidelity)
    unittest.TextTestRunner(verbosity=2).run(low_rank_mf_test_suite)
