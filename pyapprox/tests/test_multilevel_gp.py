import unittest
from pyapprox.multilevel_gp import *
import matplotlib.pyplot as plt

class TestMultilevelGP(unittest.TestCase):
    def test_multilevel_kernel(self):
        def f1(XX1):
            K1 = kernel_ff(XX1,XX1,length_scale=1)
            K1_chol_factor = np.linalg.cholesky(K1)
            nvars1 = K1.shape[0]
            samples = K1_chol_factor.dot(np.random.normal(0,1,(nvars1,nsamples)))
            return samples

        def f2(XX1,XX2):
            K2 = kernel_ff(XX2,XX2,length_scale=2)
            K2_chol_factor = np.linalg.cholesky(K2)
            nvars2 = K2.shape[0]
            samples = f1(XX1)+K2_chol_factor.dot(
                np.random.normal(0,1,(nvars2,nsamples)))
            return samples

        nsamples = int(1e5)
        XX1 = np.linspace(-1,1,3)[:,np.newaxis]
        XX2 = np.linspace(-1,1,5)[:,np.newaxis]

        YY1 = f1(XX1)
        YY2 = f2(XX1,XX2)
        
        print(np.cov(YY1))
        print(K1)
    
    def test_2_models(self):
        nvars, nmodels = 1,2
        f1 = lambda x: (2*f2(x)+x.T**2)
        f2 = lambda x: np.cos(2*np.pi*x).T

        x1 = np.atleast_2d(np.linspace(-1,1,5))
        x2 = np.atleast_2d(np.linspace(-1,1,3))
        samples = [x1,x2]
        values = [f(x) for f,x in zip([f1,f2],samples)]
        nsamples_per_model = [s.shape[1] for s in samples]

        length_scale=[1]*(nmodels*(nvars+1)-1);
        length_scale_bounds=[(1e-1,10)]*(nmodels*nvars) + [(1e-1,1)]*(nmodels-1)
        noise_level=0.02; n_restarts_optimizer=3
        mlgp_kernel  = 1*MultilevelGPKernel(
            nvars, nsamples_per_model, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)
        mlgp_kernel += WhiteKernel( # optimize gp noise
            noise_level=noise_level, noise_level_bounds=(1e-8, 1))

        gp = MultilevelGP(mlgp_kernel)
        gp.set_data(samples,values)
        gp.fit()

        print(mlgp_kernel)
        
        fig,axs = plt.subplots(1,1)
        gp.plot_1d(100,[-1,1],axs)
        xx = np.linspace(-1,1,101)[np.newaxis,:]
        print(xx.shape,f2(xx).shape)
        axs.plot(xx[0,:],f2(xx),'r')
        axs.plot(xx[0,:],f1(xx),'b--')
        axs.plot(x2[0,:],f2(x2),'ro')
        plt.show()


if __name__== "__main__":    
    multilevel_gp_test_suite=unittest.TestLoader().loadTestsFromTestCase(
         TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)
    
    
