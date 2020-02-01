import unittest
from pyapprox.multilevel_gp import *
import matplotlib.pyplot as plt
from functools import partial

class TestMultilevelGP(unittest.TestCase):
    def test_multilevel_kernel(self):
        
        nsamples = int(1e5)
        XX1 = np.linspace(-1,1,5)[:,np.newaxis]
        XX2 = np.linspace(-1,1,3)[:,np.newaxis]

        kernel1 = partial(kernel_ff,length_scale=1)
        kernel2 = partial(kernel_ff,length_scale=2)
        
        nvars1 = XX1.shape[0]
        nvars2 = XX2.shape[0]
        samples1 = np.random.normal(0,1,(nvars1,nsamples))
        samples2 = np.random.normal(0,1,(nvars2,nsamples))

        #y1 = f1(x1), x1 \subseteq x2
        #y2 = p12*f1(x2)+d2(x2)
        p12=2
        shared_idx = [0,2,4]
        YY1 = np.linalg.cholesky(kernel1(XX1,XX1)).dot(samples1)
        YY2 = np.linalg.cholesky(kernel2(XX2,XX2)).dot(samples2)+\
            p12*np.linalg.cholesky(kernel1(XX2,XX2)).dot(samples1[shared_idx,:])
        
        print(YY1.mean(axis=1))
        print(YY2.mean(axis=1))

        YY1_centered = YY1-YY1.mean(axis=1)[:,np.newaxis]
        YY2_centered = YY2-YY2.mean(axis=1)[:,np.newaxis]

        cov11 = np.cov(YY1)
        cov12 = YY1[shared_idx,:].dot(YY2.T)/nsamples
        cov22 = np.cov(YY2)
        print(cov11-kernel1(XX1,XX1))
        print(kernel1(XX1,XX2))
        print(cov12-p12*kernel1(XX1[shared_idx,:],XX2))
        print(cov22-(kernel2(XX2,XX2)+p12**2*kernel1(XX2,XX2)))
    
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
    
    
