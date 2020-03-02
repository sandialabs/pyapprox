import unittest
from pyapprox.multilevel_gp import *
import matplotlib.pyplot as plt
from functools import partial

class TestMultilevelGP(unittest.TestCase):
    def test_multilevel_kernel(self):
        
        nsamples = int(1e6)
        XX1 = np.linspace(-1,1,5)[:,np.newaxis]
        shared_idx = [0,2,4]
        XX2 = XX1[shared_idx]

        length_scales = [1,2]
        kernel1 = partial(kernel_ff,length_scale=length_scales[0])
        kernel2 = partial(kernel_ff,length_scale=length_scales[1])
        
        nvars1 = XX1.shape[0]
        nvars2 = XX2.shape[0]
        samples1 = np.random.normal(0,1,(nvars1,nsamples))
        samples2 = np.random.normal(0,1,(nvars2,nsamples))

        #y1 = f1(x1), x1 \subseteq x2
        #y2 = p12*f1(x2)+d2(x2)
        p12=2
        YY1 = np.linalg.cholesky(kernel1(XX1,XX1)).dot(samples1)
        # cannout use kernel1(XX2,XX2) here because this will generate different
        # samples to those used in YY1
        dsamples = np.linalg.cholesky(kernel2(XX2,XX2)).dot(samples2)
        YY2 = p12*np.linalg.cholesky(kernel1(XX1,XX1)).dot(samples1)[shared_idx,:]+dsamples

        assert np.allclose(YY1[shared_idx],(YY2-dsamples)/p12)
        
        assert np.allclose(YY1.mean(axis=1),0,atol=1e-2)
        assert np.allclose(YY2.mean(axis=1),0,atol=1e-2)

        YY1_centered = YY1-YY1.mean(axis=1)[:,np.newaxis]
        YY2_centered = YY2-YY2.mean(axis=1)[:,np.newaxis]

        cov11 = np.cov(YY1)
        assert np.allclose(
            YY1_centered[shared_idx,:].dot(YY1_centered[shared_idx,:].T)/(nsamples-1),
            cov11[np.ix_(shared_idx,shared_idx)])
        assert np.allclose(cov11,kernel1(XX1,XX1),atol=1e-2)
        cov22 = np.cov(YY2)
        assert np.allclose(
            YY2_centered.dot(YY2.T)/(nsamples-1),cov22)
        #print(cov22-(kernel2(XX2,XX2)+p12**2*kernel1(XX2,XX2)))
        assert np.allclose(cov22,(kernel2(XX2,XX2)+p12**2*kernel1(XX2,XX2)),atol=2e-2)
        print('Ks1',kernel1(XX2,XX2))

        cov12 = YY1_centered[shared_idx,:].dot(YY2_centered.T)/(nsamples-1)
        #print(cov11-kernel1(XX1,XX1))
        #print(cov12-p12*kernel1(XX1[shared_idx,:],XX2))
        assert np.allclose(cov12,p12*kernel1(XX1[shared_idx,:],XX2),atol=1e-2)

        nvars, nmodels = 1, 2
        nsamples_per_model = [XX1.shape[0],XX2.shape[0]]
        length_scale = length_scales+[p12]
        print(length_scale)
        length_scale_bounds=[(1e-1,10)]*(nmodels*nvars) + [(1e-1,1)]*(nmodels-1)
        mlgp_kernel  = MultilevelGPKernel(
            nvars, nsamples_per_model, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)

        XX_train = np.vstack([XX1,XX2])
        np.set_printoptions(linewidth=500)
        K = mlgp_kernel(XX_train)
        assert np.allclose(K[np.ix_(shared_idx,shared_idx)],kernel1(XX1,XX2)[shared_idx])
        assert np.allclose(K[XX1.shape[0]:,XX1.shape[0]:],cov22,atol=2e-2)
        assert np.allclose(K[shared_idx,XX1.shape[0]:],cov12,atol=2e-2)

        XX_train = np.vstack([XX1,XX2])
        K = mlgp_kernel(XX1,XX_train)
        assert np.allclose(K[:,:XX1.shape[0]],p12*kernel1(XX1,XX1))
        assert np.allclose(
            K[:,XX1.shape[0]:],p12**2*kernel1(XX1,XX2)+kernel2(XX1,XX2))
        print(K)
        
    
    def test_2_models(self):
        # TODO Add Test which builds gp on two models data separately when
        # data2 is subset data and hyperparameters are fixed.
        # Then Gp should just be sum of separate GPs. 
        
        nvars, nmodels = 1,2

        np.random.seed(2)
        #np.random.seed(3)
        #n1,n2=5,3
        n1,n2=9,5
        #n1,n2=10,9
        #n1,n2=17,9
        #n1,n2=32,17
        lb,ub=-1,1
        x1 = np.atleast_2d(np.linspace(lb,ub,n1))
        x2 = x1[:,np.random.permutation(n1)[:n2]]

        
        f1 = lambda x: (1*f2(x)+x.T**2) # change 1* to some non unitary rho
        f2 = lambda x: np.cos(2*np.pi*x).T

        def f1(x):
            return ((x.T*6-2)**2)*np.sin((x.T*6-2)*2)
        def f2(x):
            return 0.5*((x.T*6-2)**2)*np.sin((x.T*6-2)*2)+(x.T-0.5)*10. - 5

        x2 = np.array([[0.0], [0.4], [0.6], [1.0]]).T
        x1 = np.array([[0.1], [0.2], [0.3], [0.5], [0.7],
                       [0.8], [0.9], [0.0], [0.4], [0.6], [1.0]]).T
        lb,ub=0,1
        
        samples = [x1,x2]
        print(samples[0].shape)
        values = [f(x) for f,x in zip([f1,f2],samples)]
        nsamples_per_model = [s.shape[1] for s in samples]

        rho = np.ones(nmodels-1)

        n_restarts_optimizer=5
        noise_level=1e-5; 
        noise_level_bounds='fixed'
        from sklearn.gaussian_process.kernels import RBF
        def efficient_recursive_multilevel_gp(samples,values):
            nmodels = len(samples)
            nvars = samples[0].shape[0]
            shift=0
            gps = []
            for ii in range(nmodels):
                gp_kernel = RBF(
                    length_scale=.3, length_scale_bounds='fixed')#(1e-1, 1e2))
                #gp_kernel += WhiteKernel( # optimize gp noise
                #    noise_level=noise_level,
                #    noise_level_bounds=noise_level_bounds)
                gp = GaussianProcessRegressor(
                    kernel=gp_kernel,n_restarts_optimizer=n_restarts_optimizer,
                    alpha=0.0)
                gp.fit(samples[ii].T, values[ii]-shift)
                gps.append(gp)
                print('eml ii',gp.kernel_)
                if ii<nmodels-1:
                    shift = rho[ii]*gps[-1].predict(samples[ii+1].T)
            return gps

        def multilevel_predict(gps,xx):
            nmodels = len(gps)
            mean, std = gps[0].predict(xx.T,return_std=True)
            ml_mean = mean
            ml_var = std**2
            prior_var = np.diag(gps[0].kernel_(xx.T))
            #print(0,ml_var[0])
            for ii in range(1,nmodels):
                mean,std = gps[ii].predict(xx.T,return_std=True)
                ml_mean = rho[ii-1]*ml_mean + mean
                #print(gps[ii].kernel_.diag(xx.T))
                #print(std[0]**2)
                ml_var = rho[ii-1]**2*ml_var + std**2
                #prior_var = gps[ii].kernel_.diag(xx.T)+rho[ii-1]**2*prior_var
            print('prior var',prior_var)
            return ml_mean.squeeze(), np.sqrt(ml_var).squeeze()

        gps = efficient_recursive_multilevel_gp(samples,values)

        #length_scale=[1]*(nmodels*(nvars+1)-1);
        length_scale = [gp.kernel_.length_scale for gp in gps]+list(rho)
        #print(length_scale)
        length_scale_bounds=[(1e-1,10)]*(nmodels*nvars)+[(1e-1,10)]*(nmodels-1)
        #length_scale_bounds='fixed'
        mlgp_kernel  = MultilevelGPKernel(
            nvars, nsamples_per_model, length_scale=length_scale,
            length_scale_bounds=length_scale_bounds)
        #noise_level_bounds=(1e-8, 1)
        # do not use noise kernel for entire kernel
        # have individual noise kernels for each model
        #mlgp_kernel += WhiteKernel( # optimize gp noise
        #    noise_level=noise_level, noise_level_bounds=noise_level_bounds)


        gp = MultilevelGP(mlgp_kernel)
        gp.set_data(samples,values)
        gp.fit()

        print('ml',gp.kernel_)
        #assert False

        
        fig,axs = plt.subplots(1,1); axs=[axs]
        gp.plot_1d(2**8+1,[lb,ub],axs[0])
        #xx = np.linspace(lb,ub,2**8+1)[np.newaxis,:]
        xx = np.linspace(lb,ub,2**8+1)[np.newaxis,:]
        axs[0].plot(xx[0,:],f2(xx),'r')
        #axs[0].plot(xx[0,:],f1(xx),'g--')
        axs[0].plot(x1[0,:],f1(x1),'gs')

        print('when n1=17,n2=9 Warning answer seems to be off by np.sqrt(5) on most of the domain. This changes depending on number of ')
        emlgp_mean, emlgp_std = multilevel_predict(gps,xx)
        axs[0].plot(xx[0,:],emlgp_mean,'b-')
        # for ii in range(len(gps)):
        #     m,s=gp.predict(xx.T,return_std=True)
        #     axs[0].plot(xx[0,:],m+2*s,'y-')
        num_stdev=2
        gp_mean, gp_std = gp.predict(xx.T,return_std=True)
        gp_cov = gp.predict(xx.T,return_cov=True)[1]
        #print(gp_cov-gp_std**2)
        #assert np.allclose(gp_cov,gp_std**2,atol=1e-4)
        #assert np.allclose(emlgp_mean,gp_mean)
        #print(emlgp_mean,gp_mean)
        #print('s1',emlgp_std)
        #print('s2',gp_std)
        print(emlgp_std/gp_std)
        plt.plot(xx[0,:],2*gp_std+gp_mean,'-y')
        #assert np.allclose(emlgp_std,gp_std)
        axs[0].fill_between(
           xx[0,:], emlgp_mean - num_stdev*emlgp_std,
           emlgp_mean + num_stdev*emlgp_std,alpha=0.25, color='b')
        axs[0].plot(
            xx[0,:], emlgp_mean + num_stdev*emlgp_std,color='b')
        # axs[0].plot(
        #     xx[0,:], emlgp_mean - num_stdev*emlgp_std,color='b')

        # length_scale = [1]
        # hfgp_kernel = RBF(
        #     length_scale=length_scale, length_scale_bounds=(1e-1, 1e2))
        # hfgp_kernel += WhiteKernel( # optimize gp noise
        #     noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        # hfgp = GaussianProcessRegressor(
        #     kernel=hfgp_kernel,n_restarts_optimizer=n_restarts_optimizer,
        #     alpha=0.0)
        # hfgp.fit(samples[-1].T,values[-1])
        # print('hf',hfgp.kernel_)
        # hfgp_mean, hfgp_std = hfgp.predict(xx.T,return_std=True)
        # axs[0].plot(xx[0,:],hfgp_mean,'y-.')
        
        plt.show()


if __name__== "__main__":    
    multilevel_gp_test_suite=unittest.TestLoader().loadTestsFromTestCase(
         TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)
    
    
