import unittest
from pyapprox.multilevel_gp import *
import matplotlib.pyplot as plt

class TestMultilevelGP(unittest.TestCase):
    def test_2_models(self):
        nvars, nmodels = 1,2
        f1 = lambda x: 2*f2(x)+x**2
        f2 = lambda x: np.cos(2*np.pi*x)

        x2 = np.atleast_2d(np.linspace(-1,1,5))
        x1 = np.atleast_2d(np.linspace(-1,1,3))
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
        fig,axs = plt.subplots(1,1)
        gp.plot_1d(100,[-1,1],axs)


if __name__== "__main__":    
    multilevel_gp_test_suite=unittest.TestLoader().loadTestsFromTestCase(
         TestMultilevelGP)
    unittest.TextTestRunner(verbosity=2).run(multilevel_gp_test_suite)
    
    
