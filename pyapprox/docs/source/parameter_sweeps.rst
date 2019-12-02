Parameter Sweeps
================

The efficacy of different uncertainty quantification methods depends on the smoothness and variation of the model being analyzed. One-dimensional parameter sweeps provide a good mechanism to visualize these characteristics. In the following we show how to conduct parameter sweeps for models of bounded random variables and correlated Gaussian random variables.

Bounded variables
-----------------

.. plot::
   :align: center
   :context:
   :include-source:

   from pyapprox.examples.parameter_sweeps_example import *
   np.random.seed(1)
   num_vars = 2
   num_samples_per_sweep,num_sweeps=50,2
   var_trans = define_iid_random_variable_transformation(
   uniform(),num_vars)
   c = np.random.uniform(0.,1.,num_vars)
   c*=20/c.sum()
   w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
   model = GenzFunction( "oscillatory", num_vars,c=c,w=w )
   samples, active_samples, W = get_hypercube_parameter_sweeps_samples(
       var_trans.get_ranges(),num_samples_per_sweep=num_samples_per_sweep,
       num_sweeps=num_sweeps)
   vals = model(samples)
   plot_parameter_sweeps(active_samples, vals, None, qoi_indices=None,
                         show=False)
   plt.show()

   
In 2D we can also plot the multivariate coordinates of the univariate parameter
sweeps

.. plot::
   :align: center
   :include-source:
   :context:
       close-figs
       
   f, ax = plt.subplots(1,1)
   ax.plot(samples[0,:],samples[1,:],'o')
   # plot the samples at the begining and end of each parameter sweep
   for ii in range(num_sweeps):
       ax.plot(
       samples[0,[ii*num_samples_per_sweep,(ii+1)*num_samples_per_sweep-1]],
       samples[1,[ii*num_samples_per_sweep,(ii+1)*num_samples_per_sweep-1]],
       'sr')
   plt.show()

Note the above code can also be used for independent unbounded variables. In this situation specify ranges to be interval that captures a certain percentage of the total probability. For example give a frozen scipy random variable
var.interval(0.95) will give such a range.


Correlated Gaussian variables
-----------------------------

.. plot::
   :align: center
   :include-source:
   :context:
      close-figs
      
   from pyapprox.examples.parameter_sweeps_example import *
   num_vars = 2
   sweep_radius = 2
   num_samples_per_sweep = 50
   num_sweeps=2
   mean = np.ones(num_vars)
   covariance = np.asarray([[1,0.7],[0.7,1.]])
   
   model = lambda x: np.sum((x-mean[:,np.newaxis])**2,axis=0)[:,np.newaxis]
   
   covariance_chol_factor = np.linalg.cholesky(covariance)
   covariance_sqrt = lambda x : np.dot(covariance_chol_factor,x)
   
   samples, active_samples, W = get_gaussian_parameter_sweeps_samples(
       mean, covariance=None, covariance_sqrt=covariance_sqrt,
       sweep_radius=sweep_radius,
       num_samples_per_sweep=num_samples_per_sweep,
       num_sweeps=num_sweeps)
   vals = model(samples)
   plot_parameter_sweeps(active_samples, vals, None, qoi_indices=None,
                         show=False)
   plt.show()

In 2D we can also plot the multivariate coordinates of the univariate parameter
sweeps

.. plot::
   :align: center
   :include-source:
   :context:
       close-figs
       
   from pyapprox.density import plot_gaussian_contours
   f, ax = plt.subplots(1,1)
   ax=plot_gaussian_contours(mean,np.linalg.cholesky(covariance),ax=ax)[1]
   ax.plot(samples[0,:],samples[1,:],'o')
   ax.plot(samples[0,[0,num_samples_per_sweep-1]],
           samples[1,[0,num_samples_per_sweep-1]],'sr')
   if num_sweeps>1:
       ax.plot(samples[0,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
               samples[1,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
               'sr')
   plt.show()


   
These plots can be obtained using convienience functions which save plots to file. To run use

.. code-block::

   from pyapprox.examples.parameter_sweeps_example import *
   bivariate_uniform_example()
   bivariate_gaussian_example()


