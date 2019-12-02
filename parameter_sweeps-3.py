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