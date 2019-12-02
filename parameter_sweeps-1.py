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