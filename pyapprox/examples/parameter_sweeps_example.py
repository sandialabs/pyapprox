import os, shutil
from pyapprox.models.genz import GenzFunction
from pyapprox.parameter_sweeps import *
from pyapprox.variable_transformations import \
    define_iid_random_variable_transformation
from scipy.stats import uniform
    
def bivariate_uniform_example():
        
    num_vars = 2
    var_trans = define_iid_random_variable_transformation(
        uniform(),num_vars)
    c = np.random.uniform(0.,1.,num_vars)
    c*=20/c.sum()
    w = np.zeros_like(c); w[0] = np.random.uniform(0.,1.,1)
    model = GenzFunction( "oscillatory", num_vars,c=c,w=w)
    generate_parameter_sweeps_and_plot(
        model,{'ranges':var_trans.get_ranges()},
        "parameter-sweeps-test-dir/genz-parameter-sweeps-test.npz",
        'hypercube',num_sweeps=2,show=False)
    # png file save in test-dir do not remove dir if want to check png file
    shutil.rmtree('parameter-sweeps-test-dir/')
    plt.show()

def bivariate_gaussian_example():
    np.random.seed(1)
    num_vars = 2
    sweep_radius = 2
    num_samples_per_sweep = 50
    num_sweeps=2
    mean = np.ones(num_vars)
    covariance = np.asarray([[1,0.7],[0.7,1.]])

    function = lambda x: np.sum((x-mean[:,np.newaxis])**2,axis=0)[:,np.newaxis]

    covariance_chol_factor = np.linalg.cholesky(covariance)
    covariance_sqrt = lambda x : np.dot(covariance_chol_factor,x)
    
    opts = {'mean':mean,'covariance_sqrt':covariance_sqrt,
            'sweep_radius':sweep_radius}
    generate_parameter_sweeps_and_plot(
        function,opts,
        "parameter-sweeps-test-dir/additive-quadaratic-parameter-sweeps-test.npz",
        'gaussian',num_sweeps=num_sweeps,show=False)
    # png file save in test-dir do not remove dir if want to check png file
    shutil.rmtree('parameter-sweeps-test-dir/')

    samples, active_samples, W = get_gaussian_parameter_sweeps_samples(
        mean,covariance,sweep_radius=sweep_radius,
        num_samples_per_sweep=num_samples_per_sweep,num_sweeps=num_sweeps)

    from pyapprox.density import plot_gaussian_contours
    f, ax = plt.subplots(1,1)
    plot_gaussian_contours(mean,np.linalg.cholesky(covariance),ax=ax)
    ax.plot(samples[0,:],samples[1,:],'o')
    ax.plot(samples[0,[0,num_samples_per_sweep-1]],
                 samples[1,[0,num_samples_per_sweep-1]],'sr')
    if num_sweeps>1:
        ax.plot(samples[0,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
                 samples[1,[num_samples_per_sweep,2*num_samples_per_sweep-1]],
                 'sr')
    plt.show()

if __name__== "__main__":
    #bivariate_uniform_example()
    bivariate_gaussian_example()
