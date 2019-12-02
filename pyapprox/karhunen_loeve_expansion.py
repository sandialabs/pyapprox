from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from scipy.optimize import brenth
def exponential_kle_eigenvalues(sigma2,corr_len,omega):
    return sigma2*2.*corr_len/(1.+(omega*corr_len)**2)

def exponential_kle_basis(x, corr_len, sigma2, omega):
    r"""
    Basis for the kernel K(x,y)=\sigma^2\exp(-|x-y|/l)

    Parameters
    ----------
    x : np.ndarray (num_spatial_locations)
        The spatial coordinates of the nodes defining the random field in [0,1] 

    corr_len : double
        correlation length l of the covariance kernel

    sigma2 : double
        variance sigma^2 of the covariance kernel

    omega : np.ndarray (num_vars)
        The roots of the characteristic equation

    Returns
    -------
    basis_vals : np.ndarray (num_spatial_locations, num_vars)
        The values of every basis at each of the spatial locations
        basis_vals are multiplied by eigvals

    eig_vals : np.ndarray (num_vars)
        The eigemvalues of the kernel. The influence of these is already
        included in basis_vals, but these values are useful for plotting.
    """
    num_vars = omega.shape[0]
    assert x.ndim==1
    num_spatial_locations = x.shape[0]
    basis_vals = np.empty((num_spatial_locations,num_vars), float)
    eigvals = exponential_kle_eigenvalues(sigma2,corr_len,omega)
    for j in range(num_vars//2):
        frac = np.sin(omega[j])/(2*omega[j])
        basis_vals[:,2*j]=np.cos(omega[j]*(x-0.5))/np.sqrt(0.5+frac)*eigvals[2*j]
        basis_vals[:,2*j+1]=np.sin(omega[j]*(x-0.5))/np.sqrt(0.5-frac)*eigvals[2*j+1]
    if num_vars%2==1:
        frac = np.sin(omega[-1])/(2*omega[-1])
        basis_vals[:,-1]=np.cos(omega[-1]*(x-0.5))/np.sqrt(0.5+frac)*eigvals[-1]
    return basis_vals

def compute_roots_of_exponential_kernel_characteristic_equation(
        corr_len, num_vars, maxw=None, plot=False):
    r"""
    Compute roots of characteristic equation of the exponential kernel.

    Parameters
    ----------
    corr_len : double
        Correlation length l of the covariance kernel

    num_vars : integer
        The number of roots to compute

    maxw : float
         The maximum range to search for roots

    Returns
    -------
    omega : np.ndarray (num_vars)
        The roots of the characteristic equation
    """
    func = lambda w: (1-corr_len*w*np.tan(w/2.))*(corr_len*w+np.tan(w/2.))
    omega = np.empty((num_vars),float)
    import scipy
    dw = 1e-2; tol = 1e-5
    if maxw is None:
        maxw=num_vars*5
    w = np.linspace(dw,maxw,maxw//dw)
    fw = func(w)
    fw_sign = np.sign(fw)
    signchange = ((np.roll(fw_sign, -1) - fw_sign) != 0).astype(int)
    I = np.where(signchange)[0]
    wI = w[I]
    fail = False
    if I.shape[0]<num_vars+1:
        msg = 'Not enough roots extend maxw'
        print (msg)
        fail = True

    if not fail:
        prev_root = 0
        for ii in range(num_vars):
            root = brenth(
                func, wI[ii], wI[ii+1], maxiter=1000, xtol=tol)
            assert root>0 and abs(root-prev_root)>tol*100
            omega[ii]=root
            prev_root = root
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(w,fw,'-ko')
        plt.plot(wI,fw[I],'ro')
        plt.plot(omega,func(omega),'og',label='roots found')
        plt.ylim([-100,100])
        plt.legend()
        plt.show()
    if fail:
        raise Exception(msg)
    return omega

def evaluate_exponential_kle(mean_field, corr_len, sigma2, x, z, basis_vals=None):
    r"""
    Return realizations of a random field with a exponential covariance kernel.

    Parameters
    ----------
    mean_field : vector (num_spatial_locations)
       The mean temperature profile

    corr_len : double
        Correlation length l of the covariance kernel

    sigma2 : double
        The variance \sigma^2  of the random field

    x : np.ndarray (num_spatial_locations)
        The spatial coordinates of the nodes defining the random field in [0,1] 

    z : np.ndarray (num_vars, num_samples)
        A set of random samples

    basis_vals : np.ndarray (num_spatial_locations, num_vars)
        The values of every basis at each of the spatial locations

    Returns
    -------
    vals : vector (num_spatial_locations x num_samples)
        The values of the temperature profile at each of the spatial locations
    """
    if z.ndim==1:
        z = z.reshape((z.shape[0],1))

    if np.isscalar(x):
        x = np.asarray([x])
        
    assert np.all((x>=0.) & (x<=1.))
    
    num_vars, num_samples = z.shape
    num_spatial_locations = x.shape[0]

    if basis_vals is None:
        omega = compute_roots_of_exponential_kernel_characteristic_equation(
            corr_len, num_vars)
        basis_vals = exponential_kle_basis(x, corr_len, sigma2, omega)

    assert num_vars == basis_vals.shape[1]
    assert basis_vals.shape[0]==x.shape[0]


    if np.isscalar(mean_field):
        mean_field = mean_field*np.ones(num_spatial_locations)
    elif callable(mean_field):
        mean_field = mean_field(x)

    assert mean_field.ndim==1
    assert mean_field.shape[0] == num_spatial_locations

    vals = mean_field[:,np.newaxis]+np.dot(basis_vals,z)
    assert vals.shape[1] == z.shape[1]
    return vals


class KLE1D(object):
    def __init__(self,kle_opts):
        self.mean_field = kle_opts['mean_field']
        self.sigma2 = kle_opts['sigma2']
        self.corr_len = kle_opts['corr_len']
        self.num_vars = kle_opts['num_vars']
        self.use_log = kle_opts.get('use_log',True)

        self.basis_vals = None

        self.omega=\
            compute_roots_of_exponential_kernel_characteristic_equation(
                self.corr_len, self.num_vars, maxw=kle_opts.get('maxw',None))

    def update_basis_vals(self,mesh):
        if self.basis_vals is None:
            self.basis_vals = exponential_kle_basis(
                mesh,self.corr_len,self.sigma2,self.omega)
        
    def __call__(self,sample,mesh):
        self.update_basis_vals(mesh)
        vals = evaluate_exponential_kle(
            self.mean_field, self.corr_len, self.sigma2, mesh, sample,
            self.basis_vals)
        if self.use_log:
            return np.exp(vals)
        else:
            return vals

def correlation_function(X,s,corr_type):
    assert X.ndim==2
    from scipy.spatial.distance import pdist, squareform
    # this is an NxD matrix, where N is number of items and D its 
    # dimensionalities
    pairwise_dists = squareform(pdist(X.T, 'euclidean'))
    if corr_type=='gauss':
        K = np.exp(-pairwise_dists ** 2 / s ** 2)
    elif corr_type=='exp':
        K = np.exp(-np.absolute(pairwise_dists) / s)
    else:
        raise Exception('incorrect corr_type')
    assert K.shape[0]==X.shape[1]
    return K

def compute_nobile_diffusivity_eigenvectors(num_vars,corr_len,mesh):
    domain_len = 1
    assert mesh.ndim==1
    mesh = mesh[:,np.newaxis]
    sqrtpi = np.sqrt(np.pi)
    Lp = max(domain_len,2*corr_len)
    L = corr_len/Lp
    sqrtpi = np.sqrt(np.pi)
    nn = np.arange(2,num_vars+1)
    eigenvalues = np.sqrt(sqrtpi*L)*np.exp(-((np.floor(nn/2)*np.pi*L))**2/8)
    eigenvectors = np.empty((mesh.shape[0],num_vars-1))
    eigenvectors[:,::2] = np.sin(((np.floor(nn[::2]/2)*np.pi*mesh))/Lp)
    eigenvectors[:,1::2] = np.cos(((np.floor(nn[1::2]/2)*np.pi*mesh))/Lp)
    eigenvectors *= eigenvalues
    return eigenvectors

def nobile_diffusivity(eigenvectors,corr_len,samples):
    if samples.ndim==1:
        samples = samples.reshape((samples.shape[0],1))
    assert samples.ndim==2
    assert samples.shape[0]==eigenvectors.shape[1]+1
    domain_len = 1
    Lp = max(domain_len,2*corr_len)
    L = corr_len/Lp
    field  = eigenvectors.dot(samples[1:,:])
    field += 1+samples[0,:]*np.sqrt(np.sqrt(np.pi)*L/2)
    field  = np.exp(field)+0.5
    return field
