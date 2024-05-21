from abc import ABC, abstractmethod

from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.optimize import brenth

from pyapprox.util.linalg import adjust_sign_eig


def exponential_kle_eigenvalues(sigma2, corr_len, omega):
    return 2*corr_len*sigma2/(1.+(omega*corr_len)**2)


def exponential_kle_basis(x, corr_len, sigma2, dom_len, omega):
    r"""
    Basis for the kernel K(x,y)=\sigma^2\exp(-|x-y|/l)
ab
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

    References
    ----------
    https://doi.org/10.1016/j.jcp.2003.09.015
    """
    num_vars = omega.shape[0]
    assert x.ndim == 1
    num_spatial_locations = x.shape[0]
    # bn = 1/((corr_len**2*omega[jj]**2+1)*dom_len/2+corr_len)
    # an = corr_len*omega*bn
    # basis_vals = an*np.cos(omega[jj]*x)+bn*np.cos(omega[jj]*x)
    basis_vals = np.empty((num_spatial_locations, num_vars), float)
    for jj in range(num_vars):
        bn = 1/((corr_len**2*omega[jj]**2+1)*dom_len/2.0+corr_len)
        bn = np.sqrt(bn)
        an = corr_len*omega[jj]*bn
        basis_vals[:, jj] = an*np.cos(omega[jj]*x)+bn*np.sin(omega[jj]*x)
    return basis_vals


def compute_roots_of_exponential_kernel_characteristic_equation(
        corr_len, num_vars, dom_len, maxw=None, plot=False):
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
    # def func(w): return (1-corr_len*w*np.tan(w/2.))*(corr_len*w+np.tan(w/2.))
    def func(w): return (
            (corr_len**2*w**2-1.0)*np.sin(w*dom_len) -
            2*corr_len*w*np.cos(w*dom_len))
    omega = np.empty((num_vars), float)
    dw = 1e-2
    tol = 1e-5
    if maxw is None:
        maxw = num_vars*5
    w = np.linspace(dw, maxw, int(maxw//dw))
    fw = func(w)
    fw_sign = np.sign(fw)
    signchange = ((np.roll(fw_sign, -1) - fw_sign) != 0).astype(int)
    I = np.where(signchange)[0]
    wI = w[I]
    fail = False
    if I.shape[0] < num_vars+1:
        msg = 'Not enough roots extend maxw'
        print(msg)
        fail = True

    if not fail:
        prev_root = 0
        for ii in range(num_vars):
            root = brenth(
                func, wI[ii], wI[ii+1], maxiter=1000, xtol=tol)
            assert root > 0 and abs(root-prev_root) > tol*100
            omega[ii] = root
            prev_root = root
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(w, fw, '-ko')
        plt.plot(wI, fw[I], 'ro')
        plt.plot(omega, func(omega), 'og', label='roots found')
        plt.ylim([-100, 100])
        plt.legend()
        plt.show()
    if fail:
        raise Exception(msg)
    return omega


def evaluate_exponential_kle(
        mean_field, corr_len, sigma2, dom_len, x, z, basis_vals=None, eig_vals=None):
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
    if z.ndim == 1:
        z = z.reshape((z.shape[0], 1))

    if np.isscalar(x):
        x = np.asarray([x])

    assert np.all((x >= 0.) & (x <= 1.))

    num_vars, num_samples = z.shape
    num_spatial_locations = x.shape[0]

    if basis_vals is None:
        omega = compute_roots_of_exponential_kernel_characteristic_equation(
            corr_len, num_vars, dom_len)
        eig_vals = exponential_kle_eigenvalues(sigma2, corr_len, omega)
        basis_vals = exponential_kle_basis(
            x, corr_len, sigma2, dom_len, omega)

    assert num_vars == basis_vals.shape[1]
    assert basis_vals.shape[0] == x.shape[0]

    if np.isscalar(mean_field):
        mean_field = mean_field*np.ones(num_spatial_locations)
    elif callable(mean_field):
        mean_field = mean_field(x)

    assert mean_field.ndim == 1
    assert mean_field.shape[0] == num_spatial_locations

    vals = mean_field[:, np.newaxis]+np.dot(
        np.sqrt(eig_vals)[:, None]*basis_vals, z)
    assert vals.shape[1] == z.shape[1]
    return vals, eig_vals


class KLE1D(object):
    def __init__(self, kle_opts):
        "Defined on [0, L]"
        self.mean_field = kle_opts['mean_field']
        self.sigma2 = kle_opts['sigma2']      # \sigma_Y^2 in paper
        self.corr_len = kle_opts['corr_len']  # \nu in paper
        self.num_vars = kle_opts['num_vars']
        self._use_log = kle_opts.get('use_log', True)
        self.dom_len = kle_opts["dom_len"]

        self.basis_vals = None
        self.omega =\
            compute_roots_of_exponential_kernel_characteristic_equation(
                self.corr_len, self.num_vars, self.dom_len,
                maxw=kle_opts.get('maxw', None))
        self.eig_vals = exponential_kle_eigenvalues(
            self.sigma2, self.corr_len, self.omega)

    def update_basis_vals(self, mesh):
        if self.basis_vals is None:
            self.basis_vals = exponential_kle_basis(
                mesh, self.corr_len, self.sigma2, self.dom_len, self.omega)

    def __call__(self, sample, mesh):
        self.update_basis_vals(mesh)
        vals = evaluate_exponential_kle(
            self.mean_field, self.corr_len, self.sigma2, self.dom_len, mesh,
            sample, self.basis_vals, self.eig_vals)
        if self._use_log:
            return np.exp(vals)
        else:
            return vals


def correlation_function(X, s, corr_type):
    assert X.ndim == 2
    # this is an NxD matrix, where N is number of items and D its
    # dimensionalities
    pairwise_dists = squareform(pdist(X.T, 'euclidean'))
    if corr_type == 'gauss':
        K = np.exp(-pairwise_dists ** 2 / s ** 2)
    elif corr_type == 'exp':
        K = np.exp(-np.absolute(pairwise_dists) / s)
    else:
        raise Exception('incorrect corr_type')
    assert K.shape[0] == X.shape[1]
    return K


def compute_nobile_diffusivity_eigenvectors(num_vars, corr_len, mesh):
    domain_len = 1
    assert mesh.ndim == 1
    mesh = mesh[:, np.newaxis]
    sqrtpi = np.sqrt(np.pi)
    Lp = max(domain_len, 2*corr_len)
    L = corr_len/Lp
    sqrtpi = np.sqrt(np.pi)
    nn = np.arange(2, num_vars+1)
    eigenvalues = np.sqrt(sqrtpi*L)*np.exp(-((np.floor(nn/2)*np.pi*L))**2/8)
    eigenvectors = np.empty((mesh.shape[0], num_vars-1))
    eigenvectors[:, ::2] = np.sin(((np.floor(nn[::2]/2)*np.pi*mesh))/Lp)
    eigenvectors[:, 1::2] = np.cos(((np.floor(nn[1::2]/2)*np.pi*mesh))/Lp)
    eigenvectors *= eigenvalues
    return eigenvectors


def nobile_diffusivity(eigenvectors, corr_len, samples):
    if samples.ndim == 1:
        samples = samples.reshape((samples.shape[0], 1))
    assert samples.ndim == 2
    assert samples.shape[0] == eigenvectors.shape[1]+1
    domain_len = 1
    Lp = max(domain_len, 2*corr_len)
    L = corr_len/Lp
    field = eigenvectors.dot(samples[1:, :])
    field += 1+samples[0, :]*np.sqrt(np.sqrt(np.pi)*L/2)
    field = np.exp(field)+0.5
    return field


class AbstractKLE(ABC):
    def __init__(self, mean_field=0, use_log=False,
                 quad_weights=None, nterms=None):
        self._use_log = use_log
        self._quad_weights = quad_weights
        if quad_weights is not None:
            assert quad_weights.ndim == 1
        self._set_mean_field(mean_field)
        self._set_nterms(nterms)
        self._compute_basis()

    def _set_mean_field(self, mean_field):
        self._mean_field = mean_field

    def _set_nterms(self, nterms):
        self._nterms = nterms

    @abstractmethod
    def _compute_kernel_matrix(self):
        raise NotImplementedError

    def _compute_basis(self):
        """
        Compute the KLE basis

        Parameters
        ----------
        num_nterms : integer
            The number of KLE modes. If None then compute all modes
        """
        K = self._compute_kernel_matrix()
        if self._quad_weights is None:
            eig_vals, eig_vecs = eigh(
                K, turbo=False,
                subset_by_index=(K.shape[0]-self._nterms, K.shape[0]-1))
        else:
            # see https://etheses.lse.ac.uk/2950/1/U615901.pdf
            # page 42
            sqrt_weights = np.sqrt(self._quad_weights)
            sym_eig_vals, sym_eig_vecs = eigh(
                sqrt_weights[:, None]*K*sqrt_weights, turbo=False,
                subset_by_index=(K.shape[0]-self._nterms, K.shape[0]-1))
            eig_vecs = 1/sqrt_weights[:, None]*sym_eig_vecs
            eig_vals = sym_eig_vals
        eig_vecs = adjust_sign_eig(eig_vecs)
        II = np.argsort(eig_vals)[::-1][:self._nterms]
        assert np.all(eig_vals[II] > 0), eig_vals[II]
        self._sqrt_eig_vals = np.sqrt(eig_vals[II])
        self._eig_vecs = eig_vecs[:, II]

    def __call__(self, coef):
        """
        Evaluate the expansion

        Parameters
        ----------
        coef : np.ndarray (nterms, nsamples)
            The coefficients of the KLE basis
        """
        assert coef.ndim == 2
        assert coef.shape[0] == self._nterms
        if self._use_log:
            return np.exp(self._mean_field[:, None]+self._eig_vecs.dot(coef))
        return self._mean_field[:, None] + self._eig_vecs.dot(coef)

    def __repr__(self):
        if self._nterms is None:
            return "{0}()".format(
                self.__class__.__name__)
        return "{0}(nterms={1})".format(self.__class__.__name__, self._nterms)


class MeshKLE(AbstractKLE):
    """
    Compute a Karhunen Loeve expansion of a covariance function.

    Parameters
    ----------
    mesh_coords : np.ndarray (nphys_vars, ncoords)
        The coordinates to evaluate the KLE basis

    mean_field : np.ndarray (ncoords)
        The mean field of the KLE

    use_log : boolean
        True - return exp(k(x))
        False - return k(x)
    """

    def __init__(self, mesh_coords, length_scale, sigma=1., mean_field=0,
                 use_log=False, matern_nu=np.inf, quad_weights=None,
                 nterms=None, use_torch=False):
        self._set_mesh_coordinates(mesh_coords)
        self._matern_nu = matern_nu
        self._set_lenscale(length_scale)
        self._sigma = sigma
        super().__init__(mean_field, use_log, quad_weights, nterms)
        # normalize the basis
        self._eig_vecs *= sigma*self._sqrt_eig_vals

    def _set_mean_field(self, mean_field):
        if np.isscalar(mean_field):
            mean_field = np.ones(self._mesh_coords.shape[1])*mean_field
        super()._set_mean_field(mean_field)

    def _set_nterms(self, nterms):
        if nterms is None:
            nterms = self._mesh_coords.shape[1]
        assert nterms <= self._mesh_coords.shape[1]
        self._nterms = nterms

    def _set_mesh_coordinates(self, mesh_coords):
        assert mesh_coords.shape[0] <= 2
        self._mesh_coords = mesh_coords

    def _set_lenscale(self, length_scale):
        length_scale = np.atleast_1d(length_scale)
        if length_scale.shape[0] == 1:
            length_scale = np.full(self._mesh_coords.shape[0], length_scale[0])
        assert length_scale.shape[0] == self._mesh_coords.shape[0]
        self._lenscale = length_scale

    def _compute_kernel_matrix(self):
        if self._matern_nu == np.inf:
            dists = pdist(self._mesh_coords.T / self._lenscale,
                          metric='sqeuclidean')
            K = squareform(np.exp(-.5 * dists))
            np.fill_diagonal(K, 1)
            return K

        dists = pdist(self._mesh_coords.T / self._lenscale, metric='euclidean')
        if self._matern_nu == 0.5:
            K = squareform(np.exp(-dists))
        elif self._matern_nu == 1.5:
            dists = np.sqrt(3)*dists
            K = squareform((1+dists)*np.exp(-dists))
        elif self._matern_nu == 2.5:
            K = squareform((1+dists+dists**2/3)*np.exp(-dists))
        np.fill_diagonal(K, 1)
        return K

    def __repr__(self):
        if self._nterms is None:
            return "{0}(nterms={1}, mu={2})".format(
                self.___class__.__name__, self._nterms, self._matern_nu)
        return "{0}(mu={1}, nterms={2}, lenscale={3}, sigma={4})".format(
            self.__class__.__name__, self._nterms, self._matern_nu,
            self._lenscale, self._sigma)


class TorchKLEWrapper(AbstractKLE):
    def __init__(self, kle):
        import torch
        self._kle = kle
        for attr in self._kle.__dict__.keys():
            setattr(self, attr, self._kle.__dict__[attr])

    def __call__(self, coef):
        import torch
        return torch.as_tensor(self._kle(coef), dtype=torch.double)

    def __repr__(self):
        return "TorchWrapper({0})".format(self._kle.__repr__())

    def _compute_kernel_matrix(self):
        import torch
        return torch.as_tensor(
            self.kle._compute_kernel_matrix(), dtype=torch.double)


class DataDrivenKLE(AbstractKLE):
    def __init__(self, field_samples, mean_field=0,
                 use_log=False, nterms=None):
        self._field_samples = field_samples
        super().__init__(mean_field, use_log, None, nterms)

    def _set_mean_field(self, mean_field):
        if np.isscalar(mean_field):
            mean_field = np.ones(self._field_samples.shape[0])*mean_field
        super()._set_mean_field(mean_field)

    def _set_nterms(self, nterms):
        if nterms is None:
            nterms = self._field_samples.shape[0]
        assert nterms <= self._field_samples.shape[0]
        self._nterms = nterms

    def _set_mesh_coordinaets(self, mesh_coords):
        self._mesh_coords = None

    def _compute_kernel_matrix(self):
        return np.cov(self._field_samples, rowvar=True, ddof=1)


def multivariate_chain_rule(jac_yu, jac_ux):
    r"""
    Given a function :math:`y(u)`

    .. math:: u = g(x) = (g_1(x), \ldots, g_m(x)), x \in R^n, u \in R^m

    compute

    .. math::

       \frac{\partial y}{\partial x_i} = \sum_{l=1}^m \frac{\partial y}{\partial u_l}\frac{\partial u_l}{\partial x_i} = \nabla f\cdot\frac{\partial u}{\partial x_i}

    Parameters
    ----------
    jac_yu: np.ndarray (ny, nu)
        The Jacobian of y with respect to u, i.e.

        .. math::\frac{\partial y}{\partial u_l}\frac{\partial u_l}{\partial x_i}

    jac_ux : np.ndarray (nx, nu)
        The Jacobian of u with respect to x, i.e.

        .. math:: [\frac{\partial u}{\partial x_j}

    Returns
    -------
    jac : np.ndarray (ny, nx)
        The Jacobian of u with respect to x, i.e.

        ..math:: \frac{\partial y}{\partial x_i}
    """
    gradient = jac_yu.dot(jac_ux)
    return gradient


def compute_kle_gradient_from_mesh_gradient(
        mesh_gradient, kle_basis_matrix, kle_mean, use_log, sample):
    r"""
    Compute the gradient of a function with respect to the coefficients of
    a Karhunen Loeve expansion from a gradient of the KLE projected onto the
    discrete set of points (mesh) on which the KLE is defined.

    Specifically given the KLE

    ..math:: k(z, x) = \mu(x) + \sigma\sum_{n=1}^N \lambda_n\phi_n(x)z

    defined at a set of points :math:`x_m, m=1,\ldots,M`

    this function computes

    ..math:: frac{\partial f(k(z))}{\partial z}

    from

    ..math:: frac{\partial f(k)}{\partial k}

    Parameters
    ----------
    mesh_gradient : np.ndarray (nmesh_points)
        The gradient of a function with respect to the the values :math:`k_i`
        which are the evaluations of the kle at :math`x_i`

    kle_basis_matrix : np.ndarray (nmesh_points, nterms)
        The normalized basis of the KLE :math`\sigma\lambda_j\phi_j(x_i)`

    kle_mean : np.ndarray (nmesh_points)
        The mean field of the KLE

    use_log : boolean
        True - the values :math`k_i = \exp(k_i)`
        False - the values :math`k_i = \exp(k_i)`

    sample : np.ndaray (nterms)
        The KLE coeficients used to compute :math`k_i`
    """
    assert sample.ndim == 1
    assert kle_mean.ndim == 1
    assert kle_basis_matrix.ndim == 2

    if use_log:
        kvals = np.exp(kle_mean+kle_basis_matrix.dot(sample))
        k_jac = kvals[:, None]*kle_basis_matrix
    else:
        k_jac = kle_basis_matrix

    return multivariate_chain_rule(mesh_gradient, k_jac)
