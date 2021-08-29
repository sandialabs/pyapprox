from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import gaussian_kde as kde, norm as normal_rv
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.special import gammaln

from pyapprox.configure_plots import *
from pyapprox.visualization import *


class Density:
    def __init__(self, num_vars, plot_limits=None):
        self.num_dims = num_vars
        self.plot_limits = plot_limits

    def pdf(self, samples):
        assert "Virtual class. must define pdf() function"

    def generate_samples(self, num_samples, rng=np.random):
        assert "Virtual class. must define generate_samples() function"

    def plot_density(self, num_samples_1d=100, plot_limits=None, show=False,
                     figname=None, ls='-', label=None, color=None, ax=None,
                     colorbar_lims=None, cmap=mpl.cm.coolwarm,
                     num_contour_levels=30):
        if plot_limits is None:
            plot_limits = self.plot_limits

        if ax is None:
            ax = plt
        if len(plot_limits) == 4:
            X, Y, Z = get_meshgrid_function_data(
                self.pdf, plot_limits, num_samples_1d)
            ax = plot_contours(X, Y, Z, ax, num_contour_levels=num_contour_levels,
                               offset=0, cmap=cmap, zorder=None)
        else:
            plot_grid = np.linspace(
                plot_limits[0], plot_limits[1], num_samples_1d)
            data = self.pdf(plot_grid.reshape(1, num_samples_1d))
            ax.plot(plot_grid.squeeze(), data.squeeze(), ls, lw=3,
                    label=label, color=color)

        if figname is not None:
            plt.savefig(figname)

        if show:
            plt.show()

        return ax

    def __call__(self, samples):
        return self.pdf(samples)

    def num_vars(self):
        import warnings
        warnings.warn("Use of `num_vars()` will be deprecated. Access property `.nvars` instead", 
                      PendingDeprecationWarning)
        return self.num_dims
    
    @property
    def nvars(self):
        return self.num_dims


class UniformDensity(Density):
    def __init__(self, ranges):
        self.ranges = np.asarray(ranges)
        num_dims = self.ranges.shape[0] // 2
        variance = (self.ranges[1]-self.ranges[0])**2/12.
        self.covariance = np.eye(num_dims)  # hack*variance
        self.chol_factor = cholesky(self.covariance, lower=True)
        self.mean = 0.5*(self.ranges[::2]+self.ranges[1::2])
        self.volume = 1.
        for d in range(num_dims):
            self.volume *= (self.ranges[2*d+1] - self.ranges[2*d])
        Density.__init__(self, num_dims, plot_limits=self.ranges)

    def pdf(self, samples):
        """
        Return the values of of the random variable PDF at a set of samples.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        density_vals : (num_samples x 1) vector
            The values of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_samples = samples.shape[1]
        density_vals = 1. / self.volume * np.ones(num_samples)
        for i in range(self.num_dims):
            I = np.where(
                (samples[i, :] < self.ranges[2*i]) |
                (samples[i, :] > self.ranges[2*i+1]))[0]
            density_vals[I] = 0.
        return density_vals

    def generate_samples(self, num_samples):
        """
        Generate random samples from the random variable.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        Returns
        -------
        samples : (num_dims x num_samples) matrix
            Random samples drawn from the density
        """
        samples = np.empty((self.num_dims, num_samples), float)
        for i in range(self.num_dims):
            samples[i, :] = np.random.uniform(
                self.ranges[2*i], self.ranges[2*i+1], (num_samples))
        return samples

    def gradient(self, samples):
        """
        Return the gradient of the random variable PDF.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        gradients : (num_dims x num_samples) matrix
            The gradients of the the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_dims, num_samples = samples.shape
        return np.zeros((self.num_dims, num_samples))

    def log_pdf(self, samples):
        """
        Return the logarithm of the random variable PDF.

        This function is used for maximum likelihood optimization.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        log_density_vals : (num_samples x 1) vector
            The values of the logarithm of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_dims, num_samples = samples.shape
        log_density_vals = -np.ones((num_samples), float)*np.log(self.volume)
        return log_density_vals

    def log_pdf_gradient(self, samples):
        """
        Return the gradient of the logarithm of the random variable PDF.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        gradients : (num_dims x num_samples) matrix
            The gradients of the logarithm of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_dims, num_samples = samples.shape
        return np.zeros((self.num_dims, num_samples))


def map_from_canonical_gaussian(stdnormal_samples, mean,
                                covariance_chol_factor=None,
                                covariance_sqrt=None):
    """
    Transform independent stanard Normal to samples drawn from a correlated 
    multivariate Gaussian distribution.

    One and only one of covariance_chol_factor and covariance_sqrt must be 
    not None.

    Parameters
    ----------
    stdnormal_samples : np.ndarray (num_vars,num_samples)
        The independent standard Normal samples


    mean : np.ndarray (num_vars)
        The mean of the multivariate Gaussian

    covariance_chol_factor : np.ndarray (num_vars,num_vars)
        The cholesky factorization of the Gaussian covariance.

    covariance_sqrt : callable
        correlated_samples = covariance_sqrt(stdnormal_samples)
        An operator that applies the sqrt of the Gaussian covariance to a set 
        of vectors. Useful for large scale applications.

    Returns
    -------
    correlated_samples : np.ndarray (num_vars,num_samples)
        The generated correlated samples

    """
    if covariance_chol_factor is None and covariance_sqrt is None:
        raise Exception('cannot specify both covariance_chol_factor and sqrt')

    if covariance_chol_factor is None and covariance_sqrt is None:
        correlated_samples = stdnormal_samples
    elif covariance_sqrt is None:
        if covariance_chol_factor.ndim == 2:
            def covariance_sqrt(x): return np.dot(covariance_chol_factor, x)
        else:
            def covariance_sqrt(x): return (x.T*covariance_chol_factor).T

    correlated_samples = covariance_sqrt(stdnormal_samples)

    assert mean.ndim == 1
    return mean[:, np.newaxis]+correlated_samples


def map_to_canonical_gaussian(correlated_samples, mean, covariance_chol_factor):
    """
    Transform samples drawn from a correlated multivariate Gaussian distribution
    into a set of independent standard Normal samples.

    Parameters
    ----------
    correlated_samples : np.ndarray (num_vars,num_samples)
        The correlated samples

    mean : np.ndarray (num_vars)
        The mean of the multivariate Gaussian

    covariance_chol_factor : np.ndarray (num_vars,num_vars)
        The cholesky factorization of the Gaussian covariance. 
        If argument is a 1D array then covariance_chol_factor is the diagonal
        of the cholesky factorization. This is useful for independent 
        multivarite Gaussians.

    Returns
    -------
    stdnormal_samples : np.ndarray (num_vars,num_samples)
        The independent standard Normal samples
    """
    assert mean.ndim == 1
    stdnormal_samples = correlated_samples-mean[:, np.newaxis]
    if covariance_chol_factor.ndim == 2:
        # stdnormal_samples = np.linalg.solve_triangular(
        stdnormal_samples = solve_triangular(
            covariance_chol_factor, stdnormal_samples, lower=True)
    else:
        stdnormal_samples = (stdnormal_samples.T/covariance_chol_factor).T
    return stdnormal_samples


class NormalDensity(Density):
    def __init__(self, mean=None, covariance=None, covariance_chol_factor=None):
        Density.__init__(self, None, None)

        # allow for density to be initialized empty
        if mean is not None:
            self.set_mean(mean)
        if covariance is not None or covariance_chol_factor is not None:
            self.set_covariance(covariance, covariance_chol_factor)

    def set_mean(self, mean):
        self.mean = mean
        if np.isscalar(self.mean):
            self.mean = np.asarray([self.mean], dtype=float)
        if self.mean.ndim == 2:
            self.mean = self.mean.squeeze()
            assert self.mean.ndim == 1

        if self.num_dims is None:
            self.num_dims = self.mean.shape[0]
        # else: we are just updating the mean

    def set_covariance(self, covariance=None, covariance_chol_factor=None):
        assert self.mean is not None

        if covariance is not None:
            if np.isscalar(covariance):
                self.covariance = np.eye(self.num_dims) * covariance
            else:
                assert covariance.shape[0] == self.num_dims
                assert covariance.shape[0] == covariance.shape[1]
                self.covariance = covariance
            self.chol_factor = cholesky(self.covariance, lower=True)
        else:
            assert covariance_chol_factor is not None
            self.chol_factor = covariance_chol_factor
            self.covariance = np.dot(self.chol_factor, self.chol_factor.T)
        self.covariance_determinant = np.linalg.det(self.covariance)
        try:
            self.normalization_factor = 1./(np.sqrt(
                (2.*np.pi)**self.num_dims*self.covariance_determinant))
        except:
            print(('normalization_factor of Gaussian PDF could not be ',))
            print('computed. Dimensionality is likely to large')
            self.normalization_factor = 1
        self.covariance_inv = np.linalg.inv(self.covariance)

        intervals = []
        for i in range(self.num_dims):
            interval = normal_rv.interval(.999, 0., 1.)
            intervals.append([interval[0], interval[1]])
        intervals = np.array(intervals)

        intervals = np.dot(self.chol_factor, intervals) +\
            np.tile(self.mean.reshape(self.mean.shape[0], 1), (1, 2))

        # rotation can make max min and vice-versa
        # when density is correlated plot limits are the limits along the
        # rotated axes. I could project back onto original axes but I have
        # not bothered
        self.plot_limits = np.empty((2*self.num_dims), float)
        for i in range(self.num_dims):
            self.plot_limits[2*i] = intervals[i, :].min()
            self.plot_limits[2*i+1] = intervals[i, :].max()

    def plot_contours(self, show=False, ls='-', color='k', label=None,
                      num_contours=4, ax=None, plot_mean=True):
        return plot_gaussian_contours(
            self.mean, self.chol_factor, show, ls, color, label, num_contours,
            ax, plot_mean)[0]

    def pdf(self, samples):
        """
        Return the values of of the random variable PDF at a set of samples.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        density_vals : (num_samples x 1) vector
            The values of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')

        #from scipy.stats import multivariate_normal
        # return multivariate_normal.pdf(samples.T, mean=self.mean, cov=self.covariance);

        # scipy.stats.norm() scale parameter is the np.sqrt(variance)
        num_dims, num_samples = samples.shape
        assert num_dims == self.num_dims
        res = samples - self.mean[:, np.newaxis]
        scaled_res = np.dot(self.covariance_inv, res)
        density_vals = np.empty((num_samples, 1), float)
        for i in range(num_samples):
            density_vals[i] = np.exp(-0.5*np.dot(res[:, i].T,
                                                 scaled_res[:, i]))
        density_vals *= self.normalization_factor
        return density_vals

    def gradient(self, samples):
        """
        Return the gradient of the random variable PDF.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        gradients : (num_dims x num_samples) matrix
            The gradients of the the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_samples = samples.shape[1]
        gradients = np.empty((self.num_dims, num_samples))
        for i in range(num_samples):
            gradients[:, i] = self.pdf(
                samples[:, i])*np.dot(
                    self.covariance_inv, (self.mean-samples[:, i]))
        return gradients

    def log_pdf(self, samples):
        """
        Return the logarithm of the random variable PDF.

        This function is used for maximum likelihood optimization.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        log_density_vals : (num_samples x 1) vector
            The values of the logarithm of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_dims, num_samples = samples.shape
        res = samples-self.mean[:, np.newaxis]
        scaled_res = np.dot(self.covariance_inv, res)
        log_density_vals = np.empty((num_samples, 1), float)
        for i in range(num_samples):
            log_density_vals[i] = -0.5*np.dot(res[:, i], scaled_res[:, i])
        log_density_vals += np.log(self.normalization_factor)
        return log_density_vals

    def log_pdf_gradient(self, samples):
        """
        Return the gradient of the logarithm of the random variable PDF.

        Parameters
        ----------
        samples : (num_dims x num_samples) matrix
            Coordinates at which to evaluate the PDF

        Returns
        -------
        gradients : (num_dims x num_samples) matrix
            The gradients of the logarithm of the PDF at samples
        """
        if samples.ndim == 1:
            if self.num_dims == 1:
                samples = samples.reshape(1, samples.shape[0])
            elif self.num_dims == samples.shape[0]:
                samples = samples.reshape(samples.shape[0], 1)
            else:
                raise Exception('samples inconsistent with dimension')
        num_dims, num_samples = samples.shape
        gradients = np.empty((self.num_dims, num_samples))
        for i in range(num_samples):
            gradients[:, i] = -np.dot(
                self.covariance_inv, (samples[:, i]-self.mean))
        return gradients

    def generate_samples(self, num_samples, return_iid_samples=False):
        """
        Generate random samples from the random variable.

        Parameters
        ----------
        num_samples : integer
            The number of samples to generate

        return_iid_samples : boolean
            True -  return the standard Normal random samples used to
                    generate the correlated samples
            False - only return samples

        Returns
        -------
        samples : (num_dims x num_samples) matrix
            Random samples drawn from the density

        iid_samples : (num_dims x num_samples) matrix
            Random samples drawn from the standard normal density which
            were used to generate samples. Only returned if
            return_iid_samples is True
        """
        iid_samples = np.random.normal(0., 1., (self.num_dims, num_samples))
        samples = map_from_canonical_gaussian(
            iid_samples, self.mean, self.chol_factor)
        if not return_iid_samples:
            return samples
        else:
            return samples, iid_samples


class ObsDataDensity(Density):
    def __init__(self, obs_data=None, obs_data_filename=None,
                 trans=True, marginalize_dims=None, num_skip_columns=0):
        """
        Parameters
        ----------
        trans : boolean
            Whether to transpose the data when it is loaded from file

        margninalize_dim : np.ndarray ()
            If provided specifies the dimension for which
            a n-dimension marginal is requested,  where n <= the dimension
            of the data
        """

        assert obs_data is not None or obs_data_filename is not None
        self.obs_data_filename = obs_data_filename
        self.data = obs_data

        if self.obs_data_filename is not None:
            self.data = self.get_data(self.obs_data_filename, trans,
                                      num_skip_columns)

        if marginalize_dims is not None:
            self.data = self.data[marginalize_dims, :]

        plot_limits = self.get_plot_limits()

        # data is num_samples x num_qoi, kde needs num_qoi x num_samples
        # print 'building kde...',
        self.kde = kde(self.data, 'silverman')
        # print 'kde built'

        num_dims = self.data.shape[0]
        Density.__init__(self, num_dims, plot_limits)

    def get_data(self, filename, trans, num_skip_columns):
        """
        num skip columns is useful if loading function values
        stored with corresponding parameter values in first d columns
        where d is the number of random variables
        """
        data_all = np.loadtxt(filename)
        samples = data_all[:, :num_skip_columns]
        data = data_all[:, num_skip_columns:]
        if trans:
            data = data.T
        return data

    def get_plot_limits(self):
        if self.data.shape[0] == 2:
            llim = self.data.min(axis=1)
            ulim = self.data.max(axis=1)
            limits = [llim[0], ulim[0], llim[1], ulim[1]]
            return limits
        else:
            return [self.data.min(), self.data.max()]

    def pdf(self, samples):
        return self.kde(samples)

    def evaluate(self, samples):
        return self.pdf(samples)


class TensorProductDensity(Density):
    """
    Tensor product of multivariate densities. The densities can all be 1D
    (typical) but they can have arbitray dimensions. This allows one to
    have tensor products of correlated multivariate densities which
    are indendent to other multivariate densities
    """

    def __init__(self, densities):
        self.densities = densities
        self.num_independent_densities = len(self.densities)
        num_dims = 0
        plot_limits = []
        for i in range(self.num_independent_densities):
            num_dims += self.densities[i].num_dims
            plot_limits += self.densities[i].plot_limits
        Density.__init__(self, num_dims, plot_limits)

    def pdf(self, samples):
        shift = 0
        density_vals = np.ones(samples.shape[1])
        for i in range(self.num_independent_densities):
            num_dims_i = self.densities[i].num_dims
            density_vals *= self.densities[i].pdf(
                samples[shift:shift+num_dims_i, :])
            shift += num_dims_i
        return density_vals

    def generate_samples(self, num_samples):
        samples = np.empty((self.num_dims, num_samples), float)
        shift = 0
        for i in range(self.num_independent_densities):
            num_dims_i = self.densities[i].num_dims
            samples[shift:shift+num_dims_i, :] =\
                self.densities[i].generate_samples(num_samples)
            shift += num_dims_i
        return samples

    def plot_1d_marginals(self, num_samples):
        k = 1
        for i in range(self.num_independent_densities):
            num_dims_i = self.densities[i].num_dims
            if num_dims_i == 1:
                plt.figure(k)
                density = self.densities[i]
                plot_grid = np.linspace(density.plot_limits[0],
                                        density.plot_limits[1],
                                        num_samples)
                plt.plot(plot_grid, density.pdf(
                    plot_grid.reshape(1, num_samples)).squeeze())
                k += 1
        plt.show()


def tensor_product_pdf(samples, univariate_pdfs):
    """
    It is assumed that samples are within the bounds on the univariate PDFs.
    Can run into trouble when mapping a Beta pdf on [0,1] to a 
    jacobi polynomial  on [-1,1] if beta_pdf is not mapped to [-1,1]
    by user.
    """
    num_vars, num_samples = samples.shape

    if callable(univariate_pdfs):
        univariate_pdfs = [univariate_pdfs]*num_vars
    if len(univariate_pdfs) == 1:
        univariate_pdfs = [univariate_pdfs[0]]*num_vars

    vals = np.ones((num_samples), dtype=float)
    for dd in range(num_vars):
        vals *= univariate_pdfs[dd](samples[dd, :])
    return vals


def multivariate_student_t_density(samples, mean, covariance, df, log=False):
    '''
    Evaluate the Multivariate t-student density.

    Parameters:
    -----------
    samples : np.ndarray (num_vars, num_samples)
        The samples at which to evaluate the PDF      

    mean : np.ndarray (num_vars)
        The mean of the distribution

    covariance : np.ndarray(num_vars,num_vars)
        The covariance of the distribution

    df : integer
        The degrees of freedom of the distribution

    log : boolean  
        True  - return log of density
        False - return density
    '''
    num_vars = samples.shape[0]
    L = np.linalg.cholesky(covariance)
    Lres = solve_triangular(L, samples-mean[:, np.newaxis], lower=True)
    sum_squares = np.sum(Lres**2, axis=0)
    logretval = gammaln(1.0*(num_vars + df)/2.) - \
        (gammaln(1.*df/2.) + np.sum(np.log(L.diagonal()))
         + num_vars/2. * np.log(np.pi * df)) - \
        0.5 * (df + num_vars) * np.log1p((sum_squares/df))
    if log == False:
        return(np.exp(logretval))
    else:
        return(logretval)


def plot_gaussian_contours(mean, chol_factor, show=False,
                           ls='-', color='k', label=None, num_contours=3, ax=None,
                           plot_mean=True):

    if ax is None:
        f, ax = plt.subplots(1, 1)
    import scipy
    if mean.ndim == 1:
        mean = mean[:, None]
    if mean.shape[0] != 2:
        return
    alpha = [0.67, 0.95, 0.99, 0.999]
    ellips_list = []
    for i in range(min(len(alpha), num_contours)):
        # Get endpoints of the range that contains
        # alpha percent of the distribution
        interval = scipy.stats.norm.interval(alpha[i], 0., 1.)
        radius = (interval[1]-interval[0])/2.
        x = np.linspace(-radius, +radius, 200)
        y = np.hstack((np.sqrt(radius**2-x**2), -np.sqrt(radius**2-x**2)))
        x = np.hstack((x, -x))
        samples = np.vstack((x[np.newaxis, :], y[np.newaxis, :]))
        assert samples.shape[0] == 2
        samples = np.dot(chol_factor, samples)+mean
        if i > 0:
            label = None
        ellips = ax.plot(samples[0, :], samples[1, :], ls, color=color,
                         lw=2, label=label)
        ellips_list.append(ellips[0])
        if plot_mean:
            ax.plot(mean[0], mean[1], 'o', color=color, ms=5)
    if show:
        plt.show()
    return ellips_list, ax


class EmpiricalCDF(object):
    def __init__(self, samples, weights=None):
        assert samples.ndim == 1
        self.samples = samples
        self.sorted_samples = np.sort(self.samples)
        if weights is None:
            self.ecdf = [
                (ii+1)/self.sorted_samples.shape[0]
                for ii in range(self.sorted_samples.shape[0])]
        else:
            assert weights.ndim == 1
            I = np.argsort(self.samples)
            self.ecdf = np.cumsum(weights[I])

        self.interp = interp1d(
            self.sorted_samples, self.ecdf, kind="zero", fill_value=(0, 1),
            bounds_error=False)

    def __call__(self, samples):

        assert np.isscalar(samples) or samples.ndim == 1
        return self.interp(samples)

    def integrate_cdf(self):
        vals = np.cumsum(np.diff(self.sorted_samples)*self.ecdf[:-1])
        return np.append(0, vals)
