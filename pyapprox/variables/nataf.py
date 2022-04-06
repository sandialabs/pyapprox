import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from functools import partial

from pyapprox.util.utilities import scipy_gauss_hermite_pts_wts_1D
from pyapprox.util.visualization import get_meshgrid_function_data


def corrcoeffij(corrij, x_inv_cdfs,  x_means, x_stdevs, quad_rule):
    """
    Based on algorithm outlined in
    Li HongShuang et al. Chinese Science Bulletin, September 2008, vol. 53,
    no. 17, 2586-2592
    """
    # define 2d correlation matrix for idim and jdim
    corr = np.asarray([[1., corrij], [corrij, 1.]])

    # do the cholesky factorization
    chol_factor = np.linalg.cholesky(corr)

    # do the gauss-hermite quadrature
    u = np.empty((2), dtype=float)
    x = np.empty((2), dtype=float)

    corrij_corrected = 0.0
    quad_x, quad_w = quad_rule

    for ii in range(quad_x.shape[0]):
        for jj in range(quad_x.shape[0]):
            # correlate gauss hermite points
            u[0] = quad_x[ii]
            u[1] = quad_x[jj]
            z = np.dot(chol_factor, u)  # equation (18)
            # do the nataf transformation: x = F^-1(Phi(z))
            # idim: z -> u -> x
            x[0] = x_inv_cdfs[0](stats.norm.cdf(z[0]))  # equation (19)
            # jdim: z -> u -> x
            x[1] = x_inv_cdfs[1](stats.norm.cdf(z[1]))  # equation (19)

            # normalize the values to obtain the correlation coefficient
            x[0] = (x[0] - x_means[0]) / x_stdevs[0]
            x[1] = (x[1] - x_means[1]) / x_stdevs[1]

            # do the quadrature, i.e
            # evaluate the double integral in equation (17)
            corrij_corrected += quad_w[ii] * quad_w[jj] * x[0] * x[1]

    return corrij_corrected


def bisection_corrij(corrij, x_inv_cdfs, x_means, x_stdevs, quad_rule,
                     bisection_opts):

    tol = bisection_opts.get('tol', 1e-7),
    max_iterations = bisection_opts.get('max_iterations', 100)

    ii = 0
    corrij_corrected = 0.0

    xerr = 0.

    # define search interval
    dx = 0.0

    if (corrij < 0):
        dx = 1.0 + corrij
    else:
        dx = 1.0 - corrij
        dx /= 4.0

    xlower = corrij - dx
    xupper = corrij + dx
    nextX = corrij

    # Bisection loop
    while True:
        # use current x as output
        x = nextX
        # do the integration
        corrij_corrected = corrcoeffij(
            nextX, x_inv_cdfs, x_means, x_stdevs, quad_rule)

        # adjust domain for possible zero
        if (corrij < corrij_corrected):
            xupper = nextX
        else:
            xlower = nextX

        # select new center
        nextX = (xlower + xupper) / 2.0
        xerr = abs(corrij - corrij_corrected)
        ii += 1
        if ((xerr <= tol) or (ii >= max_iterations)):
            break

    return x


def transform_correlations(initial_correlation, x_marginal_inv_cdfs,
                           x_marginal_means, x_marginal_stdevs,
                           quad_rule, bisection_opts=dict()):

    num_vars = len(x_marginal_inv_cdfs)
    correlation_uspace = np.empty((num_vars, num_vars), dtype=float)
    for ii in range(num_vars):
        correlation_uspace[ii, ii] = 1.0
        for jj in range(ii+1, num_vars):
            II = [ii, jj]
            x_marginal_inv_cdfs_iijj = [
                x_marginal_inv_cdfs[ii], x_marginal_inv_cdfs[jj]]
            correlation_uspace[ii, jj] = bisection_corrij(
                initial_correlation[ii, jj], x_marginal_inv_cdfs_iijj,
                x_marginal_means[II], x_marginal_stdevs[II],
                quad_rule, bisection_opts)
            correlation_uspace[jj, ii] = correlation_uspace[ii, jj]

    return correlation_uspace


def trans_x_to_u(x_samples, x_marginal_cdfs, z_correlation_cholesky_factor):
    z_samples = trans_x_to_z(x_samples, x_marginal_cdfs)
    u_samples = trans_z_to_u(z_samples, z_correlation_cholesky_factor)
    return u_samples


def trans_x_to_z(x_samples, x_marginal_cdfs):
    num_vars = x_samples.shape[0]
    z_samples = np.empty_like(x_samples)
    for ii in range(num_vars):
        x_marginal_cdf_vals = x_marginal_cdfs[ii](x_samples[ii, :])
        z_samples[ii, :] = stats.norm.ppf(x_marginal_cdf_vals)
    return z_samples


def trans_z_to_u(z_samples, z_correlation_cholesky_factor):
    u_samples = solve_triangular(
        z_correlation_cholesky_factor, z_samples, lower=True)
    return u_samples


def trans_u_to_z(u_samples, correlation_cholesky_factor):
    return np.dot(correlation_cholesky_factor, u_samples)


def trans_z_to_x(z_samples, x_inv_cdfs):
    num_vars = z_samples.shape[0]
    x_samples = np.empty_like(z_samples)
    for ii in range(num_vars):
        z_marginal_cdf_vals = stats.norm.cdf(z_samples[ii, :])
        x_samples[ii, :] = x_inv_cdfs[ii](z_marginal_cdf_vals)
    return x_samples


def trans_u_to_x(u_samples, x_inv_cdfs, correlation_cholesky_factor):
    z_samples = trans_u_to_z(u_samples, correlation_cholesky_factor)
    x_samples = trans_z_to_x(z_samples, x_inv_cdfs)
    return x_samples


def covariance_to_correlation(covariance):
    correlation = covariance.copy()
    num_vars = covariance.shape[0]
    for ii in range(num_vars):
        correlation[ii, ii] = 1
        for jj in range(ii+1, num_vars):
            correlation[ii,
                        jj] /= np.sqrt(covariance[ii, ii]*covariance[jj, jj])
            correlation[jj, ii] = correlation[ii, jj]
    return correlation


def correlation_to_covariance(correlation, stdevs):
    covariance = correlation.copy()
    num_vars = covariance.shape[0]
    for ii in range(num_vars):
        covariance[ii, ii] = stdevs[ii]**2
        for jj in range(ii+1, num_vars):
            covariance[ii, jj] *= stdevs[ii]*stdevs[jj]
            covariance[jj, ii] = covariance[ii, jj]
    return covariance


def nataf_transformation(x_samples, x_covariance, x_marginal_cdfs,
                         x_marginal_inv_cdfs, x_marginal_means,
                         x_marginal_stdevs, bisection_opts=dict()):
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    x_correlation = covariance_to_correlation(x_covariance)
    z_correlation = transform_correlations(
        x_correlation, x_marginal_inv_cdfs, x_marginal_means,
        x_marginal_stdevs, quad_rule, bisection_opts)
    z_correlation_cholesky_factor = np.linalg.cholesky(z_correlation)
    u_samples = trans_x_to_u(
        x_samples, x_marginal_cdfs, z_correlation_cholesky_factor)
    return u_samples


def inverse_nataf_transformation(u_samples, x_covariance, x_marginal_cdfs,
                                 x_marginal_inv_cdfs, x_marginal_means,
                                 x_marginal_stdevs, bisection_opts=dict()):
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    x_correlation = covariance_to_correlation(x_covariance)
    z_correlation = transform_correlations(
        x_correlation, x_marginal_inv_cdfs, x_marginal_means,
        x_marginal_stdevs, quad_rule, bisection_opts)
    z_correlation_cholesky_factor = np.linalg.cholesky(z_correlation)
    x_samples = trans_u_to_x(
        u_samples, x_marginal_inv_cdfs, z_correlation_cholesky_factor)
    return x_samples


def nataf_joint_density(x_samples, x_marginal_cdfs, x_marginal_pdfs,
                        z_joint_density):
    num_vars, num_samples = x_samples.shape
    z_samples = trans_x_to_z(x_samples, x_marginal_cdfs)
    vals = z_joint_density(z_samples)
    for ii in range(num_vars):
        vals *= x_marginal_pdfs[ii](x_samples[ii, :])
        normal_pdf_vals = stats.norm.pdf(z_samples[ii, :])
        vals /= normal_pdf_vals
    return vals


def plot_nataf_joint_density(x_marginal_cdfs, x_marginal_pdfs, z_correlation,
                             plot_limits, num_contour_levels=40,
                             num_samples_1d=100, show=True):
    num_vars = len(x_marginal_cdfs)
    z_variable = stats.multivariate_normal(
        mean=np.zeros((num_vars)), cov=z_correlation)

    def z_joint_density(x): return z_variable.pdf(x.T)

    function = partial(
        nataf_joint_density, x_marginal_cdfs=x_marginal_cdfs,
        x_marginal_pdfs=x_marginal_pdfs, z_joint_density=z_joint_density)

    X, Y, Z = get_meshgrid_function_data(
        function, plot_limits, num_samples_1d)
    plt.contourf(
        X, Y, Z, levels=np.linspace(Z.min(), Z.max(), num_contour_levels),
        cmap="coolwarm")
    if show:
        plt.show()


def generate_x_samples_using_gaussian_copula(num_vars, z_correlation,
                                             univariate_inv_cdfs, num_samples):
    num_samples = int(num_samples)
    u_samples = np.random.normal(0., 1., (num_vars, num_samples))
    z_correlation_sqrt = np.linalg.cholesky(z_correlation)
    correlated_samples = np.dot(z_correlation_sqrt, u_samples)
    z_samples = stats.norm.cdf(correlated_samples)
    x_samples = np.empty_like(u_samples)
    for ii in range(num_vars):
        x_samples[ii, :] = univariate_inv_cdfs[ii](z_samples[ii, :])
    # plt.plot(x_samples[0,:],x_samples[1,:],'sk')
    # plt.show()
    return x_samples, u_samples


def gaussian_copula_compute_x_correlation_from_z_correlation(
        x_marginal_inv_cdfs, x_marginal_means, x_marginal_stdevs,
        z_correlation):
    num_vars = z_correlation.shape[0]
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    x_correlation = np.empty_like(z_correlation)
    for ii in range(num_vars):
        x_correlation[ii, ii] = 1.0
        for jj in range(ii+1, num_vars):
            x_correlation[ii, jj] = corrcoeffij(
                z_correlation[ii, jj], x_marginal_inv_cdfs, x_marginal_means,
                x_marginal_stdevs, quad_rule)
            x_correlation[jj, ii] = x_correlation[ii, jj]
    return x_correlation
