from functools import partial
from typing import List, Tuple, Dict

import numpy as np
from scipy import stats
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

from pyapprox.util.utilities import (
    scipy_gauss_hermite_pts_wts_1D,
    get_correlation_from_covariance,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.variables.marginals import GaussianMarginal


def corrcoeffij(
    corrij: float,
    x_inv_cdfs: List[callable],
    x_means: List[Array],
    x_stdevs: List[Array],
    quad_rule: Tuple[Array, Array],
    bkd: NumpyLinAlgMixin,
) -> float:
    """
    Based on algorithm outlined in
    Li HongShuang et al. Chinese Science Bulletin, September 2008, vol. 53,
    no. 17, 2586-2592
    """
    # define 2d correlation matrix for idim and jdim
    corr = bkd.asarray([[1.0, corrij], [corrij, 1.0]])

    # do the cholesky factorization
    chol_factor = bkd.cholesky(corr)

    # do the gauss-hermite quadrature
    u = bkd.empty((2,))
    x = bkd.empty((2,))

    corrij_corrected = 0.0
    quad_x, quad_w = quad_rule

    marginal = GaussianMarginal(0, 1, backend=bkd)

    for ii in range(quad_x.shape[0]):
        for jj in range(quad_x.shape[0]):
            # correlate gauss hermite points
            u[0] = quad_x[ii]
            u[1] = quad_x[jj]
            z = chol_factor @ u  # equation (18)
            # do the nataf transformation: x = F^-1(Phi(z))
            # idim: z -> u -> x
            x[0] = x_inv_cdfs[0](marginal.cdf(z[0:1]))[0]  # equation (19)
            # jdim: z -> u -> x
            x[1] = x_inv_cdfs[1](marginal.cdf(z[1:2]))[0]  # equation (19)

            # normalize the values to obtain the correlation coefficient
            x[0] = (x[0] - x_means[0]) / x_stdevs[0]
            x[1] = (x[1] - x_means[1]) / x_stdevs[1]

            # do the quadrature, i.e
            # evaluate the double integral in equation (17)
            corrij_corrected += quad_w[ii] * quad_w[jj] * x[0] * x[1]

    return corrij_corrected


def bisection_corrij(
    corrij: float,
    x_inv_cdfs: List[callable],
    x_means: List[Array],
    x_stdevs: List[Array],
    quad_rule: Tuple[Array, Array],
    bisection_opts: Dict,
    bkd: NumpyLinAlgMixin,
) -> float:

    tol = bisection_opts.get("tol", 1e-7)
    max_iterations = bisection_opts.get("max_iterations", 100)

    ii = 0
    corrij_corrected = 0.0

    xerr = 0.0

    # define search interval
    dx = 0.0

    if corrij < 0:
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
            nextX, x_inv_cdfs, x_means, x_stdevs, quad_rule, bkd
        )

        # adjust domain for possible zero
        if corrij < corrij_corrected:
            xupper = nextX
        else:
            xlower = nextX

        # select new center
        nextX = (xlower + xupper) / 2.0
        xerr = bkd.abs(corrij - corrij_corrected)
        ii += 1
        if (xerr <= tol) or (ii >= max_iterations):
            break

    return x


def transform_correlations(
    initial_correlation: List[Array],
    x_marginal_inv_cdfs: List[callable],
    x_marginal_means: List[Array],
    x_marginal_stdevs: List[Array],
    quad_rule: Tuple[Array, Array],
    bisection_opts: Dict = dict(),
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:

    nvars = len(x_marginal_inv_cdfs)
    correlation_uspace = bkd.empty((nvars, nvars), dtype=float)
    for ii in range(nvars):
        correlation_uspace[ii, ii] = 1.0
        for jj in range(ii + 1, nvars):
            II = [ii, jj]
            x_marginal_inv_cdfs_iijj = [
                x_marginal_inv_cdfs[ii],
                x_marginal_inv_cdfs[jj],
            ]
            correlation_uspace[ii, jj] = bisection_corrij(
                initial_correlation[ii, jj],
                x_marginal_inv_cdfs_iijj,
                x_marginal_means[II],
                x_marginal_stdevs[II],
                quad_rule,
                bisection_opts,
                bkd,
            )
            correlation_uspace[jj, ii] = correlation_uspace[ii, jj]

    return correlation_uspace


def trans_x_to_u(
    x_samples: Array,
    x_marginal_cdfs: List[callable],
    z_correlation_cholesky_factor: Array,
    bkd: LinAlgMixin,
) -> Array:
    z_samples = trans_x_to_z(x_samples, x_marginal_cdfs, bkd)
    u_samples = trans_z_to_u(z_samples, z_correlation_cholesky_factor, bkd)
    return u_samples


def trans_x_to_z(
    x_samples: Array, x_marginal_cdfs: List[callable], bkd: LinAlgMixin
) -> Array:
    nvars = x_samples.shape[0]
    z_samples = bkd.empty_like(x_samples)
    for ii in range(nvars):
        x_marginal_cdf_vals = x_marginal_cdfs[ii](x_samples[ii, :])
        z_samples[ii, :] = stats.norm.ppf(x_marginal_cdf_vals)
    return z_samples


def trans_z_to_u(
    z_samples: Array, z_correlation_cholesky_factor: Array, bkd: LinAlgMixin
) -> Array:
    u_samples = bkd.solve_triangular(
        z_correlation_cholesky_factor, z_samples, lower=True
    )
    return u_samples


def trans_u_to_z(
    u_samples: Array, correlation_cholesky_factor: Array, bkd: LinAlgMixin
) -> Array:
    return bkd.dot(correlation_cholesky_factor, u_samples)


def trans_z_to_x(
    z_samples: Array, x_inv_cdfs: Array, bkd: LinAlgMixin
) -> Array:
    nvars = z_samples.shape[0]
    x_samples = bkd.empty_like(z_samples)
    for ii in range(nvars):
        z_marginal_cdf_vals = stats.norm.cdf(z_samples[ii, :])
        x_samples[ii, :] = x_inv_cdfs[ii](z_marginal_cdf_vals)
    return x_samples


def trans_u_to_x(
    u_samples: Array,
    x_inv_cdfs: List[callable],
    correlation_cholesky_factor: Array,
    bkd: LinAlgMixin,
) -> Array:
    z_samples = trans_u_to_z(u_samples, correlation_cholesky_factor, bkd)
    x_samples = trans_z_to_x(z_samples, x_inv_cdfs, bkd)
    return x_samples


def nataf_transformation(
    x_samples: Array,
    x_covariance: Array,
    x_marginal_cdfs: List[callable],
    x_marginal_inv_cdfs: List[callable],
    x_marginal_means: Array,
    x_marginal_stdevs: Array,
    bisection_opts: Dict = dict(),
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    # x_correlation = covariance_to_correlation(x_covariance)
    x_correlation = get_correlation_from_covariance(x_covariance, bkd)
    z_correlation = transform_correlations(
        x_correlation,
        x_marginal_inv_cdfs,
        x_marginal_means,
        x_marginal_stdevs,
        quad_rule,
        bisection_opts,
    )
    z_correlation_cholesky_factor = bkd.cholesky(z_correlation)
    u_samples = trans_x_to_u(
        x_samples, x_marginal_cdfs, z_correlation_cholesky_factor, bkd
    )
    return u_samples


def inverse_nataf_transformation(
    u_samples: Array,
    x_covariance: Array,
    x_marginal_cdfs: List[callable],
    x_marginal_inv_cdfs: List[callable],
    x_marginal_means: Array,
    x_marginal_stdevs: Array,
    bisection_opts: Dict = dict(),
    bkd: LinAlgMixin = NumpyLinAlgMixin,
) -> Array:
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    x_correlation = get_correlation_from_covariance(x_covariance, bkd)
    z_correlation = transform_correlations(
        x_correlation,
        x_marginal_inv_cdfs,
        x_marginal_means,
        x_marginal_stdevs,
        quad_rule,
        bisection_opts,
    )
    z_correlation_cholesky_factor = bkd.cholesky(z_correlation)
    x_samples = trans_u_to_x(
        u_samples, x_marginal_inv_cdfs, z_correlation_cholesky_factor, bkd
    )
    return x_samples


def nataf_joint_density(
    x_samples: Array,
    x_marginal_cdfs: List[callable],
    x_marginal_pdfs: List[callable],
    z_joint_density: callable,
    bkd: LinAlgMixin,
) -> Array:
    nvars, num_samples = x_samples.shape
    z_samples = trans_x_to_z(x_samples, x_marginal_cdfs, bkd)
    vals = z_joint_density(z_samples)
    for ii in range(nvars):
        vals *= x_marginal_pdfs[ii](x_samples[ii, :])
        normal_pdf_vals = stats.norm.pdf(z_samples[ii, :])
        vals /= normal_pdf_vals
    return vals


def generate_x_samples_using_gaussian_copula(
    nvars: int,
    z_correlation: Array,
    univariate_inv_cdfs: List[callable],
    num_samples: int,
    bkd: LinAlgMixin,
):
    num_samples = int(num_samples)
    u_samples = bkd.asarray(np.random.normal(0.0, 1.0, (nvars, num_samples)))
    z_correlation_sqrt = bkd.cholesky(z_correlation)
    correlated_samples = z_correlation_sqrt @ u_samples
    z_samples = stats.norm.cdf(correlated_samples)
    x_samples = bkd.empty_like(u_samples)
    for ii in range(nvars):
        x_samples[ii, :] = univariate_inv_cdfs[ii](z_samples[ii, :])
    # plt.plot(x_samples[0,:],x_samples[1,:],'sk')
    # plt.show()
    return x_samples, u_samples


def gaussian_copula_compute_x_correlation_from_z_correlation(
    x_marginal_inv_cdfs: List[callable],
    x_marginal_means: Array,
    x_marginal_stdevs: Array,
    z_correlation: Array,
    bkd: LinAlgMixin,
) -> Array:
    nvars = z_correlation.shape[0]
    quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
    x_correlation = bkd.empty_like(z_correlation)
    for ii in range(nvars):
        x_correlation[ii, ii] = 1.0
        for jj in range(ii + 1, nvars):
            x_correlation[ii, jj] = corrcoeffij(
                z_correlation[ii, jj],
                x_marginal_inv_cdfs,
                x_marginal_means,
                x_marginal_stdevs,
                quad_rule,
                bkd,
            )
            x_correlation[jj, ii] = x_correlation[ii, jj]
    return x_correlation
