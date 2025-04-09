import unittest
import numpy as np
from functools import partial
from scipy import stats

from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.variables.joint import (
    RejectionSamplingVariable,
    IndependentMarginalsVariable,
    JointVariable,
)
from pyapprox.variables._rosenblatt import (
    invert_cdf,
    rosenblatt_transformation,
    marginalized_cumulative_distribution_function,
    marginal_pdf,
    inverse_rosenblatt_transformation,
    combine_samples_with_fixed_data,
)
from pyapprox.util.backends.numpy import NumpyMixin


def cwum_2d(x, c=0.0):
    """
    Correlated density with uniform marginals (CWUM)
    x in [0,1], 0<=c<2
    c = 1. is tensor product uniform
    c = 0. # if c=0 max pdf(x) = 2. min= 0.
    """
    normalization = (13.0 * c + 3) / 16.0
    return (
        c
        - 2.0
        * (c - 1.0)
        * (
            (x[0, :] + 1) / 2.0
            + (x[1, :] + 1) / 2.0
            - 2.0 * (x[0, :] + 1) / 2.0 * (x[1, :] + 1) / 2.0
        )
        / 4.0
    ) / normalization


class First3DExampleDensity(JointVariable):
    def nvars(self):
        return 3

    def rvs(self, nsamples):
        raise NotImplementedError

    def pdf(self, samples):
        return (
            80.0
            / 21.0
            * (samples[0, :] ** 4 + samples[1, :] ** 3 * samples[2, :] ** 3)[
                :, None
            ]
        )


def rosenblatt_example_3d(nsamples=10000, run_tests=False):
    nvars = 3
    joint_density = First3DExampleDensity(NumpyMixin)

    def marginal_pdf_x(x):
        return 5.0 / 21.0 * (16.0 * x**4 + 1.0)

    # marginal_pdf_y = lambda y: 4./21.*(5.*y**3+4.)
    # marginal_pdf_z = lambda z: 4./21.*(5.*z**3+4.)
    def marginal_pdf_xy(x, y):
        return 20.0 / 21.0 * (y**3 + 4.0 * x**4)

    def marginal_cdf_x(x):
        return 1.0 / 21.0 * x * (16.0 * x**4 + 5)

    def marginal_cdf_y_evaluated_at_x(x, y):
        return 5.0 / 21.0 * y * (y**3 + 16.0 * x**4)

    def marginal_cdf_z_evaluated_at_xy(x, y, z):
        return 20.0 / 21.0 * z * (y**3 * z**3 + 4.0 * x**4)

    limits = np.asarray([0, 1, 0, 1, 0, 1], dtype=float)

    # get samples from joint density using rejection sampling
    envelope_factor = 160.0 / 21.0 * 1.01

    proposal_density = IndependentMarginalsVariable(
        [stats.uniform(0, 1)] * nvars
    )
    rej_variable = RejectionSamplingVariable(
        joint_density, proposal_density, envelope_factor
    )
    samples = rej_variable.rvs(nsamples)
    exact_mean = np.asarray([95.0 / 126.0, 4.0 / 7.0, 4.0 / 7.0])
    if run_tests:
        assert np.allclose(exact_mean, samples.mean(axis=1), atol=1e-2)

    # Use rosenblatt transformation to map to uniform random variables
    trans_samples = np.empty_like(samples)
    # x_1 = F_1(x_1)
    trans_samples[0, :] = marginal_cdf_x(samples[0, :])
    # x_2 = F_2(x_2|x_1)
    trans_samples[1, :] = marginal_cdf_y_evaluated_at_x(
        samples[0, :], samples[1, :]
    ) / marginal_pdf_x(samples[0, :])
    # x_3 = F_3(x_3|x_1,x_2)
    trans_samples[2, :] = marginal_cdf_z_evaluated_at_xy(
        samples[0, :], samples[1, :], samples[2, :]
    ) / marginal_pdf_xy(samples[0, :], samples[1, :])
    if run_tests:
        assert np.allclose(
            [0.5, 0.5, 0.5], trans_samples.mean(axis=1), atol=1e-2
        )
    return samples, trans_samples, joint_density, limits


class First2DExampleDensity(JointVariable):
    def nvars(self):
        return 2

    def rvs(self, nsamples):
        raise NotImplementedError

    def pdf(self, samples):
        return 20.0 / 9.0 * (samples[0, :] ** 4 + samples[1, :] ** 3)[:, None]


class Second2DExampleDensity(JointVariable):
    def nvars(self):
        return 2

    def rvs(self, nsamples):
        raise NotImplementedError

    def pdf(self, samples):
        return 6.0 / 5.0 * (samples[0, :] ** 2 + samples[1, :])[:, None]


def rosenblatt_example_2d(nsamples=10000, density_num=1, run_tests=False):
    nvars = 2
    limits = np.asarray([0, 1, 0, 1], dtype=float)
    if density_num == 1:
        # joint_density = rosenblatt_example_2d_joint_density_1
        joint_density = First2DExampleDensity(NumpyMixin)

        def marginal_pdf_x(x):
            return 5.0 / 9.0 * (4.0 * x**4 + 1.0)

        # marginal_pdf_y = lambda y: 4./9.*(5.*y**3+1.)
        def marginal_cdf_x(x):
            return 1.0 / 9.0 * x * (4.0 * x**4 + 5)

        def marginal_cdf_y_evaluated_at_x(x, y):
            return 5.0 / 9.0 * y * (y**3 + 4.0 * x**4)

        envelope_factor = 40.0 / 9.0 * 1.01
        exact_mean = np.asarray([35.0 / 54.0, 2.0 / 3.0])
    elif density_num == 2:
        joint_density = Second2DExampleDensity(NumpyMixin)

        def marginal_pdf_x(x):
            return 3.0 / 5.0 * (2.0 * x**2 + 1.0)

        # marginal_pdf_y = lambda y: 2./5.*(3.*y+1.)
        def marginal_cdf_x(x):
            return 1.0 / 5.0 * x * (2.0 * x**2 + 3)

        def marginal_cdf_y_evaluated_at_x(x, y):
            return 3.0 / 5.0 * y * (2.0 * x**2 + y)

        envelope_factor = 12.0 / 5.0 * 1.01
        exact_mean = np.asarray([3.0 / 5.0, 3.0 / 5.0])
    else:
        raise Exception

    # import matplotlib.pyplot as plt
    # from PyDakota.plot_3d import get_meshgrid_function_data
    # X,Y,Z = get_meshgrid_function_data(joint_density, [0,1,0,1], 100)
    # ncontour_levels = 20
    # plt.contourf(X,Y,Z,levels=np.linspace(Z.min(),Z.max(),ncontour_levels),
    #                 cmap=mpl.cm.coolwarm)
    # plt.colorbar()
    # plt.show()

    # get samples from joint density using rejection sampling
    proposal_density = IndependentMarginalsVariable(
        [stats.uniform(0, 1)] * nvars
    )
    rej_variable = RejectionSamplingVariable(
        joint_density, proposal_density, envelope_factor
    )
    samples = rej_variable.rvs(nsamples)

    if run_tests:
        assert np.allclose(exact_mean, samples.mean(axis=1), atol=1e-2)

    # Use rosenblatt transformation to map to uniform random variables
    trans_samples = np.empty_like(samples)
    # x_1 = F_1(x_1)
    trans_samples[0, :] = marginal_cdf_x(samples[0, :])
    # x_2 = F_2(x_2|x_1)
    trans_samples[1, :] = marginal_cdf_y_evaluated_at_x(
        samples[0, :], samples[1, :]
    ) / marginal_pdf_x(samples[0, :])
    if run_tests:
        assert np.allclose([0.5, 0.5], trans_samples.mean(axis=1), atol=1e-2)

    # plt.plot(samples[0,:],samples[1,:],'s')
    # plt.plot(trans_samples[0,:],trans_samples[1,:],'o')
    # plt.show()

    inverse_trans_samples = np.empty_like(trans_samples)
    for jj in range(nsamples):
        val = invert_cdf(
            trans_samples[0, jj : jj + 1], marginal_cdf_x, limits[:2]
        )[0]
        inverse_trans_samples[0, jj] = val

    for jj in range(nsamples):

        def cdffun(y):
            if np.isscalar(y):
                y = np.array([y])
            cdf_val = marginal_cdf_y_evaluated_at_x(
                inverse_trans_samples[0, jj : jj + 1], y
            )
            pdf_val = marginal_pdf_x(inverse_trans_samples[0, jj : jj + 1])
            val = cdf_val / pdf_val
            return val

        inverse_trans_samples[1, jj] = invert_cdf(
            trans_samples[1, jj : jj + 1], cdffun, limits[2:]
        )[0]
    assert np.allclose(inverse_trans_samples, samples)

    return samples, trans_samples, joint_density, limits


class TestRosenblattTransform:
    def setUp(self):
        np.random.seed(1)

    def test_combine_samples_with_fixed_data(self):
        nsamples = 3
        nfixed_vars = 2
        nvars = 5
        fixed_data = np.ones((nfixed_vars))
        fixed_data_indices = np.array([1, 3])
        sub_samples = np.zeros((nvars - nfixed_vars, nsamples))
        samples = combine_samples_with_fixed_data(
            fixed_data, fixed_data_indices, sub_samples
        )
        true_samples = np.zeros((nvars, nsamples))
        true_samples[fixed_data_indices] = 1.0
        assert np.allclose(samples, true_samples)

    def test_marginal_pdf(self):
        limits = np.asarray([0.0, 1.0, 0, 1, 0, 1])

        def joint_density(x):
            return 80.0 / 21.0 * (x[0, :] ** 4 + x[1, :] ** 3 * x[2, :] ** 3)

        def marginal_pdf_x(x):
            return 5.0 / 21.0 * (16.0 * x**4 + 1.0)

        def marginal_pdf_y(y):
            return 4.0 / 21.0 * (5.0 * y**3 + 4.0)

        def marginal_pdf_xy(x, y):
            return 20.0 / 21.0 * (y**3 + 4.0 * x**4)

        samples = np.linspace(0.0, 1.0, 11)[np.newaxis, :]
        active_vars = np.asarray([0])
        values = marginal_pdf(joint_density, active_vars, limits, samples)
        assert np.allclose(values, marginal_pdf_x(samples[0, :]))

        samples = np.linspace(0.0, 1.0, 11)[np.newaxis, :]
        active_vars = np.asarray([1])
        values = marginal_pdf(joint_density, active_vars, limits, samples)
        assert np.allclose(values, marginal_pdf_y(samples[0, :]))

        samples = np.random.uniform(0.0, 1.0, (2, 10))
        active_vars = np.asarray([0, 1])
        values = marginal_pdf(joint_density, active_vars, limits, samples)
        assert np.allclose(
            values, marginal_pdf_xy(samples[0, :], samples[1, :])
        )

    def test_marginalized_cumulative_distribution_function(self):
        nvars = 2

        # independent uniform density all variables active
        def joint_density(x):
            return 0.5 ** x.shape[0] * np.ones((x.shape[1]), dtype=float)

        limits = np.asarray([-1, 1, -1, 1])
        active_var_samples = np.asarray([[1, 1], [0, 1]]).T
        active_vars = np.arange(nvars)
        inactive_vars = np.empty((0), dtype=int)
        fixed_var_samples = np.empty(
            (0, active_var_samples.shape[1]), dtype=float
        )
        values = marginalized_cumulative_distribution_function(
            joint_density,
            limits,
            active_vars,
            active_var_samples,
            inactive_vars,
            fixed_var_samples,
        )
        assert np.allclose(values, [1, 0.5])

        # independent normals all variables active
        nvars = 2
        variable = DenseCholeskyMultivariateGaussian(
            mean=np.zeros((nvars, 1)), cov=np.eye(nvars)
        )
        limits = np.asarray([-5, 5, -5, 5])
        active_var_samples = np.asarray([[5, 5], [0, 5]]).T
        active_vars = np.arange(nvars)
        fixed_var_samples = np.empty(
            (0, active_var_samples.shape[1]), dtype=float
        )
        values = marginalized_cumulative_distribution_function(
            lambda x: variable.pdf(x)[:, 0],
            limits,
            active_vars,
            active_var_samples,
            inactive_vars,
            fixed_var_samples,
        )
        assert np.allclose(values, [1, 0.5])

        # dependent variables NOT all variables active
        nvars = 2

        def joint_density(x):
            return 20.0 / 9.0 * (x[0, :] ** 4 + x[1, :] ** 3)

        def marginal_cdf_y_evaluated_at_x(x, y):
            return 5.0 / 9.0 * y * (y**3 + 4.0 * x**4)

        limits = np.asarray([0, 1, 0, 1])
        active_var_samples = np.asarray([[1, 0.5]])
        active_vars = np.asarray([1])
        inactive_vars = np.empty((0), dtype=int)
        fixed_var_samples = np.asarray([[0.25, 0.75]])
        values = marginalized_cumulative_distribution_function(
            joint_density,
            limits,
            active_vars,
            active_var_samples,
            inactive_vars,
            fixed_var_samples,
        )
        assert np.allclose(
            values,
            marginal_cdf_y_evaluated_at_x(
                fixed_var_samples[0, :], active_var_samples[0, :]
            ),
        )

    def test_invert_cdf(self):
        samples = np.linspace(0.0, 1.0, 11)
        true_icdf_vals = stats.beta.ppf(samples, 2, 2)
        cdffun = partial(stats.beta.cdf, a=2, b=2)
        icdf_vals = invert_cdf(samples, cdffun, [0, 1])
        assert np.allclose(true_icdf_vals, icdf_vals)

        nvars = 2

        def joint_density(x):
            return 20.0 / 9.0 * (x[0, :] ** 4 + x[1, :] ** 3)

        def marginal_pdf_x(x):
            return 5.0 / 9.0 * (4.0 * x**4 + 1.0)

        def marginal_cdf_y_evaluated_at_x(x, y):
            return 5.0 / 9.0 * y * (y**3 + 4.0 * x**4)

        limits = np.asarray([0, 1, 0, 1])
        active_var_samples = np.asarray([[1, 0.5]])
        active_vars = np.asarray([1])
        inactive_vars = np.empty((0), dtype=int)
        fixed_var_samples = np.asarray([[0.25]])

        # check with exact conditional cdf
        def cdffun(y):
            return marginal_cdf_y_evaluated_at_x(
                fixed_var_samples[0, :], y
            ) / marginal_pdf_x(fixed_var_samples[0, :])

        icdf_vals = invert_cdf(samples, cdffun, [0, 1])
        assert np.allclose(
            marginal_cdf_y_evaluated_at_x(fixed_var_samples[0, :], icdf_vals)
            / marginal_pdf_x(fixed_var_samples[0, :]),
            samples,
        )

        # check with numerical conditional cdf
        def cdffun(x):
            if np.isscalar(x):
                x = np.asarray([[x]])
            if x.ndim == 1:
                x = x[np.newaxis, :]
            return (
                marginalized_cumulative_distribution_function(
                    joint_density,
                    limits,
                    active_vars,
                    x,
                    inactive_vars,
                    fixed_var_samples,
                )
                / marginal_pdf_x(fixed_var_samples)[0]
            )

        icdf_vals = invert_cdf(samples, cdffun, [0, 1])
        assert np.allclose(
            marginal_cdf_y_evaluated_at_x(fixed_var_samples[0, :], icdf_vals)
            / marginal_pdf_x(fixed_var_samples[0, :]),
            samples,
        )

    def test_rosenblatt_transformation(self):
        # uncorrelated normal
        nvars = 2
        nsamples = 9

        def joint_density(x):
            return stats.norm.pdf(x[0, :]) * stats.norm.pdf(x[1, :])

        limits = np.asarray([-6, 6] * nvars)
        samples = np.random.normal(0.0, 1.0, (nvars, nsamples))
        assert samples.min() > limits[0] and samples.max() < limits[1]
        trans_samples = rosenblatt_transformation(
            samples, joint_density, limits, nquad_samples_1d=200
        )
        true_trans_samples = np.empty_like(trans_samples)
        for ii in range(nvars):
            true_trans_samples[ii, :] = stats.norm.cdf(samples[ii, :])
        assert np.allclose(true_trans_samples, trans_samples)

        samples, true_trans_samples, joint_density, limits = (
            rosenblatt_example_2d(nsamples=10)
        )
        trans_samples = rosenblatt_transformation(
            samples,
            lambda x: joint_density.pdf(x)[:, 0],
            limits,
            nquad_samples_1d=20,
        )
        assert np.allclose(true_trans_samples, trans_samples)

        samples, true_trans_samples, joint_density, limits = (
            rosenblatt_example_3d(nsamples=10)
        )
        trans_samples = rosenblatt_transformation(
            samples,
            lambda x: joint_density.pdf(x)[:, 0],
            limits,
            nquad_samples_1d=20,
        )
        assert np.allclose(true_trans_samples, trans_samples)

        # import matplotlib.pyplot as plt
        # plt.plot(trans_samples[0,:],trans_samples[1,:],'o')
        # plt.plot(true_trans_samples[0,:],true_trans_samples[1,:],'s')
        # plt.show()

    def test_inverse_rosenblatt_transformation(self):
        np.random.seed(3)
        nvars = 2

        def joint_density(x):
            return stats.norm.pdf(x[0, :]) * stats.norm.pdf(x[1, :])

        # decreasing limits induces an error
        limits = np.asarray([-6, 6] * nvars)

        nsamples = 10
        samples = np.random.normal(0.0, 1.0, (nvars, nsamples))
        assert samples.min() > limits[0] and samples.max() < limits[1]
        # canonical_samples = rosenblatt_transformation(
        #    samples,joint_density,limits,nquad_samples_1d=200)

        canonical_samples = np.empty_like(samples)
        for ii in range(nvars):
            canonical_samples[ii, :] = stats.norm.cdf(samples[ii, :])
        user_samples = inverse_rosenblatt_transformation(
            canonical_samples, joint_density, limits, nquad_samples_1d=30
        )

        assert np.allclose(user_samples, samples)

        samples, true_trans_samples, joint_density, limits = (
            rosenblatt_example_2d(nsamples=10)
        )

        user_samples = inverse_rosenblatt_transformation(
            true_trans_samples,
            lambda x: joint_density.pdf(x)[:, 0],
            limits,
            nquad_samples_1d=20,
        )
        assert np.allclose(samples, user_samples)


class TestNumpyRosenblattTransform(TestRosenblattTransform, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
