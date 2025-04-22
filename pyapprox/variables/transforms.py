from typing import List, Dict

import numpy as np
from scipy import stats

from pyapprox.variables._rosenblatt import (
    rosenblatt_transformation,
    inverse_rosenblatt_transformation,
)
from pyapprox.variables._nataf import (
    trans_x_to_u,
    trans_u_to_x,
    transform_correlations,
    scipy_gauss_hermite_pts_wts_1D,
    gaussian_copula_compute_x_correlation_from_z_correlation,
    nataf_joint_density,
)
from pyapprox.variables.marginals import Marginal, ContinuousMarginalMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.misc import covariance_to_correlation
from pyapprox.util.transforms import Transform
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.variables.joint import define_iid_random_variable


class AffineTransform(Transform):
    r"""
    Apply an affine transformation to a
    :py:class:`pyapprox.variables.IndependentMarginalsVariable`
    """

    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        enforce_bounds: bool = False,
    ):
        """
        Variable uniquness dependes on both the type of random variable
        e.g. beta, gaussian, etc. and the parameters of that distribution
        e.g. loc and scale parameters as well as any additional parameters
        """
        super().__init__(variable._bkd)
        if not isinstance(variable, IndependentMarginalsVariable):
            variable = IndependentMarginalsVariable(variable)
        self._variable = variable
        self.enforce_bounds = enforce_bounds
        self.identity_map_indices = None

        self.scale_parameters = self._bkd.empty(
            (self._variable._nunique_vars, 2)
        )
        for ii in range(self._variable._nunique_vars):
            marginal = self._variable._unique_marginals[ii]
            self.scale_parameters[ii, :] = self._bkd.array(
                marginal._transform_scale_parameters()
            )

    def variable(self) -> IndependentMarginalsVariable:
        return self._variable

    def set_identity_maps(self, identity_map_indices: Array):
        """
        Set the dimensions we do not want to map to and from
        canonical space

        Parameters
        ----------
        identity_map_indices : iterable
            The dimensions we do not want to map to and from
            canonical space
        """
        self.identity_map_indices = identity_map_indices

    def map_to_canonical(self, user_samples: Array) -> Array:
        canonical_samples = self._bkd.copy(user_samples)
        for ii in range(self._variable._nunique_vars):
            indices = self._variable._unique_indices[ii]
            loc, scale = self.scale_parameters[ii, :]

            bounds = self._bkd.asarray([loc - scale, loc + scale])
            var = self._variable._unique_marginals[ii]
            if self.identity_map_indices is not None:
                active_indices = np.setdiff1d(
                    indices, self.identity_map_indices, assume_unique=True
                )
            else:
                active_indices = indices
            if (
                (self.enforce_bounds is True)
                and (
                    var.is_bounded()
                    and isinstance(var, ContinuousMarginalMixin)
                )
                and (
                    (
                        self._bkd.any(
                            user_samples[active_indices, :] < bounds[0]
                        )
                    )
                    or (
                        self._bkd.any(
                            user_samples[active_indices, :] > bounds[1]
                        )
                    )
                )
            ):
                II = self._bkd.where(
                    (user_samples[active_indices, :] < bounds[0])
                    | (user_samples[active_indices, :] > bounds[1])
                )[1]
                print(user_samples[np.ix_(active_indices, II)], bounds)
                raise ValueError(f"Sample outside the bounds {bounds}")

            canonical_samples[active_indices, :] = (
                user_samples[active_indices, :] - loc
            ) / scale

        return canonical_samples

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        user_samples = self._bkd.copy(canonical_samples)
        for ii in range(self._variable._nunique_vars):
            indices = self._variable._unique_indices[ii]
            loc, scale = self.scale_parameters[ii, :]
            if self.identity_map_indices is not None:
                active_indices = np.setdiff1d(
                    indices, self.identity_map_indices, assume_unique=True
                )
            else:
                active_indices = indices
            user_samples[active_indices, :] = (
                canonical_samples[active_indices, :] * scale + loc
            )

        return user_samples

    def map_to_canonical_1d(self, samples: Array, ii: int) -> Array:
        for jj in range(self._variable._nunique_vars):
            if ii in self._variable._unique_indices[jj]:
                loc, scale = self.scale_parameters[jj, :]
                if self.identity_map_indices is None:
                    return (samples - loc) / scale
                else:
                    return samples
        print(ii, self._variable._unique_indices)
        raise Exception("Should not happen")

    def map_from_canonical_1d(self, canonical_samples: Array, ii: int):
        for jj in range(self._variable._nunique_vars):
            if ii in self._variable._unique_indices[jj]:
                loc, scale = self.scale_parameters[jj, :]
                if self.identity_map_indices is None:
                    return canonical_samples * scale + loc
                else:
                    return canonical_samples
        raise Exception("Should not happen")

    def map_derivatives_from_canonical_space(
        self, derivatives: Array
    ) -> Array:
        """
        Parameters
        ----------
        derivatives : np.ndarray (nvars*nsamples, nqoi)
            Derivatives of each qoi. The ith column consists of the derivatives
            [d/dx_1 f(x^{(1)}), ..., f(x^{(M)}),
            d/dx_2 f(x^{(1)}), ..., f(x^{(M)})
            ...,
            d/dx_D f(x^{(1)}), ..., f(x^{(M)})]
            where M is the number of samples and D=nvars

            Derivatives can also be (nvars, nsamples) - transpose of Jacobian -
            Here each sample is considered a different QoI
        """
        assert derivatives.shape[0] % self.nvars() == 0
        nsamples = int(derivatives.shape[0] / self.nvars())
        mapped_derivatives = self._bkd.copy(derivatives)
        for ii in range(self._variable._nunique_vars):
            var_indices = self._variable._unique_indices[ii]
            idx = np.tile(var_indices * nsamples, nsamples) + np.tile(
                np.arange(nsamples), var_indices.shape[0]
            )
            loc, scale = self.scale_parameters[ii, :]
            mapped_derivatives[idx, :] /= scale
        return mapped_derivatives

    def map_derivatives_to_canonical_space(
        self, canonical_derivatives: Array
    ) -> Array:
        """
        derivatives : np.ndarray (nvars*nsamples, nqoi)
            Derivatives of each qoi. The ith column consists of the derivatives
            [d/dx_1 f(x^{(1)}), ..., f(x^{(M)}),
             d/dx_2 f(x^{(1)}), ..., f(x^{(M)})
             ...,
             d/dx_D f(x^{(1)}), ..., f(x^{(M)})]
            where M is the number of samples and D=nvars

            Derivatives can also be (nvars, nsamples) - transpose of Jacobian -
            Here each sample is considered a different QoI
        """
        assert canonical_derivatives.shape[0] % self.nvars() == 0
        nsamples = int(canonical_derivatives.shape[0] / self.nvars())
        derivatives = self._bkd.copy(canonical_derivatives)
        for ii in range(self._variable._nunique_vars):
            var_indices = self._variable._unique_indices[ii]
            idx = np.tile(var_indices * nsamples, nsamples) + np.tile(
                np.arange(nsamples), var_indices.shape[0]
            )
            loc, scale = self.scale_parameters[ii, :]
            derivatives[idx, :] *= scale
        return derivatives

    def nvars(self) -> int:
        return self._variable.nvars()

    def samples_of_bounded_variables_inside_domain(
        self, samples: Array
    ) -> Array:
        for ii in range(self._variable._nunique_vars):
            var = self._variable._unique_marginals[ii]
            lb, ub = var.interval(1)
            indices = self._variable._unique_indices[ii]
            if samples[indices, :].max() > ub:
                print(samples[indices, :].max(), ub, "ub violated")
                return False
            if samples[indices, :].min() < lb:
                print(samples[indices, :].min(), lb, "lb violated")
                return False
        return True

    def get_ranges(self) -> Array:
        ranges = self._bkd.empty(
            (2 * self.nvars()), dtype=self._bkd.double_type()
        )
        for ii in range(self._variable._nunique_vars):
            var = self._variable._unique_marginals[ii]
            lb, ub = var.interval(1)
            indices = self._variable._unique_indices[ii]
            ranges[2 * indices], ranges[2 * indices + 1] = lb, ub
        return ranges


class OperatorBasedGaussianTransform(Transform):
    def __init__(
        self,
        mean: Array,
        cov_sqrt_op: callable,
        inv_cov_sqrt_op: callable,
        bkd: BackendMixin = NumpyMixin,
    ):
        if mean.ndim != 2 or mean.shape[1] != 1:
            raise ValueError("Mean must be a 2D column vector")
        self._mean = mean
        self._cov_sqrt_op = cov_sqrt_op
        self._inv_cov_sqrt_op = inv_cov_sqrt_op
        super().__init__(bkd)

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        centered_samples = self._cov_sqrt_op(canonical_samples)
        if centered_samples.shape[0] != self._mean.shape[0]:
            raise RuntimeError(
                "covariance operator returns vector inconsistent with the "
                "shape of the mean provided"
            )
        return self._mean + centered_samples

    def map_to_canonical(self, samples: Array) -> Array:
        if self._inv_cov_sqrt_op is None:
            raise RuntimeError(
                "Must provide inv_cov_sqrt_op to map_to_canonical_samples"
            )
        centered_samples = samples - self._mean
        return self._inv_cov_sqrt_op(centered_samples)


class DenseGaussianTransform(OperatorBasedGaussianTransform):
    def __init__(
        self,
        mean: Array,
        cov: Array,
        bkd: BackendMixin = NumpyMixin,
    ):
        super().__init__(mean, self._cov_sqrt_op, self._inv_cov_sqrt_op, bkd)
        self._cov = cov
        self._cov_sqrt = self._bkd.cholesky(self._cov)

    def _cov_sqrt_op(self, samples: Array) -> Array:
        return self._cov_sqrt @ samples

    def _inv_cov_sqrt_op(self, samples: Array) -> Array:
        return self._bkd.solve_triangular(self._cov_sqrt, samples)


class RosenblattTransform(Transform):
    r"""
    Apply the Rosenblatt transformation to an arbitraty multivariate
    random variable.
    """

    def __init__(
        self,
        joint_density: callable,
        nvars: int,
        opts: Dict,
        backend: BackendMixin = NumpyMixin,
    ):
        self._joint_density = joint_density
        self._limits = opts["limits"]
        self._nquad_samples_1d = opts["nquad_samples_1d"]
        self._tol = opts.get("tol", 1e-12)
        self._nbins = opts.get("nbins", 101)
        self._nvars = nvars
        self._canonical_variable_types = ["uniform"] * self.nvars()
        super().__init__(backend)

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        user_samples = inverse_rosenblatt_transformation(
            canonical_samples,
            self._joint_density,
            self._limits,
            self._nquad_samples_1d,
            self._tol,
            self._nbins,
        )
        return user_samples

    def map_to_canonical(self, user_samples: Array) -> Array:
        canonical_samples = rosenblatt_transformation(
            user_samples,
            self._joint_density,
            self._limits,
            self._nquad_samples_1d,
        )
        return canonical_samples

    def nvars(self) -> int:
        return self._nvars


class UniformMarginalTransformation(Transform):
    r"""
    Transform variables to have uniform marginals on [0,1]
    """

    def __init__(
        self,
        x_marginal_cdfs: List[callable],
        x_marginal_inv_cdfs: List[callable],
        enforce_open_bounds: bool = True,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        enforce_open_bounds: boolean
            If True  - enforce that canonical samples are in (0,1)
            If False - enforce that canonical samples are in [0,1]
        """
        super().__init__(backend)
        self._nvars = len(x_marginal_cdfs)
        self._x_marginal_cdfs = x_marginal_cdfs
        self._x_marginal_inv_cdfs = x_marginal_inv_cdfs
        self._enforce_open_bounds = enforce_open_bounds

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        # there is a singularity at the boundary of the unit hypercube when
        # mapping to the (semi) unbounded distributions
        if self._enforce_open_bounds:
            assert canonical_samples.min() > 0 and canonical_samples.max() < 1
        user_samples = self._bkd.empty(canonical_samples.shape)
        for ii in range(self.nvars()):
            user_samples[ii, :] = self._x_marginal_inv_cdfs[ii](
                canonical_samples[ii, :]
            )
        return user_samples

    def map_to_canonical(self, user_samples: Array) -> Array:
        canonical_samples = self._bkd.empty(user_samples.shape)
        for ii in range(self.nvars()):
            canonical_samples[ii, :] = self._x_marginal_cdfs[ii](
                user_samples[ii, :]
            )
        return canonical_samples

    def nvars(self) -> int:
        return self._nvars


class NatafTransform(Transform):
    r"""
    Apply the Nataf transformation to an arbitraty multivariate
    random variable.
    """

    def __init__(
        self,
        x_marginals: List[Marginal],
        x_covariance: Array,
        bisection_opts: dict = dict(),
    ):
        for marginal in x_marginals:
            if not isinstance(marginal, Marginal):
                raise ValueError("marginal must be an instance of Marginal")
        super().__init__(x_marginals[0]._bkd)
        self._nvars = len(x_marginals)
        self._x_marginals = x_marginals
        self._x_correlation = covariance_to_correlation(
            x_covariance, self._bkd
        )
        self._x_marginal_stdevs = self._bkd.sqrt(self._bkd.diag(x_covariance))

        quad_rule = scipy_gauss_hermite_pts_wts_1D(11)
        self._z_correlation = transform_correlations(
            self._x_correlation,
            self._x_marginals,
            quad_rule,
            bisection_opts,
        )

        self._z_correlation_cholesky_factor = self._bkd.cholesky(
            self._z_correlation
        )

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        return trans_u_to_x(
            canonical_samples,
            self._x_marginals,
            self._z_correlation_cholesky_factor,
        )

    def map_to_canonical(self, user_samples: Array) -> Array:
        return trans_x_to_u(
            user_samples,
            self._x_marginals,
            self._z_correlation_cholesky_factor,
        )

    def nvars(self) -> int:
        return self._nvars

    def z_correlation_to_x_correlation(self, z_correlation: Array) -> Array:
        return gaussian_copula_compute_x_correlation_from_z_correlation(
            self._x_marginals, self._z_correlation
        )

    def pdf(self, samples: Array) -> Array:
        z_variable = stats.multivariate_normal(
            mean=self._bkd.zeros((self.nvars())), cov=self._z_correlation
        )
        return nataf_joint_density(
            samples, self._x_marginals, lambda x: z_variable.pdf(x.T)[:, None]
        )


class ComposeTransforms(Transform):
    r"""
    Apply a composition of transformation to an multivariate
    random variable.
    """

    def __init__(self, transformations: List[Transform]):
        """
        Parameters
        ----------
        transformations : list of transformation objects
            The transformations are applied first to last for
            map_to_canonical and in reverse order for
            map_from_canonical
        """
        self._transformations = transformations

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        user_samples = canonical_samples
        for ii in range(len(self._transformations) - 1, -1, -1):
            user_samples = self._transformations[ii].map_from_canonical(
                user_samples
            )
        return user_samples

    def map_to_canonical(self, user_samples: Array) -> Array:
        canonical_samples = user_samples
        for ii in range(len(self._transformations)):
            canonical_samples = self._transformations[ii].map_to_canonical(
                canonical_samples
            )
        return canonical_samples

    def nvars(self) -> int:
        return self._transformations[0].nvars()


class ConfigureVariableTransformation(Transform):
    """
    Class which maps one-to-one configure indices in [0, 1, 2, 3,...]
    to a set of configure values accepted by a function
    """

    def __init__(self, config_values: List, labels: List = None):
        """
        Parameters
        ----------
        nvars : integer
            The number of configure variables

        config_values : list
            The list of configure values for each configure variable.
            Each entry in the list is a 1D np.ndarray with potentiallly
            different sizes
        """

        self._nvars = len(config_values)
        assert (
            type(config_values[0]) == list
            or type(config_values[0]) == np.ndarray
        )
        self._config_values = config_values
        self._variable_labels = labels

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Map a configure multi-dimensional index to the corresponding
        configure values
        """
        assert canonical_samples.shape[0] == self.nvars()
        samples = self._bkd.empty(canonical_samples.shape, dtype=float)
        for ii in range(samples.shape[1]):
            for jj in range(self.nvars()):
                kk = canonical_samples[jj, ii]
                samples[jj, ii] = self._config_values[jj][int(kk)]
        return samples

    def map_to_canonical(self, samples: Array) -> Array:
        """
        This is the naive slow implementation that searches through all
        canonical samples to find one that matches each sample provided
        """
        assert samples.shape[0] == self.nvars()
        canonical_samples = self._bkd.empty(samples.shape, dtype=float)
        for ii in range(samples.shape[1]):
            for jj in range(self.nvars()):
                found = False
                for kk in range(len(self._config_values[jj])):
                    if samples[jj, ii] == self._config_values[jj][int(kk)]:
                        found = True
                        break
                if not found:
                    raise Exception("Configure value not found")
                canonical_samples[jj, ii] = kk
        return canonical_samples

    def nvars(self) -> int:
        """Return the number of configure variables.

        Returns
        -------
        The number of configure variables
        """
        return self._nvars

    def __repr__(self) -> str:
        return "{0}(nvars={1}, {2})".format(
            self.__class__.__name__, self.nvars(), self._config_values
        )


def define_iid_random_variable_transformation(
    marginal: Marginal, nvars: int, backend: BackendMixin = NumpyMixin
):
    variable = define_iid_random_variable(marginal, nvars, backend=backend)
    var_trans = AffineTransform(variable)
    return var_trans
