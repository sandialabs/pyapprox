from abc import ABC, abstractmethod
import math

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)
from pyapprox.util.hyperparameter import (
    CombinedHyperParameter,
    HyperParameter,
    HyperParameterList,
    IdentityHyperParameterTransform,
    LogHyperParameterTransform,
    HyperParameterTransform,
)
from pyapprox.util.transforms import SphericalCorrelationTransform


class Kernel(ABC):
    """The base class for any kernel."""

    def __init__(self, backend: LinAlgMixin):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend

        # used when computing a Jacobian
        self._samples = None

    def diag(self, X1):
        """Return the diagonal of the kernel matrix."""
        return self._bkd._la_get_diagonal(self(X1))

    @abstractmethod
    def __call__(self, X1, X2=None):
        raise NotImplementedError()

    def __mul__(self, kernel):
        return ProductKernel(self, kernel)

    def __add__(self, kernel):
        return SumKernel(self, kernel)

    def __repr__(self):
        return "{0}({1}, bkd={2})".format(
            self.__class__.__name__, self.hyp_list._short_repr(), self._bkd
        )

    def _params_eval(self, active_opt_params):
        # define function that evaluates the kernel for different parameters
        self.hyp_list.set_active_opt_params(active_opt_params)
        return self(self._samples)

    def jacobian(self, samples):
        self._samples = samples
        return self._bkd._la_jacobian(
            self._params_eval, self.hyp_list.get_active_opt_params()
        )


class CompositionKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.hyp_list = kernel1.hyp_list + kernel2.hyp_list
        if type(kernel1._bkd) is not type(kernel2._bkd):
            raise ValueError("Kernels must have the same backend.")
        self._bkd = kernel1._bkd

        # make linear algebra functions accessible via product_kernel._la_
        for attr in dir(kernel1):
            if len(attr) >= 4 and attr[:4] == "_la_":
                setattr(self, attr, getattr(self.kernel1, attr))

    def nvars(self):
        if hasattr(self.kernel1, "nvars"):
            return self.kernel1.nvars()
        return self.kernel2.nvars()


class ProductKernel(CompositionKernel):
    def diag(self, X1):
        return self.kernel1.diag(X1) * self.kernel2.diag(X1)

    def __repr__(self):
        return "{0} * {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X1, X2=None):
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)

    def jacobian(self, X):
        Kmat1 = self.kernel1(X)
        Kmat2 = self.kernel2(X)
        jac1 = self.kernel1.jacobian(X)
        jac2 = self.kernel2.jacobian(X)
        return self._bkd._la_dstack(
            [jac1 * Kmat2[..., None], jac2 * Kmat1[..., None]]
        )


class SumKernel(CompositionKernel):
    def diag(self, X1):
        return self.kernel1.diag(X1) + self.kernel2.diag(X1)

    def __repr__(self):
        return "{0} + {1}".format(self.kernel1, self.kernel2)

    def __call__(self, X1, X2=None):
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)

    def jacobian(self, X):
        jac1 = self.kernel1.jacobian(X)
        jac2 = self.kernel2.jacobian(X)
        return self._bkd._la_dstack([jac1, jac2])


class MaternKernel(Kernel):
    def __init__(
        self,
        nu: float,
        lenscale,
        lenscale_bounds,
        nvars: int,
        backend: LinAlgMixin = None,
    ):
        """The matern kernel for varying levels of smoothness."""
        super().__init__(backend)
        self._nvars = nvars
        self.nu = nu
        transform = LogHyperParameterTransform(backend=self._bkd)
        self._lenscale = HyperParameter(
            "lenscale",
            nvars,
            lenscale,
            lenscale_bounds,
            transform,
            backend=self._bkd,
        )
        self.hyp_list = HyperParameterList([self._lenscale])

    def diag(self, X1):
        return self._bkd._la_full((X1.shape[1],), 1)

    def _eval_distance_form(self, distances):
        if self.nu == self._bkd._la_inf():
            return self._bkd._la_exp(-(distances**2) / 2.0)
        if self.nu == 5 / 2:
            tmp = self._bkd._la_sqrt(5) * distances
            return (1.0 + tmp + tmp**2 / 3.0) * self._bkd._la_exp(-tmp)
        if self.nu == 3 / 2:
            tmp = self._bkd._la_sqrt(3) * distances
            return (1.0 + tmp) * self._bkd._la_exp(-tmp)
        if self.nu == 1 / 2:
            return self._bkd._la_exp(-distances)
        raise ValueError(
            "Matern kernel with nu={0} not supported".format(self.nu)
        )

    def __call__(self, X1, X2=None):
        lenscale = self._lenscale.get_values()
        if X2 is None:
            X2 = X1
        distances = self._bkd._la_cdist(X1.T / lenscale, X2.T / lenscale)
        return self._eval_distance_form(distances)

    def nvars(self):
        return self._nvars


class ConstantKernel(Kernel):
    def __init__(
        self, constant, constant_bounds=None, transform=None, backend=None
    ):
        if backend is None and transform is not None:
            backend = transform._bkd
        super().__init__(backend)
        if transform is None:
            transform = IdentityHyperParameterTransform(backend=self._bkd)
        if constant_bounds is None:
            constant_bounds = [-self._bkd._la_inf(), self._bkd._la_inf()]
        self._const = HyperParameter(
            "const", 1, constant, constant_bounds, transform, backend=self._bkd
        )
        self.hyp_list = HyperParameterList([self._const])

    def diag(self, X1):
        return self._bkd._la_full(
            (X1.shape[1],), self.hyp_list.get_values()[0]
        )

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        # full does not work when const value requires grad
        # return full((X1.shape[1], X2.shape[1]), self._const.get_values()[0])
        const = self._bkd._la_empty((X1.shape[1], X2.shape[1]))
        # const[:] = self._const.get_values()[0]
        const = self._bkd._la_up(
            const,
            self._bkd._la_arange(X1.shape[1], dtype=int),
            self._const.get_values()[0],
            axis=0,
        )
        return const


class GaussianNoiseKernel(Kernel):
    def __init__(self, constant, constant_bounds=None, backend=None):
        super().__init__(backend)
        self._const = HyperParameter(
            "const",
            1,
            constant,
            constant_bounds,
            LogHyperParameterTransform(backend=self._bkd),
            backend=self._bkd,
        )
        self.hyp_list = HyperParameterList([self._const])

    def diag(self, X):
        return self._bkd._la_full((X.shape[1],), self.hyp_list.get_values()[0])

    def __call__(self, X, Y=None):
        if Y is None:
            return self._const.get_values()[0] * self._bkd._la_eye(X.shape[1])
        # full does not work when const value requires grad
        # return full((X.shape[1], Y.shape[1]), self._const.get_values()[0])
        const = self._bkd._la_full((X.shape[1], Y.shape[1]), 0.0)
        return const


class PeriodicMaternKernel(MaternKernel):
    def __init__(
        self,
        nu: float,
        period,
        period_bounds,
        lenscale,
        lenscale_bounds,
        backend=None,
    ):
        super().__init__(nu, lenscale, lenscale_bounds, 1, backend=backend)
        period_transform = LogHyperParameterTransform(backend=self._bkd)
        self._period = HyperParameter(
            "period",
            1,
            lenscale,
            lenscale_bounds,
            period_transform,
            backend=self._bkd,
        )
        self.hyp_list += HyperParameterList([self._period])

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        lenscale = self._lenscale.get_values()
        period = self._period.get_values()
        distances = self._bkd._la_cdist(X.T / period, Y.T / period) / lenscale
        return super()._eval_distance_form(distances)

    def diag(self, X):
        return super().diag(X)


class HilbertSchmidtKernel(Kernel):
    def __init__(
        self, basis, weights, weight_bounds, transform, normalize: bool = False
    ):
        super().__init__(basis._bkd)
        self._nvars = basis.nvars()
        self._basis = basis
        self._nterms = basis.nterms() ** 2
        self._normalize = normalize
        self._weights = HyperParameter(
            "weights",
            self._nterms,
            weights,
            weight_bounds,
            transform,
            backend=self._bkd,
        )
        self.hyp_list = HyperParameterList([self._weights])

    def _get_weights(self):
        return self._bkd._la_reshape(
            self._weights.get_values(),
            (self._basis.nterms(), self._basis.nterms()),
        )

    def __call__(self, X1, X2=None):
        weights = self._get_weights()
        if X2 is None:
            X2 = X1
        X1basis_mat = self._basis(X1)
        X2basis_mat = self._basis(X2)
        if self._normalize:
            X1basis_mat /= self._bkd._la_norm(X1basis_mat, axis=1)[:, None]
            X2basis_mat /= self._bkd._la_norm(X2basis_mat, axis=1)[:, None]
        K = (X1basis_mat @ weights) @ X2basis_mat.T
        return K


class SphericalCovarianceHyperParameter(CombinedHyperParameter):
    def __init__(self, hyper_params: list):
        super().__init__(hyper_params)
        self.cov_matrix = None
        self.name = "spherical_covariance"
        self.transform = IdentityHyperParameterTransform(backend=self._bkd)
        noutputs = hyper_params[0].nvars()
        self._trans = SphericalCorrelationTransform(
            noutputs, backend=self._bkd
        )
        self._set_covariance_matrix()

    def _set_covariance_matrix(self):
        L = self._trans.map_to_cholesky(self.get_values())
        self.cov_matrix = L @ L.T

    def set_active_opt_params(self, active_params):
        super().set_active_opt_params(active_params)
        self._set_covariance_matrix()

    def __repr__(self):
        return "{0}(name={1}, nvars={2}, transform={3}, nactive={4})".format(
            self.__class__.__name__,
            self.name,
            self.nvars(),
            self.transform,
            self.nactive_vars(),
        )


class SphericalCovariance:
    def __init__(
        self,
        noutputs: int,
        radii_transform: HyperParameterTransform = None,
        angle_transform: HyperParameterTransform = None,
        radii=1,
        radii_bounds=[1e-1, 1],
        angles=math.pi / 2,
        angle_bounds=[0, math.pi],
        backend: LinAlgMixin = None,
    ):
        if backend is None:
            if radii_transform is not None:
                backend = radii_transform._bkd
            elif angle_transform is not None:
                backend = angle_transform._bkd
            else:
                backend = NumpyLinAlgMixin()

        self._bkd = backend

        # Angle bounds close to zero can create zero on the digaonal
        # E.g. for speherical coordinates sin(0) = 0
        if radii_transform is None:
            radii_transform = IdentityHyperParameterTransform(
                backend=self._bkd
            )
        if angle_transform is None:
            angle_transform = IdentityHyperParameterTransform(
                backend=self._bkd
            )

        self.noutputs = noutputs
        self._trans = SphericalCorrelationTransform(
            self.noutputs, backend=self._bkd
        )
        self._validate_bounds(radii_bounds, angle_bounds)
        self._radii = HyperParameter(
            "radii",
            self.noutputs,
            radii,
            radii_bounds,
            radii_transform,
            backend=self._bkd,
        )
        self._angles = HyperParameter(
            "angles",
            self._trans.ntheta - self.noutputs,
            angles,
            angle_bounds,
            angle_transform,
            backend=self._bkd,
        )
        self.hyp_list = HyperParameterList(
            [SphericalCovarianceHyperParameter([self._radii, self._angles])]
        )

    def _validate_bounds(self, radii_bounds, angle_bounds):
        bounds = self._trans.get_spherical_bounds()
        # all theoretical radii_bounds are the same so just check one
        radii_bounds = self._bkd._la_atleast1d(radii_bounds)
        if radii_bounds.shape[0] == 2:
            radii_bounds = self._bkd._la_repeat(radii_bounds, self.noutputs)
        radii_bounds = radii_bounds.reshape((radii_bounds.shape[0] // 2, 2))
        if self._bkd._la_any(
            radii_bounds[:, 0] < bounds[: self.noutputs, 0]
        ) or self._bkd._la_any(
            radii_bounds[:, 1] > bounds[: self.noutputs, 1]
        ):
            raise ValueError("radii bounds are inconsistent")
        # all theoretical angle_bounds are the same so just check one
        angle_bounds = self._bkd._la_atleast1d(angle_bounds)
        if angle_bounds.shape[0] == 2:
            angle_bounds = self._bkd._la_repeat(
                angle_bounds, self._trans.ntheta - self.noutputs
            )
        angle_bounds = angle_bounds.reshape((angle_bounds.shape[0] // 2, 2))
        if self._bkd._la_any(
            angle_bounds[:, 0] < bounds[self.noutputs :, 0]
        ) or self._bkd._la_any(
            angle_bounds[:, 1] > bounds[self.noutputs :, 1]
        ):
            raise ValueError("angle bounds are inconsistent")

    def get_covariance_matrix(self):
        return self.hyp_list.hyper_params[0].cov_matrix

    def __call__(self, ii, jj):
        # chol factor must be recomputed each time even if hyp_values have not
        # changed otherwise gradient graph becomes inconsistent
        return self.hyp_list.hyper_params[0].cov_matrix[ii, jj]

    def __repr__(self):
        return "{0}(radii={1}, angles={2} cov={3})".format(
            self.__class__.__name__,
            self._radii,
            self._angles,
            self.get_covariance_matrix().detach().numpy(),
        )
