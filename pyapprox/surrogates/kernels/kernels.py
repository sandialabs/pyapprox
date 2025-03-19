from abc import ABC, abstractmethod
import math
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
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
            backend = NumpyLinAlgMixin
        self._bkd = backend

        # used when computing a Jacobian
        self._samples = None

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def diag(self, X1: Array) -> Array:
        """Return the diagonal of the kernel matrix."""
        return self._bkd.get_diagonal(self(X1))

    @abstractmethod
    def __call__(self, X1: Array, X2: Array = None) -> Array:
        raise NotImplementedError()

    def __mul__(self, kernel: "Kernel") -> "ProductKernel":
        return ProductKernel(self, kernel)

    def __add__(self, kernel: "Kernel") -> "SumKernel":
        return SumKernel(self, kernel)

    def __repr__(self) -> str:
        return "{0}({1}, bkd={2})".format(
            self.__class__.__name__,
            self._hyp_list._short_repr(),
            self._bkd.__name__,
        )

    def _params_eval(self, active_opt_params: Array):
        # define function that evaluates the kernel for different parameters
        self._hyp_list.set_active_opt_params(active_opt_params)
        return self(self._samples)

    def param_jacobian(self, samples: Array) -> Array:
        self._samples = samples
        return self._bkd.jacobian(
            self._params_eval, self._hyp_list.get_active_opt_params()
        )

    def param_jacobian_implemented(self) -> bool:
        if self._bkd.jacobian_implemented():
            return True
        return False

    def input_jacobian_implemented(self) -> bool:
        if self._bkd.jacobian_implemented():
            return True
        return False

    def _input_eval(self, X1: Array):
        return self(X1[:, None], self._X2)[0]

    def input_jacobian(self, X1: Array, X2: Array) -> Array:
        self._X2 = X2
        return self._bkd.jacobian(self._input_eval, X1[:, 0])

    def input_apply_hessian(self, X1: Array, X2: Array) -> Array:
        self._X2 = X2
        return self._bkd.hvp(
            self._input_eval, self._hyp_list.get_active_opt_params()
        )


class CompositionKernel(Kernel):
    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        self._kernel1 = kernel1
        self._kernel2 = kernel2
        self._hyp_list = kernel1.hyp_list() + kernel2.hyp_list()
        if not kernel1._bkd.bkd_equal(kernel1._bkd, kernel2._bkd):
            raise ValueError("Kernels must have the same backend.")
        self._bkd = kernel1._bkd

    def nvars(self) -> int:
        if hasattr(self._kernel1, "nvars"):
            return self._kernel1.nvars()
        return self._kernel2.nvars()

    def param_jacobian_implemented(self) -> bool:
        return (
            self._kernel1.param_jacobian_implemented()
            and self._kernel2.param_jacobian_implemented()
        )

    def input_jacobian_implemented(self) -> bool:
        return (
            self._kernel1.input_jacobian_implemented()
            and self._kernel2.input_jacobian_implemented()
        )


class ProductKernel(CompositionKernel):
    def diag(self, X1: Array) -> Array:
        return self._kernel1.diag(X1) * self._kernel2.diag(X1)

    def __repr__(self) -> str:
        return "{0} * {1}".format(self._kernel1, self._kernel2)

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        return self._kernel1(X1, X2) * self._kernel2(X1, X2)

    def param_jacobian(self, X) -> Array:
        Kmat1 = self._kernel1(X)
        Kmat2 = self._kernel2(X)
        jac1 = self._kernel1.param_jacobian(X)
        jac2 = self._kernel2.param_jacobian(X)
        return self._bkd.dstack(
            [jac1 * Kmat2[..., None], jac2 * Kmat1[..., None]]
        )

    def input_jacobian(self, X1: Array, X2: Array) -> Array:
        K1 = self._kernel1(X1, X2)
        K2 = self._kernel2(X1, X2)
        jac1 = self._kernel1.input_jacobian(X1, X2)
        jac2 = self._kernel2.input_jacobian(X1, X2)
        return K2.T * jac1 + K1.T * jac2


class SumKernel(CompositionKernel):
    def diag(self, X1) -> Array:
        return self._kernel1.diag(X1) + self._kernel2.diag(X1)

    def __repr__(self) -> str:
        return "{0} + {1}".format(self._kernel1, self._kernel2)

    def __call__(self, X1, X2=None) -> Array:
        return self._kernel1(X1, X2) + self._kernel2(X1, X2)

    def param_jacobian(self, X) -> Array:
        jac1 = self._kernel1.param_jacobian(X)
        jac2 = self._kernel2.param_jacobian(X)
        return self._bkd.dstack([jac1, jac2])

    def input_jacobian(self, X1: Array, X2: Array) -> Array:
        jac1 = self._kernel1.input_jacobian(X1, X2)
        jac2 = self._kernel2.input_jacobian(X1, X2)
        return jac1 + jac2


class MaternKernel(Kernel):
    def __init__(
        self,
        nu: float,
        lenscale: float,
        lenscale_bounds: Tuple[float, float],
        nvars: int,
        fixed: bool = False,
        backend: LinAlgMixin = None,
    ):
        """The matern kernel for varying levels of smoothness."""
        super().__init__(backend)
        self._nvars = nvars
        self._nu = nu
        transform = LogHyperParameterTransform(backend=self._bkd)
        self._lenscale = HyperParameter(
            "lenscale",
            nvars,
            lenscale,
            lenscale_bounds,
            transform,
            fixed=fixed,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._lenscale])

    def diag(self, X1: Array) -> Array:
        return self._bkd.full((X1.shape[1],), 1)

    def _eval_distance_form(self, distances: Array) -> Array:
        if self._nu == self._bkd.inf():
            return self._bkd.exp(-(distances**2) / 2.0)
        if self._nu == 5 / 2:
            tmp = self._bkd.sqrt(5) * distances
            return (1.0 + tmp + tmp**2 / 3.0) * self._bkd.exp(-tmp)
        if self._nu == 3 / 2:
            tmp = self._bkd.sqrt(3) * distances
            return (1.0 + tmp) * self._bkd.exp(-tmp)
        if self._nu == 1 / 2:
            return self._bkd.exp(-distances)
        raise ValueError(
            "Matern kernel with nu={0} not supported".format(self._nu)
        )

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        lenscale = self._lenscale.get_values()
        if X2 is None:
            X2 = X1
            # note using sqaureform(pdist(X1).T) is likely faster than
            # cdist(X1.T, X1.T) but torch does not have squareform
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        return self._eval_distance_form(distances)

    def nvars(self) -> int:
        return self._nvars

    def param_jacobian(self, samples: Array) -> Array:
        self._samples = samples
        if self._nu == self._bkd.inf():
            # todo save and load K during __call__
            lenscale = self._lenscale.get_values()
            distances = self._bkd.cdist(
                samples.T / lenscale, samples.T / lenscale
            )
            Kmat = self._eval_distance_form(distances)
            distances = (
                samples.T[:, None, :] - samples.T[None, ...]
            ) ** 2 / lenscale**2
            return distances * Kmat[..., None]
        # TODO compute gradient analytically for nu = 0.5, 1.5, 2.5
        return super().param_jacobian(samples)

    def param_jacobian_implemented(self) -> bool:
        if self._nu == self._bkd.inf():
            return True
        return super().param_jacobian_implemented()

    def input_jacobian(self, X1: Array, X2: Array) -> Array:
        lenscale = self._lenscale.get_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        if self._nu == self._bkd.inf():
            tmp2 = (self._bkd.tile(X1.T, (X2.shape[1], 1)) - X2.T) / (
                lenscale**2
            )
            K = self._bkd.exp(-0.5 * distances**2)
            return -K.T * tmp2
        if self._nu == 3 / 2:
            tmp1 = math.sqrt(3) * distances
            tmp2 = (self._bkd.tile(X1.T, (X2.shape[1], 1)) - X2.T) / (
                lenscale**2
            )
            K = self._bkd.exp(-tmp1)
            return -3 * K.T * tmp2
        if self._nu == 5 / 2:
            tmp1 = math.sqrt(5) * distances
            K = self._bkd.exp(-tmp1)
            tmp2 = (self._bkd.tile(X1.T, (X2.shape[1], 1)) - X2.T) / (
                lenscale**2
            )
            return -5 / 3 * K.T * tmp2 * (math.sqrt(5) * distances.T + 1)
        return super().input_jacobian(X1, X2)

    def __repr__(self) -> str:
        return "{0}(nu={1}, {2}, bkd={3})".format(
            self.__class__.__name__,
            self._nu,
            self._hyp_list._short_repr(),
            self._bkd.__name__,
        )


class ConstantKernel(Kernel):
    def __init__(
        self,
        constant: float,
        constant_bounds: Tuple[float, float] = None,
        transform: HyperParameterTransform = None,
        fixed: bool = False,
        backend: LinAlgMixin = None,
    ):
        if backend is None and transform is not None:
            backend = transform._bkd
        super().__init__(backend)
        if transform is None:
            transform = IdentityHyperParameterTransform(backend=self._bkd)
        if constant_bounds is None:
            constant_bounds = [-self._bkd.inf(), self._bkd.inf()]
        self._const = HyperParameter(
            "const",
            1,
            constant,
            constant_bounds,
            transform,
            fixed=fixed,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._const])

    def diag(self, X1) -> Array:
        return self._bkd.full((X1.shape[1],), self._hyp_list.get_values()[0])

    def __call__(self, X1, X2=None) -> Array:
        if X2 is None:
            X2 = X1
        # full does not work when const value requires grad
        # return full((X1.shape[1], X2.shape[1]), self._const.get_values()[0])
        const = self._bkd.empty((X1.shape[1], X2.shape[1]))
        # const[:] = self._const.get_values()[0]
        const = self._bkd.up(
            const,
            self._bkd.arange(X1.shape[1], dtype=int),
            self._const.get_values()[0],
            axis=0,
        )
        return const

    def param_jacobian_implemented(self) -> bool:
        return True

    def param_jacobian(self, samples: Array) -> Array:
        return self._bkd.full((samples.shape[1], samples.shape[1], 1), 1.0)


class GaussianNoiseKernel(Kernel):
    def __init__(
        self,
        constant: float,
        constant_bounds: Tuple[float, float] = None,
        fixed: bool = False,
        backend: LinAlgMixin = None,
    ):
        super().__init__(backend)
        self._const = HyperParameter(
            "const",
            1,
            constant,
            constant_bounds,
            LogHyperParameterTransform(backend=self._bkd),
            fixed=fixed,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._const])

    def diag(self, X: Array) -> Array:
        return self._bkd.full((X.shape[1],), self._hyp_list.get_values()[0])

    def __call__(self, X: Array, Y: Array = None) -> Array:
        if Y is None:
            return self._const.get_values()[0] * self._bkd.eye(X.shape[1])
        # full does not work when const value requires grad
        # return full((X.shape[1], Y.shape[1]), self._const.get_values()[0])
        const = self._bkd.full((X.shape[1], Y.shape[1]), 0.0)
        return const

    def param_jacobian_implemented(self) -> bool:
        return True

    def param_jacobian(self, samples: Array) -> Array:
        return self._bkd.eye(samples.shape[1])[..., None]


class PeriodicMaternKernel(MaternKernel):
    def __init__(
        self,
        nu: float,
        period,
        period_bounds: Tuple[float, float],
        lenscale,
        lenscale_bounds: Tuple[float, float],
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
        self._hyp_list += HyperParameterList([self._period])

    def __call__(self, X: Array, Y: Array = None) -> Array:
        if Y is None:
            Y = X
        lenscale = self._lenscale.get_values()
        period = self._period.get_values()
        distances = self._bkd.cdist(X.T / period, Y.T / period) / lenscale
        return super()._eval_distance_form(distances)

    def diag(self, X) -> Array:
        return super().diag(X)


class HilbertSchmidtKernel(Kernel):
    def __init__(
        self,
        basis1,
        basis2,
        weights,
        weight_bounds: Tuple[float, float],
        transform=None,
        normalize: bool = False,
    ):
        super().__init__(basis1._bkd)
        self._nvars = basis1.nvars()
        self._basis1 = basis1
        self._basis2 = basis2
        self._nterms = basis1.nterms() * basis2.nterms()
        self._normalize = normalize
        self._weights = HyperParameter(
            "weights",
            self._nterms,
            weights,
            weight_bounds,
            transform,
            backend=self._bkd,
        )
        self._hyp_list = HyperParameterList([self._weights])

        self._X1, self._X2 = None, None
        self._X1basis_mat, self._X2basis_mat = None, None

    def _get_weights(self) -> Array:
        return self._bkd.reshape(
            self._weights.get_values(),
            (self._basis1.nterms(), self._basis2.nterms()),
        )

    def _get_basis_matrices(self, X1: Array, X2: Array) -> Tuple[Array, Array]:
        if (
            self._X1 is not None
            and self._X1.shape == X2.shape
            and self._bkd.allclose(self._X1, X2, atol=1e-15)
        ):
            X1basis_mat = self._X1basis_mat
        else:
            X1basis_mat = self._basis1(X1)
            if self._normalize:
                X1basis_mat /= self._bkd.norm(X1basis_mat, axis=1)[:, None]
            self._X1 = self._bkd.copy(X1)
            self._X1basis_mat = self._bkd.copy(X1basis_mat)
        if (
            self._X2 is not None
            and self._X2.shape == X2.shape
            and self._bkd.allclose(self._X2, X2, atol=1e-15)
        ):
            X2basis_mat = self._X2basis_mat
        else:
            X2basis_mat = self._basis2(X2)
            if self._normalize:
                X2basis_mat /= self._bkd.norm(X2basis_mat, axis=1)[:, None]
            self._X2 = self._bkd.copy(X2)
            self._X2basis_mat = self._bkd.copy(X2basis_mat)
        return X1basis_mat, X2basis_mat

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        weights = self._get_weights()
        if X2 is None:
            X2 = X1
        X1basis_mat, X2basis_mat = self._get_basis_matrices(X1, X2)
        K = (X1basis_mat @ weights) @ (X2basis_mat.T)
        return K

    def __repr__(self) -> str:
        return "{0}({1}, inbasis={2}, outbasis={3}, bkd={4})".format(
            self.__class__.__name__,
            self._hyp_list._short_repr(),
            self._basis2,
            self._basis1,
            self._bkd.__name__,
        )


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

    def set_active_opt_params(self, active_params: Array):
        super().set_active_opt_params(active_params)
        self._set_covariance_matrix()

    def __repr__(self) -> str:
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
        radii: float = 1,
        radii_bounds: Tuple[float, float] = [1e-1, 1],
        angles: float = math.pi / 2,
        angle_bounds: Tuple[float, float] = [0, math.pi],
        backend: LinAlgMixin = None,
    ):
        if backend is None:
            if radii_transform is not None:
                backend = radii_transform._bkd
            elif angle_transform is not None:
                backend = angle_transform._bkd
            else:
                backend = NumpyLinAlgMixin

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
        self._hyp_list = HyperParameterList(
            [SphericalCovarianceHyperParameter([self._radii, self._angles])]
        )

    def hyp_list(self) -> HyperParameterList:
        return self._hyp_list

    def _validate_bounds(
        self,
        radii_bounds: Tuple[float, float],
        angle_bounds: Tuple[float, float],
    ):
        bounds = self._trans.get_spherical_bounds()
        # all theoretical radii_bounds are the same so just check one
        radii_bounds = self._bkd.asarray(radii_bounds)
        if radii_bounds.shape[0] == 2:
            radii_bounds = self._bkd.tile(radii_bounds, (self.noutputs,))
        radii_bounds = radii_bounds.reshape((radii_bounds.shape[0] // 2, 2))
        if self._bkd.any(
            radii_bounds[:, 0] < bounds[: self.noutputs, 0]
        ) or self._bkd.any(radii_bounds[:, 1] > bounds[: self.noutputs, 1]):
            raise ValueError("radii bounds are inconsistent")
        # all theoretical angle_bounds are the same so just check one
        angle_bounds = self._bkd.asarray(angle_bounds)
        if angle_bounds.shape[0] == 2:
            angle_bounds = self._bkd.tile(
                angle_bounds, (self._trans.ntheta - self.noutputs,)
            )
        angle_bounds = angle_bounds.reshape((angle_bounds.shape[0] // 2, 2))
        if self._bkd.any(
            angle_bounds[:, 0] < bounds[self.noutputs :, 0]
        ) or self._bkd.any(angle_bounds[:, 1] > bounds[self.noutputs :, 1]):
            raise ValueError("angle bounds are inconsistent")

    def get_covariance_matrix(self) -> Array:
        return self._hyp_list.hyper_params[0].cov_matrix

    def __call__(self, ii: int, jj: int) -> float:
        # chol factor must be recomputed each time even if hyp_values have not
        # changed otherwise gradient graph becomes inconsistent
        return self._hyp_list.hyper_params[0].cov_matrix[ii, jj]

    def __repr__(self) -> str:
        return "{0}(radii={1}, angles={2} cov={3})".format(
            self.__class__.__name__,
            self._radii,
            self._angles,
            self.get_covariance_matrix().detach().numpy(),
        )
