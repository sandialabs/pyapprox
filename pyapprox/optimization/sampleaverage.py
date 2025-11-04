from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.interface.model import Model, expand_samples_from_indices
from pyapprox.optimization.minimize import ConstraintFromModel


# TODO consider merging with multifidelity.stat
class SampleAverageStat(ABC):
    def __init__(self, backend: BackendMixin):
        self._bkd = backend

    def jacobian_implemented(self) -> bool:
        return False

    def apply_jacobian_implemented(self) -> bool:
        return False

    def hessian_implemented(self) -> bool:
        return False

    def apply_hessian_implemented(self) -> bool:
        return False

    @abstractmethod
    def _values(self, values: Array, weights: Array) -> Array:
        """
        User defined function to compute the sample average statistic.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic
        """
        raise NotImplementedError

    def _check_weights(self, values, weights: Array):
        nsamples = values.shape[0]
        if weights.shape != (nsamples, 1):
            raise ValueError(f"{weights.shape=} but must be {(nsamples, 1)}")

    def __call__(self, values: Array, weights: Array) -> Array:
        """
        Compute the sample average statistic.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (1, nqoi)
            Estimate of the statistic
        """
        nsamples, nqoi = values.shape
        self._check_weights(values, weights)
        vals = self._values(values, weights)
        if vals.shape != (1, nqoi):
            raise ValueError(
                f"{self}._values returned 2D array with shape {vals.shape} "
                f"but must have shape {(1, nqoi)}"
            )
        return vals

    def _jacobian(self, values: Array, jac_values: Array, weights: Array):
        """
        User defined function to compute the sample average jacobian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
            Jacobian values at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars)
            Estimate of the statistic jacobian
        """
        raise NotImplementedError

    def _check_jac_values_shape(self, values: Array, jac_values: Array):
        nsamples, nqoi = values.shape
        if jac_values.shape[0] != nsamples:
            raise ValueError(
                f"shape of first dimension of values {nsamples} and jac_values"
                f" {jac_values.shape[0]} must match"
            )

        if jac_values.shape[1] != nqoi:
            raise ValueError(
                f"{self}: shape of second dimension of values {nsamples} and "
                f"jac_values {jac_values.shape[1]} must match"
            )

    def jacobian(self, values: Array, jac_values: Array, weights: Array):
        """
        Compute the sample average jacobian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
            Jacobian values at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars)
            Estimate of the statistic jacobian
        """
        nqoi = values.shape[1]
        self._check_jac_values_shape(values, jac_values)
        self._check_weights(values, weights)
        jac = self._jacobian(values, jac_values, weights)
        if jac.shape != (nqoi, jac_values.shape[2]):
            raise ValueError(
                f"_jacobian returned matrix with shape {jac.shape}"
                f" but must have shape {(nqoi, jac_values.shape[2])}"
            )
        return jac

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        """
        Use defined function to compute the sample average jacobian dot
        product with a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            Values of the jacobian vector product (jvp) at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        raise NotImplementedError

    def _check_jv_values(self, values: Array, jv_values: Array):
        nsamples, nqoi = values.shape
        if jv_values.shape[0] != nsamples:
            raise ValueError(
                f"shape of first dimension of values {nsamples} and jv_values"
                f" {jv_values.shape[0]} must match"
            )

        if jv_values.shape[1] != nqoi:
            raise ValueError(
                f"shape of second dimension of values {nsamples} and "
                f"jv_values {jv_values.shape[1]} must match"
            )
        if jv_values.shape[2] != 1:
            raise ValueError("jv_values.shape[2] must == 1")

    def apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        """
        Compute the sample average jacobian dot product with
        a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            Values of the jacobian vector product (jvp) at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        nsamples, nqoi = values.shape
        self._check_jv_values(values, jv_values)
        self._check_weights(values, weights)
        jvp = self._apply_jacobian(values, jv_values, weights)
        if jvp.shape != (nqoi, 1):
            raise ValueError(
                f"_apply_jacobian returned matrix with shape {jvp.shape}"
                f" but must have shape {(nqoi, 1)}"
            )
        return jvp

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        """
        User defined function to compute the sample average hessian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           Function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
           Jacobian at each sample.

        hess_values: array (nsamples, nqoi, nvars, nvars)
           Hessian at each sample.

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars, nvars)
            Estimate of the statistic hessian
        """
        raise NotImplementedError

    def hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        """
        Compute the sample average hessian.

        Parameters
        ----------
        values: array (nsamples, nqoi)
           Function values at each sample

        jac_values: array (nsamples, nqoi, nvars)
           Jacobian at each sample

        hess_values: array (nsamples, nqoi, nvars, nvars)
            Hessian at each sample.

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        Returns
        -------
        estimate: array (nqoi, nvars, nvars)
            Estimate of the statistic hessian
        """
        nqoi = values.shape[1]
        nvars = jac_values.shape[2]
        self._check_jac_values_shape(values, jac_values)
        self._check_weights(values, weights)
        hess = self._hessian(values, jac_values, hess_values, weights)
        if hess.shape != (nqoi, nvars, nvars):
            raise ValueError(
                f"_hessian returned a 3D tensor with shape {hess.shape}"
                f" but must have shape {(nqoi, 1)}"
            )
        return hess

    def _apply_hessian(
        self,
        values: Array,
        jv_values: Array,
        hv_values: Array,
        weights: Array,
        lagrange: Array,
    ) -> Array:
        """
        User defined function to compute the sample average weighted
        combination of the QoI, dot product with a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            Values of the jacobian vector product (jvp) at each sample

        hv_values : array (nsamples, nqoi, 1)
            Values of the hessian vector product (hvp) at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        lagrange: array (nqoi, 1)
            The weights defining the combination of QoI.

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        raise NotImplementedError

    def apply_hessian(
        self,
        values: Array,
        jv_values: Array,
        hv_values: Array,
        weights: Array,
        lagrange: Array,
    ) -> Array:
        """
        Compute the sample average weighted combination of the
        Qoi, dot product with a vector.

        Parameters
        ----------
        values: array (nsamples, nqoi)
            Function values at each sample

        jv_values : array (nsamples, nqoi, 1)
            Values of the jacobian vector product (jvp) at each sample

        hv_values : array (nsamples, nqoi, 1)
            Values of the hessian vector product (hvp) at each sample

        weights: array(nsamples, 1)
            Quadrature weight for each sample

        lagrange: array (nqoi, 1)
            The weights defining the combination of QoI.

        Returns
        -------
        estimate: array (nqoi, 1)
            Estimate of the statistic jacobian vector product
        """
        nsamples, nqoi = values.shape
        self._check_jv_values(values, jv_values)
        self._check_weights(values, weights)
        if hv_values.shape != (nsamples, nqoi, 1):
            raise ValueError(
                f"{hv_values.shape=} but must have shape {(nsamples, nqoi, 1)}"
            )
        if lagrange.shape != (nqoi, 1):
            raise ValueError(
                f"{lagrange.shape=} but must have shape {(nqoi, 1)}"
            )
        hvp = self._apply_hessian(
            values, jv_values, hv_values, weights, lagrange
        )
        if hvp.shape != (nqoi, 1):
            raise ValueError(
                f"_apply_hessian returned array with shape {hvp.shape}"
                f"but must have shape {(nqoi, 1)}"
            )
        return hvp

    def __repr__(self) -> str:
        return "{0}()".format(self.__class__.__name__)

    def label(self) -> str:
        """
        Return short label typically used for plotting
        """
        if not hasattr(self, "_label"):
            return self.__repr__()
        return self._label()


class SampleAverageMean(SampleAverageStat):
    def jacobian_implemented(self) -> bool:
        return True

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        # values.shape (nsamples, ncontraints)
        return (values.T @ weights).T

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        return self._bkd.einsum("ijk,i->jk", jac_values, weights[:, 0])

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        # jac_values.shape (nsamples, ncontraints)
        return (jv_values[..., 0].T @ weights[:, 0])[:, None]

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        return self._bkd.einsum("ijkl,i->jkl", hess_values, weights[:, 0])

    def _label(self) -> str:
        return "Mean"


class SampleAverageVariance(SampleAverageStat):
    def __init__(self, backend: BackendMixin):
        super().__init__(backend=backend)
        self._mean_stat = SampleAverageMean(backend=backend)

    def jacobian_implemented(self) -> bool:
        return True

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _diff(self, values: Array, weights: Array) -> Array:
        mean = self._mean_stat(values, weights).T
        return (values - mean[:, 0]).T

    def _values(self, values: Array, weights: Array) -> Array:
        # values.shape (nsamples, ncontraints)
        return (self._diff(values, weights) ** 2 @ weights).T

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        # jac_values.shape (nsamples, ncontraints, ndesign_vars)
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)[
            None, :
        ]
        tmp = jac_values - mean_jac
        tmp = 2 * self._diff(values, weights).T[..., None] * tmp
        return self._bkd.einsum("ijk,i->jk", tmp, weights[:, 0])

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        mean_jv = self._mean_stat.apply_jacobian(values, jv_values, weights)
        tmp = jv_values - mean_jv
        tmp = 2 * self._diff(values, weights).T * tmp[..., 0]
        return self._bkd.einsum("ij,i->j", tmp, weights[:, 0])

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)[
            None, :
        ]
        mean_hess = self._mean_stat.hessian(
            values, jac_values, hess_values, weights
        )[None, :]
        tmp_jac = jac_values - mean_jac
        tmp_hess = hess_values - mean_hess
        tmp1 = 2 * self._diff(values, weights).T[..., None, None] * tmp_hess
        tmp2 = 2 * self._bkd.einsum("ijk, ijl->ijkl", tmp_jac, tmp_jac)
        return self._bkd.einsum("ijkl,i->jkl", tmp1 + tmp2, weights[:, 0])

    def _label(self) -> str:
        return "Variance"


class SampleAverageStdev(SampleAverageVariance):
    def jacobian_implemented(self) -> bool:
        return True

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _values(self, samples: Array, weights: Array) -> Array:
        return self._bkd.sqrt(super()._values(samples, weights))

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        variance_jac = super()._jacobian(values, jac_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super()._values(values, weights).T))
        return tmp * variance_jac

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        variance_jv = super()._apply_jacobian(values, jv_values, weights)
        # d/dx y^{1/2} = 0.5y^{-1/2}
        tmp = 1 / (2 * self._bkd.sqrt(super()._values(values, weights).T))
        return tmp * variance_jv[:, None]

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        variance_jac = super()._jacobian(values, jac_values, weights)
        variance_hess = super()._hessian(
            values, jac_values, hess_values, weights
        )
        # f:R^n->R, g(R->R) h(x) = g(f(x))
        # d^2h(x)/dx^2 g'(f(x))\nabla^2 f(x)+g''(f(x))\nabla f(x)\nabla f(x)^T
        # g(x)=sqrt(x), g'(x) = 1/(2x^{1/2}), g''(x)=-1/(4x^{3/2})
        tmp0 = self._bkd.sqrt(super()._values(values, weights).T)
        tmp1 = (
            1.0
            / (4.0 * tmp0[..., None] ** 3.0)
            * self._bkd.einsum("ij,ik->ijk", variance_jac, variance_jac)
        )
        tmp2 = 1.0 / (2.0 * tmp0[..., None]) * variance_hess
        return tmp2 - tmp1

    def _label(self) -> str:
        return "StDev"


class SampleAverageMeanPlusStdev(SampleAverageStat):
    def __init__(self, safety_factor: float, backend: BackendMixin):
        super().__init__(backend=backend)
        self._mean_stat = SampleAverageMean(backend=self._bkd)
        self._stdev_stat = SampleAverageStdev(backend=self._bkd)
        self._safety_factor = safety_factor

    def jacobian_implemented(self) -> bool:
        return True

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        return self._mean_stat(
            values, weights
        ) + self._safety_factor * self._stdev_stat(values, weights)

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        return self._mean_stat.jacobian(
            values, jac_values, weights
        ) + self._safety_factor * self._stdev_stat.jacobian(
            values, jac_values, weights
        )

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        return self._mean_stat.apply_jacobian(
            values, jv_values, weights
        ) + self._safety_factor * self._stdev_stat.apply_jacobian(
            values, jv_values, weights
        )

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        return self._mean_stat.hessian(
            values, jac_values, hess_values, weights
        ) + self._safety_factor * self._stdev_stat.hessian(
            values, jac_values, hess_values, weights
        )

    def __repr__(self) -> str:
        return "{0}(factor={1})".format(
            self.__class__.__name__, float(self._safety_factor)
        )

    def _label(self) -> str:
        return "Mean+StDev"


class SampleAverageEntropicRisk(SampleAverageStat):
    def __init__(self, alpha: float, backend: BackendMixin):
        super().__init__(backend)
        self._alpha = alpha

    def jacobian_implemented(self) -> bool:
        return True

    def apply_jacobian_implemented(self) -> bool:
        return True

    def hessian_implemented(self) -> bool:
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        # values (nsamples, noutputs)
        return (
            self._bkd.log(self._bkd.exp(self._alpha * values.T) @ weights).T
            / self._alpha
        )

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        # jac_values (nsamples, noutputs, nvars)
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        # h = g(f(x))
        # dh/dx = g'(f(x))\nabla f(x)
        # g(y) = log(y)/alpha, g'(y) = 1/(alpha*y)
        jac = (
            1
            / tmp
            * self._bkd.einsum(
                "ijk,i->jk",
                (self._alpha * exp_values[..., None] * jac_values),
                weights[:, 0],
            )
        )
        return jac / self._alpha

    def _apply_jacobian(
        self, values: Array, jv_values: Array, weights: Array
    ) -> Array:
        exp_values = self._bkd.exp(self._alpha * values)
        tmp = exp_values.T @ weights
        jv = (
            1
            / tmp
            * self._bkd.einsum(
                "ij,i->j",
                (self._alpha * exp_values * jv_values[..., 0]),
                weights[:, 0],
            )[:, None]
        )
        return jv / self._alpha

    def _hessian(
        self,
        values: Array,
        jac_values: Array,
        hess_values: Array,
        weights: Array,
    ) -> Array:
        exp_values = self._bkd.exp(self._alpha * values)
        exp_jac = self._alpha * self._bkd.einsum(
            "ijk,i->jk", (exp_values[..., None] * jac_values), weights[:, 0]
        )
        exp_jac_outprod = self._bkd.einsum("ij,ik->ijk", exp_jac, exp_jac)
        jac_values_outprod = self._bkd.einsum(
            "ijk,ijl->ijkl", jac_values, jac_values
        )
        exp_hess = self._alpha * self._bkd.einsum(
            "ijkl,i->jkl",
            (exp_values[..., None, None] * hess_values),
            weights[:, 0],
        ) + self._alpha**2 * self._bkd.einsum(
            "ijkl,i->jkl",
            (exp_values[..., None, None] * jac_values_outprod),
            weights[:, 0],
        )
        tmp0 = exp_values.T @ weights
        tmp1 = 1.0 / tmp0[..., None] * exp_hess
        tmp2 = 1.0 / tmp0[..., None] ** 2 * exp_jac_outprod
        return (tmp1 - tmp2) / self._alpha

    def __repr__(self) -> str:
        return "{0}(alpha={1})".format(
            self.__class__.__name__, float(self._alpha)
        )

    def _label(self) -> str:
        return "Entropic"


class SampleAverageSmoothedAverageValueAtRisk(SampleAverageStat):
    """
    Compute average value at risk without the need to estimate
    the value at risk.

    delta controls accuracy. Larger delta produces more accurate estimate
    """

    def __init__(
        self,
        alpha: float,
        backend: BackendMixin,
        delta: float = 100,
        chunk_size=10000,
    ):
        super().__init__(backend)
        alpha = self._bkd.atleast1d(self._bkd.asarray(alpha))
        self._alpha = alpha
        self._delta = delta
        self._chunk_size = chunk_size
        # some optimization algorithms can use nonzero alpha but hard code here
        # so any solver can be used
        self._lambda = 0.0

    def jacobian_implemented(self) -> bool:
        return True

    def _project_slow(self, values: Array, weights: Array) -> Array:
        # this function uses too much memory and makes unncessary
        # calculations
        # Compute all possible kinks
        lbnd = 0.0
        ubnd = 1.0 / (1.0 - self._alpha)
        dvalues = values / weights
        K = self._bkd.sort(self._bkd.hstack([lbnd - dvalues, ubnd - dvalues]))
        # Compute residuals directly for all kinks
        values_res = self._bkd.sort(
            1.0
            - weights
            @ self._bkd.maximum(
                lbnd, self._bkd.minimum(ubnd, dvalues[:, None] + K)
            )
        )
        # # broadcasting above can cause memory to run out so have to
        # # operate over chunks
        # values_res = self._bkd.zeros(len(K))  # Initialize residuals array
        # for ii in range(0, len(K), self._chunk_size):
        #     K_chunk = K[ii : ii + self._chunk_size]  # Process a chunk of K
        #     chunk_result = self._bkd.maximum(
        #         lbnd, self._bkd.minimum(ubnd, dvalues[:, None] + K_chunk)
        #     )
        #     values_res[ii : ii + self._chunk_size] = 1 - weights @ chunk_result

        # Bracket zero using numpy.searchsorted
        idx_end = self._bkd.searchsorted(values_res, 0, side="right")
        idx_beg = idx_end - 1

        # Ensure valid indices for bracketing
        x_beg = K[idx_beg]
        x_end = K[idx_end]
        values_beg = values_res[idx_beg]
        values_end = values_res[idx_end]

        # Compute zero using linear interpolation
        lam = (values_end * x_beg - values_beg * x_end) / (
            values_end - values_beg
        )
        # print(lam, "lam", idx_beg, idx_end)
        # print(K, "K")
        # print(values_res)

        # Compute projection
        proj = weights * self._bkd.maximum(
            lbnd, self._bkd.minimum(ubnd, dvalues + lam)
        )
        return proj

    def _project(self, values: Array, weights: Array) -> Array:
        """
        Compute the projection of y onto the CVaR risk envelope.

        Parameters:
            values (Array): Vector to be projected (length = number of samples).

        Returns:
            Array: Projection of values onto CVaR risk envelope (length = number of samples).
        """
        # Compute all possible kinks
        lbnd = self._bkd.asarray(0.0)
        ubnd = 1.0 / (1.0 - self._alpha)
        dvalues = values / weights
        K = self._bkd.flip(
            self._bkd.sort(self._bkd.hstack([lbnd - dvalues, ubnd - dvalues]))
        )

        # Bracket zero
        nsamp = len(values)

        def res(x):
            return 1.0 - weights @ self._bkd.maximum(
                lbnd, self._bkd.minimum(ubnd, dvalues + x)
            )

        ibeg = 0
        imid = nsamp
        iend = 2 * nsamp
        x1 = K[ibeg]
        y1 = res(x1)
        x2 = K[imid]
        y2 = res(x2)
        while True:
            # print(ibeg, imid, iend, "i", self._alpha)
            # print(x1, x2, "x")
            # print(y1, y2, "y")
            if self._bkd.sign(y1) != self._bkd.sign(y2):
                iend = imid
            else:
                ibeg = imid
                x1 = x2
                y1 = y2
            if iend - ibeg == 1:
                imid = iend
            else:
                imid = ibeg + round((iend - ibeg) / 2)
            x2 = K[imid]
            y2 = res(x2)
            if iend - ibeg == 1:
                # print(round((iend - ibeg) / 2), ((iend - ibeg) / 2))
                # print(ibeg, imid, iend)
                break

        assert self._bkd.sign(y1) != self._bkd.sign(y2)

        # Compute value of x that produces zero residual
        lam = (y2 * x1 - y1 * x2) / (y2 - y1)

        # Return projection
        return weights * self._bkd.maximum(
            lbnd, self._bkd.minimum(ubnd, dvalues + lam)
        )

    def _check_values_weights(self, values: Array, weights: Array):
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("values must be a 2D array with a single column")
        if values.shape != weights.shape:
            raise ValueError(f"{values.shape=} but {weights.shape=}")
        if not self._bkd.allclose(
            self._bkd.ones((1,)), self._bkd.sum(weights), atol=1e-15
        ):
            raise ValueError(
                "weights must sum to one but sum to {0}".format(
                    self._bkd.sum(weights)
                )
            )

    def _evaluate_single(self, values: Array, weights: Array) -> Array:
        self._check_values_weights(values, weights)
        proj_values = self._project(
            weights[:, 0] * values[:, 0] * self._delta + self._lambda,
            weights[:, 0],
        )
        return self._bkd.sum(proj_values * values[:, 0]) - 1.0 / (
            2.0 * self._delta
        ) * (proj_values - self._lambda) / weights[:, 0] @ (
            proj_values - self._lambda
        )

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Parameters
        ----------
        values : Array (nsamples, nqoi)
            Samples of the random variable.

        weights : Array (nsamples, 1)
            Importance sampling weights.

        Returns
        -------
        vals : Array (1, nqoi)
            The values of CVaR for each QoI.
        """
        vals = self._bkd.stack(
            [
                self._evaluate_single(values[:, ii : ii + 1], weights)
                for ii in range(values.shape[1])
            ]
        )[None, :]
        return vals

    def _jacobian_single(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        proj_values = self._project(
            weights[:, 0] * values[:, 0] * self._delta + self._lambda,
            weights[:, 0],
        )
        return self._bkd.einsum("i,ij->j", proj_values, jac_values)

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        jac = self._bkd.stack(
            [
                self._jacobian_single(
                    values[:, ii : ii + 1], jac_values[:, ii], weights
                )
                for ii in range(values.shape[1])
            ]
        )
        return jac

    def __repr__(self) -> str:
        return "{0}(alpha={1})".format(
            self.__class__.__name__, float(self._alpha)
        )

    def _label(self) -> str:
        return f"AVaR_{float(self._alpha):.2f}"


class SampleAverageSmoothedAverageValueAtRiskDeviation(
    SampleAverageSmoothedAverageValueAtRisk
):
    """
    Compute average value at risk deviation without the need to estimate
    the value at risk.
    """

    def set_mean(self, mean: Array):
        self._mean = mean

    def __call__(self, values: Array, weights: Array) -> Array:
        return super().__call__(values, weights) - self._mean


class SampleAverageConstraint(ConstraintFromModel):
    def __init__(
        self,
        model: Model,
        samples: Array,
        weights: Array,
        stat: SampleAverageStat,
        design_bounds: Array,
        nvars: int,
        design_indices: Array,
        backend: BackendMixin,
        keep_feasible: bool = False,
    ):
        self._stat = stat
        if not model._bkd.bkd_equal(model._bkd, backend):
            raise ValueError(
                "model._bkd {0} is inconsistent with backend {1}".format(
                    model._bkd.__name__, backend.__name__
                )
            )
        if not model._bkd.bkd_equal(model._bkd, stat._bkd):
            raise ValueError(
                "model._bkd {0} is inconsistent with stat._bkd {1}".format(
                    model._bkd.__name__, stat._bkd.__name__
                )
            )
        super().__init__(model, design_bounds, keep_feasible)
        if samples.ndim != 2 or weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError("shapes of samples and/or weights are incorrect")
        if samples.shape[1] != weights.shape[0]:
            raise ValueError("samples and weights are inconsistent")
        self._weights = weights
        self._samples = samples
        self._nvars = nvars
        self._design_indices = design_indices
        self._random_indices = self._bkd.delete(
            self._bkd.arange(nvars, dtype=int), design_indices
        )
        # warning self._joint_samples must be recomputed if self._samples
        # is changed.
        self._joint_samples = expand_samples_from_indices(
            self._samples,
            self._random_indices,
            self._design_indices,
            self._bkd.zeros((design_indices.shape[0], 1)),
            bkd=self._bkd,
        )

    def nvars(self) -> int:
        # optimizers obtain nvars from here so must be size
        # of design variables
        return self._design_indices.shape[0]

    def _update_attributes(self):
        for attr in [
            "apply_jacobian_implemented",
            "jacobian_implemented",
        ]:
            setattr(self, attr, getattr(self._model, attr))

    def hessian_implemented(self) -> bool:
        return (
            self._model.hessian_implemented()
            and self._stat.hessian_implemented()
        )

    def _random_samples_at_design_sample(self, design_sample: Array) -> Array:
        # this is slow so only update design samples as self._samples is
        # always fixed
        # return ActiveSetVariableModel._expand_samples_from_indices(
        #     self._samples, self._random_indices, self._design_indices,
        #     design_sample)
        self._joint_samples[self._design_indices, :] = self._bkd.asarray(
            np.repeat(
                self._bkd.to_numpy(design_sample),
                self._samples.shape[1],
                axis=1,
            )
        )
        return self._joint_samples

    def _values(self, design_sample: Array) -> Array:
        self._check_sample(design_sample)
        samples = self._random_samples_at_design_sample(design_sample)
        values = self._model(samples)
        return self._stat(values, self._weights)

    def _jacobian(self, design_sample: Array) -> Array:
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple jacobians. Also how to take advantage of
        # adjoint methods that compute model values to then
        # compute jacobian
        # TODO: reuse values if design sample is the same as used to last call
        # to _values
        values = self._model(samples)
        jac_values = self._bkd.stack(
            [
                self._model.jacobian(sample[:, None])[:, self._design_indices]
                for sample in samples.T
            ]
        )
        return self._stat.jacobian(values, jac_values, self._weights)

    def _apply_jacobian(self, design_sample: Array, vec: Array) -> Array:
        samples = self._random_samples_at_design_sample(design_sample)
        # todo take advantage of model prallelism to compute
        # multiple apply_jacs
        expanded_vec = self._bkd.zeros((self._nvars, 1))
        expanded_vec[self._design_indices] = vec
        values = self._model(samples)
        jv_values = self._bkd.array(
            [
                self._model._apply_jacobian(sample[:, None], expanded_vec)
                for sample in samples.T
            ]
        )
        return self._stat.apply_jacobian(values, jv_values, self._weights)

    def _hessian(self, design_sample: Array) -> Array:
        # TODO: reuse values if design sample is the same as used to last call
        # to _values same for jac_values
        samples = self._random_samples_at_design_sample(design_sample)
        values = self._model(samples)
        jac_values = self._bkd.stack(
            [
                self._model.jacobian(sample[:, None])[:, self._design_indices]
                for sample in samples.T
            ]
        )
        idx = np.ix_(self._design_indices, self._design_indices)
        hess_values = self._bkd.stack(
            [
                self._model.hessian(sample[:, None])[..., idx[0], idx[1]]
                for sample in samples.T
            ]
        )
        return self._stat.hessian(
            values, jac_values, hess_values, self._weights
        )

    def __repr__(self) -> str:
        return "{0}(model={1}, stat={2})".format(
            self.__class__.__name__, self._model, self._stat
        )

    def _label(self) -> str:
        return f"AVaRDev_{float(self._alpha):.2f}"
