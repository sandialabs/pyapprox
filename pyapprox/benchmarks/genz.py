import math
from typing import Tuple

from scipy import stats
from scipy import special

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.benchmarks.base import SingleModelBenchmark
from pyapprox.interface.model import Model
from pyapprox.variables.joint import IndependentMarginalsVariable


class GenzModel(Model):
    def __init__(self, name: str, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self.set_name(name)
        self._min_c = 5e-6
        self._funs = {
            "oscillatory": (self._oscillatory, self._oscillatory_integrate),
            "product_peak": (self._product_peak, self._product_peak_integrate),
            "corner_peak": (self._corner_peak, self._corner_peak_integrate),
            "gaussian": (self._gaussian, self._gaussian_integrate),
            "c0continuous": (
                self._c0_continuous,
                self._c0_continuous_integrate,
            ),
            "discontinuous": (
                self._discontinuous,
                self._discontinuous_integrate,
            ),
        }

    def set_name(self, name: str):
        self._name = name

    def jacobian_implemented(self) -> bool:
        return self._name not in [
            "c0continuous",
            "discontinuous",
        ]

    def _get_c_coefficients(self, decay: str, nvars: int):
        ind = self._bkd.arange(nvars, dtype=self._bkd.double_type())[:, None]
        if decay == "none":
            return (ind + 0.5) / nvars
        if decay == "quadratic":
            return 1.0 / (ind + 1.0) ** 2
        if decay == "quartic":
            return 1.0 / (ind + 1.0) ** 4
        if decay == "exp":
            # smallest value will be self._min_c
            return self._bkd.exp((ind + 1) * math.log(self._min_c) / nvars)
        if decay == "sqexp":
            # smallest value will be self._min_c
            return 10 ** (math.log10(self._min_c) * ((ind + 1) / nvars) ** 2)
        msg = f"decay: {decay} not supported"
        raise ValueError(msg)

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return 1

    def _set_coefficients(self, nvars, c, w):
        if c.shape != (nvars, 1):
            raise ValueError("c has the wrong shape")
        if not (w.ndim == 2 and w.shape[1] == 1):
            raise ValueError("w has the wrong shape")
        self._nvars = nvars
        self._w = w
        self._c = c

    def set_coefficients(
        self, nvars: int, cfactor: float, decay: str, wfactor: float = 0.5
    ):
        w = self._bkd.full((nvars, 1), wfactor)
        c = self._get_c_coefficients(decay, nvars)
        c *= cfactor / c.sum()
        self._set_coefficients(nvars, c, w)

    def _oscillatory(self, samples: Array, return_grad: bool) -> Array:
        tmp = 2.0 * math.pi * self._w[0] + samples.T @ self._c
        result = self._bkd.cos(tmp)
        if not return_grad:
            return result
        grad = -self._c * self._bkd.sin(tmp)
        return result, grad.T

    def _product_peak(self, samples: Array, return_grad: bool) -> Array:
        result = (
            1
            / self._bkd.prod(
                (1 / self._c**2 + (samples - self._w) ** 2), axis=0
            )[:, None]
        )
        if not return_grad:
            return result
        grad = (
            -2.0
            * (samples - self._w)
            / (1 / self._c**2 + (samples - self._w) ** 2)
            * result[0, 0]
        )
        return result, grad.T

    def _corner_peak(self, samples: Array, return_grad: bool) -> Array:
        tmp = 1 + samples.T @ self._c
        result = tmp ** (-(self._nvars + 1))
        if not return_grad:
            return result
        grad = -self._c * (self._nvars + 1) / tmp ** (self._nvars + 2)
        return result, grad.T

    def _gaussian(self, samples: Array, return_grad: bool) -> Array:
        tmp = -self._bkd.sum(self._c**2 * (samples - self._w) ** 2, axis=0)
        result = self._bkd.exp(tmp)[:, None]
        if not return_grad:
            return result
        grad = 2.0 * self._c.T**2 * (self._w.T - samples.T) * result
        return result, grad

    def _c0_continuous(self, samples: Array, return_grad: bool) -> Array:
        tmp = -self._bkd.sum(
            self._c * self._bkd.abs(samples - self._w), axis=0
        )
        result = self._bkd.exp(tmp)[:, None]
        if not return_grad:
            return result
        msg = "grad of c0_continuous function is not supported"
        raise ValueError(msg)

    def _discontinuous(self, samples: Array, return_grad: bool) -> Array:
        result = self._bkd.exp(samples.T @ self._c)
        II = self._bkd.where(
            (samples[0] > self._w[0]) | (samples[1] > self._w[1])
        )
        result[II] = 0.0
        if not return_grad:
            return result
        msg = "grad of discontinuous function is not supported"
        raise ValueError(msg)

    def _values(self, samples: Array) -> Array:
        return self._funs[self._name][0](samples, False)

    def _jacobian(self, sample: Array) -> Array:
        return self._funs[self._name][0](sample, True)[1]

    def _oscillatory_recursive_integrate_alternate(self, var_id, integral):
        if var_id > 0:
            return (
                self._oscillatory_recursive_integrate(
                    var_id - 1, integral + self._c[var_id - 1]
                )
                - self._oscillatory_recursive_integrate(var_id - 1, integral)
            ) / self._c[var_id - 1]
        case = self._nvars % 4
        if case == 0:
            return self._bkd.cos(2.0 * math.pi * self._w[0] + integral)
        if case == 1:
            return self._bkd.sin(2.0 * math.pi * self._w[0] + integral)
        if case == 2:
            return -self._bkd.cos(2.0 * math.pi * self._w[0] + integral)
        return -self._bkd.sin(2.0 * math.pi * self._w[0] + integral)

    def _oscillatory_integrate_alternate(self):
        return self._oscillatory_recursive_integrate(self._nvars, 0.0)

    def _oscillatory_recursive_integrate(self, var_id, cosine):
        C1 = self._bkd.sin(self._c[var_id]) / self._c[var_id]
        C2 = (1 - self._bkd.cos(self._c[var_id])) / self._c[var_id]
        if var_id == self._nvars - 1:
            if cosine:
                return C1
            return C2
        if cosine:
            return C1 * self._oscillatory_recursive_integrate(
                var_id + 1, True
            ) - C2 * self._oscillatory_recursive_integrate(var_id + 1, False)
        return C2 * self._oscillatory_recursive_integrate(
            var_id + 1, True
        ) + C1 * self._oscillatory_recursive_integrate(var_id + 1, False)

    def _oscillatory_integrate(self):
        """
        This is Better conditioned than alternate implementation
        use
        cos(x+y)=cos(x)cos(y)-sin(x)sin(y)
        sin(x+y)=sin(x)cos(y)+cos(x)sin(y)
        and if y=w+z
        sin(x+w+z)=sin(x)sin(w+z)+cos(x)cos(w+z)
        so expand sin(w+z)
        then exploit separability
        and
        int_0^1 cos(ax)dx = sin(a)/a
        int_0^1 sin(ax)dx = (1-cos(a))/a
        """
        C1 = self._bkd.cos(2.0 * math.pi * self._w[0])
        C2 = self._bkd.sin(2.0 * math.pi * self._w[0])
        integral = C1 * self._oscillatory_recursive_integrate(
            0, True
        ) - C2 * self._oscillatory_recursive_integrate(0, False)
        return integral

    def _product_peak_integrate(self):
        return self._bkd.prod(
            self._c
            * (
                self._bkd.arctan(self._c * (1.0 - self._w))
                + self._bkd.arctan(self._c * self._w)
            )
        )

    def _corner_peak_integrate_recursive(self, integral, D):
        if D == 0:
            return 1.0 / (1.0 + integral)
        return (
            1
            / (D * self._c[D - 1])
            * (
                self._corner_peak_integrate_recursive(integral, D - 1)
                - self._corner_peak_integrate_recursive(
                    integral + self._c[D - 1], D - 1
                )
            )
        )

    def _corner_peak_integrate(self):
        r"""
        int_0^1 ((c+ax)^{-d-1}dx = c^{-d}-(a+c)^{-d})/(ad)
        let c = b*y
        int_0^1 \int_0^1 ((by+ax)^{-d-1}dxdy =
           1/(ad) \int_0^1 ((by)^{-d}-(a+by)^{-d})dy
        """
        if self._c.prod() < 1e-14:
            msg = "coefficients to small for corner_peak integral to be "
            msg += " computedaccurately with recursion. increase self._min_c"
            raise ValueError(msg)
        return self._corner_peak_integrate_recursive(0.0, self._nvars)

    def _gaussian_integrate(self):
        result = self._bkd.prod(
            (
                special.erf(self._c * self._w)
                + special.erf(self._c - self._c * self._w)
            )
            * math.sqrt(math.pi)
            / (2 * self._c)
        )
        return result

    def _c0_continuous_integrate(self):
        return self._bkd.prod(
            (
                2.0
                - self._bkd.exp(-self._c * self._w)
                - self._bkd.exp(self._c * (self._w - 1.0))
            )
            / self._c
        )

    def _discontinuous_integrate(self):
        assert self._nvars >= 2
        idx = min(self._nvars, 2)
        tmp = self._bkd.prod(
            (self._bkd.exp(self._c[:idx] * self._w[:idx]) - 1.0)
            / self._c[:idx]
        )
        if self._nvars <= 2:
            return tmp
        return tmp * self._bkd.prod(
            (self._bkd.exp(self._c[2:]) - 1) / self._c[2:]
        )

    def integrate(self):
        return self._funs[self._name][1]()


class GenzBenchmark(SingleModelBenchmark):
    r"""
    Setup one of the six Genz integration benchmarks
    :math:`f_d(x):\mathbb{R}^D\to\mathbb{R}`,
    where :math:`x=[x_1,\ldots,x_D]^\top`.
    The number of inputs :math:`D` and the anisotropy (relative importance of
    each variable and interactions) of the functions can be adjusted.
    The definition of each function is in the Notes section.

    References
    ----------
    .. [Genz1984] `Genz, A. Testing multidimensional integration routines. In Proc. of international conference on Tools, methods and languages for scientific and engineering computation (pp. 81-94), 1984 <https://dl.acm.org/doi/10.5555/2837.2842>`_

    Notes
    -----
    The six Genz test function are:

    Oscillatory ('oscillatory')

    .. math:: f(z) = \cos\left(2\pi w_1 + \sum_{d=1}^D c_dz_d\right)

    Product Peak ('product_peak')

    .. math:: f(z) = \prod_{d=1}^D \left(c_d^{-2}+(z_d-w_d)^2\right)^{-1}

    Corner Peak ('corner_peak')

    .. math:: f(z)=\left( 1+\sum_{d=1}^D c_dz_d\right)^{-(D+1)}

    Gaussian Peak ('gaussian')

    .. math:: f(z) = \exp\left( -\sum_{d=1}^D c_d^2(z_d-w_d)^2\right)

    C0 Continuous ('c0continuous')

    .. math:: f(z) = \exp\left( -\sum_{d=1}^D c_d\lvert z_d-w_d\rvert\right)

    Discontinuous ('discontinuous')

    .. math:: f(z) = \begin{cases}0 & z_1>w_1 \;\mathrm{or}\; z_2>w_2\\\exp\left(\sum_{d=1}^D c_d z_d\right) & \mathrm{otherwise}\end{cases}

    Increasing :math:`\lVert c \rVert` will in general make
    the integrands more difficult.

    The :math:`0\le w_d \le 1` parameters do not affect the difficulty
    of the integration problem. We set :math:`w_1=w_2=\ldots=W_D`.

    The coefficient types implement different decay rates for :math:`c_d`.
    This allows testing of methods that can identify and exploit anisotropy.
    They are as follows:

    No decay (none)

    .. math:: \hat{c}_d=\frac{d+0.5}{D}

    Quadratic decay (qudratic)

    .. math:: \hat{c}_d = \frac{1}{(d + 1)^2}

    Quartic decay (quartic)

    .. math:: \hat{c}_d = \frac{1}{(d + 1)^4}

    Exponential decay (exp)

    .. math:: \hat{c}_d=\exp\left(\log(c_\mathrm{min})\frac{d+1}{D}\right)

    Squared-exponential decay (sqexp)

    .. math:: \hat{c}_d=10^{\left(\log_{10}(c_\mathrm{min})\frac{(d+1)^2}{D^2}\right)}

    Here :math:`c_\mathrm{min}` is argument that sets the minimum value of :math:`c_D`.

    Once the formula are used the coefficients are normalized such that

    .. math:: c_d = c_\text{factor}\frac{\hat{c}_d}{\sum_{d=1}^D \hat{c}_d}.
    """

    def __init__(
        self,
        name: str,
        nvars: int,
        decay: str = "none",
        cfactor: float = 1.0,
        wfactor: float = 0.25,
        coefs: Tuple[Array, Array] = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._name = name
        self._nvars = nvars
        self._decay = decay
        self._cfactor = cfactor
        self._wfactor = wfactor
        self._coefs = coefs
        super().__init__(backend)

    def _set_model(self):
        self._model = GenzModel(self._name, self._bkd)
        if self._coefs is None:
            self._model.set_coefficients(
                self.nvars(), self._cfactor, self._decay, self._wfactor
            )
        else:
            self._model._set_coefficients(self._nvars, *self._coefs)

    def _set_variable(self):
        marginals = [stats.uniform(0, 1)] * self._nvars
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def integral(self):
        return self._model.integrate()
