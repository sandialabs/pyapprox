import math
from typing import Tuple

from scipy import stats
from scipy import special

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.benchmarks.base import SingleModelBenchmark
from pyapprox.interface.model import Model
from pyapprox.variables.joint import IndependentMarginalsVariable


class GenzModel(Model):
    """
    Genz test function model.

    This class implements the six Genz test functions, which are widely used
    for benchmarking multidimensional integration routines. The functions are
    parameterized by coefficients and weights, allowing control over their
    anisotropy and difficulty.

    Parameters
    ----------
    name : str
        Name of the Genz test function. Must be one of:
        'oscillatory', 'product_peak', 'corner_peak', 'gaussian',
        'c0continuous', 'discontinuous'.
    backend : BackendMixin
        Backend for numerical computations.

        Compute the integral of the Genz test function.
    """

    def __init__(self, name: str, backend: BackendMixin):
        """
        Initialize the Genz test function model.

        Parameters
        ----------
        name : str
            Name of the Genz test function. Must be one of:
            'oscillatory', 'product_peak', 'corner_peak', 'gaussian',
            'c0continuous', 'discontinuous'.
        backend : BackendMixin
            Backend for numerical computations.
        """
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
        """
        Set the name of the Genz test function.

        Parameters
        ----------
        name : str
            Name of the Genz test function.
        """
        self._name = name

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented for the selected function.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return self._name not in ["c0continuous", "discontinuous"]

    def nvars(self) -> int:
        """
        Return the number of uncertain variables.

        Returns
        -------
        nvars : int
            Number of uncertain variables.
        """
        return self._nvars

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI. For this model, it is always 1.
        """
        return 1

    def _get_c_coefficients(self, decay: str, nvars: int) -> Array:
        r"""
        Compute the coefficients :math:`c_d` for the Genz test function.

        The coefficients control the anisotropy of the function and are computed
        based on the specified decay type.

        Parameters
        ----------
        decay : str
            Type of decay for the coefficients. Must be one of:
            'none', 'quadratic', 'quartic', 'exp', 'sqexp'.
        nvars : int
            Number of uncertain variables.

        Returns
        -------
        _get_c_coefficients : Array
            Array of shape (nvars, 1) containing the computed coefficients.

        Raises
        ------
        ValueError
            If the specified decay type is not supported.

        Notes
        -----
        The decay types are defined as follows:

        - No decay ('none'):
          .. math:: \hat{c}_d = \frac{d + 0.5}{D}

        - Quadratic decay ('quadratic'):
          .. math:: \hat{c}_d = \frac{1}{(d + 1)^2}

        - Quartic decay ('quartic'):
          .. math:: \hat{c}_d = \frac{1}{(d + 1)^4}

        - Exponential decay ('exp'):
          .. math:: \hat{c}_d = \exp\left(\log(c_\mathrm{min})\frac{d + 1}{D}\right)

        - Squared-exponential decay ('sqexp'):
          .. math:: \hat{c}_d = 10^{\left(\log_{10}(c_\mathrm{min})\frac{(d + 1)^2}{D^2}\right)}

        Here :math:`c_\mathrm{min}` is the minimum value of :math:`c_D`, set by
        `self._min_c`.
        """
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

    def _set_coefficients(self, nvars: int, c: Array, w: Array):
        """
        Set the coefficients and weights for the Genz test function.

        This method validates and assigns the coefficients and weights, which
        control the anisotropy and location of the function's features.

        Parameters
        ----------
        nvars : int
            Number of uncertain variables.
        c : Array
            Array of shape (nvars, 1) containing the coefficients.
        w : Array
            Array of shape (nvars, 1) containing the weights.

        Raises
        ------
        ValueError
            If the shape of `c` or `w` is invalid.

        Notes
        -----
        The coefficients :math:`c_d` control the relative importance of each
        variable, while the weights :math:`w_d` control the location of the
        function's features. Both must have the same number of variables
        (`nvars`) and be column vectors.
        """
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
        """
        Set the coefficients and weights for the Genz test function.

        Parameters
        ----------
        nvars : int
            Number of uncertain variables.
        cfactor : float
            Scaling factor for the coefficients.
        decay : str
            Type of decay for the coefficients. Must be one of:
            'none', 'quadratic', 'quartic', 'exp', 'sqexp'.
        wfactor : float, optional
            Scaling factor for the weights. Default is 0.5.
        """
        w = self._bkd.full((nvars, 1), wfactor)
        c = self._get_c_coefficients(decay, nvars)
        c *= cfactor / c.sum()
        self._set_coefficients(nvars, c, w)

    def integrate(self):
        """
        Compute the integral of the Genz test function.

        Returns
        -------
        integral : float
            The integral of the Genz test function.
        """
        return self._funs[self._name][1]()

    def _oscillatory(self, samples: Array, return_grad: bool) -> Array:
        """
        Evaluate the oscillatory Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.
        grad : Array, optional
            Gradient of the function (if `return_grad` is True).
        """
        tmp = 2.0 * math.pi * self._w[0] + samples.T @ self._c
        result = self._bkd.cos(tmp)
        if not return_grad:
            return result
        grad = -self._c * self._bkd.sin(tmp)
        return result, grad.T

    def _product_peak(self, samples: Array, return_grad: bool) -> Array:
        """
        Evaluate the product peak Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.
        grad : Array, optional
            Gradient of the function (if `return_grad` is True).
        """
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
        """
        Evaluate the corner peak Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.
        grad : Array, optional
            Gradient of the function (if `return_grad` is True).
        """
        tmp = 1 + samples.T @ self._c
        result = tmp ** (-(self._nvars + 1))
        if not return_grad:
            return result
        grad = -self._c * (self._nvars + 1) / tmp ** (self._nvars + 2)
        return result, grad.T

    def _gaussian(self, samples: Array, return_grad: bool) -> Array:
        """
        Evaluate the Gaussian peak Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.
        grad : Array, optional
            Gradient of the function (if `return_grad` is True).
        """
        tmp = -self._bkd.sum(self._c**2 * (samples - self._w) ** 2, axis=0)
        result = self._bkd.exp(tmp)[:, None]
        if not return_grad:
            return result
        grad = 2.0 * self._c.T**2 * (self._w.T - samples.T) * result
        return result, grad

    def _c0_continuous(self, samples: Array, return_grad: bool) -> Array:
        """
        Evaluate the C0 continuous Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.

        Raises
        ------
        ValueError
            If `return_grad` is True, as the gradient is not supported.
        """
        tmp = -self._bkd.sum(
            self._c * self._bkd.abs(samples - self._w), axis=0
        )
        result = self._bkd.exp(tmp)[:, None]
        if not return_grad:
            return result
        msg = "grad of c0_continuous function is not supported"
        raise ValueError(msg)

    def _discontinuous(self, samples: Array, return_grad: bool) -> Array:
        """
        Evaluate the discontinuous Genz function.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.
        return_grad : bool
            If True, also return the gradient.

        Returns
        -------
        result : Array
            Function evaluations.

        Raises
        ------
        ValueError
            If `return_grad` is True, as the gradient is not supported.
        """
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
        """
        Evaluate the Genz test function for given samples.

        Parameters
        ----------
        samples : Array
            Array of shape (nvars, nsamples) containing the input samples.

        Returns
        -------
        vals : Array
            Array of shape (nsamples, 1) containing the function evaluations.
        """
        return self._funs[self._name][0](samples, False)

    def _jacobian(self, sample: Array) -> Array:
        """
        Compute the Jacobian of the Genz test function at a given sample.

        Parameters
        ----------
        sample : Array
            Array of shape (nvars, 1) containing the input sample.

        Returns
        -------
        jac : Array
            Array of shape (1, nvars) containing the Jacobian matrix.
        """
        return self._funs[self._name][0](sample, True)[1]

    def _oscillatory_recursive_integrate_alternate(self, var_id, integral):
        """
        Recursively compute the integral of the oscillatory Genz function using
        an alternate approach.

        Parameters
        ----------
        var_id : int
            Index of the variable to integrate.
        integral : float
            Current value of the integral.

        Returns
        -------
        integral : float
            Result of the recursive integration.
        """
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
        """
        Compute the integral of the oscillatory Genz function using an alternate
        recursive approach.

        Returns
        -------
        integral : float
            Result of the integral.
        """
        return self._oscillatory_recursive_integrate(self._nvars, 0.0)

    def _oscillatory_recursive_integrate(self, var_id, cosine):
        """
        Recursively compute the integral of the oscillatory Genz function.

        Parameters
        ----------
        var_id : int
            Index of the variable to integrate.
        cosine : bool
            Whether to use the cosine term in the integration.

        Returns
        -------
        integral : float
            Result of the recursive integration.
        """
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
        Compute the integral of the oscillatory Genz function using a better
        conditioned approach.

        Returns
        -------
        integral : float
            Result of the integral.
        """
        C1 = self._bkd.cos(2.0 * math.pi * self._w[0])
        C2 = self._bkd.sin(2.0 * math.pi * self._w[0])
        integral = C1 * self._oscillatory_recursive_integrate(
            0, True
        ) - C2 * self._oscillatory_recursive_integrate(0, False)
        return integral

    def _product_peak_integrate(self):
        """
        Compute the integral of the product peak Genz function.

        Returns
        -------
        integral : float
            Result of the integral.
        """
        return self._bkd.prod(
            self._c
            * (
                self._bkd.arctan(self._c * (1.0 - self._w))
                + self._bkd.arctan(self._c * self._w)
            )
        )

    def _corner_peak_integrate_recursive(self, integral, D):
        """
        Recursively compute the integral of the corner peak Genz function.

        Parameters
        ----------
        integral : float
            Current value of the integral.
        D : int
            Dimension of the variable to integrate.

        Returns
        -------
        integral : float
            Result of the recursive integration.
        """
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
        Compute the integral of the corner peak Genz function.

        Returns
        -------
        integral : float
            Result of the integral.

        Raises
        ------
        ValueError
            If the coefficients are too small for accurate computation.
        """
        if self._c.prod() < 1e-14:
            msg = "coefficients too small for corner_peak integral to be "
            msg += "computed accurately with recursion. Increase self._min_c."
            raise ValueError(msg)
        return self._corner_peak_integrate_recursive(0.0, self._nvars)

    def _gaussian_integrate(self):
        """
        Compute the integral of the Gaussian peak Genz function.

        Returns
        -------
        integral : float
            Result of the integral.
        """
        result = self._bkd.prod(
            (
                self._bkd.asarray(
                    special.erf(self._bkd.to_numpy(self._c * self._w))
                    + special.erf(
                        (self._bkd.to_numpy(self._c - self._c * self._w))
                    )
                )
            )
            * math.sqrt(math.pi)
            / (2 * self._c)
        )
        return result

    def _c0_continuous_integrate(self):
        """
        Compute the integral of the C0 continuous Genz function.

        Returns
        -------
        integral : float
            Result of the integral.
        """
        return self._bkd.prod(
            (
                2.0
                - self._bkd.exp(-self._c * self._w)
                - self._bkd.exp(self._c * (self._w - 1.0))
            )
            / self._c
        )

    def _discontinuous_integrate(self):
        """
        Compute the integral of the discontinuous Genz function.

        Returns
        -------
        integral : float
            Result of the integral.
        """
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


class GenzBenchmark(SingleModelBenchmark):
    r"""
    Genz integration benchmark.

    This class sets up one of the six Genz integration benchmarks:
    'oscillatory', 'product_peak', 'corner_peak', 'gaussian',
    'c0continuous', 'discontinuous'. The number of inputs and the anisotropy
    of the functions can be adjusted.

    The inputs to these functions are independent uniform random variables on
    [0,1].

    Parameters
    ----------
    name : str
        Name of the Genz test function.
    nvars : int
        Number of uncertain variables.
    decay : str, optional
        Type of decay for the coefficients. Must be one of:
        'none', 'quadratic', 'quartic', 'exp', 'sqexp'. Default is 'none'.
    cfactor : float, optional
        Scaling factor for the coefficients. Default is 1.0.
    wfactor : float, optional
        Scaling factor for the weights. Default is 0.25.
    coefs : Tuple[Array, Array], optional
        Predefined coefficients and weights. Default is None.
    backend : BackendMixin
        Backend for numerical computations.

    Notes
    -----
    The six Genz test functions are:

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

    Quadratic decay (quadratic)

    .. math:: \hat{c}_d = \frac{1}{(d + 1)^2}

    Quartic decay (quartic)

    .. math:: \hat{c}_d = \frac{1}{(d + 1)^4}

    Exponential decay (exp)

    .. math:: \hat{c}_d=\exp\left(\log(c_\mathrm{min})\frac{d+1}{D}\right)

    Squared-exponential decay (sqexp)

    .. math:: \hat{c}_d=10^{\left(\log_{10}(c_\mathrm{min})\frac{(d+1)^2}{D^2}\right)}

    Here :math:`c_\mathrm{min}` is an argument that sets the minimum value of :math:`c_D`.

    Once the formulas are used, the coefficients are normalized such that:

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
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the Genz benchmark.

        Parameters
        ----------
        name : str
            Name of the Genz test function.
        nvars : int
            Number of uncertain variables.
        decay : str, optional
            Type of decay for the coefficients. Must be one of:
            'none', 'quadratic', 'quartic', 'exp', 'sqexp'. Default is 'none'.
        cfactor : float, optional
            Scaling factor for the coefficients. Default is 1.0.
        wfactor : float, optional
            Scaling factor for the weights. Default is 0.25.
        coefs : Tuple[Array, Array], optional
            Predefined coefficients and weights. Default is None.
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._name = name
        self._nvars = nvars
        self._decay = decay
        self._cfactor = cfactor
        self._wfactor = wfactor
        self._coefs = coefs
        super().__init__(backend)

    def _set_model(self):
        """
        Set the Genz test function model.

        Returns
        -------
        None
        """
        self._model = GenzModel(self._name, self._bkd)
        if self._coefs is None:
            self._model.set_coefficients(
                self.nvars(), self._cfactor, self._decay, self._wfactor
            )
        else:
            self._model._set_coefficients(self._nvars, *self._coefs)

    def _set_prior(self):
        """
        Define the input variable for the Genz test function.
        """
        marginals = [stats.uniform(0, 1)] * self._nvars
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def integral(self):
        """
        Compute the integral of the Genz test function.

        Returns
        -------
        integral : float
            The integral of the Genz test function.
        """
        return self._model.integrate()
