import numpy as np
from scipy import special


class GenzFunction(object):
    def __init__(self):
        self._nvars = None
        self._c = None
        self._w = None
        self._min_c = 5e-6

        self._funs = {
            "oscillatory": (self._oscillatory, self._oscillatory_integrate),
            "product_peak": (self._product_peak, self._product_peak_integrate),
            "corner_peak": (self._corner_peak, self._corner_peak_integrate),
            "gaussian": (self._gaussian, self._gaussian_integrate),
            "c0continuous":
            (self._c0_continuous, self._c0_continuous_integrate),
            "discontinuous":
            (self._discontinuous, self._discontinuous_integrate)}

    @staticmethod
    def _get_c_coefficients(coef_type, nvars, min_c):
        ind = np.arange(nvars)[:, None]
        if (coef_type == "no_decay"):
            return (ind+0.5)/nvars
        if (coef_type == "quadratic_decay"):
            return 1.0 / (ind + 1.)**2
        if (coef_type == "quartic_decay"):
            return 1.0 / (ind + 1.)**4
        if (coef_type == "exponential_decay"):
            # smallest value will be 1e-8
            return np.exp((ind+1)*np.log(min_c)/nvars)
        if (coef_type == "squared_exponential_decay"):
            # smallest value will be 1e-8
            return 10**(np.log10(min_c)*((ind+1)/nvars)**2)
        msg = f"coef_type: {coef_type} not supported"
        raise ValueError(msg)

    def set_coefficients(self, nvars, c_factor, coef_type, w_factor=0.5,
                         seed=0):
        self._nvars = nvars
        self._w = np.full((self._nvars, 1), w_factor, dtype=np.double)
        self._c = self._get_c_coefficients(coef_type, self._nvars, self._min_c)
        self._c *= c_factor/self._c.sum()

    def _oscillatory(self, samples, jac):
        result = 2.0 * np.pi * self._w[0]
        tmp = samples.T.dot(self._c)
        result = np.cos(tmp)
        if not jac:
            return result
        grad = -self._c*np.sin(tmp)
        return grad

    def _product_peak(self, samples, jac):
        result = 1/np.prod(
            (1/self._c**2+(samples-self._w)**2), axis=0)[:, None]
        if not jac:
            return result
        grad = 2.*(samples.T - self._w)*result
        return result, grad

    def _corner_peak(self, samples, jac):
        tmp = 1+samples.T.dot(self._c)
        result = tmp**(-(self._nvars+1))
        if not jac:
            return result
        grad = -self._c*(self._nvars+1)/tmp**(self._nvars+2)
        return result, grad

    def _gaussian(self, samples, jac):
        tmp = -np.sum(self._c**2*(samples-self._w)**2, axis=0)
        result = np.exp(tmp)[:, None]
        if not jac:
            return result
        grad = 2.*self._c**2*(self._w-samples.T)*result
        return result, grad

    def _c0_continuous(self, samples, jac):
        tmp = -np.sum(self._c*np.abs(samples-self._w), axis=0)
        result = np.exp(tmp)[:, None]
        if not jac:
            return result
        msg = "grad of c0_continuous function is not supported"
        raise ValueError(msg)

    def _discontinuous(self, samples, jac):
        result = np.exp(samples.T.dot(self._c))
        II = np.where((samples[0] > self._w[0]) & (samples[1] > self._w[1]))
        result[II] = 0.0
        if not jac:
            return result
        msg = "grad of discontinuous function is not supported"
        raise ValueError(msg)

    def __call__(self, name, samples, jac=False):
        return self._funs[name][0](samples, jac)

    def _oscillatory_recursive_integrate(self, var_id, cosine):
        C1 = np.sin(self._c[var_id])/self._c[var_id]
        C2 = (1-np.cos(self._c[var_id]))/self._c[var_id]
        if var_id == self._nvars-1:
            if cosine:
                return C1
            return C2
        if cosine:
            return (C1*self._oscillatory_recursive_integrate(var_id+1, True) -
                    C2*self._oscillatory_recursive_integrate(var_id+1, False))
        return (C2*self._oscillatory_recursive_integrate(var_id+1, True) +
                C1*self._oscillatory_recursive_integrate(var_id+1, False))

    def _oscillatory_integrate(self):
        """
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
        C1 = np.cos(2.*np.pi*self._w[0])
        C2 = np.sin(2.*np.pi*self._w[0])
        integral = (C1*self._oscillatory_recursive_integrate(0, True) -
                    C2*self._oscillatory_recursive_integrate(0, False))
        return integral

    def _product_peak_integrate(self):
        return np.prod(
            self._c*(np.arctan(self._c*(1.0-self._w)) +
                     np.arctan(self._c*self._w)))

    def _corner_peak_integrate_recursive(self, integral, D):
        if D == 0:
            return 1.0 / (1.0 + integral)
        return 1/(D*self._c[D-1])*(
            self._corner_peak_integrate_recursive(integral, D-1) -
            self._corner_peak_integrate_recursive(integral+self._c[D-1], D-1))

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
        result = np.prod(
            (special.erf(self._c*self._w)+special.erf(self._c-self._c*self._w)) *
            np.sqrt(np.pi)/(2*self._c))
        return result

    def _c0_continuous_integrate(self):
        return np.prod(
            (2.0-np.exp(-self._c*self._w)-np.exp(self._c*(self._w-1.0)))/self._c)

    def _discontinuous_integrate(self):
        assert self._nvars >= 2
        idx = min(self._nvars, 2)
        tmp = np.prod(
            (np.exp(self._c[:idx]*self._w[:idx])-1)/self._c[:idx])
        if self._nvars <= 2:
            return tmp
        return tmp*np.prod((np.exp(self._c[2:])-1)/self._c[2:])

    def integrate(self, name):
        return self._funs[name][1]()

    def __reduce__(self):
        return (type(self), (self._nvars, self._c, self._w))
