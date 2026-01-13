"""Sobol G-function for sensitivity analysis benchmarks.

The Sobol G-function is a standard benchmark for global sensitivity analysis,
with analytically known Sobol indices that depend on the importance parameters.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

from typing import Generic, Sequence

from pyapprox.typing.util.backends.protocols import Array, Backend


class SobolGFunction(Generic[Array]):
    """Sobol G-function: f(x) = prod_{i=1}^{d} g_i(x_i).

    where g_i(x_i) = (|4*x_i - 2| + a_i) / (1 + a_i)

    The parameters a_i control the importance of each variable:
    - a_i = 0: Most important (first-order index = 1)
    - Larger a_i: Less important

    Standard domain: [0, 1]^d

    This function implements FunctionWithJacobianAndHVPProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    a : Sequence[float]
        Importance parameters for each variable. Length determines nvars.

    References
    ----------
    Sobol, I.M. (1993). "Sensitivity analysis for non-linear mathematical
    models."
    """

    def __init__(
        self,
        bkd: Backend[Array],
        a: Sequence[float],
    ) -> None:
        if len(a) < 1:
            raise ValueError("Must have at least one variable")
        self._bkd = bkd
        self._a = list(a)
        self._nvars = len(a)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def a(self) -> Sequence[float]:
        """Return the importance parameters."""
        return self._a

    def _g(self, xi: Array, ai: float) -> Array:
        """Compute g_i(x_i) = (|4*x_i - 2| + a_i) / (1 + a_i)."""
        bkd = self._bkd
        return (bkd.abs(4 * xi - 2) + ai) / (1 + ai)

    def _dg(self, xi: Array, ai: float) -> Array:
        """Compute derivative of g_i w.r.t. x_i.

        dg_i/dx_i = 4 * sign(4*x_i - 2) / (1 + a_i)
        """
        bkd = self._bkd
        return 4 * bkd.sign(4 * xi - 2) / (1 + ai)

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at multiple samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values of shape (1, nsamples).
        """
        bkd = self._bkd
        result = bkd.ones((1, samples.shape[1]))
        for i in range(self._nvars):
            xi = samples[i:i+1, :]
            result = result * self._g(xi, self._a[i])
        return result

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix of shape (1, nvars).

        Raises
        ------
        ValueError
            If sample is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )

        bkd = self._bkd

        # Compute all g_i values
        g_values = []
        for i in range(self._nvars):
            xi = sample[i, 0]
            g_values.append(self._g(xi, self._a[i]))

        # Product of all g_i
        total_product = bkd.prod(bkd.stack(g_values))

        # Jacobian: df/dx_i = (product of all g_j for j != i) * dg_i/dx_i
        #                   = f / g_i * dg_i/dx_i
        grad_components = []
        for i in range(self._nvars):
            xi = sample[i, 0]
            gi = g_values[i]
            dgi = self._dg(xi, self._a[i])
            # Avoid division by zero when gi is very small
            grad_i = total_product / gi * dgi if gi != 0 else dgi
            grad_components.append(grad_i)

        return bkd.reshape(bkd.stack(grad_components), (1, self._nvars))

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        For the Sobol G function, the Hessian is:
        H[i,i] = (f / g_i^2) * (dg_i)^2 * 0 = 0  (second derivative of g_i is 0)
        H[i,j] = (f / (g_i * g_j)) * dg_i * dg_j  for i != j

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            Hessian-vector product of shape (nvars, 1).

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (nvars, 1).
        """
        if sample.ndim != 2 or sample.shape != (self._nvars, 1):
            raise ValueError(
                f"sample must have shape ({self._nvars}, 1), got {sample.shape}"
            )
        if vec.ndim != 2 or vec.shape != (self._nvars, 1):
            raise ValueError(
                f"vec must have shape ({self._nvars}, 1), got {vec.shape}"
            )

        bkd = self._bkd
        n = self._nvars

        # Compute all g_i and dg_i values
        g_values = []
        dg_values = []
        for i in range(n):
            xi = sample[i, 0]
            g_values.append(self._g(xi, self._a[i]))
            dg_values.append(self._dg(xi, self._a[i]))

        # Product of all g_i
        total_product = bkd.prod(bkd.stack(g_values))

        # Compute H @ v
        # H[i,j] = f / (g_i * g_j) * dg_i * dg_j  for i != j
        # H[i,i] = 0 (since d^2 g_i / dx_i^2 = 0 for |4x-2|, except at x=0.5)
        result_components = []
        for i in range(n):
            hvp_i = bkd.asarray([0.0])
            for j in range(n):
                if i != j:
                    # H[i,j] * v[j]
                    gi = g_values[i]
                    gj = g_values[j]
                    dgi = dg_values[i]
                    dgj = dg_values[j]
                    vj = vec[j, 0]
                    if gi != 0 and gj != 0:
                        h_ij = total_product / (gi * gj) * dgi * dgj
                    else:
                        h_ij = bkd.asarray([0.0])
                    hvp_i = hvp_i + h_ij * vj
            result_components.append(hvp_i)

        return bkd.reshape(bkd.stack(result_components), (n, 1))


def sobol_g_indices(
    a: Sequence[float],
    bkd: Backend[Array],
) -> tuple[Array, Array, float]:
    """Compute analytical Sobol indices for Sobol G function.

    Parameters
    ----------
    a : Sequence[float]
        Importance parameters.
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    tuple[Array, Array, float]
        (main_effects, total_effects, variance) where main_effects and
        total_effects have shape (nvars, 1).
    """
    a_arr = bkd.asarray(a)
    n = len(a)

    # Variance of g_i: Var(g_i) = 1 / (3 * (1 + a_i)^2)
    var_gi = 1.0 / (3.0 * (1 + a_arr) ** 2)

    # Total variance: prod(1 + Var(g_i)) - 1
    total_var = float(bkd.prod(1 + var_gi) - 1)

    # First-order indices: S_i = Var(g_i) / total_var
    main_effects = bkd.reshape(var_gi / total_var, (n, 1))

    # Total indices: ST_i = 1 - prod_{j != i}(1 + Var(g_j)) / (1 + total_var)
    total_list = []
    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        other_product = bkd.prod(bkd.asarray([1 + var_gi[j] for j in other_indices]))
        total_list.append(1 - (other_product - 1) / total_var)
    total_effects = bkd.reshape(bkd.asarray(total_list), (n, 1))

    return main_effects, total_effects, total_var
