"""Sobol G-function for sensitivity analysis benchmarks.

The Sobol G-function is a standard benchmark for global sensitivity analysis,
with analytically known Sobol indices that depend on the importance parameters.

Implements FunctionWithJacobianAndHVPProtocol directly (no inheritance).
"""

from typing import Generic, List, Sequence

from pyapprox.util.backends.protocols import Array, Backend


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
            xi = samples[i : i + 1, :]
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
            raise ValueError(f"vec must have shape ({self._nvars}, 1), got {vec.shape}")

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


class SobolGSensitivityIndices(Generic[Array]):
    """Analytical sensitivity indices for the Sobol G-function.

    Computes exact Sobol indices for the Sobol G-function:
    f(x) = prod_{i=1}^{d} g_i(x_i)
    where g_i(x_i) = (|4*x_i - 2| + a_i) / (1 + a_i)

    with x uniformly distributed on [0, 1]^d.

    This class computes main effects, total effects, and all interaction
    indices up to second order (pairwise interactions).

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
        self._compute_indices()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._nvars

    def a(self) -> Sequence[float]:
        """Return the importance parameters."""
        return self._a

    def _compute_indices(self) -> None:
        """Compute and cache all sensitivity indices."""
        bkd = self._bkd
        n = self._nvars
        a_arr = bkd.asarray(self._a)

        # Mean: E[f] = 1 (since E[g_i] = 1 for all i)
        self._mean = bkd.asarray([1.0])

        # Variance of each g_i: Var(g_i) = 1 / (3 * (1 + a_i)^2)
        var_gi = 1.0 / (3.0 * (1 + a_arr) ** 2)

        # Total variance: prod(1 + Var(g_i)) - 1
        self._variance = bkd.asarray([bkd.prod(1 + var_gi) - 1])
        var_total = float(self._variance[0])

        # Main effects (first-order Sobol indices): S_i = Var(g_i) / Var(f)
        self._main_effects = bkd.reshape(var_gi / var_total, (n, 1))

        # Total effects: ST_i = Var(g_i) * prod_{j!=i}(1 + Var(g_j)) / Var(f)
        total_product = bkd.prod(1 + var_gi)
        self._total_effects = bkd.reshape(
            var_gi * total_product / ((1 + var_gi) * var_total), (n, 1)
        )

        # Build interaction indices: main effects + pairwise interactions
        self._interaction_indices: List[List[int]] = []

        # Main effects
        for ii in range(n):
            self._interaction_indices.append([ii])

        # Pairwise interactions
        for ii in range(n):
            for jj in range(ii + 1, n):
                self._interaction_indices.append([ii, jj])

        # Compute all Sobol indices (main + pairwise interactions)
        sobol_list = []
        for idx in self._interaction_indices:
            # S_{idx} = prod_{i in idx} Var(g_i) / Var(f)
            prod_var = bkd.prod(bkd.asarray([var_gi[i] for i in idx]))
            sobol_list.append(prod_var / var_total)

        self._sobol_indices = bkd.reshape(bkd.asarray(sobol_list), (-1, 1))

    def mean(self) -> Array:
        """Return the mean of the Sobol G-function.

        Returns
        -------
        Array
            Mean value of shape (1,).
        """
        return self._mean

    def variance(self) -> Array:
        """Return the variance of the Sobol G-function.

        Returns
        -------
        Array
            Variance of shape (1,).
        """
        return self._variance

    def main_effects(self) -> Array:
        """Return the main effects (first-order Sobol indices).

        Returns
        -------
        Array
            Main effects of shape (nvars, 1).
        """
        return self._main_effects

    def total_effects(self) -> Array:
        """Return the total effects (total Sobol indices).

        Returns
        -------
        Array
            Total effects of shape (nvars, 1).
        """
        return self._total_effects

    def sobol_indices(self) -> Array:
        """Return all Sobol indices (main effects and pairwise interactions).

        Returns
        -------
        Array
            Sobol indices of shape (nindices, 1) where
            nindices = nvars + nvars*(nvars-1)/2 (main + pairwise).
            Order: S_1, ..., S_n, S_12, S_13, ..., S_{n-1,n}.
        """
        return self._sobol_indices

    def sobol_interaction_indices(self) -> Array:
        """Return the interaction indices as a binary matrix.

        Returns
        -------
        Array
            Binary matrix of shape (nvars, nindices) where entry (i, j)
            is 1 if variable i is involved in interaction j.
        """
        bkd = self._bkd
        nvars = self._nvars
        # Build as list of columns, then stack
        columns = []
        for compressed_index in self._interaction_indices:
            col = [1.0 if i in compressed_index else 0.0 for i in range(nvars)]
            columns.append(bkd.asarray(col))
        return bkd.stack(columns, axis=1)
