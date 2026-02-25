"""Polynomial model ensemble for multifidelity benchmarks.

Implements a hierarchy of polynomial models: x^5, x^4, x^3, x^2, x
with decreasing fidelity and cost.
"""

from typing import Generic, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class PolynomialModelFunction(Generic[Array]):
    """Single polynomial model f(x) = x^degree.

    Implements FunctionProtocol.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    degree : int
        Polynomial degree.
    """

    def __init__(self, bkd: Backend[Array], degree: int) -> None:
        self._bkd = bkd
        self._degree = degree

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def degree(self) -> int:
        """Return polynomial degree."""
        return self._degree

    def __call__(self, samples: Array) -> Array:
        """Evaluate the polynomial model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (1, nsamples).
        """
        return samples**self._degree

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (1, 1).

        Returns
        -------
        Array
            Jacobian of shape (1, 1).
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian expects single sample with shape (1, 1), "
                f"got shape {sample.shape}"
            )
        x = sample[0, 0]
        return self._bkd.array([[self._degree * x ** (self._degree - 1)]])

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (1, 1).
        vec : Array
            Direction vector of shape (1, 1).

        Returns
        -------
        Array
            HVP result of shape (1, 1).
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"hvp expects single sample with shape (1, 1), "
                f"got shape {sample.shape}"
            )
        if vec.shape[1] != 1:
            raise ValueError(
                f"hvp expects direction vector with shape (1, 1), "
                f"got shape {vec.shape}"
            )
        x = sample[0, 0]
        v = vec[0, 0]
        d = self._degree
        if d <= 1:
            hess_val = 0.0
        else:
            hess_val = d * (d - 1) * x ** (d - 2)
        return self._bkd.array([[hess_val * v]])

    def mean(self) -> float:
        """Analytical mean for U[0,1] input.

        Returns
        -------
        float
            Mean value E[x^degree] = 1/(degree+1).
        """
        return 1.0 / (self._degree + 1)

    def variance(self) -> float:
        """Analytical variance for U[0,1] input.

        Returns
        -------
        float
            Variance Var[x^degree] = 1/(2*degree+1) - 1/(degree+1)^2.
        """
        d = self._degree
        return 1.0 / (2 * d + 1) - 1.0 / (d + 1) ** 2


class PolynomialEnsemble(Generic[Array]):
    """Ensemble of polynomial models for multifidelity testing.

    Models: f_k(x) = x^(nmodels - k) for k = 0, ..., nmodels-1
    where k=0 is highest fidelity (highest degree).

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    nmodels : int, optional
        Number of models in ensemble. Default is 5.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        nmodels: int = 5,
    ) -> None:
        self._bkd = bkd
        self._nmodels = nmodels
        # Create models: degrees 5, 4, 3, 2, 1 for nmodels=5
        self._models = [
            PolynomialModelFunction(bkd, degree=nmodels - k)
            for k in range(nmodels)
        ]

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nmodels(self) -> int:
        """Return number of models in ensemble."""
        return self._nmodels

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of QoI per model."""
        return 1

    def models(self) -> Sequence[PolynomialModelFunction[Array]]:
        """Return the list of models."""
        return self._models

    def __getitem__(self, idx: int) -> PolynomialModelFunction[Array]:
        """Get model by index."""
        return self._models[idx]

    def costs(self) -> Array:
        """Return costs of each model.

        Costs decrease logarithmically: 1, 0.1, 0.01, ...

        Returns
        -------
        Array
            Costs of shape (nmodels,).
        """
        return self._bkd.logspace(0, -self._nmodels + 1, self._nmodels)

    def means(self) -> Array:
        """Return analytical means of all models.

        Returns
        -------
        Array
            Means of shape (nmodels,).
        """
        return self._bkd.array([m.mean() for m in self._models])

    def variances(self) -> Array:
        """Return analytical variances of all models.

        Returns
        -------
        Array
            Variances of shape (nmodels,).
        """
        return self._bkd.array([m.variance() for m in self._models])

    def covariance(self, i: int, j: int) -> float:
        """Compute analytical covariance between models i and j.

        Cov[x^d1, x^d2] = 1/(d1+d2+1) - 1/(d1+1)/(d2+1)

        Parameters
        ----------
        i : int
            Index of first model.
        j : int
            Index of second model.

        Returns
        -------
        float
            Covariance between models.
        """
        d1 = self._models[i].degree()
        d2 = self._models[j].degree()
        return 1.0 / (d1 + d2 + 1) - 1.0 / (d1 + 1) / (d2 + 1)

    def covariance_matrix(self) -> Array:
        """Return full covariance matrix between all models.

        Returns
        -------
        Array
            Covariance matrix of shape (nmodels, nmodels).
        """
        n = self._nmodels
        cov = self._bkd.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov[i, j] = self.covariance(i, j)
        return cov

    def correlation(self, i: int, j: int) -> float:
        """Compute correlation between models i and j.

        Parameters
        ----------
        i : int
            Index of first model.
        j : int
            Index of second model.

        Returns
        -------
        float
            Correlation coefficient.
        """
        cov = self.covariance(i, j)
        var_i = self._models[i].variance()
        var_j = self._models[j].variance()
        return cov / (var_i * var_j) ** 0.5

    def correlation_matrix(self) -> Array:
        """Return correlation matrix between all models.

        Returns
        -------
        Array
            Correlation matrix of shape (nmodels, nmodels).
        """
        n = self._nmodels
        corr = self._bkd.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr[i, j] = self.correlation(i, j)
        return corr
