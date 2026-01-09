"""Adapters for wrapping legacy benchmark models.

These adapters convert legacy pyapprox.benchmarks models to work with
the new typing module's protocol-based design without modifying the
original legacy code.
"""

from typing import Any, Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class LegacyFunctionAdapter(Generic[Array]):
    """Adapt a legacy Model to FunctionProtocol.

    Wraps legacy models that use BackendMixin to work with the new
    Backend[Array] protocol. Conversion to/from numpy is used as an
    intermediate step for compatibility.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    legacy_model : Any
        Legacy Model instance (from pyapprox.benchmarks).
    nvars : int
        Number of input variables.
    nqoi : int
        Number of quantities of interest.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        legacy_model: Any,
        nvars: int,
        nqoi: int,
    ) -> None:
        self._bkd = bkd
        self._legacy = legacy_model
        self._nvars = nvars
        self._nqoi = nqoi

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._nvars

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function at samples.

        Parameters
        ----------
        samples : Array
            Input samples of shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values of shape (nqoi, nsamples).
        """
        # Convert to numpy for legacy model
        np_samples = self._bkd.to_numpy(samples)
        # Call legacy model
        np_result = self._legacy(np_samples)
        # Legacy returns (nsamples, nqoi), we need (nqoi, nsamples)
        np_result = np_result.T
        # Convert back to backend array
        return self._bkd.asarray(np_result)


class LegacyFunctionWithJacobianAdapter(LegacyFunctionAdapter[Array]):
    """Adapt a legacy Model with Jacobian to FunctionWithJacobianAndHVPProtocol.

    Extends LegacyFunctionAdapter to also wrap Jacobian and Hessian methods.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    legacy_model : Any
        Legacy Model instance with jacobian and optionally hessian methods.
    nvars : int
        Number of input variables.
    nqoi : int
        Number of quantities of interest.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        legacy_model: Any,
        nvars: int,
        nqoi: int,
    ) -> None:
        super().__init__(bkd, legacy_model, nvars, nqoi)
        # Check what's available
        self._has_jacobian = (
            hasattr(legacy_model, "jacobian_implemented")
            and legacy_model.jacobian_implemented()
        )
        self._has_hessian = (
            hasattr(legacy_model, "hessian_implemented")
            and legacy_model.hessian_implemented()
        )

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian of shape (nqoi, nvars).

        Raises
        ------
        ValueError
            If sample has more than one column or Jacobian not implemented.
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"jacobian expects single sample with shape (nvars, 1), "
                f"got shape {sample.shape}"
            )
        if not self._has_jacobian:
            raise ValueError("Jacobian not implemented in legacy model")

        # Convert to numpy
        np_sample = self._bkd.to_numpy(sample)
        # Call legacy jacobian - returns (1, nvars) for nqoi=1
        np_jac = self._legacy.jacobian(np_sample)
        # Convert back
        return self._bkd.asarray(np_jac)

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample of shape (nvars, 1).
        vec : Array
            Direction vector of shape (nvars, 1).

        Returns
        -------
        Array
            HVP result of shape (nvars, 1).

        Raises
        ------
        ValueError
            If inputs have wrong shape, nqoi > 1, or Hessian not implemented.
        """
        if sample.shape[1] != 1:
            raise ValueError(
                f"hvp expects single sample with shape (nvars, 1), "
                f"got shape {sample.shape}"
            )
        if vec.shape[1] != 1:
            raise ValueError(
                f"hvp expects direction vector with shape (nvars, 1), "
                f"got shape {vec.shape}"
            )
        if self._nqoi != 1:
            raise ValueError(
                f"hvp only supported for nqoi=1, got nqoi={self._nqoi}"
            )
        if not self._has_hessian:
            raise ValueError("Hessian not implemented in legacy model")

        # Convert to numpy
        np_sample = self._bkd.to_numpy(sample)
        np_vec = self._bkd.to_numpy(vec)

        # Get legacy Hessian - returns (1, nvars, nvars) for nqoi=1
        np_hess = self._legacy.hessian(np_sample)
        # Remove batch dimension: (nvars, nvars)
        np_hess = np_hess[0]
        # Compute HVP: (nvars, nvars) @ (nvars, 1) = (nvars, 1)
        np_hvp = np_hess @ np_vec

        # Convert back
        return self._bkd.asarray(np_hvp)
