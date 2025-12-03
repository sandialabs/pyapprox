from typing import Union, Tuple, Generic, Optional, cast

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter.transforms import (
    IdentityHyperParameterTransform,
    HyperParameterTransformProtocol,
)
from pyapprox.typing.util.validate_backend import validate_backends


class HyperParameter(Generic[Array]):
    """
    A possibly vector-valued hyperparameter to be used with optimization.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.
    nparams : int
        Number of variables (parameters) in the hyperparameter.
    values : Array
        Initial values of the hyperparameter.
    bounds : Union[Tuple[float, float], Array]
        Bounds for the hyperparameter values.
    transform : HyperParameterTransform, optional
        Transformation for the hyperparameter values (default is identity).
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).
    backend : Backend
        Backend for numerical computations.
    """

    def __init__(
        self,
        name: str,
        nparams: int,
        values: Array,
        bounds: Union[Tuple[float, float], Array],
        bkd: Backend[Array],
        transform: Optional[HyperParameterTransformProtocol[Array]] = None,
        fixed: bool = False,
    ):
        self._bkd = bkd

        if transform is None:
            _transform: HyperParameterTransformProtocol[Array] = (
                IdentityHyperParameterTransform(self._bkd)
            )
        else:
            _transform = transform
        if not isinstance(_transform, HyperParameterTransformProtocol):
            raise TypeError(
                "The provided 'transform' object must implement the "
                "HyperParameterTransformProtocol. Expected an object "
                "conforming to the protocol, but got an object of "
                f"type {type(transform).__name__}."
            )
        validate_backends([_transform.bkd(), bkd])
        self._transform = _transform

        self._name = name
        self._nparams = nparams

        self._values = self._bkd.atleast_1d(self._bkd.asarray(values))
        if self._values.shape[0] == 1:
            self._values = self._bkd.tile(self._values, (self.nparams(),))
        if self._values.ndim == 2:
            raise ValueError("values is not a 1D array")
        if self._values.shape[0] != self.nparams():
            raise ValueError(
                "values shape {0} inconsistent with nparams {1}".format(
                    self._values.shape, self.nparams()
                )
            )
        self.set_bounds(bounds)
        if fixed:
            self.set_all_inactive()
        else:
            self.set_all_active()

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
        Backend for numerical computations.
        """
        return self._bkd

    def set_active_indices(self, indices: Array) -> None:
        """
        Set the active indices for the hyperparameter.

        Parameters
        ----------
        indices : Array
            Indices of the active hyperparameters.
        """
        if indices.shape[0] == 0:
            self._active_indices = indices
            return

        if max(indices) >= self.nparams():
            raise ValueError("indices exceed nparams")
        if min(indices) < 0:
            raise ValueError("Ensure indices >= 0")
        self._active_indices = indices

    def get_active_indices(self) -> Array:
        """
        Get the active indices for the hyperparameter.

        Returns
        -------
        active_indices : Array
            Indices of the active hyperparameters.
        """
        return self._active_indices

    def set_all_inactive(self) -> None:
        """
        Set all hyperparameters to inactive.
        """
        self.set_active_indices(self._bkd.zeros((0,), dtype=int))

    def set_all_active(self) -> None:
        """
        Set all hyperparameters to active.
        """
        frozen_indices = self._bkd.isnan(self._bounds[:, 0])
        self.set_active_indices(
            self._bkd.arange(self.nparams(), dtype=int)[~frozen_indices]
        )

    def set_bounds(self, bounds: Union[Tuple[float, float], Array]) -> None:
        """
        Set the bounds for the hyperparameter values.

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Array]
            Bounds for the hyperparameter values.
        """
        self._bounds = self._bkd.atleast_1d(self._bkd.asarray(bounds))
        if self._bounds.shape[0] == 2:
            self._bounds = self._bkd.tile(self._bounds, (self.nparams(),))
        if self._bounds.shape[0] != 2 * self.nparams():
            msg = "bounds shape {0} inconsistent with 2*nparams={1}".format(
                self._bounds.shape, 2 * self.nparams()
            )
            raise ValueError(msg)
        self._bounds = self._bkd.reshape(
            self._bounds, (self._bounds.shape[0] // 2, 2)
        )

    def nparams(self) -> int:
        """
        Return the number of hyperparameters.

        Returns
        -------
        nparams : int
            Number of hyperparameters.
        """
        return self._nparams

    def nactive_params(self) -> int:
        """
        Return the number of active hyperparameters.

        Returns
        -------
        nactive_params : int
            Number of active hyperparameters.
        """
        return cast(int, self._active_indices.shape[0])

    def set_active_opt_params(self, active_params: Array) -> None:
        """
        Set the values of the active parameters in the optimization space.

        Parameters
        ----------
        active_params : Array
            Values of the active parameters in the optimization space.
        """
        if active_params.ndim != 1:
            raise ValueError("active_params must be a 1D array")
        # self._values = self._bkd.copy(self._bkd.detach(self._values))
        self._values[self._active_indices] = self._transform.from_opt_space(
            active_params
        )

    def get_active_opt_params(self) -> Array:
        """
        Get the values of the active parameters in the optimization space.

        Returns
        -------
        active_opt_params : Array
            Values of the active parameters in the optimization space.
        """
        return self._transform.to_opt_space(self._values[self._active_indices])

    def get_active_opt_bounds(self) -> Array:
        """
        Get the bounds of the active parameters in the optimization space.

        Returns
        -------
        active_opt_bounds : Array
            Bounds of the active parameters in the optimization space.
        """
        return self._transform.to_opt_space(
            self._bounds[self._active_indices, :]
        )

    def get_bounds(self) -> Array:
        """
        Get the bounds of the parameters in the user space.

        Returns
        -------
        bounds : Array
            Bounds of the parameters in the user space.
        """
        return self._bkd.flatten(self._bounds)

    def get_values(self) -> Array:
        """
        Get the values of the parameters in the user space.

        Returns
        -------
        values : Array
            Values of the parameters in the user space.
        """
        return self._values

    def set_values(self, values: Array) -> None:
        """
        Set the values of the parameters in the user space.

        Parameters
        ----------
        values : Array
            Values of the parameters in the user space.
        """
        if values.ndim != 1:
            raise ValueError("values must be 1D")
        self._values = values

    def _short_repr(self) -> str:
        """
        Return a short string representation of the hyperparameter.

        Returns
        -------
        repr : str
            Short string representation of the hyperparameter.
        """
        if self.nparams() > 5:
            return "{0}:nparams={1}".format(self._name, self.nparams())

        return "{0}={1}".format(
            self._name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
        )

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the hyperparameter.

        Returns
        -------
        repr : str
            Detailed string representation of the hyperparameter.
        """
        if self.nparams() > 5:
            return "{0}(name={1}, nparams={2}, transform={3}, nactive={4})".format(
                self.__class__.__name__,
                self._name,
                self.nparams(),
                self._transform,
                self.nactive_params(),
            )
        return "{0}(name={1}, values={2}, transform={3}, active={4})".format(
            self.__class__.__name__,
            self._name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
            self._transform,
            "[" + ", ".join(map("{0}".format, self._active_indices)) + "]",
        )

    # def detach(self) -> None:
    #     """
    #     Detach the hyperparameter values from the computational graph if in use.
    #     """
    #     self.set_values(self._bkd.detach(self.get_values()))
