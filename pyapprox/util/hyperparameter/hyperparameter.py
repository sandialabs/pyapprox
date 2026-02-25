from typing import Generic, Tuple, Union, cast

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class HyperParameter(Generic[Array]):
    """
    A possibly vector-valued hyperparameter to be used with optimization.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.
    nparams : int
        Number of variables (parameters) in the hyperparameter.
    values : Union[float, Array]
        Initial values of the hyperparameter (can be a scalar or an array).
    bounds : Union[Tuple[float, float], Array]
        Bounds for the hyperparameter values.
    bkd : Backend
        Backend for numerical computations.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).
    """

    def __init__(
        self,
        name: str,
        nparams: int,
        values: Union[float, Array],
        bounds: Union[Tuple[float, float], Array],
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        validate_backend(bkd)
        self._bkd = bkd

        self._name = name
        self._nparams = nparams

        # Parse and validate values
        self._values = self._parse_values(values)
        self._bounds = self._parse_bounds(bounds)

        if fixed:
            self.set_all_inactive()
        else:
            self.set_all_active()

    def _parse_values(self, values: Union[float, Array]) -> Array:
        """
        Parse and validate the input values.

        Parameters
        ----------
        values : Union[float, Array]
            Input values (can be a scalar or an array).

        Returns
        -------
        parsed_values : Array
            Parsed values as a 1D array of correct shape.

        Raises
        ------
        ValueError
            If the input values are not valid (e.g., incorrect shape).
        """
        parsed_values = self._bkd.atleast_1d(self._bkd.asarray(values))
        if parsed_values.shape[0] == 1:
            parsed_values = self._bkd.tile(parsed_values, (self.nparams(),))
        if parsed_values.ndim != 1:
            raise ValueError("values must be a 1D array")
        if parsed_values.shape[0] != self.nparams():
            raise ValueError(
                "values shape {0} inconsistent with nparams {1}".format(
                    parsed_values.shape, self.nparams()
                )
            )
        return parsed_values

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
        self.set_active_indices(self._bkd.arange(self.nparams(), dtype=int))

    def _parse_bounds(
        self, bounds: Union[Tuple[float, float], Array]
    ) -> Array:
        """
        Parse and validate the bounds for the hyperparameter values.

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Array]
            Bounds for the hyperparameter values.

        Raises
        ------
        ValueError
            If the bounds shape is inconsistent with the number of parameters.
        """
        if isinstance(bounds, tuple):
            _bounds = self._bkd.tile(
                self.bkd().asarray(bounds), (self.nparams(), 1)
            )
        else:
            _bounds = bounds
        if _bounds.shape != (self.nparams(), 2):
            msg = "bounds shape {0} must be {1}".format(
                _bounds.shape, (self.nparams(), 2)
            )
            raise ValueError(msg)
        return _bounds

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

    def get_values(self) -> Array:
        """
        Get the values of the parameters.

        Returns
        -------
        values : Array
            Values of the parameters.
        """
        return self._values

    def set_values(self, values: Array) -> None:
        """
        Set all parameter values (both active and inactive).

        Parameters
        ----------
        values : Array
            Values for all parameters. Shape: (nparams,)

        Raises
        ------
        ValueError
            If values shape doesn't match nparams.
        """
        if values.ndim != 1:
            raise ValueError("values must be a 1D array")
        if values.shape[0] != self.nparams():
            raise ValueError(
                f"values shape {values.shape} inconsistent"
                f" with nparams {self.nparams()}"
            )
        self._values = values

    def get_active_values(self) -> Array:
        """
        Get the values of the active parameters.

        Returns
        -------
        active_values : Array
            Values of the active parameters.
        """
        return self._values[self._active_indices]

    def set_active_values(self, active_values: Array) -> None:
        """
        Set the values of the active parameters.

        Parameters
        ----------
        active_values : Array
            Values of the active parameters.

        Raises
        ------
        ValueError
            If the active_values shape is inconsistent with
            the number of active parameters.
        """
        if active_values.ndim != 1:
            raise ValueError("active_values must be a 1D array")
        if active_values.shape[0] != self.nactive_params():
            raise ValueError(
                "active_values shape {0} inconsistent with nactive_params {1}".format(
                    active_values.shape, self.nactive_params()
                )
            )
        self._values[self._active_indices] = active_values

    def get_bounds(self) -> Array:
        """
        Get the bounds of the parameters.

        Returns
        -------
        bounds : Array
            Bounds of the parameters.
        """
        return self._bounds

    def set_bounds(self, bounds: "Union[Tuple[float, float], Array]") -> None:
        """
        Set the bounds for the hyperparameter values.

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Array]
            Bounds for the hyperparameter values. Either a tuple
            (lower, upper) applied to all parameters, or an array
            of shape (nparams, 2).
        """
        self._bounds = self._parse_bounds(bounds)

    def get_active_bounds(self) -> Array:
        """
        Get the bounds of the active parameters.

        Returns
        -------
        active_bounds : Array
            Bounds of the active parameters.
        """
        return self._bounds[self._active_indices, :]

    def __str__(self) -> str:
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
            return "{0}(name={1}, nparams={2}, nactive={3})".format(
                self.__class__.__name__,
                self._name,
                self.nparams(),
                self.nactive_params(),
            )
        return "{0}(name={1}, values={2}, active={3})".format(
            self.__class__.__name__,
            self._name,
            "[" + ", ".join(map("{0:.2g}".format, self.get_values())) + "]",
            "[" + ", ".join(map("{0}".format, self._active_indices)) + "]",
        )
