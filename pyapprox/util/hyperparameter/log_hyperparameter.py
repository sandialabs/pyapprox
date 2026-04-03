from typing import Tuple, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter import HyperParameter


class LogHyperParameter(HyperParameter[Array]):
    """
    A hyperparameter that stores log-transformed values internally but allows
    users to interact with values in their original (exponential) form.

    Parameters
    ----------
    name : str
        Name of the hyperparameter.
    nparams : int
        Number of variables (parameters) in the hyperparameter.
    user_values : Union[float, Array]
        Values provided by the user in their original (exponential) form.
    user_bounds : Union[Tuple[float, float], Array]
        Bounds provided by the user in their original (exponential) form.
    bkd : Backend
        Backend for numerical computations.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).
    """

    def __init__(
        self,
        name: str,
        nparams: int,
        user_values: Union[float, Array],
        user_bounds: Union[Tuple[float, float], Array],
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        # Transform user values to log space
        self._bkd = bkd
        self._nparams = nparams
        log_values = bkd.log(super()._parse_values(user_values))
        log_bounds = bkd.log(super()._parse_bounds(user_bounds))

        # Initialize the base HyperParameter class with log-transformed values
        super().__init__(
            name=name,
            nparams=nparams,
            values=log_values,
            bounds=log_bounds,
            bkd=bkd,
            fixed=fixed,
        )

    def exp_values(self) -> Array:
        """
        Get the values in their original (exponential) form.

        Returns
        -------
        exp_values : Array
            Values in their original (exponential) form.
        """
        log_values = self.get_values()
        return self._bkd.exp(log_values)

    def exp_bounds(self) -> Union[Tuple[float, float], Array]:
        """
        Get the bounds in their original (exponential) form.

        Returns
        -------
        exp_bounds : Union[Tuple[float, float], Array]
            Bounds in their original (exponential) form.
        """
        log_bounds = self.get_bounds()
        if isinstance(log_bounds, tuple):
            return (self._bkd.exp(log_bounds[0]), self._bkd.exp(log_bounds[1]))
        else:
            return self._bkd.exp(log_bounds)

# TODO: add __repr__ that prints bounds in user space exp_values
# or refactor to make values always returned user units and have canonical
# units e.g. log values
