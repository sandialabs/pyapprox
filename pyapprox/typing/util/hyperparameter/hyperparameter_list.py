from typing import Union, Tuple, List

from pyapprox.typing.util.backends.protocols import (
    Array,
    Backend,
    ArrayProtocol,
)
from pyapprox.typing.util.hyperparameter.hyperparameter import HyperParameter
from pyapprox.typing.util.validate_backend import validate_backends


class HyperParameterList:
    """
    A list of hyperparameters to be used with optimization.

    Parameters
    ----------
    hyperparam_list : list
        List of HyperParameter objects.
    """

    def __init__(self, hyperparam_list: List[HyperParameter]):
        """Initialize the HyperParameterList."""
        self._validate_hyperparameters(hyperparam_list)
        self._hyperparam_list = hyperparam_list
        self._bkd = self._hyperparam_list[0]._bkd

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
        Backend for numerical computations.
        """
        return self._bkd

    def _validate_hyperparameters(
        self, hyperparam_list: List[HyperParameter]
    ) -> None:
        """
        Validate the list of hyperparameters.

        Ensures that all hyperparameters use the same backend and are valid.
        """
        if not hyperparam_list:
            raise ValueError("The hyperparameter list cannot be empty.")

        # Ensure all hyperparameters use the same backend
        validate_backends([hyperparam.bkd() for hyperparam in hyperparam_list])

    def hyperparameters(self) -> List[HyperParameter]:
        """
        Return the list of hyperparameters.

        Returns
        -------
        hyperparam_list : list
            List of HyperParameter objects.
        """
        return self._hyperparam_list

    def set_active_opt_params(self, active_params: Array) -> None:
        """
        Set the values of the active parameters in the optimization space.

        Parameters
        ----------
        active_params : Array
            Values of the active parameters in the optimization space.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            hyp.set_active_opt_params(
                active_params[cnt : cnt + hyp.nactive_params()]
            )
            cnt += hyp.nactive_params()

    def nparams(self) -> int:
        """
        Return the total number of hyperparameters (active and inactive).

        Returns
        -------
        nparams : int
            Total number of hyperparameters.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            cnt += hyp.nparams()
        return cnt

    def nactive_params(self) -> int:
        """
        Return the number of active (to be optimized) hyperparameters.

        Returns
        -------
        nactive_params : int
            Number of active hyperparameters.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            cnt += hyp.nactive_params()
        return cnt

    def get_active_opt_params(self) -> Array:
        """
        Get the values of the active parameters in the optimization space.

        Returns
        -------
        active_opt_params : Array
            Values of the active parameters in the optimization space.
        """
        return self._bkd.hstack(
            [hyp.get_active_opt_params() for hyp in self._hyperparam_list]
        )

    def get_active_opt_bounds(self) -> Array:
        """
        Get the bounds of the active parameters in the optimization space.

        Returns
        -------
        active_opt_bounds : Array
            Bounds of the active parameters in the optimization space.
        """
        return self._bkd.vstack(
            [hyp.get_active_opt_bounds() for hyp in self._hyperparam_list]
        )

    def get_bounds(self) -> Array:
        """
        Get the flattened bounds of the parameters in the user space.

        Returns
        -------
        bounds : Array
            Flattened bounds of the parameters in the user space.
        """
        return self._bkd.hstack(
            [hyp.get_bounds() for hyp in self._hyperparam_list]
        )

    def get_values(self) -> Array:
        """
        Get the values of the parameters in the user space.

        Returns
        -------
        values : Array
            Values of the parameters in the user space.
        """
        return self._bkd.hstack(
            [hyp.get_values() for hyp in self._hyperparam_list]
        )

    def get_active_indices(self) -> ArrayProtocol:
        """
        Get the active indices of the hyperparameters.

        Returns
        -------
        active_indices : Array
            Active indices of the hyperparameters.
        """
        cnt = 0
        active_indices = []
        for hyp in self._hyperparam_list:
            active_indices.append(hyp.get_active_indices() + cnt)
            cnt += hyp.nparams()
        return self._bkd.hstack(active_indices)

    def set_active_indices(self, active_indices: Array) -> None:
        """
        Set the active indices for the hyperparameters.

        Parameters
        ----------
        active_indices : Array
            Active indices to set.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            hyp_indices = self._bkd.nonzero(
                (active_indices >= cnt)
                & (active_indices < cnt + hyp.nparams())
            )[0]
            hyp.set_active_indices(active_indices[hyp_indices] - cnt)
            cnt += hyp.nparams()

    def set_all_inactive(self) -> None:
        """
        Set all hyperparameters to inactive.
        """
        for hyp in self._hyperparam_list:
            hyp.set_all_inactive()

    def set_all_active(self) -> None:
        """
        Set all hyperparameters to active.
        """
        for hyp in self._hyperparam_list:
            hyp.set_all_active()

    def set_values(self, values: Array) -> None:
        """
        Set the values of the hyperparameters in the user space.

        Parameters
        ----------
        values : Array
            Values to set.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            hyp.set_values(values[cnt : cnt + hyp.nparams()])
            cnt += hyp.nparams()

    def __add__(self, hyp_list: "HyperParameterList") -> "HyperParameterList":
        """
        Add two HyperParameterLists.

        Parameters
        ----------
        hyp_list : HyperParameterList
            Another HyperParameterList to add.

        Returns
        -------
        result : HyperParameterList
            Combined HyperParameterList.
        """
        return self.__class__(
            self._hyperparam_list + hyp_list.hyperparameters()
        )

    def __radd__(
        self, hyp_list: Union[int, "HyperParameterList"]
    ) -> "HyperParameterList":
        """
        Add two HyperParameterLists (reverse addition).

        Parameters
        ----------
        hyp_list : Union[int, HyperParameterList]
            Another HyperParameterList to add.

        Returns
        -------
        result : HyperParameterList
            Combined HyperParameterList.
        """
        if hyp_list == 0:
            return self
        return self.__class__(
            hyp_list.hyperparameters() + self._hyperparam_list
        )

    def _short_repr(self) -> str:
        """
        Return a short string representation of the HyperParameterList.

        Returns
        -------
        repr : str
            Short string representation.
        """
        return ", ".join(
            map(
                "{0}".format,
                [hyp._short_repr() for hyp in self._hyperparam_list],
            )
        )

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the HyperParameterList.

        Returns
        -------
        repr : str
            Detailed string representation.
        """
        return (
            "{0}(".format(self.__class__.__name__)
            + ",\n\t\t   ".join(map("{0}".format, self._hyperparam_list))
            + ")"
        )

    def set_bounds(self, bounds: Union[Tuple[float, float], Array]) -> None:
        """
        Set the bounds for the hyperparameters.

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Array]
            Bounds to set.
        """
        _bounds = self._bkd.atleast1d(bounds)
        if _bounds.shape[0] == 2:
            _bounds = self._bkd.tile(_bounds, (self.nparams(),))
        if _bounds.shape[0] != 2 * self.nparams():
            msg = "_bounds shape {0} inconsistent with 2*nparams={1}".format(
                _bounds.shape, 2 * self.nparams()
            )
            raise ValueError(msg)
        cnt = 0
        for hyp in self._hyperparam_list:
            hyp.set_bounds(_bounds[cnt : cnt + 2 * hyp.nparams()])
            cnt += 2 * hyp.nparams()
