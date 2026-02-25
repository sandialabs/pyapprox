from typing import Union, Tuple, List, Optional

from pyapprox.util.backends.protocols import (
    Array,
    Backend,
    ArrayProtocol,
)
from pyapprox.util.hyperparameter.hyperparameter import HyperParameter
from pyapprox.util.backends.validation import validate_backends


class HyperParameterList:
    """
    A list of hyperparameters to be used with optimization.

    Parameters
    ----------
    hyperparam_list : list
        List of HyperParameter objects.
    bkd : Backend[Array], optional
        Backend to use if hyperparam_list is empty. If not provided and
        hyperparam_list is empty, raises ValueError.
    """

    def __init__(
        self,
        hyperparam_list: List[HyperParameter],
        bkd: Optional[Backend[Array]] = None
    ):
        """Initialize the HyperParameterList."""
        self._validate_hyperparameters(hyperparam_list, bkd)
        self._hyperparam_list = hyperparam_list
        if hyperparam_list:
            self._bkd = self._hyperparam_list[0].bkd()
        elif bkd is not None:
            self._bkd = bkd
        else:
            raise ValueError(
                "Backend must be provided for empty hyperparameter list"
            )

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
        self,
        hyperparam_list: List[HyperParameter],
        bkd: Optional[Backend[Array]]
    ) -> None:
        """
        Validate the list of hyperparameters.

        Ensures that all hyperparameters use the same backend and are valid.
        """
        if not hyperparam_list and bkd is None:
            raise ValueError(
                "Backend must be provided for empty hyperparameter list"
            )

        # Ensure all hyperparameters use the same backend (skip if empty)
        if hyperparam_list:
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

    def get_active_indices(self) -> ArrayProtocol:
        """
        Get the active indices of the hyperparameters.

        Returns
        -------
        active_indices : Array
            Active indices of the hyperparameters.
        """
        if not self._hyperparam_list:
            return self._bkd.array([])

        cnt = 0
        active_indices = []
        for hyp in self._hyperparam_list:
            active_indices.append(hyp.get_active_indices() + cnt)
            cnt += hyp.nparams()
        return self._bkd.hstack(active_indices)

    def set_all_active(self) -> None:
        """
        Set all hyperparameters to active.
        """
        for hyp in self._hyperparam_list:
            hyp.set_all_active()

    def set_all_inactive(self) -> None:
        """
        Set all hyperparameters to inactive.
        """
        for hyp in self._hyperparam_list:
            hyp.set_all_inactive()

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

    def get_values(self) -> ArrayProtocol:
        """
        Get the values of the parameters in the user space.

        Returns
        -------
        values : Array
            Values of the parameters in the user space.
        """
        if not self._hyperparam_list:
            return self._bkd.array([])

        return self._bkd.hstack(
            [hyp.get_values() for hyp in self._hyperparam_list]
        )

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

    def get_bounds(self) -> ArrayProtocol:
        """
        Get the flattened bounds of the parameters in the user space.

        Returns
        -------
        bounds : Array
            Flattened bounds of the parameters in the user space.
        """
        if not self._hyperparam_list:
            return self._bkd.array([]).reshape((0, 2))

        return self._bkd.vstack(
            [hyp.get_bounds() for hyp in self._hyperparam_list]
        )

    def get_active_values(self) -> ArrayProtocol:
        """
        Get the values of the active parameters in the optimization space.

        Returns
        -------
        active_values : Array
            Values of the active parameters in the optimization space.
        """
        if not self._hyperparam_list:
            return self._bkd.array([])

        return self._bkd.hstack(
            [hyp.get_active_values() for hyp in self._hyperparam_list]
        )

    def get_active_bounds(self) -> ArrayProtocol:
        """
        Get the bounds of the active parameters.

        Returns
        -------
        active_bounds : Array
            Bounds of the active parameters, shape (nactive, 2).
        """
        if not self._hyperparam_list:
            return self._bkd.array([]).reshape((0, 2))

        return self._bkd.vstack(
            [hyp.get_active_bounds() for hyp in self._hyperparam_list]
        )

    def extract_active(self, full_array: Array) -> ArrayProtocol:
        """
        Extract elements corresponding to active parameters from a full array.

        This is useful for extracting active gradient elements from a gradient
        computed over all parameters. Works with 1D arrays (gradients) and
        2D arrays with parameters in the last dimension.

        Parameters
        ----------
        full_array : Array
            Array with values for all parameters.
            - 1D: shape (nparams,)
            - 2D: shape (..., nparams)

        Returns
        -------
        active_array : Array
            Array with values for active parameters only.
            - 1D: shape (nactive,)
            - 2D: shape (..., nactive)

        Examples
        --------
        >>> # Extract active gradient elements
        >>> full_grad = kernel.jacobian_wrt_params(X)  # shape (n, n, nparams)
        >>> active_grad = hyp_list.extract_active(full_grad)  # shape (n, n, nactive)
        """
        active_indices = self.get_active_indices()
        if full_array.ndim == 1:
            return full_array[active_indices]
        else:
            # For multi-dimensional arrays, extract along last axis
            return full_array[..., active_indices]

    def expand_to_full(
        self, active_array: Array, fill_value: float = 0.0
    ) -> ArrayProtocol:
        """
        Expand an active-parameter array to full parameter space.

        Fixed parameters are filled with the specified fill value (default 0).
        This is useful for expanding a direction vector for HVP computation.

        Parameters
        ----------
        active_array : Array
            Array with values for active parameters only, shape (nactive,).
        fill_value : float, optional
            Value to use for fixed parameters (default 0.0).

        Returns
        -------
        full_array : Array
            Array with values for all parameters, shape (nparams,).

        Examples
        --------
        >>> # Expand active direction for HVP
        >>> active_dir = bkd.ones(hyp_list.nactive_params())
        >>> full_dir = hyp_list.expand_to_full(active_dir)  # zeros for fixed
        """
        full_array = self._bkd.full((self.nparams(),), fill_value)
        active_indices = self.get_active_indices()
        # Use scatter-like update
        for i, idx in enumerate(active_indices):
            full_array[idx] = active_array[i]
        return full_array

    def set_active_values(self, active_params: Array) -> None:
        """
        Set the values of the active parameters in the optimization space.

        Parameters
        ----------
        active_params : Array
            Values of the active parameters in the optimization space.
        """
        cnt = 0
        for hyp in self._hyperparam_list:
            hyp.set_active_values(
                active_params[cnt : cnt + hyp.nactive_params()]
            )
            cnt += hyp.nactive_params()

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

    def __radd__(self, hyp_list: "HyperParameterList") -> "HyperParameterList":
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
        return self.__class__(
            hyp_list.hyperparameters() + self._hyperparam_list
        )

    def __str__(self) -> str:
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
                [str(hyp) for hyp in self._hyperparam_list],
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
