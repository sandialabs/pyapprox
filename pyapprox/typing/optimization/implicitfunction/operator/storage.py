from typing import Generic

from pyapprox.typing.util.backend import Array, Backend


class AdjointOperatorStorage(Generic[Array]):
    """
    Storage for adjoint operator data.

    This class manages the storage and retrieval of adjoint operator data,
    including state and QoI Jacobians, forward states, and adjoint states.
    """

    def __init__(self, backend: Backend):
        """
        Initialize the AdjointOperatorStorage object.

        Parameters
        ----------
        backend : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        self._bkd = backend
        self._attribute_names = [
            "_drdy",  # State equation state Jacobian
            "_drdp",  # State equation parameter Jacobian
            "_dqdy",  # QoI state Jacobian
            "_dqdp",  # QoI parameter Jacobian
            "_fwd_state",  # Forward state
            "_adj_state",  # Adjoint state
        ]

    def set_parameter(self, param: Array) -> None:
        """
        Set the parameters for the adjoint operator.

        Parameters
        ----------
        param : Array
            Parameters to set.
        """
        self._param = param
        self._clear()

    def _clear(self) -> None:
        """
        Clear all stored attributes.
        """
        for attr_name in self._attribute_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

    def has_parameter(self, param: Array) -> bool:
        """
        Check if the given parameters match the stored parameters.

        Parameters
        ----------
        param : Array
            Parameters to check.

        Returns
        -------
        bool
            True if the parameters match, False otherwise.
        """
        if not hasattr(self, "_param"):
            return False
        return self._bkd.allclose(param, self._param, atol=3e-16, rtol=3e-16)

    def set_forward_state(self, fwd_state: Array) -> None:
        """
        Set the forward state.

        Parameters
        ----------
        fwd_state : Array
            Forward state to set.
        """
        self._fwd_state = fwd_state

    def has_forward_state(self) -> bool:
        """
        Check if the forward state is set.

        Returns
        -------
        bool
            True if the forward state is set, False otherwise.
        """
        return hasattr(self, "_fwd_state")

    def get_forward_state(self) -> Array:
        """
        Get the forward state.

        Returns
        -------
        Array
            Forward state.

        Raises
        ------
        AttributeError
            If the forward state is not set.
        """
        if not self.has_forward_state():
            raise AttributeError("must call set_forward_state")
        return self._fwd_state

    def set_state_eq_state_jacobian(self, drdy: Array) -> None:
        """
        Set the state equation state Jacobian.

        Parameters
        ----------
        drdy : Array
            State equation state Jacobian to set.
        """
        self._drdy = drdy

    def get_state_eq_state_jacobian(self) -> Array:
        """
        Get the state equation state Jacobian.

        Returns
        -------
        Array
            State equation state Jacobian.

        Raises
        ------
        AttributeError
            If the state equation state Jacobian is not set.
        """
        if not hasattr(self, "_drdy"):
            raise AttributeError("must call set_state_eq_state_jacobian")
        return self._drdy

    def set_state_eq_param_jacobian(self, drdp: Array) -> None:
        """
        Set the state equation parameter Jacobian.

        Parameters
        ----------
        drdp : Array
            State equation parameter Jacobian to set.
        """
        self._drdp = drdp

    def has_state_eq_param_jacobian(self) -> bool:
        """
        Check if the state equation parameter Jacobian is set.

        Returns
        -------
        bool
            True if the state equation parameter Jacobian is set, False otherwise.
        """
        return hasattr(self, "_drdp")

    def get_state_eq_param_jacobian(self) -> Array:
        """
        Get the state equation parameter Jacobian.

        Returns
        -------
        Array
            State equation parameter Jacobian.

        Raises
        ------
        AttributeError
            If the state equation parameter Jacobian is not set.
        """
        if not self.has_state_eq_param_jacobian():
            raise AttributeError("must call set_state_eq_param_jacobian")
        return self._drdp

    def set_qoi_state_jacobian(self, dqdy: Array) -> None:
        """
        Set the QoI state Jacobian.

        Parameters
        ----------
        dqdy : Array
            QoI state Jacobian to set.
        """
        self._dqdy = dqdy

    def get_qoi_state_jacobian(self) -> Array:
        """
        Get the QoI state Jacobian.

        Returns
        -------
        Array
            QoI state Jacobian.

        Raises
        ------
        AttributeError
            If the QoI state Jacobian is not set.
        """
        if not hasattr(self, "_dqdy"):
            raise AttributeError("must call set_qoi_state_jacobian")
        return self._dqdy

    def set_qoi_param_jacobian(self, dqdp: Array) -> None:
        """
        Set the QoI parameter Jacobian.

        Parameters
        ----------
        dqdp : Array
            QoI parameter Jacobian to set.
        """
        self._dqdp = dqdp

    def get_qoi_param_jacobian(self) -> Array:
        """
        Get the QoI parameter Jacobian.

        Returns
        -------
        Array
            QoI parameter Jacobian.

        Raises
        ------
        AttributeError
            If the QoI parameter Jacobian is not set.
        """
        if not hasattr(self, "_dqdp"):
            raise AttributeError("must call set_qoi_param_jacobian")
        return self._dqdp

    def set_adjoint_state(self, adj_state: Array) -> None:
        """
        Set the adjoint state.

        Parameters
        ----------
        adj_state : Array
            Adjoint state to set.
        """
        self._adj_state = adj_state

    def has_adjoint_state(self) -> bool:
        """
        Check if the adjoint state is set.

        Returns
        -------
        bool
            True if the adjoint state is set, False otherwise.
        """
        return hasattr(self, "_adj_state")

    def get_adjoint_state(self) -> Array:
        """
        Get the adjoint state.

        Returns
        -------
        Array
            Adjoint state.

        Raises
        ------
        AttributeError
            If the adjoint state is not set.
        """
        if not self.has_adjoint_state():
            raise AttributeError("must call set_adjoint_state")
        return self._adj_state

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"{self.__class__.__name__}()"
