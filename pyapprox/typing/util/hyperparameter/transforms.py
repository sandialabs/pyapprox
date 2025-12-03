from typing import Generic, Protocol, runtime_checkable
from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class HyperParameterTransformProtocol(Generic[Array], Protocol):
    """
    Protocol for hyperparameter transformations.

    Defines the interface for transforming hyperparameters between user space
    and optimization space.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
        Backend for numerical computations.
        """
        ...

    def to_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from user space to optimization space.

        Parameters
        ----------
        params : Array
            Hyperparameters in user space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in optimization space.
        """
        ...

    def from_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from optimization space to user space.

        Parameters
        ----------
        params : Array
            Hyperparameters in optimization space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in user space.
        """
        ...


class IdentityHyperParameterTransform(Generic[Array]):
    """
    Identity transformation for hyperparameters.

    This transformation leaves the hyperparameters unchanged when converting
    between user space and optimization space.

    Parameters
    ----------
    bkd : Backend
        Backend for numerical computations.
    """

    def __init__(self, bkd: Backend) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
        Backend for numerical computations.
        """
        return self._bkd

    def to_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from user space to optimization space.

        Parameters
        ----------
        params : Array
            Hyperparameters in user space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in optimization space (unchanged).
        """
        return params

    def from_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from optimization space to user space.

        Parameters
        ----------
        params : Array
            Hyperparameters in optimization space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in user space (unchanged).
        """
        return params

    def __repr__(self) -> str:
        """
        Return a string representation of the IdentityHyperParameterTransform.

        Returns
        -------
        repr : str
            String representation of the class.
        """
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__class__.__name__
        )


class LogHyperParameterTransform(Generic[Array]):
    """
    Logarithmic transformation for hyperparameters.

    This transformation applies the logarithm when converting from user space
    to optimization space, and the exponential when converting back.

    Parameters
    ----------
    bkd : Backend
        Backend for numerical computations.
    """

    def __init__(self, bkd: Backend) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
        Backend for numerical computations.
        """
        return self._bkd

    def to_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from user space to optimization space.

        Parameters
        ----------
        params : Array
            Hyperparameters in user space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in optimization space (logarithm applied).
        """
        return self._bkd.log(params)

    def from_opt_space(self, params: Array) -> Array:
        """
        Transform hyperparameters from optimization space to user space.

        Parameters
        ----------
        params : Array
            Hyperparameters in optimization space.

        Returns
        -------
        transformed_params : Array
            Hyperparameters in user space (exponential applied).
        """
        return self._bkd.exp(params)

    def __repr__(self) -> str:
        """
        Return a string representation of the LogHyperParameterTransform.

        Returns
        -------
        repr : str
            String representation of the class.
        """
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__class__.__name__
        )
