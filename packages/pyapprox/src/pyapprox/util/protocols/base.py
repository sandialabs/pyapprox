"""
Base protocol hierarchy for PyApprox typing module.

This module defines foundational protocols that establish common interfaces
for computational objects throughout the PyApprox typing system. These protocols
support the backend abstraction pattern that enables code to work seamlessly
with both NumPy arrays and PyTorch tensors.

Protocol Hierarchy
------------------
ComputationalObject
├── CallableObject
├── DimensionalObject
└── ParameterizedObject

All protocols use Generic[Array] to support both NumPy and PyTorch backends.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ComputationalObject(Protocol, Generic[Array]):
    """
    Base protocol for all computational objects with backend support.

    This is the root protocol in the hierarchy. All computational objects
    that use the backend abstraction should implement this protocol.

    Methods
    -------
    bkd() -> Backend[Array]
        Returns the backend used for numerical computations.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        Backend[Array]
            Backend for numerical computations (e.g., NumPy or PyTorch).
        """
        ...


@runtime_checkable
class CallableObject(ComputationalObject[Array], Protocol):
    """
    Protocol for callable computational objects.

    Extends ComputationalObject to add callable functionality. Objects
    implementing this protocol can be invoked with array inputs.

    Methods
    -------
    bkd() -> Backend[Array]
        Inherited from ComputationalObject.
    __call__(samples: Array) -> Array
        Evaluate the object with given samples.
    """

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the object with given samples.

        Parameters
        ----------
        samples : Array
            Input samples for evaluation.

        Returns
        -------
        Array
            Evaluation results.
        """
        ...


@runtime_checkable
class DimensionalObject(ComputationalObject[Array], Protocol):
    """
    Protocol for objects with input/output dimensions.

    Extends ComputationalObject to add dimension inquiry methods. This
    protocol is used for objects that have well-defined input and output
    dimensions (e.g., functions, operators, surrogate models).

    Methods
    -------
    bkd() -> Backend[Array]
        Inherited from ComputationalObject.
    nvars() -> int
        Returns the number of input variables.
    nqoi() -> int
        Returns the number of quantities of interest (outputs).
    """

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Returns
        -------
        int
            Number of input variables (input dimension).
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (outputs).

        Returns
        -------
        int
            Number of quantities of interest (output dimension).
        """
        ...


@runtime_checkable
class ParameterizedObject(ComputationalObject[Array], Protocol):
    """
    Protocol for parameterized computational objects.

    Extends ComputationalObject to add parameter management methods. This
    protocol is used for objects that have tunable parameters (e.g.,
    parameterized functions, neural networks, kernels with hyperparameters).

    Methods
    -------
    bkd() -> Backend[Array]
        Inherited from ComputationalObject.
    nparams() -> int
        Returns the number of parameters.
    set_parameter(param: Array) -> None
        Sets the parameter values.
    """

    def nparams(self) -> int:
        """
        Return the number of parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...

    def set_parameter(self, param: Array) -> None:
        """
        Set the parameter values.

        Parameters
        ----------
        param : Array
            Parameter values to set. Shape should be (nparams(),).
        """
        ...
