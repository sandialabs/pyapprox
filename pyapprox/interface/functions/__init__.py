"""
Function interface module for PyApprox typing.

This module provides protocols, wrappers, and utilities for working with
functions in a type-safe manner with backend abstraction (NumPy/PyTorch).

Submodules
----------
- protocols: Protocol definitions for functions
- parameterized: Parameterized function support
- fromcallable: Create functions from callable objects
- derivative_checks: Derivative validation utilities
- plot: Plotting utilities for functions
- numpy: NumPy-specific function wrappers

Examples
--------
>>> from pyapprox.interface.functions.protocols import FunctionProtocol
>>> from pyapprox.interface.functions.parameterized import (
...     convert_to_function_of_parameters
... )
"""

__all__ = []
