# PyApprox Typing Module: Protocol Naming Conventions

This document defines standard naming conventions for protocols and methods across the PyApprox typing module to ensure consistency and clarity.

## Protocol Hierarchy

Base protocols are defined in `pyapprox.typing.util.protocols.base`:

```
ComputationalObject (root)
├── CallableObject
├── DimensionalObject
└── ParameterizedObject
```

All protocols should:
- Use `@runtime_checkable` decorator
- Inherit from `Protocol, Generic[Array]`
- Include comprehensive docstrings

## Standard Method Names

### Backend Accessor
**Method**: `bkd() -> Backend[Array]`

All computational objects must implement this method.

**Usage**:
```python
def bkd(self) -> Backend[Array]:
    """Return the backend used for computations."""
    ...
```

### Dimension Inquiry Methods

#### Universal Dimensions
Use these consistently across ALL domains:

- **`nvars() -> int`**: Number of input variables
- **`nqoi() -> int`**: Number of quantities of interest (outputs)

#### Domain-Specific Dimensions
Use these for specialized contexts:

- **`nparams() -> int`**: Number of parameters (for parameterized functions)
- **`nstates() -> int`**: Number of state variables (for PDEs, state equations)
- **`nterms() -> int`**: Number of terms (for basis expansions)
- **`nunique_params() -> int`**: Number of unique parameters (for shared parameters)

**Example**:
```python
# Function protocols use:
def nvars(self) -> int: ...  # input dimension
def nqoi(self) -> int: ...   # output dimension

# Parameterized function protocols add:
def nparams(self) -> int: ...  # parameter dimension

# State equation protocols use:
def nstates(self) -> int: ...  # state dimension
def nparams(self) -> int: ...  # parameter dimension
```

### Evaluation Method
**Method**: `__call__(samples: Array) -> Array`

Make objects callable for evaluation.

**Usage**:
```python
def __call__(self, samples: Array) -> Array:
    """Evaluate the object with given samples."""
    ...
```

### Derivative Methods

#### First-Order Derivatives
- **`jacobian(sample: Array) -> Array`**: Full Jacobian matrix
- **`jvp(sample: Array, vec: Array) -> Array`**: Jacobian-vector product
- **`gradient(sample: Array) -> Array`**: Gradient (for scalar outputs)

#### Second-Order Derivatives
- **`hessian(sample: Array) -> Array`**: Full Hessian matrix
- **`hvp(sample: Array, vec: Array) -> Array`**: Hessian-vector product
- **`whvp(sample: Array, vec: Array, weights: Array) -> Array`**: Weighted Hessian-vector product

#### Partial Derivatives
For multi-argument functions (e.g., `f(state, param)`):

- **`state_jacobian(state: Array, param: Array) -> Array`**: ∂f/∂state
- **`param_jacobian(state: Array, param: Array) -> Array`**: ∂f/∂param

#### Second-Order Partial Derivatives
Use this naming pattern for Hessian-vector products:

- **`state_state_hvp(...)`**: ∂²f/∂state²
- **`param_param_hvp(...)`**: ∂²f/∂param²
- **`state_param_hvp(...)`**: ∂²f/∂state∂param
- **`param_state_hvp(...)`**: ∂²f/∂param∂state

**Example**:
```python
# For implicit function operator protocols:
def state_jacobian(self, state: Array, param: Array) -> Array: ...
def param_jacobian(self, state: Array, param: Array) -> Array: ...
def state_state_hvp(self, state: Array, param: Array, vec: Array) -> Array: ...
def state_param_hvp(self, state: Array, param: Array, vec: Array) -> Array: ...
```

### Parameter Management
For parameterized objects:

- **`set_parameter(param: Array) -> None`**: Set parameter values
- **`get_parameter() -> Array`**: Get current parameter values (optional)
- **`nparams() -> int`**: Number of parameters

### Storage and State Management
For objects that maintain internal state:

- **`storage() -> StorageType`**: Return storage object
- **`reset() -> None`**: Reset internal state
- **`state() -> Dict`**: Return current state as dictionary

## Naming Patterns

### Protocol Naming
- Use descriptive names ending with `Protocol`
- Examples: `FunctionProtocol`, `KernelProtocol`, `BasisProtocol`

### Specific Capability Protocols
When extending protocols with additional capabilities:

- **Pattern**: `BaseWithCapabilityProtocol`
- Examples:
  - `FunctionWithJacobianProtocol`
  - `FunctionWithHessianProtocol`
  - `KernelWithJacobianProtocol`

### "Has" Protocols
For protocols that indicate presence of a feature (used with composition):

- **Pattern**: `HasCapabilityProtocol`
- Examples:
  - `HasJacobianProtocol`
  - `HasParameterJacobianProtocol`

## Return Type Specifications

### Backend Return Type
Always specify the type parameter:

```python
# Correct:
def bkd(self) -> Backend[Array]: ...

# Incorrect:
def bkd(self) -> Backend: ...  # Missing type parameter
```

### Array Return Types
Use `Array` from `pyapprox.typing.util.backends.protocols`:

```python
from pyapprox.typing.util.backends.protocols import Array, Backend

def jacobian(self, sample: Array) -> Array: ...
```

## Protocol Inheritance

### Inheriting from Base Protocols
When appropriate, inherit from base protocols:

```python
from pyapprox.typing.util.protocols.base import (
    ComputationalObject,
    DimensionalObject
)

@runtime_checkable
class FunctionProtocol(
    DimensionalObject[Array],  # Provides nvars(), nqoi(), bkd()
    Protocol
):
    """Function protocol with dimensions."""

    def __call__(self, samples: Array) -> Array:
        """Evaluate function."""
        ...
```

### Composition Over Repetition
When multiple protocols share methods, use composition:

```python
@runtime_checkable
class FunctionWithJacobianProtocol(
    FunctionProtocol[Array],
    HasJacobianProtocol[Array],
    Protocol
):
    pass  # Inherits all methods from both protocols
```

## Examples

### Simple Function Protocol
```python
@runtime_checkable
class FunctionProtocol(DimensionalObject[Array], Protocol):
    """
    Protocol for function implementations.

    Methods
    -------
    bkd() -> Backend[Array]
        Inherited from ComputationalObject via DimensionalObject.
    nvars() -> int
        Inherited from DimensionalObject.
    nqoi() -> int
        Inherited from DimensionalObject.
    __call__(samples: Array) -> Array
        Evaluate the function.
    """

    def __call__(self, samples: Array) -> Array:
        """Evaluate the function with given samples."""
        ...
```

### Parameterized Function Protocol
```python
@runtime_checkable
class ParameterizedFunctionProtocol(
    DimensionalObject[Array],
    ParameterizedObject[Array],
    Protocol
):
    """
    Protocol for parameterized functions.

    Inherits nvars(), nqoi(), bkd() from DimensionalObject.
    Inherits nparams(), set_parameter() from ParameterizedObject.
    """

    def __call__(self, samples: Array) -> Array: ...
```

### Kernel Protocol
```python
@runtime_checkable
class KernelProtocol(ComputationalObject[Array], Protocol):
    """
    Protocol for kernel implementations.

    Methods
    -------
    bkd() -> Backend[Array]
        Inherited from ComputationalObject.
    __call__(X1: Array, X2: Optional[Array] = None) -> Array
        Compute kernel matrix.
    diag(X: Array) -> Array
        Compute diagonal of kernel matrix.
    """

    def __call__(self, X1: Array, X2: Optional[Array] = None) -> Array:
        """Compute kernel matrix."""
        ...

    def diag(self, X: Array) -> Array:
        """Compute diagonal of kernel matrix."""
        ...
```

## Migration Guidelines

When updating existing protocols:

1. **Add base protocol inheritance** where appropriate
2. **Maintain backward compatibility** - don't remove existing methods
3. **Update return types** to include type parameters (e.g., `Backend[Array]`)
4. **Add @runtime_checkable** if missing
5. **Update docstrings** to reflect inheritance

## Deprecation Policy

When renaming methods:

1. Keep old method with deprecation warning
2. Add new method with correct naming
3. Provide transition period (1-2 releases)
4. Document in changelog

Example:
```python
import warnings

def old_method_name(self) -> Array:
    warnings.warn(
        "old_method_name is deprecated, use new_method_name instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method_name()

def new_method_name(self) -> Array:
    """New method following naming conventions."""
    ...
```

## Questions?

For questions about protocol design or naming conventions, refer to:
1. Existing examples in well-organized modules (e.g., `util.hyperparameter`)
2. Base protocol definitions in `util.protocols.base`
3. Project maintainers
