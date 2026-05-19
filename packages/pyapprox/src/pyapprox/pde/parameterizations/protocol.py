"""Protocols for parameterizations module."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array


@runtime_checkable
class DerivativeMatrixBasisProtocol(Protocol, Generic[Array]):
    """Minimal basis interface needed by parameterization factories.

    Any TensorProductBasisProtocol or BasisProtocol satisfies this
    via structural subtyping.
    """

    def ndim(self) -> int: ...

    def derivative_matrix(self, order: int, dim: int) -> Array: ...


@runtime_checkable
class ParameterizationProtocol(Protocol, Generic[Array]):
    """Protocol for physics parameterizations.

    Required methods:
        nparams() -> int
        apply(physics, params_1d) -> None

    Optional methods (detected via hasattr):
        param_jacobian(physics, state, time, params_1d) -> Array  (nstates, nparams)
        initial_param_jacobian(physics, params_1d) -> Array  (nstates, nparams)
        bc_flux_param_sensitivity(physics, state, time, params_1d,
            bc_indices, normals) -> Array  (n_bc, nparams)
        param_param_hvp(physics, state, time, params_1d, adj_state, vvec) -> Array
        state_param_hvp(physics, state, time, params_1d, adj_state, vvec) -> Array
        param_state_hvp(physics, state, time, params_1d, adj_state, wvec) -> Array
    """

    def nparams(self) -> int: ...

    def apply(self, physics: object, params_1d: Array) -> None: ...
