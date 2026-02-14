"""CompositeParameterization: chains multiple parameterizations."""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.forward_models.parameterizations.protocol import (
    ParameterizationProtocol,
)


class CompositeParameterization(Generic[Array]):
    """Chains multiple parameterizations over contiguous parameter slices.

    Parameters
    ----------
    parts : List[ParameterizationProtocol]
        List of parameterization components.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        parts: List[ParameterizationProtocol[Array]],
        bkd: Backend[Array],
    ) -> None:
        for part in parts:
            if not isinstance(part, ParameterizationProtocol):
                raise TypeError(
                    f"Each part must satisfy ParameterizationProtocol, "
                    f"got {type(part).__name__}"
                )
        self._parts: List[ParameterizationProtocol[Array]] = list(parts)
        self._bkd = bkd
        self._recompute_offsets()
        self._bind_optional_methods()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _recompute_offsets(self) -> None:
        """Recompute contiguous parameter slice offsets."""
        self._offsets: List[int] = []
        offset = 0
        for part in self._parts:
            self._offsets.append(offset)
            offset += part.nparams()
        self._total_nparams = offset

    def _bind_optional_methods(self) -> None:
        """Bind optional methods if ALL parts support them."""
        # param_jacobian: only if ALL parts have it
        if all(hasattr(p, "param_jacobian") for p in self._parts):
            self.param_jacobian = self._param_jacobian
        elif hasattr(self, "param_jacobian"):
            del self.param_jacobian

        # initial_param_jacobian: only if ALL parts have it
        if all(hasattr(p, "initial_param_jacobian") for p in self._parts):
            self.initial_param_jacobian = self._initial_param_jacobian
        elif hasattr(self, "initial_param_jacobian"):
            del self.initial_param_jacobian

        # bc_flux_param_sensitivity: only if ALL parts have it
        if all(
            hasattr(p, "bc_flux_param_sensitivity") for p in self._parts
        ):
            self.bc_flux_param_sensitivity = (
                self._bc_flux_param_sensitivity
            )
        elif hasattr(self, "bc_flux_param_sensitivity"):
            del self.bc_flux_param_sensitivity

        # HVP methods: only if ALL parts have them
        for method_name, impl_name in [
            ("param_param_hvp", "_param_param_hvp"),
            ("state_param_hvp", "_state_param_hvp"),
            ("param_state_hvp", "_param_state_hvp"),
        ]:
            if all(hasattr(p, method_name) for p in self._parts):
                setattr(self, method_name, getattr(self, impl_name))
            elif hasattr(self, method_name):
                delattr(self, method_name)

    def nparams(self) -> int:
        return self._total_nparams

    def apply(self, physics: object, params_1d: Array) -> None:
        """Apply all parameterizations in sequence."""
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            part.apply(physics, params_1d[offset:offset + np_i])

    def append(self, part: ParameterizationProtocol[Array]) -> None:
        """Append a parameterization. Re-binds optional methods."""
        if not isinstance(part, ParameterizationProtocol):
            raise TypeError(
                f"part must satisfy ParameterizationProtocol, "
                f"got {type(part).__name__}"
            )
        self._parts.append(part)
        self._recompute_offsets()
        self._bind_optional_methods()

    def _param_jacobian(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Block-column assembly of param Jacobian. Shape: (npts, total_nparams)."""
        npts = state.shape[0]
        result = self._bkd.zeros((npts, self._total_nparams))
        result = self._bkd.copy(result)
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            block = part.param_jacobian(physics, state, time, sub_params)
            for col in range(np_i):
                for row in range(npts):
                    result[row, offset + col] = block[row, col]
        return result

    def _initial_param_jacobian(
        self, physics: object, params_1d: Array
    ) -> Array:
        """Block-column assembly of initial param Jacobian."""
        # Get npts from first part's result
        first_offset = self._offsets[0]
        np_0 = self._parts[0].nparams()
        sub_params_0 = params_1d[first_offset:first_offset + np_0]
        block_0 = self._parts[0].initial_param_jacobian(physics, sub_params_0)
        npts = block_0.shape[0]

        result = self._bkd.zeros((npts, self._total_nparams))
        result = self._bkd.copy(result)
        # Fill first block
        for col in range(np_0):
            for row in range(npts):
                result[row, first_offset + col] = block_0[row, col]
        # Fill remaining blocks
        for ii in range(1, len(self._parts)):
            part = self._parts[ii]
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            block = part.initial_param_jacobian(physics, sub_params)
            for col in range(np_i):
                for row in range(npts):
                    result[row, offset + col] = block[row, col]
        return result

    def _param_param_hvp(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Block assembly of param-param HVP."""
        result = self._bkd.zeros((self._total_nparams,))
        result = self._bkd.copy(result)
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            sub_vvec = vvec[offset:offset + np_i]
            sub_result = part.param_param_hvp(
                physics, state, time, sub_params, adj_state, sub_vvec
            )
            for k in range(np_i):
                result[offset + k] = sub_result[k]
        return result

    def _state_param_hvp(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Block assembly of state-param HVP."""
        result = self._bkd.zeros((self._total_nparams,))
        result = self._bkd.copy(result)
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            sub_vvec = vvec[offset:offset + np_i]
            sub_result = part.state_param_hvp(
                physics, state, time, sub_params, adj_state, sub_vvec
            )
            for k in range(np_i):
                result[offset + k] = sub_result[k]
        return result

    def _param_state_hvp(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Block assembly of param-state HVP."""
        result = self._bkd.zeros((self._total_nparams,))
        result = self._bkd.copy(result)
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            sub_result = part.param_state_hvp(
                physics, state, time, sub_params, adj_state, wvec
            )
            for k in range(np_i):
                result[offset + k] = sub_result[k]
        return result

    def _bc_flux_param_sensitivity(
        self,
        physics: object,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Block-column assembly of BC flux param sensitivity."""
        nbnd = bc_indices.shape[0]
        result = self._bkd.zeros((nbnd, self._total_nparams))
        result = self._bkd.copy(result)
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            sub_params = params_1d[offset:offset + np_i]
            block = part.bc_flux_param_sensitivity(
                physics, state, time, sub_params, bc_indices, normals
            )
            if block is not None:
                for col in range(np_i):
                    for i in range(nbnd):
                        result[i, offset + col] = block[i, col]
        return result
