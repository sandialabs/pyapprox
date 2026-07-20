"""CompositeParameterization: chains multiple parameterizations."""

from typing import Generic, List, Optional, Sequence, TypeVar

from pyapprox.pde.parameterizations.derivatives import (
    BCFluxParamSensitivityFn,
    InitialParamJacobianFn,
    ParamDerivatives,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

_F = TypeVar("_F")


def _all_or_none(fns: Sequence[Optional[_F]]) -> Optional[List[_F]]:
    """Return the callables when every entry is populated, else None.

    A composite capability exists iff EVERY part provides it.
    """
    result: List[_F] = []
    for fn in fns:
        if fn is None:
            return None
        result.append(fn)
    return result


class CompositeParameterization(Generic[Array]):
    """Chains multiple parameterizations over contiguous parameter slices.

    Derivative capability is composed field-by-field from the parts'
    :class:`ParamDerivatives` bundles: a field is populated iff EVERY
    part's bundle has it. The bundle is rebuilt (never mutated) when a
    part is appended.

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
            self._validate_part(part, parts[0])
        self._parts: List[ParameterizationProtocol[Array]] = list(parts)
        self._bkd = bkd
        self._recompute_offsets()
        self._build_derivatives()

    @staticmethod
    def _validate_part(
        part: ParameterizationProtocol[Array],
        first_part: ParameterizationProtocol[Array],
    ) -> None:
        """Validate protocol conformance and shared physics identity."""
        if not isinstance(part, ParameterizationProtocol):
            raise TypeError(
                f"Each part must satisfy ParameterizationProtocol, "
                f"got {type(part).__name__}"
            )
        if part.physics() is not first_part.physics():
            raise ValueError(
                f"All parts must bind the SAME physics instance; "
                f"{type(part).__name__} binds a different physics than "
                f"{type(first_part).__name__}. Ensembles must construct "
                f"one composite per physics."
            )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def physics(self) -> object:
        """Return the physics instance shared by all parts."""
        if not self._parts:
            raise RuntimeError(
                "CompositeParameterization has no parts; physics() is "
                "undefined until a part is appended"
            )
        return self._parts[0].physics()

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the composed derivative capability bundle."""
        return self._derivs

    def _recompute_offsets(self) -> None:
        """Recompute contiguous parameter slice offsets."""
        self._offsets: List[int] = []
        offset = 0
        for part in self._parts:
            self._offsets.append(offset)
            offset += part.nparams()
        self._total_nparams = offset

    def _build_derivatives(self) -> None:
        """Compose the bundle field-by-field from the parts' bundles."""
        part_derivs = [p.param_derivatives() for p in self._parts]
        self._part_param_jacs: Optional[List[ParamJacobianFn[Array]]] = (
            _all_or_none([d.param_jacobian for d in part_derivs])
        )
        self._part_initial_param_jacs: Optional[
            List[InitialParamJacobianFn[Array]]
        ] = _all_or_none([d.initial_param_jacobian for d in part_derivs])
        self._part_param_param_hvps: Optional[List[ParamHVPFn[Array]]] = (
            _all_or_none([d.param_param_hvp for d in part_derivs])
        )
        self._part_state_param_hvps: Optional[List[ParamHVPFn[Array]]] = (
            _all_or_none([d.state_param_hvp for d in part_derivs])
        )
        self._part_param_state_hvps: Optional[List[ParamHVPFn[Array]]] = (
            _all_or_none([d.param_state_hvp for d in part_derivs])
        )
        self._part_bc_flux_fns: Optional[
            List[BCFluxParamSensitivityFn[Array]]
        ] = _all_or_none(
            [d.bc_flux_param_sensitivity for d in part_derivs]
        )
        self._derivs: ParamDerivatives[Array] = ParamDerivatives(
            param_jacobian=(
                self._param_jacobian
                if self._part_param_jacs is not None
                else None
            ),
            initial_param_jacobian=(
                self._initial_param_jacobian
                if self._part_initial_param_jacs is not None
                else None
            ),
            param_param_hvp=(
                self._param_param_hvp
                if self._part_param_param_hvps is not None
                else None
            ),
            state_param_hvp=(
                self._state_param_hvp
                if self._part_state_param_hvps is not None
                else None
            ),
            param_state_hvp=(
                self._param_state_hvp
                if self._part_param_state_hvps is not None
                else None
            ),
            bc_flux_param_sensitivity=(
                self._bc_flux_param_sensitivity
                if self._part_bc_flux_fns is not None
                else None
            ),
        )

    def nparams(self) -> int:
        return self._total_nparams

    def apply(self, params_1d: Array) -> None:
        """Apply all parameterizations in sequence."""
        for ii, part in enumerate(self._parts):
            offset = self._offsets[ii]
            np_i = part.nparams()
            part.apply(params_1d[offset : offset + np_i])

    def append(self, part: ParameterizationProtocol[Array]) -> None:
        """Append a parameterization. Rebuilds the capability bundle."""
        self._validate_part(
            part, self._parts[0] if self._parts else part
        )
        self._parts.append(part)
        self._recompute_offsets()
        self._build_derivatives()

    def _param_jacobian(
        self,
        state: Array,
        time: float,
        params_1d: Array,
    ) -> Array:
        """Block-column assembly of param Jacobian. Shape: (npts, total_nparams)."""
        fns = self._part_param_jacs
        if fns is None:
            raise RuntimeError(
                "param_jacobian is unavailable; check param_derivatives() "
                "before calling"
            )
        npts = state.shape[0]
        result = self._bkd.zeros((npts, self._total_nparams))
        result = self._bkd.copy(result)
        for ii, fn in enumerate(fns):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            block = fn(state, time, sub_params)
            for col in range(np_i):
                for row in range(npts):
                    result[row, offset + col] = block[row, col]
        return result

    def _initial_param_jacobian(self, params_1d: Array) -> Array:
        """Block-column assembly of initial param Jacobian."""
        fns = self._part_initial_param_jacs
        if fns is None:
            raise RuntimeError(
                "initial_param_jacobian is unavailable; check "
                "param_derivatives() before calling"
            )
        # Get npts from first part's result
        first_offset = self._offsets[0]
        np_0 = self._parts[0].nparams()
        sub_params_0 = params_1d[first_offset : first_offset + np_0]
        block_0 = fns[0](sub_params_0)
        npts = block_0.shape[0]

        result = self._bkd.zeros((npts, self._total_nparams))
        result = self._bkd.copy(result)
        # Fill first block
        for col in range(np_0):
            for row in range(npts):
                result[row, first_offset + col] = block_0[row, col]
        # Fill remaining blocks
        for ii in range(1, len(self._parts)):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            block = fns[ii](sub_params)
            for col in range(np_i):
                for row in range(npts):
                    result[row, offset + col] = block[row, col]
        return result

    def _param_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Block assembly of param-param HVP. Shape: (total_nparams,).

        The block-diagonal Hessian assumption is valid because parts
        parameterize distinct additive residual terms.
        """
        fns = self._part_param_param_hvps
        if fns is None:
            raise RuntimeError(
                "param_param_hvp is unavailable; check param_derivatives() "
                "before calling"
            )
        result = self._bkd.zeros((self._total_nparams,))
        result = self._bkd.copy(result)
        for ii, fn in enumerate(fns):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            sub_vvec = vvec[offset : offset + np_i]
            sub_result = fn(
                state, time, sub_params, adj_state, sub_vvec
            )
            for k in range(np_i):
                result[offset + k] = sub_result[k]
        return result

    def _state_param_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Sum of the parts' state-shaped HVPs. Shape: (nstates,).

        lambda^T (d^2R/dy dp) v is state-shaped and additive over parts
        (each part parameterizes a distinct additive residual term), so
        the composite result is the SUM of part results, each contracted
        with its own parameter slice of ``vvec`` — not a per-slot
        assembly into a parameter-shaped vector.
        """
        fns = self._part_state_param_hvps
        if fns is None:
            raise RuntimeError(
                "state_param_hvp is unavailable; check param_derivatives() "
                "before calling"
            )
        result = self._bkd.zeros((state.shape[0],))
        for ii, fn in enumerate(fns):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            sub_vvec = vvec[offset : offset + np_i]
            result = result + fn(
                state, time, sub_params, adj_state, sub_vvec
            )
        return result

    def _param_state_hvp(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Block assembly of param-state HVP. Shape: (total_nparams,)."""
        fns = self._part_param_state_hvps
        if fns is None:
            raise RuntimeError(
                "param_state_hvp is unavailable; check param_derivatives() "
                "before calling"
            )
        result = self._bkd.zeros((self._total_nparams,))
        result = self._bkd.copy(result)
        for ii, fn in enumerate(fns):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            sub_result = fn(
                state, time, sub_params, adj_state, wvec
            )
            for k in range(np_i):
                result[offset + k] = sub_result[k]
        return result

    def _bc_flux_param_sensitivity(
        self,
        state: Array,
        time: float,
        params_1d: Array,
        bc_indices: Array,
        normals: Array,
    ) -> Array:
        """Block-column assembly of BC flux param sensitivity."""
        fns = self._part_bc_flux_fns
        if fns is None:
            raise RuntimeError(
                "bc_flux_param_sensitivity is unavailable; check "
                "param_derivatives() before calling"
            )
        nbnd = bc_indices.shape[0]
        result = self._bkd.zeros((nbnd, self._total_nparams))
        result = self._bkd.copy(result)
        for ii, fn in enumerate(fns):
            offset = self._offsets[ii]
            np_i = self._parts[ii].nparams()
            sub_params = params_1d[offset : offset + np_i]
            block = fn(
                state, time, sub_params, bc_indices, normals
            )
            for col in range(np_i):
                for i in range(nbnd):
                    result[i, offset + col] = block[i, col]
        return result
