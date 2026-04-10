"""Time-evolving vector field for flow matching.

At each training time t_k, an orthonormal polynomial basis is built from
the marginal distribution of x_t via the Stieltjes/Lanczos algorithm.

Two coefficient fitting strategies are supported:

- **Kronecker** (default when ``per_slice=False``): Global Legendre time
  expansion ``v(x, t) = sum_{n,j} c_{nj} phi_n^t(x) psi_j(t)``.
- **Per-slice** (default when ``per_slice=True``): Independent lstsq at
  each training t_k, with Lagrange interpolation of coefficients for
  non-training t values.
"""

from typing import Generic, Optional, Protocol, Union, runtime_checkable

from pyapprox.generative.flowmatching.basis_factory import (
    StieltjesBasisFactory,
)
from pyapprox.generative.flowmatching.basis_interp import (
    IdentityInterpolator,
    LagrangeInterp1D,
    RecurrenceInterpolator,
)
from pyapprox.generative.flowmatching.basis_state import (
    StieltjesBasisState,
)
from pyapprox.generative.flowmatching.protocols import (
    ProbabilityPathProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.protocols import (
    LinearSystemSolverProtocol,
)
from pyapprox.surrogates.affine.univariate import create_basis_1d
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)

BasisInterpolator = Union[
    IdentityInterpolator[Array], RecurrenceInterpolator[Array]
]


def _make_legendre_basis(n_legendre: int, bkd: Backend[Array]):
    """Create orthonormal Legendre basis on [0, 1] with n_legendre terms."""
    basis = create_basis_1d(UniformMarginal(0.0, 1.0, bkd), bkd)
    basis.set_nterms(n_legendre)
    return basis


# ------------------------------------------------------------------ #
#  Coefficient strategy protocol and implementations                  #
# ------------------------------------------------------------------ #

@runtime_checkable
class CoefficientStrategy(Protocol[Array]):
    """Protocol for coefficient fitting strategies."""

    def fit(
        self,
        interpolator: BasisInterpolator,
        samples: Array,
        values: Array,
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]],
    ) -> None: ...

    def evaluate(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array: ...

    def evaluate_deriv(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array: ...


class PerSliceStrategy(Generic[Array]):
    """Independent lstsq fit at each training t_k.

    Coefficients at non-training t values are obtained by Lagrange
    interpolation of the per-slice coefficient vectors.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._slice_coefs: Optional[list[Array]] = None
        self._t_nodes: Optional[Array] = None
        self._coef_interps: Optional[list[LagrangeInterp1D[Array]]] = None
        self._n_basis: int = 0
        self._nqoi: int = 0

    def fit(
        self,
        interpolator: BasisInterpolator,
        samples: Array,
        values: Array,
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        """Independent lstsq at each unique t."""
        t_row = samples[0, :]
        x_row = samples[1:2, :]
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        # Extract global weights from solver if available
        global_weights = solver._weights if solver is not None else None

        unique_t = bkd.unique(t_row)
        slice_coefs: list[Array] = []

        for ii in range(len(unique_t)):
            t_k = unique_t[ii]
            mask = bkd.abs(t_row - t_k) < 1e-12
            indices = bkd.where(mask)[0]
            t_val = bkd.to_float(t_k)

            state = interpolator(t_val)
            phi = state.eval(x_row[:, indices])    # (n_k, n_basis)
            u = values[:, indices].T               # (n_k, nqoi)

            if solver is not None:
                from pyapprox.optimization.linear import LeastSquaresSolver
                solver_k = LeastSquaresSolver(bkd)
                if global_weights is not None:
                    solver_k.set_weights(global_weights[indices])
                c_k = solver_k.solve(phi, u)
            else:
                from pyapprox.optimization.linear import LeastSquaresSolver
                solver_k = LeastSquaresSolver(bkd)
                c_k = solver_k.solve(phi, u)

            slice_coefs.append(c_k)

        self._t_nodes = bkd.asarray(
            [bkd.to_float(unique_t[ii]) for ii in range(len(unique_t))]
        )
        self._slice_coefs = slice_coefs
        self._n_basis = slice_coefs[0].shape[0]
        self._nqoi = slice_coefs[0].shape[1]
        self._build_coef_interpolators(bkd)

    def _build_coef_interpolators(self, bkd: Backend[Array]) -> None:
        """Build Lagrange interpolators for each coefficient component."""
        assert self._slice_coefs is not None
        assert self._t_nodes is not None
        coef_stack = bkd.stack(self._slice_coefs, axis=0)  # (n_t, n_basis, nqoi)
        self._coef_interps = []
        for n in range(self._n_basis):
            for q in range(self._nqoi):
                interp = LagrangeInterp1D(bkd)
                interp.fit(self._t_nodes, coef_stack[:, n, q])
                self._coef_interps.append(interp)

    def _get_coefs(self, t_val: float, bkd: Backend[Array]) -> Array:
        """Get coefficients at t_val (exact lookup or interpolation)."""
        assert self._t_nodes is not None
        assert self._slice_coefs is not None
        assert self._coef_interps is not None

        diffs = bkd.abs(self._t_nodes - t_val)
        idx = int(bkd.to_float(bkd.argmin(diffs)))
        if bkd.to_float(diffs[idx]) < 1e-12:
            return self._slice_coefs[idx]

        # Interpolate
        t_arr = bkd.asarray([t_val])
        c = bkd.zeros((self._n_basis, self._nqoi))
        ii = 0
        for n in range(self._n_basis):
            for q in range(self._nqoi):
                c[n, q] = self._coef_interps[ii](t_arr)[0]
                ii += 1
        return c

    def evaluate(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array:
        """v(x, t) = phi(x) @ c(t), returns (nqoi, n_k)."""
        phi = state.eval(x_row)
        c = self._get_coefs(t_val, bkd)
        return (phi @ c).T

    def evaluate_deriv(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array:
        """dv/dx at (x, t), returns (n_k, nqoi)."""
        dphi = state.eval_derivatives(x_row, order=1)
        c = self._get_coefs(t_val, bkd)
        return dphi @ c

    def get_slice_coefs(self) -> list[Array]:
        """Return per-slice coefficient list."""
        assert self._slice_coefs is not None
        return self._slice_coefs

    def set_slice_coefs(self, coefs: list[Array]) -> None:
        """Set per-slice coefficients and rebuild interpolators."""
        self._slice_coefs = coefs
        self._n_basis = coefs[0].shape[0]
        self._nqoi = coefs[0].shape[1]
        self._build_coef_interpolators(self._bkd)

    def n_basis(self) -> int:
        return self._n_basis

    def nqoi(self) -> int:
        return self._nqoi


class KroneckerStrategy(Generic[Array]):
    """Global Kronecker product fitting with Legendre time expansion.

    Fits ``v(x, t) = sum_{n,j} c_{nj} phi_n^t(x) psi_j(t)`` via a
    single global least squares solve.
    """

    def __init__(self, bkd: Backend[Array], n_legendre: int = 1) -> None:
        self._bkd = bkd
        self._n_legendre = n_legendre
        self._legendre = _make_legendre_basis(n_legendre, bkd)
        self._coef: Optional[Array] = None
        self._n_basis: int = 0
        self._n_total: int = 0
        self._initialized: bool = False

    def _eval_legendre(self, t_vals: Array) -> Array:
        t_2d = self._bkd.reshape(t_vals, (1, -1))
        return self._legendre(t_2d)

    def _kronecker_block(
        self, phi: Array, psi: Array, n_k: int
    ) -> Array:
        bkd = self._bkd
        return bkd.reshape(
            bkd.reshape(phi, (n_k, self._n_basis, 1))
            * bkd.reshape(psi, (n_k, 1, self._n_legendre)),
            (n_k, self._n_total),
        )

    def basis_matrix(
        self,
        interpolator: BasisInterpolator,
        vf_input: Array,
        bkd: Backend[Array],
    ) -> Array:
        """Compute the Kronecker basis matrix."""
        t_row = vf_input[0, :]
        x_row = vf_input[1:2, :]
        n_quad = vf_input.shape[1]

        # Fast path: single t
        if bkd.to_float(bkd.max(bkd.abs(t_row - t_row[0]))) < 1e-12:
            t_val = bkd.to_float(t_row[0])
            state = interpolator(t_val)
            phi = state.eval(x_row)
            psi = self._eval_legendre(t_row)
            return self._kronecker_block(phi, psi, n_quad)

        result = bkd.zeros((n_quad, self._n_total))
        unique_t = bkd.unique(t_row)

        for ii in range(len(unique_t)):
            t_val = bkd.to_float(unique_t[ii])
            mask = bkd.abs(t_row - unique_t[ii]) < 1e-12
            indices = bkd.where(mask)[0]
            n_k = len(indices)

            state = interpolator(t_val)
            phi_slice = state.eval(x_row[:, indices])
            psi_slice = self._eval_legendre(t_row[indices])
            block = self._kronecker_block(phi_slice, psi_slice, n_k)
            result[indices, :] = block

        return result

    def fit(
        self,
        interpolator: BasisInterpolator,
        samples: Array,
        values: Array,
        bkd: Backend[Array],
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        """Fit via global Kronecker lstsq."""
        if values.ndim == 1:
            values = bkd.reshape(values, (1, -1))

        if solver is None:
            from pyapprox.optimization.linear import LeastSquaresSolver
            solver = LeastSquaresSolver(bkd)

        basis_mat = self.basis_matrix(interpolator, samples, bkd)
        self._coef = solver.solve(basis_mat, values.T)

    def evaluate(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array:
        """v(x, t) = (phi kron psi) @ c, returns (nqoi, n_k)."""
        coef = self.get_coefficients()
        n_k = x_row.shape[1]
        phi = state.eval(x_row)
        t_arr = bkd.full((n_k,), t_val)
        psi = self._eval_legendre(t_arr)
        basis = self._kronecker_block(phi, psi, n_k)
        return (basis @ coef).T

    def evaluate_deriv(
        self,
        t_val: float,
        state: StieltjesBasisState[Array],
        x_row: Array,
        bkd: Backend[Array],
    ) -> Array:
        """dv/dx, returns (n_k, nqoi)."""
        coef = self.get_coefficients()
        n_k = x_row.shape[1]
        dphi = state.eval_derivatives(x_row, order=1)
        t_arr = bkd.full((n_k,), t_val)
        psi = self._eval_legendre(t_arr)
        dbasis_dx = self._kronecker_block(dphi, psi, n_k)
        return dbasis_dx @ coef

    def get_coefficients(self) -> Array:
        if self._coef is None:
            return self._bkd.zeros((self._n_total, 1))
        return self._coef

    def set_coefficients(self, coef: Array) -> None:
        self._coef = coef

    def set_n_basis(self, n_basis: int, nqoi: int = 1) -> None:
        """Set spatial basis count (called during VF construction)."""
        self._n_basis = n_basis
        self._n_total = n_basis * self._n_legendre
        if self._coef is None:
            self._coef = self._bkd.zeros((self._n_total, nqoi))

    def n_legendre(self) -> int:
        return self._n_legendre


CoefficientStrategyType = Union[
    PerSliceStrategy[Array], KroneckerStrategy[Array]
]


class StieltjesFlowVF(Generic[Array]):
    """Vector field with time-evolving 1D Stieltjes polynomial basis.

    Delegates coefficient fitting and evaluation to a ``CoefficientStrategy``:
    either ``PerSliceStrategy`` (independent fit per t_k) or
    ``KroneckerStrategy`` (global Legendre time expansion).

    Parameters
    ----------
    interpolator : IdentityInterpolator or RecurrenceInterpolator
        Maps t values to ``StieltjesBasisState`` objects.
    strategy : PerSliceStrategy or KroneckerStrategy
        Coefficient fitting strategy.
    nqoi : int
        Number of output dimensions.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        interpolator: BasisInterpolator,
        strategy: CoefficientStrategyType,
        nqoi: int,
        bkd: Backend[Array],
    ) -> None:
        self._interp = interpolator
        self._strategy = strategy
        self._nqoi = nqoi
        self._bkd = bkd
        self._n_basis = interpolator._states[0].n_basis()

        # For backward compat: track n_total and coef for Kronecker
        if isinstance(strategy, KroneckerStrategy):
            strategy.set_n_basis(self._n_basis, nqoi)
            self._n_total = self._n_basis * strategy.n_legendre()
        else:
            self._n_total = self._n_basis

        self._coef: Array = bkd.zeros((self._n_total, nqoi))
        self._hyp_list: Optional[HyperParameterList] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Input dimensionality (t, x)."""
        return 2

    def nterms(self) -> int:
        """Number of basis terms."""
        return self._n_total

    def n_basis(self) -> int:
        """Number of Stieltjes spatial basis terms."""
        return self._n_basis

    def n_legendre(self) -> int:
        """Number of Legendre time basis terms."""
        if isinstance(self._strategy, KroneckerStrategy):
            return self._strategy.n_legendre()
        return 1

    def nqoi(self) -> int:
        """Number of output dimensions."""
        return self._nqoi

    def get_coefficients(self) -> Array:
        """Return coefficients.

        For KroneckerStrategy: shape ``(n_total, nqoi)``.
        For PerSliceStrategy: returns the first slice's coefficients
        (use ``strategy.get_slice_coefs()`` for all slices).
        """
        if isinstance(self._strategy, KroneckerStrategy):
            return self._strategy.get_coefficients()
        return self._coef

    def set_coefficients(self, coef: Array) -> None:
        """Set coefficients.

        Parameters
        ----------
        coef : Array
            Shape ``(n_total, nqoi)`` or ``(n_total,)`` for nqoi=1.
        """
        if coef.ndim == 1:
            coef = self._bkd.reshape(coef, (-1, 1))
        if isinstance(self._strategy, KroneckerStrategy):
            if coef.shape != (self._n_total, self._nqoi):
                raise ValueError(
                    f"Expected shape ({self._n_total}, {self._nqoi}), "
                    f"got {coef.shape}"
                )
            self._strategy.set_coefficients(coef)
        self._coef = coef
        if self._hyp_list is not None:
            self._hyp_list.set_values(self._bkd.flatten(coef))

    def hyp_list(self) -> HyperParameterList:
        """Return hyperparameter list for coefficient optimization.

        Only meaningful for KroneckerStrategy.
        """
        if isinstance(self._strategy, PerSliceStrategy):
            raise NotImplementedError(
                "hyp_list() is not supported for PerSliceStrategy. "
                "Use KroneckerStrategy for optimizer-based fitting."
            )
        if self._hyp_list is None:
            ncoeffs = self._n_total * self._nqoi
            self._hyp_list = HyperParameterList(
                [
                    HyperParameter(
                        name="coefficients",
                        nparams=ncoeffs,
                        values=self._bkd.flatten(self._coef),
                        bounds=(-1e10, 1e10),
                        bkd=self._bkd,
                    )
                ],
                self._bkd,
            )
        return self._hyp_list

    def _sync_from_hyp_list(self) -> None:
        """Sync coefficients from hyp_list values."""
        if self._hyp_list is not None:
            values = self._hyp_list.get_values()
            self._coef = self._bkd.reshape(
                values, (self._n_total, self._nqoi)
            )
            if isinstance(self._strategy, KroneckerStrategy):
                self._strategy.set_coefficients(self._coef)

    def _get_state(self, t_val: float) -> StieltjesBasisState[Array]:
        """Evaluate basis state at a given t value."""
        return self._interp(t_val)

    def basis_matrix(self, vf_input: Array) -> Array:
        """Compute basis matrix for input ``[t; x]``.

        Only supported for KroneckerStrategy.

        Parameters
        ----------
        vf_input : Array
            Shape ``(2, n_quad)`` with row 0 = t, row 1 = x.

        Returns
        -------
        Array
            Basis matrix, shape ``(n_quad, n_total)``.
        """
        if isinstance(self._strategy, PerSliceStrategy):
            raise NotImplementedError(
                "basis_matrix() is not supported for PerSliceStrategy. "
                "Use KroneckerStrategy for callers that need a global "
                "basis matrix (e.g. OptimizerFitter)."
            )
        return self._strategy.basis_matrix(self._interp, vf_input, self._bkd)

    def __call__(self, vf_input: Array) -> Array:
        """Evaluate vector field.

        Parameters
        ----------
        vf_input : Array
            Shape ``(2, n)`` with row 0 = t, row 1 = x.

        Returns
        -------
        Array
            Shape ``(nqoi, n)``.
        """
        bkd = self._bkd
        t_row = vf_input[0, :]
        x_row = vf_input[1:2, :]
        n = vf_input.shape[1]

        # Fast path: single t
        if bkd.to_float(bkd.max(bkd.abs(t_row - t_row[0]))) < 1e-12:
            t_val = bkd.to_float(t_row[0])
            state = self._get_state(t_val)
            return self._strategy.evaluate(t_val, state, x_row, bkd)

        # General path: multiple unique t values
        result = bkd.zeros((self._nqoi, n))
        unique_t = bkd.unique(t_row)
        for ii in range(len(unique_t)):
            t_val = bkd.to_float(unique_t[ii])
            mask = bkd.abs(t_row - unique_t[ii]) < 1e-12
            indices = bkd.where(mask)[0]
            state = self._get_state(t_val)
            result[:, indices] = self._strategy.evaluate(
                t_val, state, x_row[:, indices], bkd,
            )
        return result

    def fit(
        self,
        samples: Array,
        values: Array,
        solver: Optional[LinearSystemSolverProtocol[Array]] = None,
    ) -> None:
        """Fit coefficients via least squares.

        Delegates to the coefficient strategy.

        Parameters
        ----------
        samples : Array
            VF input, shape ``(2, n_quad)``.
        values : Array
            Target values, shape ``(nqoi, n_quad)`` or ``(n_quad,)``.
        solver : LinearSystemSolverProtocol, optional
            If None, uses default ``LeastSquaresSolver``.
        """
        self._strategy.fit(self._interp, samples, values, self._bkd, solver)

        # Sync internal _coef for Kronecker backward compat
        if isinstance(self._strategy, KroneckerStrategy):
            self._coef = self._strategy.get_coefficients()
            if self._hyp_list is not None:
                self._hyp_list.set_values(self._bkd.flatten(self._coef))

    def jacobian_batch(self, vf_input: Array) -> Array:
        """Jacobian of VF output w.r.t. ``(t, x)``.

        Only the ``dv/dx`` entry (index 1) is used by
        ``compute_flow_density`` for divergence tracking.
        The ``dv/dt`` entry (index 0) is set to zero.

        Parameters
        ----------
        vf_input : Array
            Shape ``(2, n)``.

        Returns
        -------
        Array
            Shape ``(n, nqoi, 2)``.
        """
        bkd = self._bkd
        t_row = vf_input[0, :]
        x_row = vf_input[1:2, :]
        n = vf_input.shape[1]

        # Fast path: single t
        if bkd.to_float(bkd.max(bkd.abs(t_row - t_row[0]))) < 1e-12:
            t_val = bkd.to_float(t_row[0])
            state = self._get_state(t_val)
            dvdx = self._strategy.evaluate_deriv(
                t_val, state, x_row, bkd,
            )  # (n, nqoi)
            result = bkd.zeros((n, self._nqoi, 2))
            result[:, :, 1] = dvdx
            return result

        # General path: multiple unique t values
        result = bkd.zeros((n, self._nqoi, 2))
        unique_t = bkd.unique(t_row)

        for ii in range(len(unique_t)):
            t_val = bkd.to_float(unique_t[ii])
            mask = bkd.abs(t_row - unique_t[ii]) < 1e-12
            indices = bkd.where(mask)[0]

            state = self._get_state(t_val)
            dvdx = self._strategy.evaluate_deriv(
                t_val, state, x_row[:, indices], bkd,
            )  # (n_k, nqoi)
            result[indices, :, 1] = dvdx

        return result


def build_stieltjes_flow_vf(
    quad_data: FlowMatchingQuadData[Array],
    path: ProbabilityPathProtocol[Array],
    nterms: int,
    bkd: Backend[Array],
    nqoi: int = 1,
    ortho_tol: float = 1e-10,
    n_legendre: int = 1,
    interpolate_rcoefs: bool = False,
    strategy_factory: Optional[object] = None,
    per_slice: bool = False,
) -> StieltjesFlowVF[Array]:
    """Build a StieltjesFlowVF from quad data.

    Extracts unique t values from ``quad_data``, computes the interpolated
    state ``x_t = path.interpolate(t, x0, x1)`` at each t, and runs the
    Lanczos algorithm to build orthonormal polynomial bases.

    For ODE integration at arbitrary t, set ``interpolate_rcoefs=True``
    to use ``RecurrenceInterpolator`` (Lagrange interpolation of
    recurrence coefficients by default).  When ``interpolate_rcoefs=False``
    (the default), ``IdentityInterpolator`` is used and the ODE solver
    must step at exactly the training t nodes.

    Parameters
    ----------
    quad_data : FlowMatchingQuadData
        Must have a discrete t-axis with multiple (x0, x1) pairs per t.
    path : ProbabilityPathProtocol
        Probability path for interpolation.
    nterms : int
        Number of orthonormal polynomial basis terms.
    bkd : Backend[Array]
        Computational backend.
    nqoi : int
        Number of output dimensions.
    ortho_tol : float
        Lanczos orthonormality tolerance.
    n_legendre : int
        Number of Legendre time basis terms.  Only used when
        ``per_slice=False``.
    interpolate_rcoefs : bool
        If True, use ``RecurrenceInterpolator`` for smooth basis evolution
        at arbitrary t.  If False, use ``IdentityInterpolator`` (exact
        t-node matching only).
    strategy_factory : callable, optional
        Factory ``() -> ScalarInterp1DProtocol`` for creating per-coefficient
        interpolation strategies.  Only used when ``interpolate_rcoefs=True``.
        Defaults to ``LagrangeInterp1D``.
    per_slice : bool
        If True, use ``PerSliceStrategy`` (independent fit at each t_k).
        If False (default), use ``KroneckerStrategy`` (global Legendre
        time expansion).

    Returns
    -------
    StieltjesFlowVF
        Constructed vector field ready for fitting.

    Raises
    ------
    ValueError
        If quad_data has insufficient points per t value.
    """
    t_all = quad_data.t()[0, :]
    x0_all = quad_data.x0()
    x1_all = quad_data.x1()
    w_all = quad_data.weights()

    unique_t = bkd.unique(t_all)

    # Validate t-axis structure
    for ii in range(len(unique_t)):
        mask = bkd.abs(t_all - unique_t[ii]) < 1e-12
        count = int(bkd.to_float(bkd.sum(mask)))
        if count < nterms + 1:
            t_val = bkd.to_float(unique_t[ii])
            raise ValueError(
                f"Insufficient points per t value for Stieltjes with "
                f"nterms={nterms}. quad_data must be built from a tensor "
                f"product rule with a discrete t axis "
                f"(found {count} points at t={t_val:.6f}, "
                f"need at least {nterms + 1})."
            )

    factory = StieltjesBasisFactory(nterms, bkd, ortho_tol)
    t_vals_list: list[float] = []
    states: list[StieltjesBasisState[Array]] = []

    for ii in range(len(unique_t)):
        t_k = unique_t[ii]
        mask = bkd.abs(t_all - t_k) < 1e-12
        indices = bkd.where(mask)[0]

        t_slice = bkd.full((1, len(indices)), bkd.to_float(t_k))
        x0_slice = x0_all[:, indices]
        x1_slice = x1_all[:, indices]
        w_slice = w_all[indices]

        x_t_slice = path.interpolate(t_slice, x0_slice, x1_slice)
        x_t_1d = x_t_slice[0, :]

        state = factory.build(x_t_1d, w_slice)
        t_vals_list.append(bkd.to_float(t_k))
        states.append(state)

    t_arr = bkd.asarray(t_vals_list)

    if interpolate_rcoefs:
        interp: BasisInterpolator = RecurrenceInterpolator(
            bkd, strategy_factory=strategy_factory,
        )
    else:
        interp = IdentityInterpolator(bkd)

    interp.fit(t_arr, states)

    if per_slice:
        strategy: CoefficientStrategyType = PerSliceStrategy(bkd)
    else:
        strategy = KroneckerStrategy(bkd, n_legendre)

    return StieltjesFlowVF(interp, strategy, nqoi, bkd)
