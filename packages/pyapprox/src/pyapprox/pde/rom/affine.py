r"""Reduced residuals of systems affine in the state and the parameters.

When the raw residual is affine in the state :math:`u` and in the
parameters :math:`p`,

.. math:: F(u, p) = c + A u + \sum_q p_q (b_q + B_q u),

its projection onto a linear trial space
:math:`u = g + \bar{u} + V z` is fixed by a handful of reduced pieces,
computed once from ordinary evaluations of the full-order residual at a
reference parameter :math:`\bar{p}` and offset
:math:`o = g + \bar{u}`:

.. math::
    \hat{c} = V^T F(o, \bar{p}), \quad
    \hat{A} = V^T \frac{\partial F}{\partial u} V, \quad
    \hat{G} = V^T \frac{\partial F}{\partial p}(o), \quad
    T_{:, q, k} = V^T \Big(\frac{\partial F}{\partial p_q}(o + v_k)
                  - \frac{\partial F}{\partial p_q}(o)\Big),

after which every evaluation is :math:`r \times r` arithmetic,

.. math::
    \hat{F}(z, p) = \hat{c} + \hat{A} z + (\hat{G} + T z)(p - \bar{p}).

The parameter Jacobian :math:`\hat{G} + T z` comes out exactly as well,
which is what reduced adjoints need.

The construction is only correct when the affinity really holds, so it
is required rather than assumed: the spatial operator must declare
itself affine in the state (``StateDerivatives.affine``), the
parameterization affine in the parameters (``AffineInParamsProtocol``),
both time-invariant, and the trial space linear. A check from
``pyapprox.pde.rom.checks``, run on the result against the exact
projection, also catches a declaration that is wrong rather than
missing.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.parameterizations.protocol import (
    AffineInParamsProtocol,
    ParameterizationProtocol,
)
from pyapprox.pde.rom.lift import LiftProtocol
from pyapprox.pde.rom.projection import (
    ParameterizedResidualProtocol,
    ReducedResidualDerivatives,
)
from pyapprox.pde.rom.trial_space import LinearTrialSpaceProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ComposedParameterizedResidualProtocol(
    ParameterizedResidualProtocol[Array], Protocol
):
    """A raw residual that exposes the operator and parameterization
    it is built from, so their declarations can be read.

    Satisfied by ``GalerkinParameterizedResidual``.
    """

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return the operator whose residual this is."""
        ...

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """Return the parameterization applied before each evaluation."""
        ...


class AffineProjectedResidual(Generic[Array]):
    r"""A reduced residual evaluated from precomputed affine pieces.

    Satisfies ``ReducedResidualProtocol``. Built by
    :func:`fit_affine_projection`; every evaluation is independent of
    the full-order dimension.

    Parameters
    ----------
    constant : Array
        :math:`\hat{c}`. Shape: (ncodes,).
    state_matrix : Array
        :math:`\hat{A}`. Shape: (ncodes, ncodes).
    param_matrix : Array
        :math:`\hat{G}`. Shape: (ncodes, nparams).
    coupling : Array
        :math:`T`. Shape: (ncodes, nparams, ncodes).
    reference_params : Array
        :math:`\bar{p}`. Shape: (nparams,).
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        constant: Array,
        state_matrix: Array,
        param_matrix: Array,
        coupling: Array,
        reference_params: Array,
        bkd: Backend[Array],
    ) -> None:
        ncodes = int(constant.shape[0])
        nparams = int(reference_params.shape[0])
        expected = {
            "constant": ((ncodes,), constant),
            "state_matrix": ((ncodes, ncodes), state_matrix),
            "param_matrix": ((ncodes, nparams), param_matrix),
            "coupling": ((ncodes, nparams, ncodes), coupling),
        }
        for name, (shape, value) in expected.items():
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(value.shape)}"
                )
        self._constant = constant
        self._state_matrix = state_matrix
        self._param_matrix = param_matrix
        self._coupling = coupling
        self._reference_params = reference_params
        self._bkd = bkd
        self._derivs: ReducedResidualDerivatives[Array] = (
            ReducedResidualDerivatives.first_order(self._param_jacobian)
        )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        return int(self._constant.shape[0])

    def nparams(self) -> int:
        """Return the number of parameters."""
        return int(self._reference_params.shape[0])

    def reference_params(self) -> Array:
        r"""Return :math:`\bar{p}`. Shape: (nparams,)."""
        return self._reference_params

    def residual(self, codes: Array, time: float, params: Array) -> Array:
        """Return the reduced residual. Shape: (ncodes,)."""
        bkd = self._bkd
        shift = params - self._reference_params
        return (
            self._constant
            + bkd.dot(self._state_matrix, codes)
            + bkd.dot(self._param_jacobian(codes, time, params), shift)
        )

    def state_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        """Return its derivative in the codes. Shape: (ncodes, ncodes)."""
        shift = params - self._reference_params
        return self._state_matrix + self._bkd.einsum("iqk,q->ik", self._coupling, shift)

    def derivatives(self) -> ReducedResidualDerivatives[Array]:
        """Return the parameter Jacobian, always present."""
        return self._derivs

    def _param_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        return self._param_matrix + self._bkd.einsum("iqk,k->iq", self._coupling, codes)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ncodes={self.ncodes()}, nparams={self.nparams()})"
        )


def _require_affine(
    residual: ComposedParameterizedResidualProtocol[Array],
    trial_space: LinearTrialSpaceProtocol[Array],
    lift: LiftProtocol[Array],
) -> None:
    """Raise unless every declaration the affine form relies on holds."""
    if not isinstance(trial_space, LinearTrialSpaceProtocol):
        raise TypeError(
            "affine projection needs a constant basis; trial_space must "
            "satisfy LinearTrialSpaceProtocol, got "
            f"{type(trial_space).__name__}"
        )
    operator = residual.spatial_operator()
    if not operator.state_derivatives().affine:
        raise ValueError(
            "the spatial operator does not declare itself affine in the "
            "state (StateDerivatives.affine), so its projection cannot be "
            "precomputed; use ProjectedResidual or a hyper-reduced residual"
        )
    parameterization = residual.parameterization()
    if not (
        isinstance(parameterization, AffineInParamsProtocol)
        and parameterization.is_affine_in_params()
    ):
        raise ValueError(
            f"{type(parameterization).__name__} does not declare the "
            "residual affine in the parameters (AffineInParamsProtocol); "
            "a nonlinear field map needs hyper-reduction or an affine "
            "approximation of the field"
        )
    if not (operator.is_time_invariant() and lift.is_time_invariant()):
        raise ValueError(
            "affine projection precomputes at one time, so the operator "
            "and the lift must be declared time-invariant"
        )
    if residual.derivatives().param_jacobian is None:
        raise ValueError(
            "the parameter terms are read from the residual's parameter "
            "Jacobian, which this residual does not provide"
        )


def fit_affine_projection(
    residual: ComposedParameterizedResidualProtocol[Array],
    trial_space: LinearTrialSpaceProtocol[Array],
    lift: LiftProtocol[Array],
    reference_params: Array,
) -> AffineProjectedResidual[Array]:
    r"""Precompute the affine reduced pieces of a raw residual.

    Costs one residual, one Jacobian and ``ncodes + 1`` parameter
    Jacobians of the full-order system. The full-order parameter
    Jacobian is a dense (nstates, nparams) array, contracted with the
    basis as soon as it is formed.

    Parameters
    ----------
    residual : ComposedParameterizedResidualProtocol
        The raw residual, e.g. ``GalerkinParameterizedResidual``.
    trial_space : LinearTrialSpaceProtocol
        A linear trial space fitted to homogenized snapshots.
    lift : LiftProtocol
        The time-invariant lift.
    reference_params : Array
        :math:`\bar{p}`, where the operator is assembled; it must be a
        valid parameter (a positive diffusivity, say). Shape: (nparams,).

    Raises
    ------
    TypeError
        If the arguments do not satisfy their protocols.
    ValueError
        If an affinity or time-invariance declaration is missing.

    Notes
    -----
    To verify a fit, compare it with the exact projection afterwards,
    e.g. ``AgreementCheck(params).check(fitted,
    ProjectedResidual(residual, trial_space, lift))``. Value and Jacobian
    pin an affine form, so agreement at the offset for a few parameters
    verifies every piece.
    """
    if not isinstance(residual, ComposedParameterizedResidualProtocol):
        raise TypeError(
            "residual must satisfy ComposedParameterizedResidualProtocol, "
            f"got {type(residual).__name__}"
        )
    if not isinstance(lift, LiftProtocol):
        raise TypeError(f"lift must satisfy LiftProtocol, got {type(lift).__name__}")
    _require_affine(residual, trial_space, lift)
    param_jacobian = residual.derivatives().param_jacobian
    if param_jacobian is None:  # guaranteed above; narrows the type
        raise ValueError("the residual provides no parameter Jacobian")
    bkd = residual.bkd()
    ncodes = trial_space.ncodes()
    time = 0.0
    offset = lift.lift(time) + trial_space.reference()
    columns = trial_space.apply(bkd.eye(ncodes))

    constant = trial_space.apply_transpose(
        residual.residual(offset, time, reference_params)[:, None]
    )[:, 0]
    jac_columns = residual.state_jacobian(offset, time, reference_params) @ columns
    state_matrix = trial_space.apply_transpose(bkd.asarray(jac_columns))
    param_at_offset = trial_space.apply_transpose(
        param_jacobian(offset, time, reference_params)
    )
    coupling_columns = [
        trial_space.apply_transpose(
            param_jacobian(offset + columns[:, kk], time, reference_params)
        )
        - param_at_offset
        for kk in range(ncodes)
    ]
    reduced = AffineProjectedResidual(
        constant,
        state_matrix,
        param_at_offset,
        bkd.stack(coupling_columns, axis=2),
        bkd.copy(reference_params),
        bkd,
    )
    return reduced
