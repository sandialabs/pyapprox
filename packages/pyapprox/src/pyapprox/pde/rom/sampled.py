r"""Reduced residuals evaluated on a sample of elements.

The exact projection :math:`V^T F(o + V z)` is a sum of element
contributions, :math:`\sum_e V_e^T F_e(o_e + V_e z)`, over every
element. Hyper-reduction keeps a few elements and reduces each one with
its own matrix,

.. math:: \hat{F}(z, t, p) = \sum_{e \in \tilde{E}}
          C_e F_e(o_e + V_e z, t, p),

where :math:`V_e` and :math:`o_e` are the trial basis and offset on the
element's DOFs. ECSW uses :math:`C_e = \xi_e V_e^T` with non-negative
weights; DEIM would supply its own :math:`C_e`. Which elements and which
:math:`C_e` is a fitting decision; evaluating the sum is the same for
every method, so it lives here once, in :class:`ElementSampledResidual`.

Everything is element-local: the basis and offset are taken on the
selected elements' DOFs once, and each evaluation calls the restricted
element operator on local states. The one global step left is applying
the parameters, which today writes the whole coefficient field.
"""

from typing import Generic, List

from pyapprox.pde.elementwise import (
    RestrictedElementOperatorProtocol,
    check_restricted_operator,
)
from pyapprox.pde.parameterizations.protocol import ParameterizationProtocol
from pyapprox.pde.rom.lift import LiftProtocol
from pyapprox.pde.rom.projection import ReducedResidualDerivatives
from pyapprox.pde.rom.trial_space import LinearTrialSpaceProtocol
from pyapprox.util.backends.protocols import Array, Backend


class ElementSampledResidual(Generic[Array]):
    r"""A reduced residual summed over a sample of elements.

    Satisfies ``ReducedResidualProtocol``:

    .. math::
        \hat{F} = \sum_e C_e F_e, \qquad
        \frac{\partial \hat{F}}{\partial z} = \sum_e C_e J_e V_e.

    The parameter Jacobian is absent: it needs element-level parameter
    derivatives, which are not yet supplied.

    Parameters
    ----------
    restricted : RestrictedElementOperatorProtocol
        The element operator over the sampled elements.
    parameterization : ParameterizationProtocol
        Applied before each evaluation.
    trial_space : LinearTrialSpaceProtocol
        Its rows on the sampled elements' DOFs are taken once.
    lift : LiftProtocol
        Must be time-invariant; it is taken once on those DOFs.
    reductions : Array
        :math:`C_e`. Shape: (nselected, ncodes, nrow).

    Raises
    ------
    ValueError
        If the lift is time-dependent or the shapes disagree.
    """

    def __init__(
        self,
        restricted: RestrictedElementOperatorProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        trial_space: LinearTrialSpaceProtocol[Array],
        lift: LiftProtocol[Array],
        reductions: Array,
    ) -> None:
        check_restricted_operator(restricted)
        if not isinstance(trial_space, LinearTrialSpaceProtocol):
            raise TypeError(
                "trial_space must satisfy LinearTrialSpaceProtocol, got "
                f"{type(trial_space).__name__}"
            )
        if not lift.is_time_invariant():
            raise ValueError(
                "the lift is taken once on the sampled DOFs, so it must be "
                "time-invariant"
            )
        bkd = restricted.bkd()
        global_dofs = restricted.global_dofs()
        dof_list: List[int] = [int(dof) for dof in bkd.to_numpy(global_dofs)]
        basis_local = trial_space.rows(dof_list)
        offset_local = (lift.lift(0.0) + trial_space.reference())[global_dofs]
        self._row_dofs = restricted.element_row_dofs()
        self._state_dofs = restricted.element_state_dofs()
        nselected = int(self._row_dofs.shape[0])
        ncodes = trial_space.ncodes()
        expected = (nselected, ncodes, int(self._row_dofs.shape[1]))
        if tuple(reductions.shape) != expected:
            raise ValueError(
                f"reductions must have shape {expected}, got {tuple(reductions.shape)}"
            )
        self._restricted = restricted
        self._parameterization = parameterization
        self._bkd = bkd
        self._basis_local = basis_local
        self._offset_local = offset_local
        self._basis_state = basis_local[self._state_dofs]
        self._reductions = reductions
        self._ncodes = ncodes

    @classmethod
    def with_weights(
        cls,
        restricted: RestrictedElementOperatorProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        trial_space: LinearTrialSpaceProtocol[Array],
        lift: LiftProtocol[Array],
        weights: Array,
    ) -> "ElementSampledResidual[Array]":
        r"""The ECSW form, :math:`C_e = \xi_e V_e^T` on the rows.

        ``weights`` has shape (nselected,); all ones over every element
        gives the exact projection.
        """
        check_restricted_operator(restricted)
        bkd = restricted.bkd()
        global_dofs = restricted.global_dofs()
        dof_list: List[int] = [int(dof) for dof in bkd.to_numpy(global_dofs)]
        basis_rows = trial_space.rows(dof_list)[restricted.element_row_dofs()]
        reductions = weights[:, None, None] * bkd.transpose(basis_rows, (0, 2, 1))
        return cls(restricted, parameterization, trial_space, lift, reductions)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def ncodes(self) -> int:
        """Return the reduced dimension."""
        return self._ncodes

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def restricted(self) -> RestrictedElementOperatorProtocol[Array]:
        """Return the element operator over the sampled elements."""
        return self._restricted

    def _local_states(self, codes: Array, params: Array) -> Array:
        self._parameterization.apply(params)
        local = self._offset_local + self._bkd.dot(self._basis_local, codes)
        return local[self._state_dofs]

    def element_contributions(self, codes: Array, time: float, params: Array) -> Array:
        r"""Return :math:`C_e F_e` per element. Shape: (nselected, ncodes)."""
        residuals = self._restricted.element_residuals(
            self._local_states(codes, params), time
        )
        return self._bkd.einsum("erk,ek->er", self._reductions, residuals)

    def element_jacobian_contributions(
        self, codes: Array, time: float, params: Array
    ) -> Array:
        r"""Return :math:`C_e J_e V_e` per element.

        Shape: (nselected, ncodes, ncodes).
        """
        # The action on the basis rows, k = ncodes directions: J_e is never
        # formed when the reduced dimension is below the element's size.
        applied = self._restricted.element_jacobian_actions(
            self._local_states(codes, params), time, self._basis_state
        )
        return self._bkd.einsum("erk,eks->ers", self._reductions, applied)

    def residual(self, codes: Array, time: float, params: Array) -> Array:
        """Return the reduced residual. Shape: (ncodes,)."""
        return self._bkd.sum(self.element_contributions(codes, time, params), axis=0)

    def state_jacobian(self, codes: Array, time: float, params: Array) -> Array:
        """Return its derivative in the codes. Shape: (ncodes, ncodes)."""
        return self._bkd.sum(
            self.element_jacobian_contributions(codes, time, params), axis=0
        )

    def derivatives(self) -> ReducedResidualDerivatives[Array]:
        """No parameter Jacobian yet: element-level parameter derivatives
        are not supplied."""
        return ReducedResidualDerivatives.none()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nselected={int(self._row_dofs.shape[0])}, "
            f"ncodes={self._ncodes})"
        )
