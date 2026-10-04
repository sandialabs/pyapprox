r"""The design objective: inference, relaxation and criterion together.

For design weights :math:`w`, the relaxation gives the independent-noise
variances :math:`\nu(w)`, the joint Gaussian gives the observation of one
target at :math:`(w, \nu(w))`, and the criterion scores it. The gradient is
the chain rule through :math:`\nu`, which depends on :math:`w_i` alone:

.. math::

    \frac{df}{dw_i} = \frac{\partial f}{\partial w_i}
    + \frac{\partial f}{\partial \nu_i}\,\frac{d\nu_i}{dw_i}.

This is the only place the three meet; each can be swapped independently.
"""

from typing import Generic

from pyapprox.expdesign.protocols.gaussian_criterion import (
    GaussianDesignCriterionProtocol,
)
from pyapprox.expdesign.protocols.relaxation import ObservationRelaxationProtocol
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.inverse.joint_gaussian import JointGaussian, LinearGaussianObservation
from pyapprox.util.backends.protocols import Array, Backend


class DesignObjective(Generic[Array]):
    """A criterion of one target, as a function of the design weights.

    Satisfies ``OEDObjectiveProtocol``: weights of shape (nobs, 1) give a
    value of shape (1, 1), and ``derivatives()`` declares a Jacobian of
    shape (1, nobs).

    Parameters
    ----------
    joint : JointGaussian[Array]
        Targets and noisy observations.
    relaxation : ObservationRelaxationProtocol[Array]
        Maps weights to independent-noise variances.
    criterion : GaussianDesignCriterionProtocol[Array]
        The quantity to minimize.
    index : int
        Which target block of ``joint`` the criterion scores.
    """

    def __init__(
        self,
        joint: JointGaussian[Array],
        relaxation: ObservationRelaxationProtocol[Array],
        criterion: GaussianDesignCriterionProtocol[Array],
        index: int,
    ) -> None:
        if not isinstance(joint, JointGaussian):
            raise TypeError(
                f"joint must be a JointGaussian, got {type(joint).__name__}"
            )
        if not isinstance(relaxation, ObservationRelaxationProtocol):
            raise TypeError(
                "relaxation must satisfy ObservationRelaxationProtocol, got "
                f"{type(relaxation).__name__}"
            )
        if not isinstance(criterion, GaussianDesignCriterionProtocol):
            raise TypeError(
                "criterion must satisfy GaussianDesignCriterionProtocol, got "
                f"{type(criterion).__name__}"
            )
        if relaxation.nobs() != joint.nobs():
            raise ValueError(
                f"relaxation has {relaxation.nobs()} observations but joint has "
                f"{joint.nobs()}"
            )
        ntargets = len(joint.target_sizes())
        if not 0 <= index < ntargets:
            raise ValueError(f"target {index} does not exist; there are {ntargets}")
        self._joint = joint
        self._relaxation = relaxation
        self._criterion = criterion
        self._index = index
        self._derivatives = Derivatives.first_order(jacobian=self._jacobian)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._joint.bkd()

    def nvars(self) -> int:
        """Number of design weights (= number of observations)."""
        return self._joint.nobs()

    def nqoi(self) -> int:
        """Always 1."""
        return 1

    def observation(self, weights: Array) -> LinearGaussianObservation[Array]:
        """The target seen through the relaxed observation at ``weights``."""
        return self._joint.observe(
            weights, self._relaxation.variances(weights), self._index
        )

    def __call__(self, design_weights: Array) -> Array:
        """The criterion. Weights (nobs, n) give values of shape (1, n)."""
        bkd = self.bkd()
        nobs = self.nvars()
        if design_weights.ndim != 2 or design_weights.shape[0] != nobs:
            raise ValueError(
                f"design_weights must have shape ({nobs}, n), got "
                f"{tuple(design_weights.shape)}"
            )
        values = [
            self._criterion.value(self.observation(design_weights[:, ii : ii + 1]))
            for ii in range(design_weights.shape[1])
        ]
        return bkd.reshape(bkd.hstack(values), (1, -1))

    def _jacobian(self, design_weights: Array) -> Array:
        """``(df/dw + df/dnu * dnu/dw)^T``. Shape: (1, nobs)"""
        dw, dnu = self._criterion.gradient(self.observation(design_weights))
        dnu_dw = self._relaxation.variances_jacobian_diagonal(design_weights)
        return self.bkd().reshape(dw + dnu * dnu_dw, (1, -1))

    def derivatives(self) -> Derivatives[Array]:
        """First-order bundle: the chain-rule Jacobian."""
        return self._derivatives
