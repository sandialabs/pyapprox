"""Bending-stiffness parameterization of the Euler-Bernoulli beam.

``BendingStiffnessApplyOnlyParameterization`` takes the bending
stiffness :math:`EI` of each element as its parameters. It is apply-only:
the beam has no typed assembly of its residual's derivative in
:math:`EI`, so the derivative bundle is empty.
"""

from typing import Generic, Tuple

import numpy as np

from pyapprox.pde.galerkin.physics.euler_bernoulli import EulerBernoulliBeamFEM
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


class BendingStiffnessApplyOnlyParameterization(Generic[Array]):
    r"""Apply-only parameterization of a beam by its element stiffnesses.

    The parameters are :math:`EI` on each element, shape
    ``(nelements,)``; ``apply`` hands them to the beam's ``set_EI``. A
    caller mapping its own inputs (a random field, a mixture rule) to
    stiffnesses applies them through a
    :class:`~pyapprox.pde.parameterizations.parameter_state.ParameterState`
    over this parameterization, so solves that key on the parameters'
    version see the change.

    Parameters
    ----------
    beam : EulerBernoulliBeamFEM
        The bound beam.
    bkd : Backend
        Computational backend.

    Raises
    ------
    TypeError
        If ``beam`` is not an ``EulerBernoulliBeamFEM``.
    """

    def __init__(self, beam: EulerBernoulliBeamFEM[Array], bkd: Backend[Array]) -> None:
        if not isinstance(beam, EulerBernoulliBeamFEM):
            raise TypeError(
                f"beam must be an EulerBernoulliBeamFEM, got {type(beam).__name__}"
            )
        self._beam = beam
        self._nelements = beam.basis().mesh().nelements()
        self._bkd = bkd
        self._derivs: ParamDerivatives[Array] = ParamDerivatives.none()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def targets(self) -> Tuple[object, ...]:
        """Return the objects whose coefficients ``apply`` writes."""
        return (self._beam,)

    def nparams(self) -> int:
        """Return the number of parameters, one per element."""
        return self._nelements

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the parameterized coefficient fields."""
        return ("EI",)

    def apply(self, params_1d: Array) -> None:
        """Set the beam's element stiffnesses to ``params_1d``, shape
        ``(nelements,)``.

        Raises
        ------
        ValueError
            If ``params_1d`` is not of shape ``(nelements,)``.
        """
        if tuple(params_1d.shape) != (self._nelements,):
            raise ValueError(
                f"params_1d must have shape ({self._nelements},), got "
                f"{tuple(params_1d.shape)}"
            )
        self._beam.set_EI(np.array(self._bkd.to_numpy(params_1d), dtype=float))

    def param_derivatives(self) -> ParamDerivatives[Array]:
        """Return the empty derivative capability bundle."""
        return self._derivs
