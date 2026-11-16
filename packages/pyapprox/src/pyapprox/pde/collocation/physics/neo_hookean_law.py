r"""The compressible neo-Hookean law in the form collocation
hyperelasticity reads it.

The physics holds the Lame parameters as nodal coefficients and passes
their values to the law with the deformation gradient; the law holds no
material state. :class:`NeoHookeanLaw` satisfies the physics'
``HyperelasticLawProtocol`` and ``LameSensitiveLawProtocol`` by building
a :class:`~pyapprox.pde.constitutive.neo_hookean.NeoHookeanStress` from
the values on each call (it only stores them) and converting its
per-dimension results to the nested ``P[i][J]`` and ``A[i][J][k][L]``
forms.
"""

from typing import Generic, List

from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.util.backends.protocols import Array, Backend


class NeoHookeanLaw(Generic[Array]):
    r"""Compressible neo-Hookean law,
    :math:`P = \mu F + (\lambda \ln J - \mu) F^{-T}`, in 1D, 2D and 3D.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def stress(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        """``P[i][J]``, nested as ``F``."""
        model = NeoHookeanStress(lamda, mu)
        bkd = self._bkd
        ndim = len(F)
        if ndim == 1:
            return [[model.compute_stress_1d(F[0][0], bkd)]]
        if ndim == 2:
            P11, P12, P21, P22 = model.compute_stress_2d(
                F[0][0], F[0][1], F[1][0], F[1][1], bkd
            )
            return [[P11, P12], [P21, P22]]
        P3 = model.compute_stress_3d(tuple(tuple(row) for row in F), bkd)
        return [list(row) for row in P3]

    def tangent(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[List[List[Array]]]]:
        r""":math:`A_{iJkL} = \partial P_{iJ}/\partial F_{kL}`, nested
        ``A[i][J][k][L]``."""
        model = NeoHookeanStress(lamda, mu)
        bkd = self._bkd
        ndim = len(F)
        if ndim == 1:
            return [[[[model.compute_tangent_1d(F[0][0], bkd)]]]]
        if ndim == 2:
            A2 = model.compute_tangent_2d(F[0][0], F[0][1], F[1][0], F[1][1], bkd)
            return [
                [
                    [
                        [A2[f"A_{i + 1}{J + 1}{k + 1}{L + 1}"] for L in range(2)]
                        for k in range(2)
                    ]
                    for J in range(2)
                ]
                for i in range(2)
            ]
        A3 = model.compute_tangent_3d(tuple(tuple(row) for row in F), bkd)
        return [
            [[[A3[i, J, k, L] for L in range(3)] for k in range(3)] for J in range(3)]
            for i in range(3)
        ]

    def stress_mu_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        r""":math:`\partial P_{iJ}/\partial \mu`, nested as ``F``.

        Raises
        ------
        NotImplementedError
            In 3D.
        """
        model = NeoHookeanStress(lamda, mu)
        bkd = self._bkd
        ndim = len(F)
        if ndim == 1:
            return [[model.stress_sensitivity_mu_1d(F[0][0], bkd)]]
        if ndim == 2:
            d11, d12, d21, d22 = model.stress_sensitivity_mu_2d(
                F[0][0], F[0][1], F[1][0], F[1][1], bkd
            )
            return [[d11, d12], [d21, d22]]
        raise NotImplementedError(
            "the neo-Hookean mu sensitivity is implemented in 1D and 2D only"
        )

    def stress_lamda_sensitivity(
        self, F: List[List[Array]], mu: Array, lamda: Array
    ) -> List[List[Array]]:
        r""":math:`\partial P_{iJ}/\partial \lambda`, nested as ``F``.

        Raises
        ------
        NotImplementedError
            In 3D.
        """
        model = NeoHookeanStress(lamda, mu)
        bkd = self._bkd
        ndim = len(F)
        if ndim == 1:
            return [[model.stress_sensitivity_lamda_1d(F[0][0], bkd)]]
        if ndim == 2:
            d11, d12, d21, d22 = model.stress_sensitivity_lamda_2d(
                F[0][0], F[0][1], F[1][0], F[1][1], bkd
            )
            return [[d11, d12], [d21, d22]]
        raise NotImplementedError(
            "the neo-Hookean lamda sensitivity is implemented in 1D and 2D only"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
