"""Linear elasticity: the collocation physics' Jacobian, and the sympy
manufactured-solution pipeline checked by hand in 3D.

Solves against manufactured solutions are verified by the collocation
linear-elasticity catalog
(``tests/pde/models/collocation/test_verification_linear_elasticity.py``).
"""

from typing import Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics import LinearElasticityPhysics
from pyapprox.pde.manufactured import (
    ManufacturedLinearElasticityEquations,
)
from pyapprox.util.backends.protocols import Array, Backend


class _Residual(Generic[Array]):
    """The physics residual at time 0 as a function of the state, for
    DerivativeChecker: samples ``(nstates, 1)``, values ``(nstates, 1)``."""

    def __init__(self, physics: LinearElasticityPhysics[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nvars(self) -> int:
        return self._physics.nstates()

    def nqoi(self) -> int:
        return self._physics.nstates()

    def __call__(self, sample: Array) -> Array:
        return self._physics.residual(sample[:, 0], 0.0)[:, None]

    def jacobian(self, sample: Array) -> Array:
        return self._physics.jacobian(sample[:, 0], 0.0)

    def derivatives(self) -> Derivatives[Array]:
        return Derivatives.first_order(jacobian=self.jacobian)


@pytest.mark.parametrize(
    "lamda, mu",
    [(1.0, 1.0), (2.5, 0.5), (0.5, 2.5)],
    ids=["equal", "high_lambda", "high_mu"],
)
def test_jacobian_2d(bkd: Backend[Array], lamda: float, mu: float) -> None:
    """The 2D state Jacobian on 6 x 6 nodes against finite differences:
    error ratio at most 1e-5."""
    mesh = TransformedMesh2D(6, 6, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    physics = LinearElasticityPhysics(basis, bkd, lamda=lamda, mu=mu)
    nstates = physics.nstates()
    sample = bkd.asarray([[0.1 * float(i) for i in range(nstates)]]).T
    checker = DerivativeChecker(_Residual(physics))
    errors = checker.check_derivatives(sample, verbosity=0)
    assert checker.error_ratio(errors[0]) <= 1e-5


def test_ncomponents(bkd: Backend[Array]) -> None:
    """The 2D physics has two displacement components, stacked."""
    basis = ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
    physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)
    assert physics.ncomponents() == 2
    assert physics.nstates() == 2 * basis.npts()


class TestManufacturedLinearElasticity3DSympy:
    """Hand-checkable verification of the 3D sympy pipeline."""

    def test_uniaxial_quadratic_forcing(self, bkd: Backend[Array]) -> None:
        """u = (x**2, 0, 0): sigma_xx = (lam+2mu)*2x, sigma_yy = sigma_zz
        = 2*lam*x, so f = -div(sigma) = (-(2lam+4mu), 0, 0)."""
        lam, mu = 2.0, 3.0
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["x**2", "0", "0"],
            nvars=3,
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=bkd,
            oned=True,
        )
        pts = bkd.asarray([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8], [0.3, 0.6, 0.7]])
        forcing = man_sol.functions["forcing"](pts)  # (npts, 3)
        expected_fx = -(2.0 * lam + 4.0 * mu)
        bkd.assert_allclose(forcing[:, 0], bkd.full((3,), expected_fx), rtol=1e-12)
        bkd.assert_allclose(forcing[:, 1], bkd.zeros((3,)), atol=1e-14)
        bkd.assert_allclose(forcing[:, 2], bkd.zeros((3,)), atol=1e-14)

    def test_cross_coupled_solution_zero_forcing(self, bkd: Backend[Array]) -> None:
        """u = (y*z, x*z, x*y): all diagonal strains vanish and the shear
        stresses 2mu*(z, y, x) are divergence-free, so f = 0."""
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["y*z", "x*z", "x*y"],
            nvars=3,
            lambda_str="2.5",
            mu_str="1.5",
            bkd=bkd,
            oned=True,
        )
        pts = bkd.asarray([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8], [0.3, 0.6, 0.7]])
        forcing = man_sol.functions["forcing"](pts)
        bkd.assert_allclose(forcing, bkd.zeros((3, 3)), atol=1e-14)

    def test_traction_3d(self, bkd: Backend[Array]) -> None:
        """Traction on an x-normal face for u = (x**2, 0, 0) is
        ((lam+2mu)*2x, 0, 0)."""
        lam, mu = 2.0, 3.0
        man_sol = ManufacturedLinearElasticityEquations(
            sol_strs=["x**2", "0", "0"],
            nvars=3,
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=bkd,
            oned=True,
        )
        pts = bkd.asarray([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8], [0.3, 0.6, 0.7]])
        normals = bkd.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        traction = man_sol.traction_values(pts, normals)  # (npts, 3)
        expected_tx = (lam + 2.0 * mu) * 2.0 * pts[0, :]
        bkd.assert_allclose(traction[:, 0], expected_tx, rtol=1e-12)
        bkd.assert_allclose(traction[:, 1:], bkd.zeros((3, 2)), atol=1e-14)

    def test_invalid_nvars_raises(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="nvars in"):
            ManufacturedLinearElasticityEquations(
                sol_strs=["x", "y", "z", "x"],
                nvars=4,
                lambda_str="1.0",
                mu_str="1.0",
                bkd=bkd,
            )
