r"""Integrated-density state terms and the radial-displacement observation.

Setups:

- 1D: 15 Chebyshev nodes on :math:`[0, 2]`, :math:`E = 3`,
  :math:`\nu = 0.3`; the material is affine in three parameters,
  :math:`m(p) = m_0 + M p` with :math:`M` 5% random multiples of
  :math:`m_0` (seed 0), at :math:`p = (0.3, -0.2, 0.5)`. Displacement
  :math:`u = 0.05 x + 0.01 \sin(2 x)` plus noise of size 1e-3 (seed 1).
  Linear and neo-Hookean strain energies over the whole bar and the linear
  stress averaged over :math:`[0.4, 1.5]` (:math:`s = 1/1.1`).
- 2D: 5 x 5 nodes on :math:`[0, 1]^2`, the curvilinear basis a rotation
  by :math:`\theta = \operatorname{atan2}(y + 1, x + 1)` at each node, a
  displacement of size 1e-2 (seed 2), Lame fields affine in three
  parameters about :math:`\mu = 1.5, \lambda = 2`, positive weights
  (seed 3) and area 0.7. Linear and hyperelastic hoop stress and strain
  energy.

Tests: :math:`Q_u` and :math:`Q_p` by ``DerivativeChecker``, central
differences (minimum relative error below 1e-6), for every 1D density
and both 2D post-processors; derivation-free values on the bar at
:math:`p = 0` (:math:`u = e x`, :math:`e = 0.02`): the linear energy
:math:`\tfrac12 E e^2 \cdot 2`, the averaged linear stress :math:`E e`,
and zero neo-Hookean energy at zero displacement (1e-12); the neo-Hookean
stress density is the energy's derivative in :math:`F` and its
:math:`F`-derivative the second (central differences, 1e-8); the average
radial displacement is the identity for a pure radial field; every term
and observation survives a pickle round trip; refusals.
"""

import pickle
from typing import Callable, Generic, List, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.functional_terms import LinearizationProtocol
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.functionals.density_terms import (
    GradientDensity1D,
    HoopStressDensity,
    IntegratedDensity,
    StrainEnergyDensity2D,
    gradient_density_term_1d,
)
from pyapprox.pde.collocation.functionals.observations import (
    average_radial_displacement_observation,
)
from pyapprox.pde.collocation.functionals.strain_energy_1d import (
    GradientDensity1DProtocol,
    LinearStrainEnergyDensity1D,
    LinearStressDensity1D,
    NeoHookeanStrainEnergyDensity1D,
    NeoHookeanStressDensity1D,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.collocation.post_processing.stress import (
    HyperelasticStressPostProcessor2D,
    StressPostProcessor2D,
)
from pyapprox.util.backends.protocols import Array, Backend

_NPARAMS = 3
_PARAM = [0.3, -0.2, 0.5]


class _AffineMap(Generic[Array]):
    """:math:`m(p) = m_0 + M p`, :math:`M` 5% random multiples of
    :math:`m_0`."""

    def __init__(self, bkd: Backend[Array], base: Array) -> None:
        rng = np.random.default_rng(0)
        self._base = base
        self._modes = bkd.asarray(
            0.05 * rng.normal(size=(base.shape[0], _NPARAMS))
        ) * bkd.reshape(base, (-1, 1))

    def nvars(self) -> int:
        return _NPARAMS

    def __call__(self, params_1d: Array) -> Array:
        return self._base + self._modes @ params_1d

    def jacobian(self, params_1d: Array) -> Array:
        return self._modes


class _Parameters(Generic[Array]):
    """The parameters currently set, as a forward model's would be."""

    def __init__(self, param: Array) -> None:
        self._param = param

    def set(self, param: Array) -> None:
        self._param = param

    def current(self) -> Array:
        return self._param


def _row(
    bkd: Backend[Array], linearization: LinearizationProtocol[Array, Array]
) -> Array:
    """A scalar term's derivative as a row, through its transpose."""
    return linearization.apply_transpose(bkd.ones((1, 1))).T


def _first(
    bkd: Backend[Array], term: IntegratedDensity[Array], state: Array
) -> Tuple[Array, Array]:
    """``(Q_u, Q_p)`` as rows."""
    bundle = term.derivatives()
    assert bundle.wrt_state is not None and bundle.wrt_param is not None
    wrt_param = bundle.wrt_param(state)
    assert wrt_param is not None
    return _row(bkd, bundle.wrt_state(state)), _row(bkd, wrt_param)


def _min_error(
    bkd: Backend[Array],
    sample: Array,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
) -> float:
    # Steps from 1e-2 down: a larger state step can drive the deformation
    # gradient 1 + u' negative, where the neo-Hookean log is undefined.
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=1, nvars=sample.shape[0], fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(
        sample,
        fd_eps=bkd.flip(bkd.logspace(-13, -2, 12)),
        relative=True,
        central=True,
    )[0]
    return bkd.to_float(bkd.min(errors))


def _check_derivatives(
    bkd: Backend[Array],
    term: IntegratedDensity[Array],
    parameters: _Parameters[Array],
    state: Array,
    param: Array,
) -> None:
    """Q_u along the state and Q_p along the parameters, by central
    differences."""

    def of_state(u: Array) -> Array:
        return term.value(u)

    def state_row(u: Array) -> Array:
        return _first(bkd, term, u)[0]

    def of_param(p: Array) -> Array:
        parameters.set(p[:, 0])
        return term.value(state)

    def param_row(p: Array) -> Array:
        parameters.set(p[:, 0])
        return _first(bkd, term, state)[1]

    assert _min_error(bkd, state, of_state, state_row) < 1e-6
    assert _min_error(bkd, param[:, None], of_param, param_row) < 1e-6
    parameters.set(param)


class _Bar(Generic[Array]):
    """The 1D bar of the module docstring."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        mesh = TransformedMesh1D(15, bkd, AffineTransform1D((0.0, 2.0), bkd))
        self.basis = ChebyshevBasis1D(mesh, bkd)
        x = mesh.points()[0]
        noise = np.random.default_rng(1).normal(0.0, 1e-3, 15)
        self.state = bkd.reshape(
            0.05 * x + 0.01 * bkd.sin(2.0 * x) + bkd.asarray(noise), (15, 1)
        )
        self.param = bkd.asarray(_PARAM)
        self.parameters = _Parameters(self.param)
        nu, young = 0.3, 3.0
        mu = young / (2.0 * (1.0 + nu))
        lamda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.modulus = _AffineMap(bkd, bkd.full((15,), young))
        self.lame = _AffineMap(
            bkd, bkd.concatenate([bkd.full((15,), mu), bkd.full((15,), lamda)])
        )

    def term(
        self,
        density: GradientDensity1DProtocol[Array],
        material: _AffineMap[Array],
        bounds: Tuple[float, float] | None = None,
        scale: float = 1.0,
    ) -> IntegratedDensity[Array]:
        return gradient_density_term_1d(
            self.basis,
            density,
            material,
            self.parameters,
            self.bkd,
            bounds=bounds,
            scale=scale,
        )

    def material(self, density: GradientDensity1DProtocol[Array]) -> _AffineMap[Array]:
        """The modulus for a one-field density, the Lame pair otherwise."""
        return self.modulus if density.nfields() == 1 else self.lame


class _Plate(Generic[Array]):
    """The 2D plate of the module docstring."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        mesh = TransformedMesh2D(5, 5, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        self.npts = basis.npts()
        points = mesh.points()
        theta = bkd.arctan2(points[1] + 1.0, points[0] + 1.0)
        cos, sin = bkd.cos(theta), bkd.sin(theta)
        self.curvilinear = bkd.stack(
            [bkd.stack([cos, sin], axis=1), bkd.stack([-sin, cos], axis=1)], axis=1
        )
        self.dx = basis.derivative_matrix(1, 0)
        self.dy = basis.derivative_matrix(1, 1)
        self.state = bkd.asarray(
            np.random.default_rng(2).normal(0.0, 1e-2, (2 * self.npts, 1))
        )
        self.weights = bkd.asarray(
            np.random.default_rng(3).uniform(0.5, 1.5, self.npts)
        )
        self.area = 0.7
        self.param = bkd.asarray(_PARAM)
        self.parameters = _Parameters(self.param)
        self.lame = _AffineMap(
            bkd,
            bkd.concatenate([bkd.full((self.npts,), 1.5), bkd.full((self.npts,), 2.0)]),
        )

    def linear(self) -> StressPostProcessor2D[Array]:
        return StressPostProcessor2D(
            self.dx, self.dy, self.bkd, curvilinear_basis=self.curvilinear
        )

    def hyperelastic(self) -> HyperelasticStressPostProcessor2D[Array]:
        return HyperelasticStressPostProcessor2D(
            self.dx, self.dy, NeoHookeanLaw(self.bkd), self.bkd, self.curvilinear
        )


_DENSITIES_1D: List[Callable[[Backend[Array]], GradientDensity1DProtocol[Array]]] = [
    LinearStrainEnergyDensity1D,
    LinearStressDensity1D,
    NeoHookeanStrainEnergyDensity1D,
    NeoHookeanStressDensity1D,
]


class TestDerivatives:
    @pytest.mark.parametrize("density_cls", _DENSITIES_1D)
    def test_1d_densities(
        self,
        bkd: Backend[Array],
        density_cls: Callable[[Backend[Array]], GradientDensity1DProtocol[Array]],
    ) -> None:
        """Over the whole bar, and the stresses also averaged over
        [0.4, 1.5]."""
        bar = _Bar(bkd)
        density = density_cls(bkd)
        terms = [bar.term(density, bar.material(density))]
        if density_cls in (LinearStressDensity1D, NeoHookeanStressDensity1D):
            terms.append(
                bar.term(
                    density, bar.material(density), bounds=(0.4, 1.5), scale=1 / 1.1
                )
            )
        for term in terms:
            _check_derivatives(bkd, term, bar.parameters, bar.state, bar.param)

    @pytest.mark.parametrize("kind", ["linear", "hyperelastic"])
    def test_2d_hoop_stress_and_strain_energy(
        self, bkd: Backend[Array], kind: str
    ) -> None:
        plate = _Plate(bkd)
        source = plate.linear() if kind == "linear" else plate.hyperelastic()
        for term in (
            IntegratedDensity(
                bkd,
                HoopStressDensity(bkd, source),
                plate.lame,
                plate.parameters,
                plate.weights / plate.area,
            ),
            IntegratedDensity(
                bkd,
                StrainEnergyDensity2D(bkd, source),
                plate.lame,
                plate.parameters,
                plate.weights,
            ),
        ):
            _check_derivatives(bkd, term, plate.parameters, plate.state, plate.param)


class TestValues:
    """Derivation-free, on the bar at p = 0 (E = 3)."""

    def _uniform_strain(self, bar: _Bar[Array], strain: float) -> Array:
        x = bar.basis.mesh().points()[0]
        return bar.bkd.reshape(strain * x, (15, 1))

    def test_linear_energy_and_stress(self, bkd: Backend[Array]) -> None:
        bar = _Bar(bkd)
        bar.parameters.set(bkd.zeros((_NPARAMS,)))
        state = self._uniform_strain(bar, 0.02)
        energy = bar.term(LinearStrainEnergyDensity1D(bkd), bar.modulus)
        bkd.assert_allclose(
            energy.value(state), bkd.asarray([[0.5 * 3.0 * 0.02**2 * 2.0]]), rtol=1e-12
        )
        stress = bar.term(
            LinearStressDensity1D(bkd), bar.modulus, bounds=(0.4, 1.5), scale=1 / 1.1
        )
        bkd.assert_allclose(
            stress.value(state), bkd.asarray([[3.0 * 0.02]]), rtol=1e-12
        )

    def test_neo_hookean_energy_vanishes_at_rest(self, bkd: Backend[Array]) -> None:
        bar = _Bar(bkd)
        energy = bar.term(NeoHookeanStrainEnergyDensity1D(bkd), bar.lame)
        bkd.assert_allclose(
            energy.value(bkd.zeros((15, 1))), bkd.zeros((1, 1)), atol=1e-12
        )

    def test_neo_hookean_stress_is_energy_derivative(self, bkd: Backend[Array]) -> None:
        """The stress density is d(psi)/dF of the energy density, and its
        F-derivative the energy's second derivative (central differences
        in F)."""
        energy = NeoHookeanStrainEnergyDensity1D(bkd)
        stress = NeoHookeanStressDensity1D(bkd)
        F = bkd.asarray([0.8, 1.0, 1.3])
        material = bkd.asarray([1.2, 0.7, 2.0, 0.5, 1.5, 0.9])
        h = 1e-6
        bkd.assert_allclose(
            stress.value(F, material),
            (energy.value(F + h, material) - energy.value(F - h, material)) / (2.0 * h),
            rtol=1e-8,
            atol=1e-9,
        )
        bkd.assert_allclose(
            stress.arg_derivative(F, material),
            (stress.value(F + h, material) - stress.value(F - h, material)) / (2.0 * h),
            rtol=1e-8,
            atol=1e-9,
        )


class TestRadialDisplacement:
    def _angles(self, bkd: Backend[Array]) -> Tuple[List[int], Array, Array]:
        indices = [0, 4, 7, 12]
        theta = bkd.asarray([0.1, 0.9, 1.7, 2.6])
        return indices, bkd.cos(theta), bkd.sin(theta)

    def test_a_radial_field(self, bkd: Backend[Array]) -> None:
        """Derivation-free: u = r_k (cos, sin) at each node averages to the
        mean of r_k."""
        indices, cos, sin = self._angles(bkd)
        npts = 16
        radii = [0.5, 1.0, 1.5, 2.0]
        ux, uy = [0.0] * npts, [0.0] * npts
        for k, index in enumerate(indices):
            ux[index] = radii[k] * bkd.to_float(cos[k])
            uy[index] = radii[k] * bkd.to_float(sin[k])
        state = bkd.reshape(bkd.asarray(ux + uy), (2 * npts, 1))
        observation = average_radial_displacement_observation(
            bkd, indices, cos, sin, npts
        )
        bkd.assert_allclose(
            observation.apply(state), bkd.asarray([[sum(radii) / 4]]), rtol=1e-14
        )


class TestPickle:
    def test_terms_and_observations(self, bkd: Backend[Array]) -> None:
        bar, plate = _Bar(bkd), _Plate(bkd)
        term = bar.term(NeoHookeanStrainEnergyDensity1D(bkd), bar.lame)
        copy = pickle.loads(pickle.dumps(term))
        bkd.assert_allclose(copy.value(bar.state), term.value(bar.state), rtol=0.0)
        hoop = IntegratedDensity(
            bkd,
            HoopStressDensity(bkd, plate.hyperelastic()),
            plate.lame,
            plate.parameters,
            plate.weights,
        )
        copy = pickle.loads(pickle.dumps(hoop))
        bkd.assert_allclose(copy.value(plate.state), hoop.value(plate.state), rtol=0.0)
        observation = average_radial_displacement_observation(
            bkd, [0, 3], bkd.ones((2,)), bkd.zeros((2,)), plate.npts
        )
        restored = pickle.loads(pickle.dumps(observation))
        bkd.assert_allclose(
            restored.apply(plate.state), observation.apply(plate.state), rtol=0.0
        )


class TestRefusals:
    def test_term_arguments(self, bkd: Backend[Array]) -> None:
        bar = _Bar(bkd)
        density = GradientDensity1D(bar.basis, LinearStrainEnergyDensity1D(bkd))
        with pytest.raises(TypeError, match="density must satisfy"):
            IntegratedDensity(
                bkd,
                object(),  # type: ignore[arg-type]
                bar.modulus,
                bar.parameters,
                bkd.ones((15,)),
            )
        with pytest.raises(TypeError, match="parameters must satisfy"):
            IntegratedDensity(
                bkd,
                density,
                bar.modulus,
                object(),  # type: ignore[arg-type]
                bkd.ones((15,)),
            )
        with pytest.raises(ValueError, match="weights must have shape"):
            IntegratedDensity(
                bkd, density, bar.modulus, bar.parameters, bkd.ones((14,))
            )
        term = IntegratedDensity(
            bkd, density, bar.lame, bar.parameters, bkd.ones((15,))
        )
        with pytest.raises(ValueError, match="nfields\\*npts = 15"):
            term.value(bar.state)
        with pytest.raises(ValueError, match="increasing"):
            gradient_density_term_1d(
                bar.basis,
                LinearStrainEnergyDensity1D(bkd),
                bar.modulus,
                bar.parameters,
                bkd,
                bounds=(1.0, 0.5),
            )

    def test_radial_displacement(self, bkd: Backend[Array]) -> None:
        ones = bkd.ones((2,))
        with pytest.raises(ValueError, match="at least one node"):
            average_radial_displacement_observation(bkd, [], ones[:0], ones[:0], 4)
        with pytest.raises(ValueError, match="not all in"):
            average_radial_displacement_observation(bkd, [0, 4], ones, ones, 4)
        with pytest.raises(ValueError, match="sin_theta must have shape"):
            average_radial_displacement_observation(bkd, [0, 1], ones, ones[:1], 4)
