"""Binding by membership: a parameterization's targets must belong to
what is solved."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC, RobinBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.ownership import owns
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.composite import (
    CompositeParameterization,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Backend


def _zero(x: NDArray[Any]) -> NDArray[Any]:
    return np.zeros(x.shape[1])


def _physics(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    basis = LagrangeBasis(StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), 1)
    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=1.0,
        bkd=bkd,
        forcing=TimeIndependent(_zero),
        boundary_conditions=[
            DirichletBC(basis, "left", 0.0, bkd),
            RobinBC(basis, "right", alpha=1.0, value_func=0.0, bkd=bkd),
        ],
    )


class _Param:
    """A parameterization writing ``coefficient`` of each target."""

    def __init__(self, targets: Tuple[object, ...], coefficient: str) -> None:
        self._targets = targets
        self._coefficient = coefficient

    def nparams(self) -> int:
        return 1

    def targets(self) -> Tuple[object, ...]:
        return self._targets

    def owned_coefficients(self) -> Tuple[str, ...]:
        return (self._coefficient,)

    def apply(self, params_1d: Any) -> None:
        pass

    def param_derivatives(self) -> ParamDerivatives[Any]:
        return ParamDerivatives()


class TestOwnership:
    def test_physics_owns_itself_and_its_bcs(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        assert owns(physics, physics)
        for bc in physics._boundary_conditions:
            assert owns(physics, bc)
        assert not owns(physics, _physics(bkd))

    def test_system_owns_its_interior_and_terms(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        system = physics.system()
        assert owns(system, physics)
        for term in physics.weak_form_bcs():
            assert owns(system, term)
        assert not owns(system, _physics(bkd))

    def test_object_without_owns_owns_only_itself(self) -> None:
        owner = object()
        assert owns(owner, owner)
        assert not owns(owner, object())


class TestRequireOwnedTargets:
    def test_parameterization_of_the_interior_is_accepted_by_its_system(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        require_owned_targets(_Param((physics,), "diffusion"), physics)
        require_owned_targets(
            _Param((physics,), "diffusion"), physics.system()
        )

    def test_foreign_physics_is_rejected(self, bkd: Backend[Array]) -> None:
        physics, foreign = _physics(bkd), _physics(bkd)
        with pytest.raises(ValueError, match="does not hold"):
            require_owned_targets(_Param((foreign,), "diffusion"), physics)
        with pytest.raises(ValueError, match="does not hold"):
            require_owned_targets(
                _Param((foreign,), "diffusion"), physics.system()
            )


class TestCompositeTargets:
    def test_two_terms_with_the_same_coefficient_name_are_accepted(
        self, bkd: Backend[Array]
    ) -> None:
        basis = LagrangeBasis(
            StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), 1
        )
        left = RobinBC(basis, "left", alpha=1.0, value_func=0.0, bkd=bkd)
        right = RobinBC(basis, "right", alpha=1.0, value_func=0.0, bkd=bkd)
        composite = CompositeParameterization(
            [_Param((left,), "alpha"), _Param((right,), "alpha")], bkd
        )
        assert composite.targets() == (left, right)

    def test_same_coefficient_of_the_same_target_is_rejected(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        with pytest.raises(ValueError, match="'diffusion' of the same"):
            CompositeParameterization(
                [
                    _Param((physics,), "diffusion"),
                    _Param((physics,), "diffusion"),
                ],
                bkd,
            )

    def test_targets_are_deduplicated_by_identity(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        composite = CompositeParameterization(
            [
                _Param((physics,), "diffusion"),
                _Param((physics,), "reaction"),
            ],
            bkd,
        )
        assert composite.targets() == (physics,)

    def test_composite_of_physics_and_term_is_accepted_by_the_system(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _physics(bkd)
        robin = physics.weak_form_bcs()[0]
        composite = CompositeParameterization(
            [_Param((physics,), "diffusion"), _Param((robin,), "alpha")], bkd
        )
        require_owned_targets(composite, physics.system())
