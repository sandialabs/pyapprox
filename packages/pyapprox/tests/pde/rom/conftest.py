"""Fixtures running the physics-agnostic ROM tests over every factory.

The factories and the :class:`RomProblem` bundle live in
``tests/_helpers/rom_problems.py``; adding a physics to the suite means
adding a factory there and listing it.
"""

from typing import Callable, List

import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.optional_deps import package_available

_Factory = Callable[[NumpyBkd], object]


# The factories import skfem, so collecting them without it would error;
# an empty list makes the dependent tests skip instead.
def _all_factories() -> List[_Factory]:
    if not package_available("skfem"):
        return []
    from tests._helpers.rom_problems import ALL_ROM_FACTORIES

    return list(ALL_ROM_FACTORIES)


def _affine_factories() -> List[_Factory]:
    if not package_available("skfem"):
        return []
    from tests._helpers.rom_problems import AFFINE_ROM_FACTORIES

    return list(AFFINE_ROM_FACTORIES)


def _elementwise_factories() -> List[_Factory]:
    if not package_available("skfem"):
        return []
    from tests._helpers.rom_problems import ELEMENTWISE_ROM_FACTORIES

    return list(ELEMENTWISE_ROM_FACTORIES)


def _factory_name(factory: _Factory) -> str:
    return factory.__name__


@pytest.fixture(params=_all_factories(), ids=_factory_name)
def rom_problem(request: pytest.FixtureRequest, numpy_bkd: NumpyBkd) -> object:
    """Every ROM test problem."""
    return request.param(numpy_bkd)


@pytest.fixture(params=_affine_factories(), ids=_factory_name)
def affine_rom_problem(request: pytest.FixtureRequest, numpy_bkd: NumpyBkd) -> object:
    """The ROM test problems affine in state and parameters."""
    problem = request.param(numpy_bkd)
    if not problem.affine:
        raise ValueError(f"{problem.name} is listed as affine but declares not")
    return problem


@pytest.fixture(params=_elementwise_factories(), ids=_factory_name)
def elementwise_rom_problem(
    request: pytest.FixtureRequest, numpy_bkd: NumpyBkd
) -> object:
    """The ROM test problems whose discretization is element-wise."""
    problem = request.param(numpy_bkd)
    if problem.elementwise is None:
        raise ValueError(f"{problem.name} is listed as element-wise but has none")
    return problem
