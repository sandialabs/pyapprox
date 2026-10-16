"""Fixtures running the physics-agnostic ROM tests over every factory.

The :class:`RomProblem` bundle lives in ``tests/_helpers/rom_problem.py``,
the skfem-built factories in ``tests/_helpers/rom_problems.py`` and the
mock element-wise factories, which need no assembly library, in
``tests/_helpers/mock_elementwise.py``; adding a physics to the suite
means adding a factory and listing it.
"""

from typing import Callable, List

import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.optional_deps import package_available
from tests._helpers.rom_problem import RomProblem

_Factory = Callable[[NumpyBkd], RomProblem]


# The skfem-built factories import skfem, so collecting them without it
# would error; they are left out then. The mock factories need no
# assembly library and always run.
def _all_factories() -> List[_Factory]:
    from tests._helpers.mock_elementwise import MOCK_ROM_FACTORIES

    if not package_available("skfem"):
        return list(MOCK_ROM_FACTORIES)
    from tests._helpers.rom_problems import ALL_ROM_FACTORIES

    return list(ALL_ROM_FACTORIES) + list(MOCK_ROM_FACTORIES)


def _affine_factories() -> List[_Factory]:
    from tests._helpers.mock_elementwise import MOCK_AFFINE_ROM_FACTORIES

    if not package_available("skfem"):
        return list(MOCK_AFFINE_ROM_FACTORIES)
    from tests._helpers.rom_problems import AFFINE_ROM_FACTORIES

    return list(AFFINE_ROM_FACTORIES) + list(MOCK_AFFINE_ROM_FACTORIES)


def _elementwise_factories() -> List[_Factory]:
    from tests._helpers.mock_elementwise import MOCK_ELEMENTWISE_ROM_FACTORIES

    if not package_available("skfem"):
        return list(MOCK_ELEMENTWISE_ROM_FACTORIES)
    from tests._helpers.rom_problems import ELEMENTWISE_ROM_FACTORIES

    return list(ELEMENTWISE_ROM_FACTORIES) + list(MOCK_ELEMENTWISE_ROM_FACTORIES)


def _factory_name(factory: _Factory) -> str:
    return factory.__name__


@pytest.fixture(params=_all_factories(), ids=_factory_name)
def rom_problem(request: pytest.FixtureRequest, numpy_bkd: NumpyBkd) -> RomProblem:
    """Every ROM test problem."""
    factory: _Factory = request.param
    return factory(numpy_bkd)


@pytest.fixture(params=_affine_factories(), ids=_factory_name)
def affine_rom_problem(
    request: pytest.FixtureRequest, numpy_bkd: NumpyBkd
) -> RomProblem:
    """The ROM test problems affine in state and parameters."""
    factory: _Factory = request.param
    problem = factory(numpy_bkd)
    if not problem.affine:
        raise ValueError(f"{problem.name} is listed as affine but declares not")
    return problem


@pytest.fixture(params=_elementwise_factories(), ids=_factory_name)
def elementwise_rom_problem(
    request: pytest.FixtureRequest, numpy_bkd: NumpyBkd
) -> RomProblem:
    """The ROM test problems whose discretization is element-wise."""
    factory: _Factory = request.param
    problem = factory(numpy_bkd)
    if problem.elementwise is None:
        raise ValueError(f"{problem.name} is listed as element-wise but has none")
    return problem
