"""Fixtures running the physics-agnostic ROM tests over every factory.

The factories and the :class:`RomProblem` bundle live in
``tests/_helpers/rom_problems.py``; adding a physics to the suite means
adding a factory there and listing it.
"""

from typing import Callable

import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.optional_deps import package_available


def _factory_name(factory: Callable[[NumpyBkd], object]) -> str:
    return factory.__name__


def _factories(affine_only: bool) -> list[Callable[[NumpyBkd], object]]:
    # Collected without skfem, which the factories import, would error;
    # an empty list makes the dependent tests skip instead.
    if not package_available("skfem"):
        return []
    from tests._helpers.rom_problems import (
        AFFINE_ROM_FACTORIES,
        ALL_ROM_FACTORIES,
    )

    return list(AFFINE_ROM_FACTORIES if affine_only else ALL_ROM_FACTORIES)


@pytest.fixture(params=_factories(affine_only=False), ids=_factory_name)
def rom_problem(request: pytest.FixtureRequest, numpy_bkd: NumpyBkd) -> object:
    """Every ROM test problem."""
    return request.param(numpy_bkd)


@pytest.fixture(params=_factories(affine_only=True), ids=_factory_name)
def affine_rom_problem(request: pytest.FixtureRequest, numpy_bkd: NumpyBkd) -> object:
    """The ROM test problems affine in state and parameters."""
    problem = request.param(numpy_bkd)
    if not problem.affine:
        raise ValueError(f"{problem.name} is listed as affine but declares not")
    return problem
