"""Stack-generic suites for the discretization contract.

:class:`SteadyDiscretizationSuite` checks what a steady residual reads of
a discretization
(:class:`~pyapprox.pde.galerkin.protocols.discretization.SteadyDiscretizationProtocol`):
its spaces, its essential rows and its assembled Jacobian.
:class:`TransientDiscretizationSuite` adds the assembled mass
(:class:`~pyapprox.pde.assembly.protocols.MassAssemblerProtocol`). Each
is compared with references the case computes from the system itself,
independently of the discretization's assembly, and reads the stack's
vectors only through the case's ``TypedStackCase``.

A discretization's test module subclasses a suite and defines the
``case`` fixture, with its own optional-dependency skip. This module
imports protocols only.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Sequence

import numpy as np

from pyapprox.pde.assembly.protocols import MassAssemblerProtocol
from pyapprox.pde.galerkin.protocols.discretization import (
    SteadyDiscretizationProtocol,
)
from pyapprox.pde.galerkin.protocols.system import GalerkinSteadySystemProtocol
from pyapprox.util.backends.protocols import Array
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V

_TIME = 0.3


@dataclass
class SteadyDiscretizationCase(Generic[Array, V, F, A]):
    """A steady discretization, its stack, and references from its system.

    ``Array`` is the system's state type and ``V``, ``F``, ``A`` are the
    stack's (``TypedStackCase``).
    """

    stack: TypedStackCase[V, F, A]
    discretization: SteadyDiscretizationProtocol[Array, V, F]
    system: GalerkinSteadySystemProtocol[Array]
    jacobian: Callable[[Array, float], A]
    """The system's own state Jacobian as a dense stack array."""
    essential_dofs: Sequence[int]
    """The constraint set's DOFs, in any order."""

    def state(self, seed: int) -> Array:
        """A random state of the system."""
        rng = np.random.default_rng(seed)
        return self.system.bkd().asarray(rng.normal(size=self.system.nstates()))


@dataclass
class TransientDiscretizationCase(SteadyDiscretizationCase[Array, V, F, A]):
    """A steady case whose discretization also assembles a mass."""

    mass_assembler: MassAssemblerProtocol[V, F]
    """The discretization, as a mass assembler."""
    mass: A
    """The system's mass as a dense stack array, lumped when ``lumped``."""
    lumped: bool


class SteadyDiscretizationSuite:
    def test_spaces(self, case: SteadyDiscretizationCase[Array, V, F, A]) -> None:
        disc = case.discretization
        n = case.system.nstates()
        descriptor = disc.space().descriptor()
        assert disc.system() is case.system
        assert disc.nstates() == n == descriptor.size
        assert disc.parameter_space(6).descriptor().size == 6
        assert disc.essential_rows().descriptor() == descriptor
        op = disc.assemble_jacobian(case.state(0), _TIME)
        assert (op.domain(), op.range()) == (descriptor, descriptor)

    def test_essential_rows_are_the_constraint_sets(
        self, case: SteadyDiscretizationCase[Array, V, F, A]
    ) -> None:
        rows = case.discretization.essential_rows()
        descriptor = case.discretization.space().descriptor()
        stack = case.stack
        ones = stack.vector(stack.bkd.ones((descriptor.size, 1)), descriptor)
        expected = np.ones((descriptor.size, 1))
        expected[list(case.essential_dofs)] = 0.0
        stack.assert_vector(rows.zero_rows(ones), stack.dense(expected), atol=0.0)
        assert rows.count() == len(set(case.essential_dofs)) > 0

    def test_jacobian_acts_as_the_systems(
        self, case: SteadyDiscretizationCase[Array, V, F, A]
    ) -> None:
        stack = case.stack
        state = case.state(0)
        op = case.discretization.assemble_jacobian(state, _TIME)
        reference = case.jacobian(state, _TIME)
        x = stack.vec(op.domain(), 2, 1)
        stack.assert_vector(op.apply(x), stack.times(reference, x), rtol=1e-12)
        y = stack.vec(op.range(), 2, 2)
        stack.assert_vector(
            op.apply_transpose(y),
            stack.times(stack.bkd.transpose(reference), y),
            rtol=1e-12,
        )

    def test_jacobians_are_independent_of_later_calls(
        self, case: SteadyDiscretizationCase[Array, V, F, A]
    ) -> None:
        stack = case.stack
        disc = case.discretization
        first = disc.assemble_jacobian(case.state(0), _TIME)
        x = stack.vec(first.domain(), 1, 1)
        before = stack.bkd.copy(stack.array(first.apply(x)))
        disc.assemble_jacobian(case.state(3), 2 * _TIME)
        stack.assert_vector(first.apply(x), before, rtol=1e-15)


class TransientDiscretizationSuite(SteadyDiscretizationSuite):
    def test_mass_is_the_systems(
        self, case: TransientDiscretizationCase[Array, V, F, A]
    ) -> None:
        op = case.mass_assembler.assemble_mass()
        descriptor = case.discretization.space().descriptor()
        assert (op.domain(), op.range()) == (descriptor, descriptor)
        assert isinstance(op.format(), case.stack.format_type)
        case.stack.bkd.assert_allclose(
            case.stack.format_dense(op.format()), case.mass, atol=1e-15
        )
        assert op.properties().diagonal == case.lumped

    def test_masses_are_independent_of_later_calls(
        self, case: TransientDiscretizationCase[Array, V, F, A]
    ) -> None:
        stack = case.stack
        first = case.mass_assembler.assemble_mass()
        before = stack.bkd.copy(stack.format_dense(first.format()))
        case.mass_assembler.assemble_mass()
        stack.bkd.assert_allclose(stack.format_dense(first.format()), before)
