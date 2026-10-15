"""Element-wise evaluation of a Galerkin physics agrees with its assembly.

The conformance condition of ``ElementwiseOperatorProtocol``: scattering
the element residuals of every element to their global rows gives the
assembled interior residual, and likewise the Jacobian. Both paths go
through the physics' one set of residual terms, so this checks the
shared term assembler's element path (skfem block layout, local-state
interpolation, local DOF numbering) against its global path.

Setup: AdvectionDiffusionReaction with every kind of term: a nodal
diffusivity 1 + x/2 (+ y/4 in 2D), a nodal forcing 1 + x, the nonlinear
reaction u^2 (a state load with its Jacobian), and constant-velocity
advection; P1 on 6 cells in 1D and on a 4 x 3 quadrilateral mesh in 2D;
a random state (seed 0) of size O(1).

Checks: the full scatter-sum equals the assembly (rtol 1e-12); a
two-way partition of the elements sums to the same; a subset's element
results equal the matching rows of the full element evaluation; the
local-to-global map reproduces skfem's element DOFs; the restricted
operator survives pickling unchanged. Wrong-model control: leaving out
one element breaks the identity.
"""

import pickle
from typing import Any, List, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.constitutive.coefficient_functions import (
    CallableReaction,
    NodalFieldDiffusion,
    NodalFieldForcing,
)
from pyapprox.pde.elementwise import (
    ElementSet,
    ElementwiseOperatorProtocol,
    RestrictedElementOperatorProtocol,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.term_assembly import SkfemRestrictedOperator
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray

_Points = NDArray[np.floating[Any]]


def _square(x: _Points, u: _Points) -> _Points:
    return u**2


def _square_deriv(x: _Points, u: _Points) -> _Points:
    return 2.0 * u


def _square_deriv2(x: _Points, u: _Points) -> _Points:
    return np.full_like(u, 2.0)


def _physics(bkd: NumpyBkd, ndim: int) -> AdvectionDiffusionReaction[NumpyArray]:
    if ndim == 1:
        mesh: Any = StructuredMesh1D(nx=6, bounds=(0.0, 1.0), bkd=bkd)
        velocity = bkd.asarray([0.5])
    else:
        mesh = StructuredMesh2D(nx=4, ny=3, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        velocity = bkd.asarray([0.5, -0.3])
    basis = LagrangeBasis(mesh, degree=1)
    coords = bkd.to_numpy(basis.dof_coordinates())
    diffusivity = 1.0 + 0.5 * coords[0]
    if ndim == 2:
        diffusivity = diffusivity + 0.25 * coords[1]
    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=NodalFieldDiffusion(basis, dofs=diffusivity),
        bkd=bkd,
        velocity=velocity,
        reaction=CallableReaction(_square, _square_deriv, _square_deriv2),
        forcing=NodalFieldForcing(basis, dofs=1.0 + coords[0]),
    )


def _state(physics: AdvectionDiffusionReaction[NumpyArray]) -> NumpyArray:
    rng = np.random.default_rng(0)
    state: NumpyArray = rng.uniform(0.5, 1.5, size=physics.nstates())
    return state


def _scatter(
    restricted: RestrictedElementOperatorProtocol[NumpyArray],
    state: NumpyArray,
    nstates: int,
) -> Tuple[NumpyArray, NumpyArray]:
    """Scatter element residuals and Jacobians to global rows."""
    global_dofs = restricted.global_dofs()
    rows = global_dofs[restricted.element_row_dofs()]
    cols = global_dofs[restricted.element_state_dofs()]
    local_states = state[cols]
    residual = np.zeros(nstates)
    np.add.at(residual, rows, restricted.element_residuals(local_states, 0.0))
    jacobian = np.zeros((nstates, nstates))
    blocks = restricted.element_jacobians(local_states, 0.0)
    for elem in range(int(rows.shape[0])):
        jacobian[np.ix_(rows[elem], cols[elem])] += blocks[elem]
    return residual, jacobian


def _elements(indices: List[int], bkd: NumpyBkd) -> ElementSet[NumpyArray]:
    return ElementSet(bkd.asarray(indices, dtype=bkd.int64_dtype()), bkd)


@pytest.mark.parametrize("ndim", [1, 2])
class TestConformance:
    def test_protocols(self, numpy_bkd: NumpyBkd, ndim: int) -> None:
        physics = _physics(numpy_bkd, ndim)
        assert isinstance(physics, ElementwiseOperatorProtocol)
        restricted = physics.restrict(ElementSet.all(physics.nelements(), numpy_bkd))
        assert isinstance(restricted, RestrictedElementOperatorProtocol)

    def test_all_elements_sum_to_the_assembly(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        physics = _physics(numpy_bkd, ndim)
        state = _state(physics)
        restricted = physics.restrict(ElementSet.all(physics.nelements(), numpy_bkd))
        residual, jacobian = _scatter(restricted, state, physics.nstates())
        numpy_bkd.assert_allclose(
            residual, physics.interior_residual(state, 0.0), rtol=1e-12
        )
        # Applied to the identity rather than densified: the assembled
        # Jacobian is sparse behind a protocol that says Array, and a
        # product is defined for both.
        assembled = physics.interior_jacobian(state, 0.0) @ np.eye(physics.nstates())
        numpy_bkd.assert_allclose(jacobian, assembled, rtol=1e-12, atol=1e-12)

    def test_partition_sums_to_the_assembly(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        physics = _physics(numpy_bkd, ndim)
        state = _state(physics)
        nelements = physics.nelements()
        rng = np.random.default_rng(1)
        order = rng.permutation(nelements)
        halves = [sorted(order[: nelements // 2]), sorted(order[nelements // 2 :])]
        total = np.zeros(physics.nstates())
        for half in halves:
            restricted = physics.restrict(_elements([int(e) for e in half], numpy_bkd))
            total = total + _scatter(restricted, state, physics.nstates())[0]
        numpy_bkd.assert_allclose(
            total, physics.interior_residual(state, 0.0), rtol=1e-12
        )

    def test_subset_matches_full_element_evaluation(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        physics = _physics(numpy_bkd, ndim)
        state = _state(physics)
        full = physics.restrict(ElementSet.all(physics.nelements(), numpy_bkd))
        subset_indices = [3, 0, 4]
        subset = physics.restrict(_elements(subset_indices, numpy_bkd))

        def evaluate(
            op: RestrictedElementOperatorProtocol[NumpyArray],
        ) -> Tuple[NumpyArray, NumpyArray]:
            local = state[op.global_dofs()[op.element_state_dofs()]]
            return (
                op.element_residuals(local, 0.0),
                op.element_jacobians(local, 0.0),
            )

        full_res, full_jac = evaluate(full)
        sub_res, sub_jac = evaluate(subset)
        numpy_bkd.assert_allclose(sub_res, full_res[subset_indices], rtol=1e-12)
        numpy_bkd.assert_allclose(sub_jac, full_jac[subset_indices], rtol=1e-12)

    def test_local_numbering_maps_to_skfem_dofs(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        physics = _physics(numpy_bkd, ndim)
        indices = [1, 2]
        restricted = physics.restrict(_elements(indices, numpy_bkd))
        skfem_dofs = np.asarray(physics.basis().skfem_basis().element_dofs)
        mapped = restricted.global_dofs()[restricted.element_row_dofs()]
        numpy_bkd.assert_allclose(mapped, skfem_dofs[:, indices].T)

    def test_survives_pickling(self, numpy_bkd: NumpyBkd, ndim: int) -> None:
        physics = _physics(numpy_bkd, ndim)
        state = _state(physics)
        restricted = physics.restrict(_elements([0, 2], numpy_bkd))
        clone: SkfemRestrictedOperator = pickle.loads(pickle.dumps(restricted))
        local = state[restricted.global_dofs()[restricted.element_state_dofs()]]
        numpy_bkd.assert_allclose(
            clone.element_residuals(local, 0.0),
            restricted.element_residuals(local, 0.0),
            rtol=1e-14,
        )

    def test_dropping_an_element_breaks_the_identity(
        self, numpy_bkd: NumpyBkd, ndim: int
    ) -> None:
        physics = _physics(numpy_bkd, ndim)
        state = _state(physics)
        restricted = physics.restrict(
            _elements(list(range(1, physics.nelements())), numpy_bkd)
        )
        residual = _scatter(restricted, state, physics.nstates())[0]
        gap = np.max(np.abs(residual - physics.interior_residual(state, 0.0)))
        assert gap > 1e-3


class TestElementSet:
    def test_rejects_repeats(self, numpy_bkd: NumpyBkd) -> None:
        with pytest.raises(ValueError, match="repeat"):
            _elements([0, 1, 1], numpy_bkd)

    def test_rejects_out_of_range(self, numpy_bkd: NumpyBkd) -> None:
        physics = _physics(numpy_bkd, 1)
        with pytest.raises(ValueError, match="element indices"):
            physics.restrict(_elements([0, physics.nelements()], numpy_bkd))
