"""Tests for boundary condition implementations.

Tests Dirichlet, Neumann and Robin BCs and the BoundaryConditionSet.
"""


import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.boundary import (
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import (
    LagrangeBasis,
    MixedBasis,
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.boundary import (
    BoundaryConditionSet,
    CallableDirichletBC,
    DirectDirichletBC,
    DirichletBC,
    NeumannBC,
    NormalDependentDataProtocol,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.util.backends.numpy import NumpyBkd


class _NormalX:
    """Time-independent data equal to the normal's first component."""

    def evaluate(
        self,
        coords: NDArray[np.floating[Any]],
        normal: NDArray[np.floating[Any]],
        time: float,
    ) -> NDArray[np.floating[Any]]:
        return np.asarray(normal[0])

    def is_time_dependent(self) -> bool:
        return False


class TestDirichletBCBase:
    """Tests for DirichletBC."""

    def test_1d_constant_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        """Test constant Dirichlet BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = DirichletBC(basis, "left", value_func=1.5, bkd=bkd)

        # Check boundary DOFs
        dofs = bkd.to_numpy(bc.boundary_dofs())
        assert len(dofs) == 1
        assert dofs[0] == 0  # Left boundary is DOF 0

        # Check boundary values
        values = bc.boundary_values(time=0.0)
        expected = bkd.asarray(np.array([1.5], dtype=np.float64))
        bkd.assert_allclose(values, expected)

    def test_1d_function_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        """Test function-valued Dirichlet BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # BC: u = x^2 at boundary
        def bc_func(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return np.asarray(x[0] ** 2)

        bc = DirichletBC(basis, "right", value_func=bc_func, bkd=bkd)

        # Right boundary is at x=1
        values = bc.boundary_values(time=0.0)
        expected = bkd.asarray(
            np.array([1.0], dtype=np.float64)  # 1^2 = 1
        )
        bkd.assert_allclose(values, expected)

    def test_2d_dirichlet_multiple_boundaries(self, numpy_bkd: NumpyBkd) -> None:
        """Test Dirichlet BC on multiple boundaries in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        bc_right = DirichletBC(basis, "right", value_func=1.0, bkd=bkd)

        # Check DOFs are distinct
        left_dofs = set(bkd.to_numpy(bc_left.boundary_dofs()))
        right_dofs = set(bkd.to_numpy(bc_right.boundary_dofs()))

        assert len(left_dofs & right_dofs) == 0


class TestNeumannBCBase:
    """Tests for NeumannBC."""

    def test_1d_constant_neumann(self, numpy_bkd: NumpyBkd) -> None:
        """Test constant Neumann BC in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)

        # A point boundary integrates g * phi to g at its node.
        load = bc.apply_to_load(bkd.zeros((basis.ndofs(),)), time=0.0)
        expected = np.zeros(basis.ndofs())
        expected[-1] = 1.0
        bkd.assert_allclose(load, bkd.asarray(expected))

    def test_normal_dependent_data_gets_outward_normal(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Data g(x, n, t) = n_x is -1 at the left end and +1 at the right."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        assert isinstance(_NormalX(), NormalDependentDataProtocol)
        zeros = bkd.zeros((basis.ndofs(),))
        left = NeumannBC(basis, "left", flux_func=_NormalX(), bkd=bkd)
        right = NeumannBC(basis, "right", flux_func=_NormalX(), bkd=bkd)
        bkd.assert_allclose(
            left.apply_to_load(zeros, 0.0)[0:1], bkd.asarray([-1.0])
        )
        bkd.assert_allclose(
            right.apply_to_load(zeros, 0.0)[-1:], bkd.asarray([1.0])
        )
        assert not left.is_time_dependent()

    def test_1d_neumann_apply_to_load(self, numpy_bkd: NumpyBkd) -> None:
        """Test Neumann BC adds to load vector."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Constant flux at right boundary
        bc = NeumannBC(basis, "right", flux_func=2.0, bkd=bkd)

        nstates = basis.ndofs()
        load = bkd.asarray(np.zeros(nstates))

        modified_load = bc.apply_to_load(load, time=0.0)
        modified_load_np = bkd.to_numpy(modified_load)

        # Neumann BC should add contribution at right boundary DOF
        # For 1D linear elements, the boundary integral of constant = value
        # The contribution goes to the boundary node
        assert modified_load_np[-1] > 0  # Right boundary is last DOF


class TestRobinBCBase:
    """Tests for RobinBC."""

    def test_1d_robin_reduces_to_dirichlet(self, numpy_bkd: NumpyBkd) -> None:
        """Test Robin BC with large alpha approaches Dirichlet."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        # Large alpha effectively enforces u = g/alpha
        alpha = 1000.0
        g_val = 1000.0  # g = alpha * u_desired = 1000 * 1 = 1000
        bc = RobinBC(basis, "left", alpha=alpha, value_func=g_val, bkd=bkd)

        assert bc.alpha() == alpha

    def test_1d_robin_apply_to_stiffness(self, numpy_bkd: NumpyBkd) -> None:
        """Test Robin BC modifies stiffness matrix."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc = RobinBC(basis, "left", alpha=2.0, value_func=0.0, bkd=bkd)

        nstates = basis.ndofs()
        stiffness = bkd.asarray(np.zeros((nstates, nstates)))

        modified_stiff = bc.apply_to_stiffness(stiffness, time=0.0)
        modified_stiff_np = bkd.to_numpy(modified_stiff)

        # Robin BC should add alpha * boundary_mass_matrix contribution
        # For 1D point boundary, this adds to (0,0) entry
        assert modified_stiff_np[0, 0] > 0

    def test_a_declared_time_dependent_alpha_is_evaluated_at_the_time(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        r"""On the left side of the unit square, :math:`\alpha = (1 + t)
        (1 + y)` gives a boundary stiffness at t = 1 twice that at t = 0,
        which equals the one of the time-independent :math:`1 + y` given
        bare or declared; the Jacobian is minus that stiffness, and only
        the declared time-dependent coefficient makes the term so."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(nx=3, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        zeros = bkd.zeros((basis.ndofs(), basis.ndofs()))
        growing = RobinBC(
            basis, "left", TimeDependent(_growing_alpha), value_func=0.0, bkd=bkd
        )
        steady = RobinBC(basis, "left", _profile_alpha, value_func=0.0, bkd=bkd)
        declared = RobinBC(
            basis, "left", TimeIndependent(_profile_alpha), value_func=0.0, bkd=bkd
        )
        at_zero = growing.apply_to_stiffness(zeros, 0.0)
        at_one = growing.apply_to_stiffness(zeros, 1.0)
        assert float(np.abs(bkd.to_numpy(at_zero)).max()) > 0.1
        bkd.assert_allclose(at_one, 2.0 * at_zero, rtol=1e-14)
        bkd.assert_allclose(steady.apply_to_stiffness(zeros, 1.0), at_zero)
        bkd.assert_allclose(declared.apply_to_stiffness(zeros, 1.0), at_zero)
        state = bkd.zeros((basis.ndofs(),))
        bkd.assert_allclose(growing.apply_to_jacobian(zeros, state, 1.0), -at_one)
        assert growing.is_time_dependent()
        assert not steady.is_time_dependent()
        assert not declared.is_time_dependent()

    def test_an_undeclared_alpha_that_could_take_a_time_is_refused(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        basis = LagrangeBasis(StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd), 1)
        with pytest.raises(TypeError, match="ambiguous"):
            RobinBC(basis, "left", _growing_alpha, value_func=0.0, bkd=bkd)

    def test_per_component_alpha_springs_each_component_alone(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        r"""On a 2D vector basis, :math:`\alpha = (1 + y, 0)` per component
        gives the shared coefficient's boundary stiffness on the
        x-component DOFs and none on the y-component ones, so y is
        traction free there; :math:`(1 + y, 1 + y)` gives the shared
        stiffness itself. Values for the wrong number of components are
        refused, and per-component values on a scalar basis too."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(nx=3, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        zeros = bkd.zeros((basis.ndofs(), basis.ndofs()))

        def stiffness(alpha: Any) -> NDArray[np.floating[Any]]:
            bc = RobinBC(basis, "left", TimeIndependent(alpha), 0.0, bkd)
            return np.asarray(bkd.to_numpy(bc.apply_to_stiffness(zeros, 0.0)))

        shared = stiffness(_profile_alpha)
        on_x = bkd.to_numpy(basis.dof_components(bkd.arange(basis.ndofs()))) == 0
        assert float(np.abs(shared[np.ix_(on_x, on_x)]).max()) > 0.1
        np.testing.assert_allclose(
            stiffness(_profile_alpha_on_x), shared * np.outer(on_x, on_x), atol=1e-15
        )
        np.testing.assert_allclose(stiffness(_profile_alpha_on_both), shared)
        with pytest.raises(ValueError, match="alpha must return shape"):
            stiffness(_three_alphas)
        scalar = LagrangeBasis(mesh, degree=1)
        bc = RobinBC(
            scalar, "left", TimeIndependent(_profile_alpha_on_both), 0.0, bkd
        )
        with pytest.raises(ValueError, match="alpha must return shape"):
            bc.apply_to_stiffness(bkd.zeros((scalar.ndofs(), scalar.ndofs())), 0.0)


class TestNaturalBCsOnMixedFields:
    """Neumann and Robin conditions on one field of a mixed basis."""

    def test_terms_are_the_plain_basis_terms_at_the_field_dofs(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        r"""On a Taylor-Hood ``[velocity | pressure]`` basis of the unit
        square, a Neumann and a Robin condition on either field give the
        load, stiffness, residual and Jacobian of the same condition on
        the field's own basis, placed at the field's DOFs and zero
        elsewhere. Velocity starts at offset zero, so only the padding to
        the mixed size is tested there; pressure starts after it."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(nx=3, ny=4, bounds=[(0.0, 1.0), (0.0, 1.0)], bkd=bkd)
        velocity = VectorLagrangeBasis(mesh, degree=2)
        pressure = LagrangeBasis(mesh, degree=1)
        fields: list[tuple[str, MixedFieldBasisProtocol[NDArray[Any]]]] = [
            ("velocity", velocity),
            ("pressure", pressure),
        ]
        mixed = MixedBasis(fields)
        nstates = mixed.ndofs()
        rng = np.random.default_rng(1)
        state = bkd.asarray(rng.normal(size=nstates))
        cases: list[tuple[str, Any, Any, Any]] = [
            ("velocity", velocity, _two_fluxes, _profile_alpha_on_x),
            ("pressure", pressure, _profile_alpha, _profile_alpha),
        ]
        for name, plain, flux, alpha in cases:
            offset, n = mixed.offset(name), plain.ndofs()
            field = slice(offset, offset + n)

            def placed_vector(local: Any) -> NDArray[np.floating[Any]]:
                out = np.zeros(nstates)
                out[field] = bkd.to_numpy(local)
                return out

            def placed_matrix(local: Any) -> NDArray[np.floating[Any]]:
                out = np.zeros((nstates, nstates))
                out[field, field] = bkd.to_numpy(local)
                return out

            plain_neumann = NeumannBC(plain, "right", flux, bkd)
            neumann = NeumannBC(mixed.field(name), "right", flux, bkd)
            plain_load = plain_neumann.apply_to_load(bkd.zeros((n,)), 0.0)
            assert float(np.abs(bkd.to_numpy(plain_load)).max()) > 0.01
            bkd.assert_allclose(
                neumann.apply_to_load(bkd.zeros((nstates,)), 0.0),
                bkd.asarray(placed_vector(plain_load)),
            )

            alpha_fn = TimeIndependent(alpha)
            plain_robin = RobinBC(plain, "left", alpha_fn, flux, bkd)
            robin = RobinBC(mixed.field(name), "left", alpha_fn, flux, bkd)
            plain_stiff = plain_robin.apply_to_stiffness(bkd.zeros((n, n)), 0.0)
            assert float(np.abs(bkd.to_numpy(plain_stiff)).max()) > 0.01
            bkd.assert_allclose(
                robin.apply_to_stiffness(bkd.zeros((nstates, nstates)), 0.0),
                bkd.asarray(placed_matrix(plain_stiff)),
            )
            plain_jac = plain_robin.apply_to_jacobian(
                bkd.zeros((n, n)), state[field], 0.0
            )
            bkd.assert_allclose(
                robin.apply_to_jacobian(
                    bkd.zeros((nstates, nstates)), state, 0.0
                ),
                bkd.asarray(placed_matrix(plain_jac)),
            )
            plain_res = plain_robin.apply_to_residual(
                bkd.zeros((n,)), state[field], 0.0
            )
            bkd.assert_allclose(
                robin.apply_to_residual(bkd.zeros((nstates,)), state, 0.0),
                bkd.asarray(placed_vector(plain_res)),
            )


def _two_fluxes(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    return np.stack([1.0 + x[1], 2.0 - x[1]])


def _profile_alpha(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    return np.asarray(1.0 + x[1])


def _profile_alpha_on_x(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    return np.stack([1.0 + x[1], np.zeros_like(x[1])])


def _profile_alpha_on_both(
    x: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    return np.stack([1.0 + x[1], 1.0 + x[1]])


def _three_alphas(x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    return np.stack([1.0 + x[1]] * 3)


def _growing_alpha(
    x: NDArray[np.floating[Any]], time: float
) -> NDArray[np.floating[Any]]:
    return np.asarray((1.0 + time) * (1.0 + x[1]))


class TestBoundaryConditionSetBase:
    """Tests for BoundaryConditionSet."""

    def test_empty_set(self, numpy_bkd: NumpyBkd) -> None:
        """Test empty boundary condition set."""
        bkd = numpy_bkd
        bc_set = BoundaryConditionSet(bkd)

        assert bc_set.ndirichlet() == 0
        assert bc_set.nneumann() == 0
        assert bc_set.nrobin() == 0

    def test_add_multiple_bcs(self, numpy_bkd: NumpyBkd) -> None:
        """Test adding multiple boundary conditions."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(bkd)

        bc_left = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        bc_right = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)

        bc_set.add_dirichlet(bc_left)
        bc_set.add_neumann(bc_right)

        assert bc_set.ndirichlet() == 1
        assert bc_set.nneumann() == 1

    def test_dirichlet_dofs_and_values(self, numpy_bkd: NumpyBkd) -> None:
        """Test getting all Dirichlet DOFs and values."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        bc_set = BoundaryConditionSet(bkd)
        bc_set.add_dirichlet(
            DirichletBC(basis, "left", value_func=1.0, bkd=bkd)
        )
        bc_set.add_dirichlet(
            DirichletBC(basis, "right", value_func=2.0, bkd=bkd)
        )

        dofs = bkd.to_numpy(bc_set.dirichlet_dofs())
        values = bkd.to_numpy(bc_set.dirichlet_values(time=0.0))

        assert len(dofs) == 2
        assert len(values) == 2


class TestRoleProtocols:
    """Concrete BC classes satisfy exactly one of the two disjoint roles."""

    def test_dirichlet_bcs_are_essential_only(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bcs = [
            DirichletBC(basis, "left", value_func=1.0, bkd=bkd),
            DirectDirichletBC([0], [1.0], bkd),
            CallableDirichletBC([0], lambda t: np.array([t]), bkd),
        ]
        for bc in bcs:
            assert isinstance(bc, EssentialBCProtocol), bc
            assert not isinstance(bc, WeakFormBCProtocol), bc

    def test_neumann_robin_are_weak_form_only(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bcs = [
            NeumannBC(basis, "right", flux_func=1.0, bkd=bkd),
            RobinBC(basis, "right", alpha=2.0, value_func=1.0, bkd=bkd),
        ]
        for bc in bcs:
            assert isinstance(bc, WeakFormBCProtocol), bc
            assert not isinstance(bc, EssentialBCProtocol), bc

    def test_constrained_accessors_match_legacy(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = DirichletBC(basis, "left", value_func=3.0, bkd=bkd)
        bkd.assert_allclose(
            bkd.asarray(bc.constrained_dofs(), dtype=bkd.double_dtype()),
            bkd.asarray(bc.boundary_dofs(), dtype=bkd.double_dtype()),
        )
        bkd.assert_allclose(
            bc.constrained_values(0.0), bc.boundary_values(0.0)
        )

    def test_neumann_stiffness_and_jacobian_are_identity(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        n = basis.ndofs()
        K = bkd.asarray(np.eye(n))
        state = bkd.asarray(np.zeros(n))
        assert bc.apply_to_stiffness(K, 0.0) is K
        assert bc.apply_to_jacobian(K, state, 0.0) is K

    def test_neumann_residual_adds_load(self, numpy_bkd: NumpyBkd) -> None:
        """The term enters F of M du/dt = F with a plus sign."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        n = basis.ndofs()
        state = bkd.asarray(np.zeros(n))
        res = bc.apply_to_residual(bkd.asarray(np.zeros(n)), state, 0.0)
        load = bc.apply_to_load(bkd.asarray(np.zeros(n)), 0.0)
        bkd.assert_allclose(res, load)

    def test_bc_set_role_accessors(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        bc_set = BoundaryConditionSet(bkd)
        dirichlet = DirichletBC(basis, "left", value_func=0.0, bkd=bkd)
        neumann = NeumannBC(basis, "right", flux_func=1.0, bkd=bkd)
        robin = RobinBC(basis, "right", alpha=1.0, value_func=0.0, bkd=bkd)
        bc_set.add_dirichlet(dirichlet)
        bc_set.add_neumann(neumann)
        bc_set.add_robin(robin)
        assert bc_set.essential_bcs() == [dirichlet]
        assert bc_set.weak_form_bcs() == [neumann, robin]
