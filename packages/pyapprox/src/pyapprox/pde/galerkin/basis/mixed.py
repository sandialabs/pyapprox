"""Mixed finite element basis: several fields in one state vector.

The state is the concatenation of the fields' DOFs in the order given,
e.g. ``[velocity | pressure]`` for Taylor-Hood Stokes. A field view
answers boundary-DOF queries in those global indices, so the ordinary
``DirichletBC`` constrains one field of the mixed system:

.. code-block:: python

    mixed = MixedBasis([("velocity", vel_basis), ("pressure", pres_basis)])
    DirichletBC(mixed.field("velocity"), "left", inflow, bkd)
    DirichletBC(mixed.field("pressure"), "right", 0.0, bkd)
"""

from typing import Generic, List, Optional, Sequence, Tuple, Union

from pyapprox.pde.galerkin.protocols.basis import (
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
)
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _ScalarFieldView(Generic[Array]):
    """A scalar field of a ``MixedBasis``, in global DOF indices."""

    def __init__(
        self,
        mixed: "MixedBasis[Array]",
        basis: GalerkinBasisProtocol[Array],
        offset: int,
    ) -> None:
        self._mixed = mixed
        self._basis = basis
        self._offset = offset

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def get_dofs(self, boundary_name: str) -> Array:
        """Return the field's DOFs on a boundary, as global indices."""
        return self._basis.get_dofs(boundary_name) + self._offset

    def dof_coordinates(self) -> Array:
        """Return the coordinates of every DOF of the mixed system."""
        return self._mixed.dof_coordinates()


class _VectorFieldView(Generic[Array]):
    """A vector field of a ``MixedBasis``, in global DOF indices."""

    def __init__(
        self,
        mixed: "MixedBasis[Array]",
        basis: ComponentDofsBasisProtocol[Array],
        offset: int,
    ) -> None:
        self._mixed = mixed
        self._basis = basis
        self._offset = offset

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def ncomponents(self) -> int:
        return self._basis.ncomponents()

    def get_dofs(
        self,
        boundary_name: str,
        components: Optional[tuple[int, ...]] = None,
    ) -> Array:
        """Return the field's DOFs on a boundary, as global indices,
        optionally only those of the given components."""
        return (
            self._basis.get_dofs(boundary_name, components=components)
            + self._offset
        )

    def dof_components(self, dofs: Array) -> Array:
        """Return the component of each global DOF of this field."""
        return self._basis.dof_components(dofs - self._offset)

    def dof_coordinates(self) -> Array:
        """Return the coordinates of every DOF of the mixed system."""
        return self._mixed.dof_coordinates()


FieldView = Union[_ScalarFieldView[Array], _VectorFieldView[Array]]


class MixedBasis(Generic[Array]):
    """Several named field bases on one mesh, concatenated in order.

    Parameters
    ----------
    fields : Sequence[Tuple[str, GalerkinBasisProtocol]]
        ``(name, basis)`` pairs, in state-vector order. The bases must
        share one mesh. A basis whose DOFs carry components (a vector
        basis) gives a vector field view.

    Raises
    ------
    ValueError
        If there are no fields, a name repeats, or the bases do not share
        a mesh.
    """

    def __init__(
        self, fields: Sequence[Tuple[str, GalerkinBasisProtocol[Array]]]
    ) -> None:
        if len(fields) == 0:
            raise ValueError("a MixedBasis needs at least one field")
        names = [name for name, _ in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"field names must be unique, got {names}")
        mesh = fields[0][1].mesh()
        if any(basis.mesh() is not mesh for _, basis in fields):
            raise ValueError("the field bases must share one mesh")
        self._names = names
        self._bases = [basis for _, basis in fields]
        self._offsets: List[int] = []
        offset = 0
        for basis in self._bases:
            self._offsets.append(offset)
            offset += basis.ndofs()
        self._ndofs = offset
        self._mesh = mesh
        self._bkd = mesh.bkd()
        self._dof_coords: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> GalerkinMeshProtocol[Array]:
        """Return the shared mesh."""
        return self._mesh

    def ndofs(self) -> int:
        """Return the total number of DOFs."""
        return self._ndofs

    def field_names(self) -> List[str]:
        """Return the field names, in state-vector order."""
        return list(self._names)

    def _index(self, name: str) -> int:
        if name not in self._names:
            raise KeyError(
                f"no field {name!r}; fields are {self._names}"
            )
        return self._names.index(name)

    def field_basis(self, name: str) -> GalerkinBasisProtocol[Array]:
        """Return the basis of a field."""
        return self._bases[self._index(name)]

    def offset(self, name: str) -> int:
        """Return the index of a field's first DOF in the state vector."""
        return self._offsets[self._index(name)]

    def field(self, name: str) -> FieldView[Array]:
        """Return a field's view, which answers in global DOF indices."""
        index = self._index(name)
        basis = self._bases[index]
        if isinstance(basis, ComponentDofsBasisProtocol):
            return _VectorFieldView(self, basis, self._offsets[index])
        return _ScalarFieldView(self, basis, self._offsets[index])

    def dof_coordinates(self) -> Array:
        """Return the coordinates of every DOF, fields concatenated.
        Shape: (ndim, ndofs)."""
        if self._dof_coords is None:
            self._dof_coords = self._bkd.hstack(
                [basis.dof_coordinates() for basis in self._bases]
            )
        return self._dof_coords

    def __repr__(self) -> str:
        sizes = ", ".join(
            f"{name}={basis.ndofs()}"
            for name, basis in zip(self._names, self._bases)
        )
        return f"MixedBasis({sizes})"
