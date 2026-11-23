r"""1D diffusion with a parameterized diffusivity, on a Chebyshev mesh.

.. math::

    -\frac{d}{dx}\Big(D(x; p)\, \frac{du}{dx}\Big) - r u = f(x)

(steady) or :math:`u_t = (D u')' + f` (transient) on the physical
interval :math:`[a, b]` of ``mesh`` (its transform maps the reference
:math:`[-1, 1]` there), with :math:`D` given by a field map and boundary
conditions by side specs. Field maps, forcings and
initial states are evaluated at the physical nodes,
``mesh.points()[0]``.
"""

from typing import Optional, Sequence

from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import BoundarySide, TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeAwareCallableProtocol,
)
from pyapprox.pde.field_maps.protocol import FieldMapProtocol
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_function import (
    ParameterizedCollocationSystem,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.util.backends.protocols import Array
from pyapprox_benchmarks.functions.pde.collocation.boundaries import (
    BoundaryContext,
    BoundarySpecProtocol,
    Dirichlet,
    assemble_boundaries,
)

_ZERO_ENDS = (Dirichlet(BoundarySide.LEFT, 0.0), Dirichlet(BoundarySide.RIGHT, 0.0))


def build_steady_diffusion_1d(
    mesh: TransformedMesh1D[Array],
    forcing: TimeAwareCallableProtocol[Array],
    field_map: FieldMapProtocol[Array],
    boundaries: Sequence[BoundarySpecProtocol] = _ZERO_ENDS,
    reaction: float = 0.0,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
    r"""Steady 1D diffusion, :math:`-(D u')' - r u = f`.

    Parameters
    ----------
    mesh : TransformedMesh1D
        The Chebyshev mesh; its transform sets the physical interval.
    forcing : TimeAwareCallableProtocol
        :math:`f`, called with the physical node coordinates, shape
        ``(1, npts)``, and a time, and declared time-independent or not
        (``TimeIndependent`` / ``TimeDependent``); a time-dependent one
        needs the time when the function is built.
    field_map : FieldMapProtocol
        :math:`D(p)` at the physical nodes, each value positive.
    boundaries : sequence of BoundarySpecProtocol, optional
        One spec per side (see ``boundaries``); default zero Dirichlet at
        both ends. Given, it replaces the default whole.
    reaction : float, optional
        The constant :math:`r`, entering the rate as :math:`+r u` (the
        physics' sign); default none. Periodic ends need :math:`r \ne 0`
        for a unique solution, and :math:`r < 0` keeps the operator
        coercive.

    Returns
    -------
    ParameterizedCollocationSystem
        Build its function with ``collocation_steady_function``.

    Raises
    ------
    TypeError
        If a spec does not satisfy ``BoundarySpecProtocol``.
    ValueError
        If ``boundaries`` is empty or names a side twice or off the mesh.
    """
    bkd = mesh.bkd()
    basis = ChebyshevBasis1D(mesh, bkd)
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        reaction=reaction,
        forcing=NodalSupplier(forcing, mesh.points()),
    )
    essential, closures = assemble_boundaries(
        BoundaryContext(mesh, basis, physics), boundaries
    )
    return ParameterizedCollocationSystem(
        CollocationSystem(physics, essential, closures),
        create_diffusion_parameterization(physics, bkd, field_map),
        basis,
    )


def build_transient_diffusion_1d(
    mesh: TransformedMesh1D[Array],
    field_map: FieldMapProtocol[Array],
    forcing: Optional[TimeAwareCallableProtocol[Array]] = None,
    boundaries: Sequence[BoundarySpecProtocol] = _ZERO_ENDS,
) -> ParameterizedCollocationSystem[Array, ChebyshevBasis1D[Array]]:
    r"""Transient 1D diffusion, :math:`u_t = (D u')' + f`.

    The time discretization is the caller's: build the transient system
    with ``collocation_transient_system`` (choosing how the boundary rows
    are pinned in time) and solve it with ``TransientImplicitMap``, whose
    initial state is evaluated at the physical nodes, ``mesh.points()[0]``.

    Parameters
    ----------
    mesh : TransformedMesh1D
        The Chebyshev mesh; its transform sets the physical interval.
    field_map : FieldMapProtocol
        :math:`D(p)` at the physical nodes, each value positive.
    forcing : TimeAwareCallableProtocol, optional
        :math:`f`, called with the physical node coordinates, shape
        ``(1, npts)``, and a time, and declared time-independent or not;
        default none.
    boundaries : sequence of BoundarySpecProtocol, optional
        One spec per side (see ``boundaries``); default zero Dirichlet at
        both ends. Given, it replaces the default whole.

    Returns
    -------
    ParameterizedCollocationSystem
        The system with its diffusion parameterization.

    Raises
    ------
    TypeError
        If a spec does not satisfy ``BoundarySpecProtocol``.
    ValueError
        If ``boundaries`` is empty or names a side twice or off the mesh.
    """
    bkd = mesh.bkd()
    basis = ChebyshevBasis1D(mesh, bkd)
    physics = AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=1.0,
        forcing=None if forcing is None else NodalSupplier(forcing, mesh.points()),
    )
    essential, closures = assemble_boundaries(
        BoundaryContext(mesh, basis, physics), boundaries
    )
    return ParameterizedCollocationSystem(
        CollocationSystem(physics, essential, closures),
        create_diffusion_parameterization(physics, bkd, field_map),
        basis,
    )
