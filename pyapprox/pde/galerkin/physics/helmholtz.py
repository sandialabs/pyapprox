"""Helmholtz equation physics for Galerkin FEM.

Solves the Helmholtz equation:
    -div(grad(u)) + k^2 * u = f

where:
    k = wavenumber (scalar or spatially-varying)
    f = source term

This is a frequency-domain wave equation that arises in acoustics,
electromagnetics, and other wave phenomena.
"""

from typing import Generic, Optional, Callable, List, Union

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import BoundaryConditionProtocol
from pyapprox.pde.galerkin.physics.galerkin_base import GalerkinPhysicsBase
from pyapprox.pde.galerkin.physics.helpers import ScalarMassAssembler

# Import skfem for assembly
try:
    from skfem import asm, LinearForm, BilinearForm
    from skfem.helpers import dot, grad
except ImportError:
    raise ImportError(
        "scikit-fem is required for Galerkin module. "
        "Install with: pip install scikit-fem"
    )


class Helmholtz(GalerkinPhysicsBase[Array]):
    """Helmholtz equation physics.

    Solves:
        -div(grad(u)) + k^2 * u = f

    In weak form:
        integral(grad(u) . grad(v)) + k^2 * integral(u * v) = integral(f * v)

    The stiffness matrix is:
        K_ij = integral(grad(phi_j) . grad(phi_i)) + k^2(x) * integral(phi_j * phi_i)
             = K_laplacian + k^2(x) * M

    The load vector is:
        b_i = integral(f * phi_i)

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    wavenumber : float or Callable
        Wavenumber k. If float, k is constant and k^2 is used in the
        bilinear form. If callable, it is interpreted as the **squared
        wavenumber** k^2(x): takes coordinates (ndim, npts) and returns
        (npts,) values of k^2.
    bkd : Backend
        Computational backend.
    forcing : Callable, optional
        Source term. Takes coordinates (ndim, npts) and returns (npts,).
    boundary_conditions : List[BoundaryConditionProtocol], optional
        List of boundary conditions.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin import (
    ...     StructuredMesh1D, LagrangeBasis
    ... )
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> basis = LagrangeBasis(mesh, degree=1)
    >>> physics = Helmholtz(
    ...     basis=basis,
    ...     wavenumber=2*np.pi,  # wavelength = 1
    ...     bkd=bkd,
    ... )
    """

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        wavenumber: Union[float, Callable],
        bkd: Backend[Array],
        forcing: Optional[Callable] = None,
        boundary_conditions: Optional[List[BoundaryConditionProtocol[Array]]] = None,
    ):
        super().__init__(basis, bkd, boundary_conditions)
        self._mass = ScalarMassAssembler(basis, bkd)

        # Store wavenumber
        self._wavenumber = wavenumber
        self._wavenumber_is_callable = callable(wavenumber)

        # Store forcing
        self._forcing = forcing

        # Cache assembled matrices
        self._stiffness_cached: Optional[Array] = None
        self._load_cached: Optional[Array] = None

    def wavenumber(self) -> Union[float, Callable]:
        """Return the wavenumber k (scalar) or k^2(x) (callable)."""
        return self._wavenumber

    def is_linear(self) -> bool:
        """Helmholtz equation is always linear."""
        return True

    def mass_matrix(self):
        """Return the scalar mass matrix."""
        return self._mass.mass_matrix()

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x."""
        return self._mass.mass_solve(rhs)

    def _assemble_stiffness(self, state: Array, time: float) -> Array:
        """Assemble stiffness matrix K.

        K = K_laplacian + k^2 * M
        where:
            K_laplacian_ij = integral(grad(phi_j) . grad(phi_i))
            M_ij = integral(k^2(x) * phi_j * phi_i)
        """
        if self._stiffness_cached is not None:
            return self._stiffness_cached

        skfem_basis = self._basis.skfem_basis()

        # Laplacian stiffness
        def laplacian_form(u, v, w):
            return dot(grad(u), grad(v))

        laplacian_np = asm(BilinearForm(laplacian_form), skfem_basis)

        # Mass-like term with k^2
        if self._wavenumber_is_callable:
            sqwavenum_func = self._wavenumber

            def mass_form(u, v, w):
                x_np = np.asarray(w.x)
                x_shape = x_np.shape
                if len(x_shape) == 3:
                    ndim, nelem, nquad = x_shape
                    x_flat = x_np.reshape(ndim, -1)
                    k2_flat = sqwavenum_func(x_flat)
                    k2 = k2_flat.reshape(nelem, nquad)
                else:
                    k2 = sqwavenum_func(x_np)
                return k2 * u * v
        else:
            k = self._wavenumber

            def mass_form(u, v, w):
                return k * k * u * v

        mass_np = asm(BilinearForm(mass_form), skfem_basis)

        # sparse + sparse = sparse
        stiffness = laplacian_np + mass_np

        # Cache since coefficients are constant (even callable ones are
        # spatially varying but not state-dependent)
        self._stiffness_cached = stiffness

        return stiffness

    def _assemble_load(self, state: Array, time: float) -> Array:
        """Assemble load vector b.

        b_i = integral(f * phi_i)
        """
        if self._load_cached is not None:
            return self._load_cached

        skfem_basis = self._basis.skfem_basis()

        if self._forcing is None:
            load_np = np.zeros(self.nstates())
        else:
            forcing_func = self._forcing

            def linear_form(v, w):
                # w.x shape: (ndim, nelem, nquad)
                x_shape = w.x.shape
                if len(x_shape) == 3:
                    ndim, nelem, nquad = x_shape
                    x_flat = w.x.reshape(ndim, -1)
                    forc_flat = forcing_func(x_flat)
                    forc = forc_flat.reshape(nelem, nquad)
                else:
                    forc = forcing_func(w.x)
                return forc * v

            load_np = asm(LinearForm(linear_form), skfem_basis)

        load = self._bkd.asarray(load_np.astype(np.float64))

        # Cache since forcing is not state-dependent
        self._load_cached = load

        return load

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute spatial residual without Dirichlet enforcement.

        Returns F = b - K*u with Robin/Neumann BC contributions.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Spatial residual. Shape: (nstates,)
        """
        stiffness = self._assemble_stiffness(state, time)
        load = self._assemble_load(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        load = self._apply_bc_to_load(load, time)
        return load - stiffness @ state

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute dF/du without Dirichlet enforcement.

        For the linear Helmholtz equation, dF/du = -K.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian dF/du. Shape: (nstates, nstates)
        """
        stiffness = self._assemble_stiffness(state, time)
        stiffness = self._apply_bc_to_stiffness(stiffness, time)
        return -stiffness

    def initial_condition(self, func: Callable) -> Array:
        """Create initial condition by interpolating a function.

        For Helmholtz (steady-state), this is typically used as a guess
        for iterative solvers.

        Parameters
        ----------
        func : Callable
            Function to interpolate. Takes coordinates (ndim, npts)
            and returns (npts,).

        Returns
        -------
        Array
            DOF values. Shape: (nstates,)
        """
        return self._basis.interpolate(func)

    def __repr__(self) -> str:
        return (
            f"Helmholtz("
            f"nstates={self.nstates()}, "
            f"wavenumber={self._wavenumber})"
        )
