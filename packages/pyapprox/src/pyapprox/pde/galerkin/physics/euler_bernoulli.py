"""Euler-Bernoulli beam models: analytical and FEM.

Solves the cantilever beam equation:

    EI * w''''(x) = q(x)

where w is the transverse deflection, EI is the bending stiffness,
and q(x) is the distributed transverse load.

The FEM formulation uses cubic Hermite (C1) elements with DOFs
[w, dw/dx] at each node. The weak form is:

    integral(EI * w'' * v'') dx = integral(q * v) dx

for all test functions v in the Hermite finite element space.
"""

from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Union

if TYPE_CHECKING:
    from skfem.assembly.form.form import FormExtraParams
    from skfem.element.discrete_field import DiscreteField


import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.galerkin.basis.hermite import CubicHermiteBasis
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import BilinearForm, LinearForm, asm
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )


class EulerBernoulliBeamAnalytical(Generic[Array]):
    """Analytical solution for a cantilever Euler-Bernoulli beam.

    Clamped at x=0 (w=0, w'=0), free at x=L (M=0, V=0).

    Supports two load types:
    - Uniform load: q(x) = q0
    - Linearly increasing load: q(x) = q0 * x / L

    Parameters
    ----------
    length : float
        Beam length L.
    EI : float
        Bending stiffness (Young's modulus times second moment of area).
    q0 : float
        Load magnitude.
    load_type : str, default="uniform"
        Either "uniform" for q(x)=q0 or "linear" for q(x)=q0*x/L.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        length: float,
        EI: float,
        q0: float,
        bkd: Backend[Array],
        load_type: str = "uniform",
    ):
        self._length = length
        self._EI = EI
        self._q0 = q0
        self._bkd = bkd
        if load_type not in ("uniform", "linear"):
            raise ValueError(
                f"load_type must be 'uniform' or 'linear', got '{load_type}'"
            )
        self._load_type = load_type

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def length(self) -> float:
        return self._length

    def EI(self) -> float:
        return self._EI

    def q0(self) -> float:
        return self._q0

    def deflection(self, x: Array) -> Array:
        """Compute deflection w(x).

        Parameters
        ----------
        x : Array
            Evaluation points. Shape: (npts,)

        Returns
        -------
        Array
            Deflection values. Shape: (npts,)
        """
        L = self._length
        EI = self._EI
        q0 = self._q0

        if self._load_type == "uniform":
            # w(x) = q0/(24*EI) * x^2 * (6*L^2 - 4*L*x + x^2)
            return (q0 / (24.0 * EI)) * x**2 * (6.0 * L**2 - 4.0 * L * x + x**2)
        else:
            # w(x) = q0/(120*L*EI) * x^2 * (20*L^3 - 10*L^2*x + x^3)
            return (
                (q0 / (120.0 * L * EI)) * x**2 * (20.0 * L**3 - 10.0 * L**2 * x + x**3)
            )

    def slope(self, x: Array) -> Array:
        """Compute slope dw/dx(x).

        Parameters
        ----------
        x : Array
            Evaluation points. Shape: (npts,)

        Returns
        -------
        Array
            Slope values. Shape: (npts,)
        """
        L = self._length
        EI = self._EI
        q0 = self._q0

        if self._load_type == "uniform":
            # w'(x) = q0/(6*EI) * x * (3*L^2 - 3*L*x + x^2)
            return (q0 / (6.0 * EI)) * x * (3.0 * L**2 - 3.0 * L * x + x**2)
        else:
            # w'(x) = q0/(120*L*EI) * x * (40*L^3 - 30*L^2*x + 5*x^3)
            return (
                (q0 / (120.0 * L * EI))
                * x
                * (40.0 * L**3 - 30.0 * L**2 * x + 5.0 * x**3)
            )

    def tip_deflection(self) -> float:
        """Compute tip deflection w(L).

        Returns
        -------
        float
            Deflection at x=L.
        """
        L = self._length
        EI = self._EI
        q0 = self._q0
        if self._load_type == "uniform":
            # w(L) = q0*L^4 / (8*EI)
            return q0 * L**4 / (8.0 * EI)
        else:
            # w(L) = 11*q0*L^4 / (120*EI)
            return 11.0 * q0 * L**4 / (120.0 * EI)


class EulerBernoulliBeamFEM(Generic[Array]):
    r"""Euler-Bernoulli beam interior operator on a cubic Hermite basis.

    The interior residual is :math:`F_\Omega(u) = b - K u` with
    :math:`K_{ij} = \int EI\, \phi_i'' \phi_j''` and
    :math:`b_i = \int q\, \phi_i`. The physics holds no boundary
    conditions: compose it with them, e.g. a cantilever clamped at the
    left end,

    .. code-block:: python

        system = compose_galerkin_system(
            beam, [DirichletBC(basis, "left", 0.0, bkd)]
        )

    and solve the system (``GalerkinModel(system, bkd).solve_steady``).
    An end with no condition is free (zero moment and shear, natural in
    the weak form). Read results through the basis:
    ``basis.values(u)``, ``basis.slopes(u)``, ``basis.evaluate(u, x)``.

    Parameters
    ----------
    basis : CubicHermiteBasis
        The C1 basis; DOFs ``[w_0, w'_0, w_1, w'_1, ...]``.
    EI : float or np.ndarray
        Bending stiffness: a scalar, or one value per element.
    load_func : Callable
        Distributed load :math:`q(x)`; takes a NumPy array of
        x-coordinates and returns the loads.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        basis: CubicHermiteBasis[Array],
        EI: Union[float, np.ndarray],
        load_func: Callable[..., Any],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(basis, CubicHermiteBasis):
            raise TypeError(
                "basis must be a CubicHermiteBasis (the beam needs C1 "
                f"elements), got {type(basis).__name__}"
            )
        self._basis = basis
        self._load_func = load_func
        self._bkd = bkd
        self._stiffness: Optional[Array] = None
        self._mass: Optional[Array] = None
        self._EI: Union[float, np.ndarray] = 0.0
        self.set_EI(EI)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def basis(self) -> CubicHermiteBasis[Array]:
        """Return the cubic Hermite basis."""
        return self._basis

    def nstates(self) -> int:
        """Return the number of DOFs (two per node)."""
        return self._basis.ndofs()

    def EI(self) -> Union[float, np.ndarray]:
        return self._EI

    def set_EI(self, EI: Union[float, np.ndarray]) -> None:
        """Set the bending stiffness and invalidate the stiffness cache.

        Raises
        ------
        ValueError
            If an array ``EI`` does not have one value per element.
        """
        nelements = self._basis.mesh().nelements()
        if isinstance(EI, np.ndarray) and EI.shape != (nelements,):
            raise ValueError(
                f"EI must have one value per element, shape ({nelements},), "
                f"got {EI.shape}"
            )
        self._EI = EI
        self._stiffness = None

    def set_load_func(self, load_func: Callable[..., Any]) -> None:
        """Set the distributed load :math:`q(x)`."""
        self._load_func = load_func

    def stiffness_matrix(self) -> Array:
        """Return beam stiffness matrix K_ij = integral(EI * w''_i * w''_j).

        Returns
        -------
        Array
            Stiffness matrix. Shape: (ndofs, ndofs)
        """
        if self._stiffness is not None:
            return self._stiffness

        EI_val = self._EI

        if isinstance(EI_val, np.ndarray):
            x_nodes = self._bkd.to_numpy(self._basis.mesh().nodes())[0]
            ei_array = EI_val

            def varying_stiffness_form(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                x = w.x[0]
                elem_idx = np.clip(
                    np.searchsorted(x_nodes[1:], x), 0, len(ei_array) - 1
                )
                ret: NDArray[np.floating[Any]] = (
                    ei_array[elem_idx]
                    * u.hess[0, 0]
                    * v.hess[0, 0]
                )
                return ret

            # Wrapped by call, not by decorator: skfem is unstubbed, so
            # `@BilinearForm` would erase these annotations.
            beam_stiffness_form = BilinearForm(varying_stiffness_form)
        else:

            def uniform_stiffness_form(
                u: "DiscreteField",
                v: "DiscreteField",
                w: "FormExtraParams",
            ) -> np.ndarray:
                ret: NDArray[np.floating[Any]] = EI_val * u.hess[0, 0] * v.hess[0, 0]
                return ret

            beam_stiffness_form = BilinearForm(uniform_stiffness_form)

        self._stiffness = asm(beam_stiffness_form, self._basis.skfem_basis())
        return self._stiffness

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        """Exact zero curvature: the interior is linear in u."""
        return StateDerivatives.linear(self._bkd)

    def interior_is_time_invariant(self) -> bool:
        """Always: the distributed load is ``q(x)``, with no time."""
        return True

    def mass_matrix(self) -> Array:
        """Return beam mass matrix M_ij = integral(w_i * w_j).

        Returns
        -------
        Array
            Mass matrix. Shape: (ndofs, ndofs)
        """
        if self._mass is not None:
            return self._mass

        def mass_form(
            u: "DiscreteField",
            v: "DiscreteField",
            w: "FormExtraParams",
        ) -> np.ndarray:
            ret: NDArray[np.floating[Any]] = u.value * v.value
            return ret

        self._mass = asm(BilinearForm(mass_form), self._basis.skfem_basis())
        return self._mass

    def interior_load_vector(self) -> Array:
        """Return the distributed-load vector b_i = integral(q(x) * w_i).

        Body load only; natural-BC terms are added by the composed
        spatial operator.

        Returns
        -------
        Array
            Load vector. Shape: (ndofs,)
        """
        load_func = self._load_func

        def load_form(v: "DiscreteField", w: "FormExtraParams") -> np.ndarray:
            x = w.x[0]
            ret: NDArray[np.floating[Any]] = load_func(x) * v.value
            return ret

        f = asm(LinearForm(load_form), self._basis.skfem_basis())
        return self._bkd.asarray(f.astype(np.float64))

    def interior_residual(self, state: Array, time: float = 0.0) -> Array:
        """Compute the interior residual ``F_Omega = b - K u``.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Residual. Shape: (ndofs,)
        """
        K = self.stiffness_matrix()
        b = self.interior_load_vector()
        return b - K @ state

    def interior_jacobian(self, state: Array, time: float = 0.0) -> Array:
        """Compute ``dF_Omega/du = -K``.

        Parameters
        ----------
        state : Array
            DOF vector. Shape: (ndofs,)
        time : float
            Current time.

        Returns
        -------
        Array
            Jacobian. Shape: (ndofs, ndofs)
        """
        return -self.stiffness_matrix()


def element_curvatures(
    basis: CubicHermiteBasis[Array], coeffs: Array
) -> NDArray[np.floating[Any]]:
    """Element-wise absolute curvature |w''|, from the nodal slopes.

    The difference of the slopes at an element's ends over its length,
    i.e. the element's mean curvature.

    Parameters
    ----------
    basis : CubicHermiteBasis
        The beam's basis.
    coeffs : Array
        DOF vector. Shape: (ndofs,)

    Returns
    -------
    np.ndarray
        Absolute curvature per element. Shape: (nelements,)
    """
    bkd = basis.bkd()
    slopes = bkd.to_numpy(basis.slopes(coeffs))
    x_nodes = bkd.to_numpy(basis.mesh().nodes())[0]
    curvatures: NDArray[np.floating[Any]] = np.abs(
        np.diff(slopes) / np.diff(x_nodes)
    )
    return curvatures
