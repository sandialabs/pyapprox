"""Scalar fields defined via SymPy expressions.

Provides classes for defining scalar fields (e.g., bed elevation, surface
height, initial conditions) using symbolic expressions. Supports evaluation
at mesh points and automatic gradient computation.
"""

from typing import Dict, Generic, List, Tuple

import numpy as np
import sympy as sp

from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
)
from pyapprox.pde.collocation.protocols.mesh import (
    MeshWithTransformProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class SympyField2D(Generic[Array]):
    """Scalar field defined via SymPy expression.

    Generates bed elevations, surfaces, or other scalar fields from
    symbolic expressions. Supports evaluation at mesh points and
    automatic gradient computation.

    Parameters
    ----------
    expr_str : str
        SymPy expression for the field. Coordinates can be:
        - 'xn', 'yn': Normalized coordinates in [0, 1]
        - 'x', 'y': Physical coordinates
    params : Dict[str, float]
        Parameter values to substitute (e.g., {"mean": 10.0, "amp": 0.5}).
    mesh : MeshWithTransformProtocol
        Mesh providing physical coordinates and transform.
    bkd : Backend[Array]
        Computational backend.
    coord_type : str
        "normalized" (default) or "physical" - which coordinates expr_str uses.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Bed with quadratic dip (uses normalized coords)
    >>> bed = SympyField2D(
    ...     expr_str="mean - yn * (1 - yn)",
    ...     params={"mean": 10.0},
    ...     mesh=mesh,
    ...     bkd=bkd,
    ... )
    >>> bed_values = bed.evaluate()  # Shape: (npts,)
    """

    def __init__(
        self,
        expr_str: str,
        params: Dict[str, float],
        mesh: MeshWithTransformProtocol[Array],
        bkd: Backend[Array],
        coord_type: str = "normalized",
    ):
        if coord_type not in ("normalized", "physical"):
            raise ValueError("coord_type must be 'normalized' or 'physical'")

        self._bkd = bkd
        self._mesh = mesh
        self._coord_type = coord_type

        # Define symbols
        xn, yn = sp.symbols("xn yn")  # Normalized [0,1]
        x, y = sp.symbols("x y")  # Physical

        # Build local dictionary for sympify
        local_dict = {"xn": xn, "yn": yn, "x": x, "y": y}
        for name in params:
            local_dict[name] = sp.Symbol(name)

        # Parse and substitute params
        expr = sp.sympify(expr_str, locals=local_dict)
        for name, val in params.items():
            expr = expr.subs(sp.Symbol(name), val)

        self._expr = expr

        # Compute gradient symbolically
        if coord_type == "normalized":
            self._grad_exprs = [sp.diff(expr, xn), sp.diff(expr, yn)]
            self._coord_symbols = (xn, yn)
        else:
            self._grad_exprs = [sp.diff(expr, x), sp.diff(expr, y)]
            self._coord_symbols = (x, y)

        # Lambdify for numerical evaluation
        self._eval_func = sp.lambdify(self._coord_symbols, expr, "numpy")
        self._grad_funcs = [
            sp.lambdify(self._coord_symbols, g, "numpy") for g in self._grad_exprs
        ]

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> MeshWithTransformProtocol[Array]:
        """Return the mesh."""
        return self._mesh

    def evaluate(self) -> Array:
        """Evaluate field at mesh points.

        Returns
        -------
        Array
            Field values. Shape: (npts,)
        """
        coords = self._get_coords()
        vals = self._eval_func(coords[0], coords[1])
        # Handle case where expression is constant (returns scalar)
        if np.isscalar(vals):
            vals = np.full(self._mesh.npts(), vals)
        return self._bkd.asarray(vals)

    def gradient(self) -> List[Array]:
        """Compute gradient at mesh points in physical coordinates.

        Returns
        -------
        List[Array]
            [df/dx, df/dy] in physical Cartesian coordinates. Each shape: (npts,)
        """
        coords = self._get_coords()
        npts = self._mesh.npts()

        if self._coord_type == "normalized":
            # Gradient in normalized coordinates
            grad_normalized = []
            for f in self._grad_funcs:
                val = f(coords[0], coords[1])
                if np.isscalar(val):
                    val = np.full(npts, val)
                grad_normalized.append(self._bkd.asarray(val))

            # Check if mesh has curvilinear transform
            transform = self._mesh.transform()
            if transform is None or isinstance(transform, AffineTransform2D):
                # Affine case: simple scaling
                # dxn/dx_phys = 1 / Lx, dyn/dy_phys = 1 / Ly
                bounds = self._get_physical_bounds()
                Lx = bounds[1] - bounds[0]
                Ly = bounds[3] - bounds[2]
                return [
                    grad_normalized[0] / Lx,
                    grad_normalized[1] / Ly,
                ]
            else:
                # Curvilinear case: use mesh gradient_factors
                # First convert normalized -> reference: xn = (x_ref + 1) / 2
                # So d/d(x_ref) = 0.5 * d/d(xn)
                grad_ref = [0.5 * g for g in grad_normalized]

                # Apply gradient_factors to convert ref -> physical Cartesian
                # grad_phys[d] = sum_j G[:, d, j] * grad_ref[j]
                G = self._mesh.gradient_factors()  # (npts, 2, 2)
                grad_x = G[:, 0, 0] * grad_ref[0] + G[:, 0, 1] * grad_ref[1]
                grad_y = G[:, 1, 0] * grad_ref[0] + G[:, 1, 1] * grad_ref[1]
                return [grad_x, grad_y]

        else:  # physical coordinates
            result = []
            for f in self._grad_funcs:
                val = f(coords[0], coords[1])
                if np.isscalar(val):
                    val = np.full(npts, float(val))
                result.append(self._bkd.asarray(val))
            return result

    def _get_physical_bounds(self) -> Tuple[float, float, float, float]:
        """Get physical domain bounds from mesh or transform."""
        transform = self._mesh.transform()
        if transform is not None and hasattr(transform, "physical_bounds"):
            return transform.physical_bounds()
        else:
            # Fall back to computing from mesh points
            pts = self._mesh.points()
            return (
                self._bkd.to_float(self._bkd.min(pts[0, :])),
                self._bkd.to_float(self._bkd.max(pts[0, :])),
                self._bkd.to_float(self._bkd.min(pts[1, :])),
                self._bkd.to_float(self._bkd.max(pts[1, :])),
            )

    def _get_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinates for evaluation."""
        physical_pts = self._mesh.points()  # Shape: (2, npts)

        if self._coord_type == "normalized":
            # Normalize to [0, 1]
            bounds = self._get_physical_bounds()
            x_phys = self._bkd.to_numpy(physical_pts[0, :])
            y_phys = self._bkd.to_numpy(physical_pts[1, :])
            xn = (x_phys - bounds[0]) / (bounds[1] - bounds[0])
            yn = (y_phys - bounds[2]) / (bounds[3] - bounds[2])
            return (xn, yn)
        else:
            return (
                self._bkd.to_numpy(physical_pts[0, :]),
                self._bkd.to_numpy(physical_pts[1, :]),
            )


def create_quadratic_bed(
    mesh: MeshWithTransformProtocol[Array],
    bkd: Backend[Array],
    mean: float = 10.0,
    amplitude: float = 1.0,
    direction: str = "y",
) -> SympyField2D[Array]:
    """Create bed with quadratic dip.

    bed = mean - amplitude * coord * (1 - coord)
    where coord is normalized to [0, 1].

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing coordinates.
    bkd : Backend
        Computational backend.
    mean : float
        Mean bed elevation.
    amplitude : float
        Amplitude of the dip.
    direction : str
        "x", "y", or "xy" - direction of the dip.

    Returns
    -------
    SympyField2D
        Field representing the bed elevation.
    """
    if direction == "y":
        expr = "mean - amp * yn * (1 - yn)"
    elif direction == "x":
        expr = "mean - amp * xn * (1 - xn)"
    elif direction == "xy":
        expr = "mean - amp * xn * (1 - xn) * yn * (1 - yn)"
    else:
        raise ValueError("direction must be 'x', 'y', or 'xy'")

    return SympyField2D(
        expr_str=expr,
        params={"mean": mean, "amp": amplitude},
        mesh=mesh,
        bkd=bkd,
    )


def create_polynomial_surface(
    mesh: MeshWithTransformProtocol[Array],
    bkd: Backend[Array],
    x_coefs: List[float],
    y_coefs: List[float],
) -> SympyField2D[Array]:
    """Create surface as product of polynomial profiles.

    surface = (a0 + a1*xn + a2*xn^2 + ...) * (b0 + b1*yn + b2*yn^2 + ...)

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing coordinates.
    bkd : Backend
        Computational backend.
    x_coefs : List[float]
        Polynomial coefficients [a0, a1, a2, ...] for x-direction.
    y_coefs : List[float]
        Polynomial coefficients [b0, b1, b2, ...] for y-direction.

    Returns
    -------
    SympyField2D
        Field representing the surface.
    """
    # Build expression strings
    x_terms = " + ".join(f"a{i}*xn**{i}" for i in range(len(x_coefs)))
    y_terms = " + ".join(f"b{i}*yn**{i}" for i in range(len(y_coefs)))
    expr = f"({x_terms}) * ({y_terms})"

    params: Dict[str, float] = {f"a{i}": c for i, c in enumerate(x_coefs)}
    params.update({f"b{i}": c for i, c in enumerate(y_coefs)})

    return SympyField2D(expr_str=expr, params=params, mesh=mesh, bkd=bkd)


def create_beta_surface(
    mesh: MeshWithTransformProtocol[Array],
    bkd: Backend[Array],
    a0: float,
    b0: float,
    a1: float,
    b1: float,
    scale: float = 1.0,
) -> SympyField2D[Array]:
    """Create surface using beta function profile.

    Smooth bump shape commonly used for initial water surface.
    surface = const * xn^(a0-1) * (1-xn)^(b0-1) * yn^(a1-1) * (1-yn)^(b1-1)

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing coordinates.
    bkd : Backend
        Computational backend.
    a0, b0 : float
        Beta function shape parameters for x-direction.
    a1, b1 : float
        Beta function shape parameters for y-direction.
    scale : float
        Overall scale factor.

    Returns
    -------
    SympyField2D
        Field representing the surface.
    """
    from scipy.special import beta as beta_fn

    const = scale / (beta_fn(a0, b0) * beta_fn(a1, b1))

    expr = "const * xn**(a0-1) * (1-xn)**(b0-1) * yn**(a1-1) * (1-yn)**(b1-1)"
    return SympyField2D(
        expr_str=expr,
        params={
            "a0": a0,
            "b0": b0,
            "a1": a1,
            "b1": b1,
            "const": const,
        },
        mesh=mesh,
        bkd=bkd,
    )


def create_shallow_wave_bed(
    mesh: MeshWithTransformProtocol[Array],
    bkd: Backend[Array],
) -> SympyField2D[Array]:
    """Create default shallow wave bed from legacy examples.

    bed = -1 + (-0.1 + yn*(yn-1)) * (1 - 0.9*xn)

    Elevation is higher at boundaries, lower in interior.
    Wave propagates faster in deeper water (lower bed elevation).

    Parameters
    ----------
    mesh : MeshWithTransformProtocol
        Mesh providing coordinates.
    bkd : Backend
        Computational backend.

    Returns
    -------
    SympyField2D
        Field representing the bed elevation.
    """
    expr = "-1 + (-0.1 + yn*(yn-1)) * (1 - 0.9*xn)"
    return SympyField2D(expr_str=expr, params={}, mesh=mesh, bkd=bkd)
