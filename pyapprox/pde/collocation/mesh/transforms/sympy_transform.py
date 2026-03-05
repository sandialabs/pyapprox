"""User-defined coordinate transforms via SymPy expressions.

Allows users to define arbitrary orthogonal curvilinear coordinate transforms
using symbolic expressions. The class automatically:
1. Computes symbolic derivatives (Jacobian) via SymPy
2. Generates efficient NumPy/PyTorch-compatible lambdified functions
3. Derives scale factors, unit basis, and gradient factors from the Jacobian
4. Validates orthogonality at construction time
"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Tuple

import numpy as np
import sympy as sp

from pyapprox.util.backends.protocols import Array, Backend


class BaseSympyTransform(Generic[Array], ABC):
    """Base class for user-defined orthogonal transforms via SymPy.

    Provides shared lambdification and validation logic for 2D and 3D transforms.

    Parameters
    ----------
    exprs : List[str]
        SymPy expressions for physical coordinates as functions of reference.
    params : Dict[str, float]
        Parameter values to substitute.
    bkd : Backend[Array]
        Computational backend.
    coord_names : Tuple[str, ...]
        Names of reference coordinate variables.
    bounds : Tuple[Tuple[float, float], ...]
        Reference domain bounds for each coordinate.
    inv_exprs : List[str], optional
        SymPy expressions for inverse mapping (reference from physical).
    validate : bool
        If True, validate orthogonality and inverse at construction.
    validation_npts : int
        Number of points per dimension for validation.
    """

    def __init__(
        self,
        exprs: List[str],
        params: Dict[str, float],
        bkd: Backend[Array],
        coord_names: Tuple[str, ...],
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        inv_exprs: Optional[List[str]] = None,
        validate: bool = True,
        validation_npts: int = 5,
    ):
        self._bkd = bkd
        self._bounds = bounds
        self._coord_names = coord_names
        self._has_symbolic_inverse = inv_exprs is not None

        # Setup symbols and lambdify
        self._setup_symbols(exprs, params, inv_exprs)

        # Validation (only if bounds provided)
        if validate and bounds is not None:
            test_pts = self._generate_test_points(validation_npts)
            self._validate_orthogonality(test_pts)
            if self._has_symbolic_inverse:
                self._validate_inverse(test_pts)

    @abstractmethod
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        ...

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def bounds(self) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Return reference domain bounds."""
        return self._bounds

    def _setup_symbols(
        self,
        exprs: List[str],
        params: Dict[str, float],
        inv_exprs: Optional[List[str]],
    ) -> None:
        """Setup symbolic expressions and lambdify them."""
        ndim = len(self._coord_names)

        # Create coordinate symbols
        coord_symbols = [sp.Symbol(name) for name in self._coord_names]
        self._coord_symbols = coord_symbols

        # Parse and substitute parameters
        phys_exprs = []
        for expr_str in exprs:
            expr = sp.sympify(expr_str)
            for name, val in params.items():
                expr = expr.subs(sp.Symbol(name), val)
            phys_exprs.append(sp.simplify(expr))
        self._phys_exprs = phys_exprs

        # Compute Jacobian: J[i,j] = d(phys_i)/d(coord_j)
        jac_sym = [
            [sp.diff(phys_exprs[i], coord_symbols[j]) for j in range(ndim)]
            for i in range(ndim)
        ]
        self._jac_sym = jac_sym

        # Scale factors: h_j = ||d(r)/d(q_j)|| = sqrt(sum_i (J[i,j])^2)
        h_syms = []
        for j in range(ndim):
            col = [jac_sym[i][j] for i in range(ndim)]
            h_j = sp.sqrt(sum(c**2 for c in col))
            h_syms.append(sp.simplify(h_j))
        self._h_syms = h_syms

        # Unit basis vectors: e_j = (1/h_j) * (J[:,j])
        e_syms: List[List[sp.Expr]] = []
        for j in range(ndim):
            col = [jac_sym[i][j] for i in range(ndim)]
            e_j = [sp.simplify(c / h_syms[j]) for c in col]
            e_syms.append(e_j)
        self._e_syms = e_syms

        # Jacobian determinant (for orthogonal: product of scale factors)
        det_sym = sp.simplify(sp.prod(h_syms))
        self._det_sym = det_sym

        # Lambdify all expressions for numerical evaluation
        self._map_funcs = [
            sp.lambdify(coord_symbols, expr, "numpy") for expr in phys_exprs
        ]
        self._jac_funcs = [
            [sp.lambdify(coord_symbols, jac_sym[i][j], "numpy") for j in range(ndim)]
            for i in range(ndim)
        ]
        self._h_funcs = [sp.lambdify(coord_symbols, h, "numpy") for h in h_syms]
        self._e_funcs = [
            [sp.lambdify(coord_symbols, e_syms[j][i], "numpy") for j in range(ndim)]
            for i in range(ndim)
        ]
        self._det_func = sp.lambdify(coord_symbols, det_sym, "numpy")

        # Inverse mapping (if provided)
        if inv_exprs is not None:
            phys_symbols = [sp.Symbol(name) for name in ["x", "y", "z"][:ndim]]
            self._phys_symbols = phys_symbols
            inv_parsed = []
            for expr_str in inv_exprs:
                expr = sp.sympify(expr_str)
                for name, val in params.items():
                    expr = expr.subs(sp.Symbol(name), val)
                inv_parsed.append(sp.simplify(expr))
            self._inv_funcs = [
                sp.lambdify(phys_symbols, expr, "numpy") for expr in inv_parsed
            ]
        else:
            self._inv_funcs = None

    def _generate_test_points(self, npts_per_dim: int) -> Array:
        """Generate interior test points for validation."""
        if self._bounds is None:
            raise ValueError("bounds required for validation")

        # Chebyshev nodes on [-1, 1]
        k = np.arange(npts_per_dim)
        cheb = np.cos((2 * k + 1) / (2 * npts_per_dim) * np.pi)
        # Map to [0.1, 0.9] of each bounds interval to avoid boundaries
        nodes_list = []
        for lb, ub in self._bounds:
            # Map cheb from [-1,1] to [lb + 0.1*(ub-lb), lb + 0.9*(ub-lb)]
            margin = 0.1 * (ub - lb)
            nodes = lb + margin + 0.5 * (ub - lb - 2 * margin) * (cheb + 1)
            nodes_list.append(nodes)

        # Create tensor product grid
        grids = np.meshgrid(*nodes_list, indexing="ij")
        test_pts = np.stack([g.ravel() for g in grids], axis=0)
        return self._bkd.asarray(test_pts)

    def _validate_orthogonality(
        self, test_pts: Array, rtol: float = 1e-8
    ) -> None:
        """Verify coordinate system is orthogonal."""
        jac = self.jacobian_matrix(test_pts)
        npts = test_pts.shape[1]
        ndim = self.ndim()

        for pt_idx in range(npts):
            J = jac[pt_idx, :, :]
            # Metric tensor: g = J^T @ J
            g = self._bkd.dot(J.T, J)

            # Check off-diagonal elements
            for i in range(ndim):
                for j in range(i + 1, ndim):
                    off_diag = float(g[i, j])
                    diag_scale = float(
                        self._bkd.sqrt(g[i, i] * g[j, j])
                    )
                    if diag_scale > 1e-14:
                        relative_off_diag = abs(off_diag) / diag_scale
                    else:
                        relative_off_diag = abs(off_diag)

                    if relative_off_diag > rtol:
                        pt_coords = ", ".join(
                            f"{name}={float(test_pts[d, pt_idx]):.4f}"
                            for d, name in enumerate(self._coord_names)
                        )
                        raise ValueError(
                            f"Non-orthogonal coordinate system at ({pt_coords}): "
                            f"g[{i},{j}]/sqrt(g[{i},{i}]*g[{j},{j}])="
                            f"{relative_off_diag:.2e}. "
                            f"SympyTransform requires orthogonal coordinates."
                        )

    def _validate_inverse(self, test_pts: Array, rtol: float = 1e-8) -> None:
        """Verify map_to_reference inverts map_to_physical."""
        physical = self.map_to_physical(test_pts)
        recovered = self.map_to_reference(physical)

        diff = self._bkd.abs(recovered - test_pts)
        scale = self._bkd.maximum(self._bkd.abs(test_pts), self._bkd.asarray([1e-10]))
        max_rel_error = self._bkd.to_float(self._bkd.max(diff / scale))

        if max_rel_error > rtol:
            raise ValueError(
                "Inverse mapping inconsistent: "
                f"max relative error = {max_rel_error:.2e}. "
                f"Check that inverse expressions correctly invert forward mapping."
            )

    def _get_coords_numpy(self, reference_pts: Array) -> List[np.ndarray]:
        """Extract coordinates as numpy arrays."""
        return [
            self._bkd.to_numpy(reference_pts[i, :]) for i in range(self.ndim())
        ]

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Map from reference to physical coordinates.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Physical coordinates. Shape: (ndim, npts)
        """
        coords = self._get_coords_numpy(reference_pts)
        phys = [self._bkd.asarray(np.asarray(f(*coords))) for f in self._map_funcs]
        return self._bkd.stack(phys, axis=0)

    def map_to_reference(self, physical_pts: Array) -> Array:
        """Map from physical to reference coordinates.

        Parameters
        ----------
        physical_pts : Array
            Physical coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Reference coordinates. Shape: (ndim, npts)
        """
        if self._has_symbolic_inverse and self._inv_funcs is not None:
            # Use symbolic inverse
            phys_coords = [
                self._bkd.to_numpy(physical_pts[i, :]) for i in range(self.ndim())
            ]
            ref = [
                self._bkd.asarray(np.asarray(f(*phys_coords)))
                for f in self._inv_funcs
            ]
            return self._bkd.stack(ref, axis=0)
        else:
            # Numerical inversion via optimization
            return self._numerical_inverse(physical_pts)

    def _numerical_inverse(self, physical_pts: Array) -> Array:
        """Compute inverse mapping via numerical optimization."""
        from scipy.optimize import fsolve

        if self._bounds is None:
            raise ValueError("bounds required for numerical inverse")

        npts = physical_pts.shape[1]
        ndim = self.ndim()
        phys_np = self._bkd.to_numpy(physical_pts)

        result = np.zeros((ndim, npts))

        # Initial guess: midpoint of bounds
        init_guess = np.array(
            [0.5 * (lb + ub) for lb, ub in self._bounds]
        )

        for i in range(npts):
            target = phys_np[:, i]

            def residual(ref_coords: np.ndarray) -> np.ndarray:
                phys = np.array([f(*ref_coords) for f in self._map_funcs])
                return phys - target

            sol, info, ier, msg = fsolve(residual, init_guess, full_output=True)
            if ier != 1:
                raise RuntimeError(
                    f"Inverse mapping failed at point {target}: {msg}"
                )
            result[:, i] = sol

        return self._bkd.asarray(result)

    def jacobian_matrix(self, reference_pts: Array) -> Array:
        """Compute Jacobian matrix of the mapping.

        J[i,j] = d(physical_i)/d(reference_j)

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian matrices. Shape: (npts, ndim, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self.ndim()
        coords = self._get_coords_numpy(reference_pts)

        jac = self._bkd.zeros((npts, ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                val = np.asarray(self._jac_funcs[i][j](*coords))
                jac[:, i, j] = self._bkd.asarray(val)
        return jac

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Compute Jacobian determinant.

        For orthogonal coordinates: det(J) = h_1 * h_2 * ... * h_n

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Jacobian determinants. Shape: (npts,)
        """
        coords = self._get_coords_numpy(reference_pts)
        return self._bkd.asarray(np.asarray(self._det_func(*coords)))

    def scale_factors(self, reference_pts: Array) -> Array:
        """Compute scale factors for curvilinear coordinates.

        h_j = ||d(r)/d(q_j)|| for each coordinate j.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Scale factors. Shape: (npts, ndim)
        """
        npts = reference_pts.shape[1]
        ndim = self.ndim()
        coords = self._get_coords_numpy(reference_pts)

        scales = self._bkd.zeros((npts, ndim))
        for j in range(ndim):
            scales[:, j] = self._bkd.asarray(np.asarray(self._h_funcs[j](*coords)))
        return scales

    def unit_curvilinear_basis(self, reference_pts: Array) -> Array:
        """Compute unit curvilinear basis vectors.

        Returns e_j = (1/h_j) * (d(r)/d(q_j)) for each coordinate.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Unit basis vectors. Shape: (npts, ndim, ndim)
            result[:, :, j] = e_j (j-th basis vector in Cartesian components)
        """
        npts = reference_pts.shape[1]
        ndim = self.ndim()
        coords = self._get_coords_numpy(reference_pts)

        basis = self._bkd.zeros((npts, ndim, ndim))
        for i in range(ndim):  # Cartesian component
            for j in range(ndim):  # basis vector index
                val = np.asarray(self._e_funcs[i][j](*coords))
                basis[:, i, j] = self._bkd.asarray(val)
        return basis

    def gradient_factors(self, reference_pts: Array) -> Array:
        """Compute factors for transforming gradients.

        Returns (1/h_j) * e_j for each coordinate.

        Parameters
        ----------
        reference_pts : Array
            Reference coordinates. Shape: (ndim, npts)

        Returns
        -------
        Array
            Gradient factors. Shape: (npts, ndim, ndim)
            result[:, :, j] = e_j / h_j
        """
        unit_basis = self.unit_curvilinear_basis(reference_pts)
        scale = self.scale_factors(reference_pts)
        return unit_basis / scale[:, None, :]


class SympyTransform2D(BaseSympyTransform[Array]):
    """User-defined 2D orthogonal transform from SymPy expressions.

    Parameters
    ----------
    x_expr : str
        SymPy expression for x(u, v).
    y_expr : str
        SymPy expression for y(u, v).
    params : Dict[str, float]
        Parameter values to substitute.
    bkd : Backend[Array]
        Computational backend.
    coord_names : Tuple[str, str]
        Names of reference coordinates, default ("u", "v").
    bounds : Tuple[Tuple[float, float], Tuple[float, float]], optional
        Reference domain bounds ((u_min, u_max), (v_min, v_max)).
    x_inv_expr : str, optional
        SymPy expression for u(x, y).
    y_inv_expr : str, optional
        SymPy expression for v(x, y).
    validate : bool
        If True (default), validate orthogonality at construction.
    validation_npts : int
        Number of points per dimension for validation.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import math
    >>> bkd = NumpyBkd()
    >>> # Polar coordinates
    >>> tf = SympyTransform2D(
    ...     x_expr="r * cos(theta)",
    ...     y_expr="r * sin(theta)",
    ...     params={},
    ...     bkd=bkd,
    ...     coord_names=("r", "theta"),
    ...     bounds=((0.5, 2.0), (-math.pi, math.pi)),
    ...     x_inv_expr="sqrt(x**2 + y**2)",
    ...     y_inv_expr="atan2(y, x)",
    ... )
    """

    def __init__(
        self,
        x_expr: str,
        y_expr: str,
        params: Dict[str, float],
        bkd: Backend[Array],
        coord_names: Tuple[str, str] = ("u", "v"),
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        x_inv_expr: Optional[str] = None,
        y_inv_expr: Optional[str] = None,
        validate: bool = True,
        validation_npts: int = 5,
    ):
        inv_exprs = (
            [x_inv_expr, y_inv_expr]
            if x_inv_expr is not None and y_inv_expr is not None
            else None
        )
        super().__init__(
            exprs=[x_expr, y_expr],
            params=params,
            bkd=bkd,
            coord_names=coord_names,
            bounds=bounds,
            inv_exprs=inv_exprs,
            validate=validate,
            validation_npts=validation_npts,
        )

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 2


class SympyTransform3D(BaseSympyTransform[Array]):
    """User-defined 3D orthogonal transform from SymPy expressions.

    Parameters
    ----------
    x_expr : str
        SymPy expression for x(u, v, w).
    y_expr : str
        SymPy expression for y(u, v, w).
    z_expr : str
        SymPy expression for z(u, v, w).
    params : Dict[str, float]
        Parameter values to substitute.
    bkd : Backend[Array]
        Computational backend.
    coord_names : Tuple[str, str, str]
        Names of reference coordinates, default ("u", "v", "w").
    bounds : Tuple[Tuple[float, float], ...], optional
        Reference domain bounds.
    x_inv_expr : str, optional
        SymPy expression for u(x, y, z).
    y_inv_expr : str, optional
        SymPy expression for v(x, y, z).
    z_inv_expr : str, optional
        SymPy expression for w(x, y, z).
    validate : bool
        If True (default), validate orthogonality at construction.
    validation_npts : int
        Number of points per dimension for validation.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import math
    >>> bkd = NumpyBkd()
    >>> # Spherical coordinates
    >>> tf = SympyTransform3D(
    ...     x_expr="r * sin(theta) * cos(phi)",
    ...     y_expr="r * sin(theta) * sin(phi)",
    ...     z_expr="r * cos(theta)",
    ...     params={},
    ...     bkd=bkd,
    ...     coord_names=("r", "theta", "phi"),
    ...     bounds=((1.0, 2.0), (0.1, math.pi - 0.1), (0, 2 * math.pi)),
    ... )
    """

    def __init__(
        self,
        x_expr: str,
        y_expr: str,
        z_expr: str,
        params: Dict[str, float],
        bkd: Backend[Array],
        coord_names: Tuple[str, str, str] = ("u", "v", "w"),
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        x_inv_expr: Optional[str] = None,
        y_inv_expr: Optional[str] = None,
        z_inv_expr: Optional[str] = None,
        validate: bool = True,
        validation_npts: int = 5,
    ):
        inv_exprs = (
            [x_inv_expr, y_inv_expr, z_inv_expr]
            if (
                x_inv_expr is not None
                and y_inv_expr is not None
                and z_inv_expr is not None
            )
            else None
        )
        super().__init__(
            exprs=[x_expr, y_expr, z_expr],
            params=params,
            bkd=bkd,
            coord_names=coord_names,
            bounds=bounds,
            inv_exprs=inv_exprs,
            validate=validate,
            validation_npts=validation_npts,
        )

    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return 3
