from abc import ABC, abstractmethod
from typing import List, Tuple

from scipy import stats

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.surrogates.affine.multiindex import compute_hyperbolic_indices
from pyapprox.surrogates.univariate.base import (
    UnivariateInterpolatingBasis,
    UnivariateQuadratureRule,
    UnivariateBasis,
)
from pyapprox.surrogates.univariate.orthopoly import (
    OrthonormalPolynomial1D,
    TrigonometricPolynomial1D,
    FourierBasis1D,
    GaussQuadratureRule,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.visualization import get_meshgrid_samples, plot_surface
from pyapprox.surrogates.univariate.local import (
    UnivariateEquidistantNodeGenerator,
    UnivariatePiecewisePolynomialQuadratureRule,
)


class Basis(ABC):
    """The base class for any multivariate Basis."""

    def __init__(self, backend: BackendMixin):
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend

    def jacobian_implemented(self) -> bool:
        return False

    def hessian_implemented(self) -> bool:
        return False

    @abstractmethod
    def nterms(self) -> int:
        """
        Return the number of basis functions.
        """
        raise NotImplementedError()

    @abstractmethod
    def nvars(self) -> int:
        """
        Return the number of inputs to the basis.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, samples: Array) -> Array:
        """
        Evaluate a multivariate basis at a set of samples.

        Parameters
        ----------
        samples : array (nvars, nsamples)
            Samples at which to evaluate the basis

        Return
        ------
        basis_matrix : array (nsamples, nterms)
            The values of the basis at the samples
        """
        raise NotImplementedError()

    def jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobians of multivariate basis at a set of samples.

        Parameters
        ----------
        samples : array (nvars, nsamples)
            Samples at which to evaluate the basis Jacobian

        Return
        ------
        jac : array (nsamples, nterms, nvars)
            The Jacobian of the basis at each sample
        """
        raise NotImplementedError("Basis jacobian not implemented")

    def __repr__(self):
        return "{0}(nterms={1})".format(self.__class__.__name__, self.nterms())


class MultiIndexBasis(Basis):
    """Multivariate basis defined by multi-indices."""

    def __init__(self, bases_1d: List[UnivariateBasis], indices: Array = None):
        # check that bases_1d are not exactly the same object
        # e.g. created using MultiIndexBasis[basis(nterms)]*nvars)
        # as this will create shallow copies preventing storage of different
        # nterms in each basis. Need to create with
        # MultiIndexBasis[basis(nterms) for ii in range(nvars)])
        for ii in range(len(bases_1d)):
            for jj in range(len(bases_1d)):
                if ii != jj and bases_1d[ii] is bases_1d[jj]:
                    raise ValueError("Each basis must be different")
        super().__init__(bases_1d[0]._bkd)
        self._bases_1d = bases_1d
        self._indices = None
        if indices is not None:
            self.set_indices(indices)

    def jacobian_implemented(self) -> bool:
        for basis_1d in self._bases_1d:
            if not basis_1d.first_derivatives_implemented():
                return False
        return True

    def hessian_implemented(self) -> bool:
        for basis_1d in self._bases_1d:
            if not basis_1d.second_derivatives_implemented():
                return False
        return True

    def set_hyperbolic_indices(self, level: int, pnorm: float):
        indices = self._bkd.asarray(
            compute_hyperbolic_indices(self.nvars(), level, pnorm),
            dtype=int,
        )
        self.set_indices(indices)

    def set_tensor_product_indices(self, nterms: List[int]):
        if len(nterms) != self.nvars():
            raise ValueError("must specify nterms for each dimension")
        self.set_indices(
            self._bkd.cartesian_product(
                [self._bkd.arange(nt) for nt in nterms]
            )
        )

    def _set_nterms(self, nterms_per_1d_basis: List[int]):
        for ii, basis_1d in enumerate(self._bases_1d):
            basis_1d.set_nterms(nterms_per_1d_basis[ii])

    def set_indices(self, indices: Array):
        """
        Set the multivariate indices of the basis functions.

        Parameters
        ----------
        indices : array (nvars, nterms)
            Multivariate indices specifying the basis functions
        """
        if indices.ndim != 2:
            raise ValueError("indices must have two dimensions")
        if indices.shape[0] != len(self._bases_1d):
            raise ValueError(
                "indices.shape[0] {0} doesnt match len(bases_1d) {1}".format(
                    indices.shape[0], len(self._bases_1d)
                )
            )
        self._indices = self._bkd.array(indices, dtype=int)
        self._set_nterms(self._bkd.max(self._indices, axis=1) + 1)

    def get_indices(self) -> Array:
        """Return the indices defining the basis terms."""
        return self._indices

    def nterms(self) -> int:
        if self._indices is None:
            return 0
        return self._indices.shape[1]

    def nvars(self) -> int:
        # use bases_1d so do not have to set indices to determine nvars
        # like is done for base class
        return len(self._bases_1d)

    def _basis_vals_1d(self, samples: Array) -> List[Array]:
        return [
            poly(samples[dd : dd + 1, :])
            for dd, poly in enumerate(self._bases_1d)
        ]

    def _basis_derivs_1d(self, samples: Array, order: int) -> List[Array]:
        return [
            poly.derivatives(samples[dd : dd + 1, :], order)
            for dd, poly in enumerate(self._bases_1d)
        ]

    def __call__(self, samples: Array) -> Array:
        if samples.shape[0] != self.nvars():
            raise ValueError(
                "samples must have nrows={0}".format(self.nvars())
            )
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_matrix = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix *= basis_vals_1d[dd][:, self._indices[dd, :]]
        return basis_matrix

    def jacobian(self, samples: Array) -> Array:
        # return jac with shape (nsamples, nterms, nvars)
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_derivs_1d = self._basis_derivs_1d(samples, 1)
        jac = []
        for dd in range(self.nvars()):
            jac_dd = basis_derivs_1d[dd][:, self._indices[dd, :]]
            for kk in range(self.nvars()):
                if kk != dd:
                    jac_dd *= basis_vals_1d[kk][:, self._indices[kk, :]]
            jac.append(jac_dd)
        return self._bkd.moveaxis(self._bkd.stack(jac, axis=0), 0, -1)

    def hessian(self, samples: Array) -> Array:
        basis_vals_1d = self._basis_vals_1d(samples)
        # todo allow basis derivs to return vals and derivs of all
        # order so bases like orthopoly do not recompute data
        fir_derivs_1d = self._basis_derivs_1d(samples, 1)
        sec_derivs_1d = self._basis_derivs_1d(samples, 2)
        hess = [
            [[] for kk in range(self.nvars())] for dd in range(self.nvars())
        ]
        for dd in range(self.nvars()):
            for kk in range(dd, self.nvars()):
                if kk == dd:
                    hess_dk = sec_derivs_1d[kk][:, self._indices[kk, :]]
                else:
                    hess_dk = fir_derivs_1d[dd][:, self._indices[dd, :]]
                    hess_dk *= fir_derivs_1d[kk][:, self._indices[kk, :]]
                for ll in range(self.nvars()):
                    if ll == kk or ll == dd:
                        continue
                    hess_dk *= basis_vals_1d[ll][:, self._indices[ll, :]]
                hess[dd][kk] = hess_dk
                hess[kk][dd] = hess_dk
        hess = self._bkd.stack(
            [self._bkd.stack(hess[dd], axis=-1) for dd in range(self.nvars())],
            axis=-1,
        )
        return hess

    def __repr__(self) -> str:
        return "{0}(nvars={1}, nterms={2})".format(
            self.__class__.__name__, self.nvars(), self.nterms()
        )


class OrthonormalPolynomialBasis(MultiIndexBasis):
    """Multivariate orthogonal polynomial basis."""

    # TODO: consider adding transform to OthogonalPolyBasis as
    # regressor should not have to know about constraints on
    # poly for instance Legendre samples must be in [-1, 1].
    # The regressor could apply another transform on top of this
    # but only for consistency with other regressors as
    # a second transform would not be necessary if poly
    # had its own.

    def __init__(self, bases_1d, indices=None):
        for poly in bases_1d:
            if not isinstance(poly, OrthonormalPolynomial1D):
                raise ValueError(
                    "poly must be instance of OrthonormalPolynomial1D"
                )
        super().__init__(bases_1d, indices)

    def univariate_quadrature(self, poly_id):
        """Return univariate gauss quadrature rule that can exactly integrate
        all polynomials in the index set associated with the dimension
        requested.
        """
        return self._bases_1d[poly_id].gauss_quadrature_rule(
            self._bkd.max(self._indices[poly_id]) + 1
        )


class TensorProductInterpolatingBasis(MultiIndexBasis):

    def __init__(self, bases_1d):
        for basis in bases_1d:
            if not isinstance(basis, UnivariateInterpolatingBasis):
                raise ValueError(
                    "basis: {0}, must be instance of Basis1D".format(basis)
                )
        super().__init__(bases_1d)
        self._nodes_1d = None

    def set_hyperbolic_indices(self, nvars, nterms, pnorm):
        raise NotImplementedError(
            "{0} cannot have a hyperbolic index set".format(self)
        )

    def tensor_product_grid(self):
        nodes_1d = []
        for basis in self._bases_1d:
            if basis._quad_samples is None:
                raise RuntimeError("must define quad_samples")
            nodes_1d.append(basis._quad_samples[0])
        return self._bkd.cartesian_product(nodes_1d)

    def nterms(self):
        return self._bkd.prod(
            self._bkd.array(
                [basis.nterms() for basis in self._bases_1d], dtype=int
            ),
        )

    def quadrature_rule(self):
        samples_1d, weights_1d = [], []
        for basis in self._bases_1d:
            xx, ww = basis.quadrature_rule()
            samples_1d.append(xx[0])
            weights_1d.append(ww[:, 0])
        samples = self._bkd.cartesian_product(samples_1d)
        weights = self._bkd.outer_product(weights_1d)[:, None]
        return samples, weights

    def _plot_single_basis(
        self,
        ax,
        ii,
        jj,
        nterms_1d,
        plot_limits,
        num_pts_1d,
        surface_cmap,
        contour_cmap,
    ):
        X, Y, pts = get_meshgrid_samples(
            plot_limits, num_pts_1d, bkd=self._bkd
        )
        idx = jj * nterms_1d[0] + ii
        basis_vals = self(pts)
        Z = self._bkd.reshape(basis_vals[:, idx], X.shape)
        if surface_cmap is not None:
            plot_surface(
                X,
                Y,
                Z,
                ax,
                axis_labels=None,
                limit_state=None,
                alpha=0.3,
                cmap=surface_cmap,
                zorder=3,
                plot_axes=False,
            )
        if contour_cmap is not None:
            num_contour_levels = 30
            offset = -(Z.max() - Z.min()) / 2
            ax.contourf(
                X,
                Y,
                Z,
                zdir="z",
                offset=offset,
                levels=self._bkd.linspace(
                    Z.min(), Z.max(), num_contour_levels
                ),
                cmap=contour_cmap,
                zorder=-1,
            )
        return offset, idx, nterms_1d, X, Y

    def _plot_nodes(self, ax, offset, X, Y, idx, nterms_1d):
        nodes = self.tensor_product_grid()
        nodes_1d = [basis.quadrature_rule()[0] for basis in self._bases_1d]
        ax.plot(
            nodes[0, :],
            nodes[1, :],
            offset * self._bkd.ones(nodes.shape[1]),
            "o",
            zorder=100,
            color="b",
        )

        x = self._bkd.linspace(-1, 1, 100)
        y = nodes[1, idx] * self._bkd.ones((x.shape[0],))
        z = self(self._bkd.vstack((x[None, :], y[None, :])))[:, idx]
        ax.plot(x, Y.max() * self._bkd.ones((x.shape[0],)), z, "-r")
        ax.plot(
            nodes_1d[0][0],
            Y.max() * self._bkd.ones((nterms_1d[0],)),
            self._bkd.zeros(
                nterms_1d[0],
            ),
            "or",
        )

        y = self._bkd.linspace(-1, 1, 100)
        x = nodes[0, idx] * self._bkd.ones((y.shape[0],))
        z = self(self._bkd.vstack((x[None, :], y[None, :])))[:, idx]
        ax.plot(X.min() * self._bkd.ones((x.shape[0],)), y, z, "-r")
        ax.plot(
            X.min() * self._bkd.ones((nterms_1d[1],)),
            nodes_1d[1][0],
            self._bkd.zeros(
                nterms_1d[1],
            ),
            "or",
        )

    def plot_single_basis(
        self,
        ax,
        ii,
        jj,
        plot_limits=[-1, 1, -1, 1],
        num_pts_1d=101,
        surface_cmap="coolwarm",
        contour_cmap="gray",
        plot_nodes=False,
    ):
        if self.nvars() != 2:
            raise ValueError("Can only be used when nvars == 2")
        # evaluate 1D basis functions once to get number of basis functions
        sample = (
            self._bkd.reshape(self._bkd.asarray(plot_limits), (2, 2))
            .T[:, :1]
            .T
        )
        nterms_1d = [basis(sample).shape[1] for basis in self._bases_1d]
        offset, idx, nterms_1d, X, Y = self._plot_single_basis(
            ax,
            ii,
            jj,
            nterms_1d,
            plot_limits,
            num_pts_1d,
            surface_cmap,
            contour_cmap,
        )
        if plot_nodes is None:
            return
        self._plot_nodes(ax, offset, X, Y, idx, nterms_1d)

    def plot_basis_1d(self, ax, plot_limits):
        if self.nvars() != 1:
            raise ValueError("Can only be used when nvars == 2")
        plot_xx = self._bkd.linspace(*plot_limits, 101)[None, :]
        ax.plot(plot_xx[0], self.__call__(plot_xx), "k--")

    def _semideep_copy(self):
        # this function can be dangerous so should be used with caution
        # when not wanting to copy all internal data
        return TensorProductInterpolatingBasis(
            [basis._semideep_copy() for basis in self._bases_1d]
        )


class QuadratureRule(ABC):
    def __init__(self, backend):
        if backend is None:
            backend = NumpyMixin
        self._bkd = backend

    @abstractmethod
    def nvars(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__name__
        )


class QuadratureRuleFromData(QuadratureRule):
    def __init__(
        self, quadx: Array, quadw: Array, backend: BackendMixin = NumpyMixin
    ):
        super().__init__(backend)
        if quadw.ndim != 2 or quadw.shape[1] != 1:
            raise ValueError("quadw must be a 2D array column vector")
        if quadx.shape[1] != quadw.shape[0]:
            raise ValueError("quadx and quadw shapes are inconsistent")
        self._quadx = self._bkd.array(quadx)
        self._quadw = self._bkd.array(quadw)

    def nvars(self) -> int:
        return self._quadx.shape[0]

    def __call__(self) -> Tuple[Array, Array]:
        return self._quadx, self._quadw


class TensorProductQuadratureRule(QuadratureRule):
    def __init__(
        self, nvars: int, univariate_quad_rules: List, store: bool = False
    ):
        super().__init__(univariate_quad_rules[0]._bkd)
        if isinstance(univariate_quad_rules, UnivariateQuadratureRule):
            univariate_quad_rules = [univariate_quad_rules] * nvars
        if len(univariate_quad_rules) != nvars:
            raise ValueError(
                "must specify a single quadrature rule or"
                " one for each dimension"
            )
        for quad_rule in univariate_quad_rules:
            if not isinstance(quad_rule, UnivariateQuadratureRule):
                raise ValueError(
                    "quad rule {0} not an instance of "
                    "UnivariateQuadratureRule".format(quad_rule)
                )
        self._nvars = nvars
        self._univariate_quad_rules = univariate_quad_rules
        self._store = store
        self._quad_samples = dict()
        self._quad_weights = dict()

    def nvars(self) -> int:
        return self._nvars

    def __call__(self, nnodes_1d: Array) -> Tuple[Array, Array]:
        if len(nnodes_1d) != self.nvars():
            raise ValueError("must specify nnodes for each dimension")
        np_array = self._bkd.to_numpy(self._bkd.asarray(nnodes_1d, dtype=int))
        key = hash(np_array.tobytes())
        if self._store and key in self._quad_samples:
            return self._quad_samples[key], self._quad_weights[key]
        samples_1d, weights_1d = [], []
        for quad_rule, nnodes in zip(self._univariate_quad_rules, nnodes_1d):
            xx, ww = quad_rule(int(nnodes))
            samples_1d.append(xx[0])
            weights_1d.append(ww[:, 0])
        samples = self._bkd.cartesian_product(samples_1d)
        weights = self._bkd.outer_product(weights_1d)[:, None]
        if self._store:
            self._quad_samples[nnodes] = samples
            self._quad_weights[nnodes] = weights
        return samples, weights

    def univariate_quad_rules(self):
        return self._univariate_quad_rules


class FixedTensorProductQuadratureRule(TensorProductQuadratureRule):
    def __init__(
        self,
        nvars: int,
        univariate_quad_rules: List[UnivariateQuadratureRule],
        nnodes_1d: List[int],
    ):
        super().__init__(nvars, univariate_quad_rules, store=True)
        self._nnodes_1d = self._bkd.asarray(nnodes_1d)

    # TODO fix class inheritance to either require arguments to call or no args
    def __call__(self) -> Tuple[Array, Array]:
        return super().__call__(self._nnodes_1d)


class FixedGaussianTensorProductQuadratureRuleFromVariable(
    FixedTensorProductQuadratureRule
):
    def __init__(
        self,
        variable: IndependentMarginalsVariable,
        nnodes_1d: List[int],
    ):
        marginal_quad_rules = [
            GaussQuadratureRule(marginal, backend=variable._bkd)
            for marginal in variable.marginals()
        ]
        super().__init__(variable.nvars(), marginal_quad_rules, nnodes_1d)


class TrigonometricBasis(MultiIndexBasis):
    def __init__(
        self,
        bounds: Array,
        indices: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__([TrigonometricPolynomial1D(bounds, backend)])


class FourierBasis(MultiIndexBasis):
    def __init__(
        self,
        bounds: Array,
        inverse: bool = True,
        indices: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        super().__init__([FourierBasis1D(bounds, inverse, backend)])


def setup_tensor_product_gauss_quadrature_rule(
    variable: IndependentMarginalsVariable,
) -> TensorProductQuadratureRule:
    return TensorProductQuadratureRule(
        variable.nvars(),
        [
            GaussQuadratureRule(marginal, backend=variable._bkd)
            for marginal in variable.marginals()
        ],
    )


def setup_tensor_product_piecewise_poly_quadrature_rule(
    variable: IndependentMarginalsVariable,
    basis_types: List[str],
    unbounded_alpha=0.999,
    weighted: bool = False,
) -> TensorProductQuadratureRule:
    # if not weighted variable only used to define ranges.
    # weighted mutiplies qaudrature weights by marginals PDF
    if len(basis_types) != variable.nvars():
        raise ValueError("must specify basis type for each dimension")
    intervals = [
        (
            marginal.interval(1.0)
            if marginal.is_bounded()
            else marginal.interval(unbounded_alpha)
        )
        for marginal in variable.marginals()
    ]
    return TensorProductQuadratureRule(
        variable.nvars(),
        [
            UnivariatePiecewisePolynomialQuadratureRule(
                basis_type,
                interval,
                UnivariateEquidistantNodeGenerator(backend=variable._bkd),
                backend=variable._bkd,
                marginal=(None if not weighted else marginal),
            )
            for marginal, basis_type, interval in zip(
                variable.marginals(), basis_types, intervals
            )
        ],
    )


class TriangleLebesqueQuadratureRule:
    def __init__(self, vertices: Array, backend: BackendMixin):
        self._bkd = backend
        if vertices.ndim != 2 or vertices.shape[0] != 2:
            raise ValueError("vertices has the wrong shape")
        self._vertices = vertices
        marginal = stats.uniform(-1, 2)
        self._sq_quad_rule = TensorProductQuadratureRule(
            2, [GaussQuadratureRule(marginal, backend=self._bkd)] * 2
        )

    def _quadrature_tuple_on_square(
        self, nsamples: Array
    ) -> Tuple[Array, Array]:
        return self._sq_quad_rule(nsamples)

    def _map_samples_from_square(self, sq_quadx: Array) -> Array:
        x1, x2, x3 = self._vertices[0]
        y1, y2, y3 = self._vertices[1]
        x = (
            x1
            + (x2 - x1) * (1.0 + sq_quadx[0]) / 2.0
            + (x3 - x1) * (1.0 - sq_quadx[0]) * (1.0 + sq_quadx[1]) / 4.0
        )

        y = (
            y1
            + (y2 - y1) * (1.0 + sq_quadx[0]) / 2.0
            + (y3 - y1) * (1.0 - sq_quadx[0]) * (1.0 + sq_quadx[1]) / 4.0
        )
        return self._bkd.stack((x, y), axis=0)

    def __call__(self, nsamples: Array) -> Tuple[Array, Array]:
        if len(nsamples) != 2:
            raise ValueError("nsamples must contain two entries")
        sq_quadx, sq_quadw = self._sq_quad_rule(nsamples)
        # adjust weights to account for 1d rules using weight function
        # w(x) = 1/2
        sq_quadw *= 4.0
        tri_quadx = self._map_samples_from_square(sq_quadx)
        area = 1
        tri_quadw = (area * sq_quadw[:, 0] * (1.0 - sq_quadx[0]) / 8.0)[
            :, None
        ]
        return tri_quadx, tri_quadw
