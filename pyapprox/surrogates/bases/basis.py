from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin,
    NumpyLinAlgMixin,
)
from pyapprox.surrogates.bases.multiindex import compute_hyperbolic_indices
from pyapprox.surrogates.bases.univariate import UnivariateInterpolatingBasis
from pyapprox.surrogates.orthopoly.poly import OrthonormalPolynomial1D

from pyapprox.util.visualization import get_meshgrid_samples, plot_surface


class Basis(ABC):
    """The base class for any multivariate Basis."""

    def __init__(self, backend: LinAlgMixin):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend
        self._jacobian_implemented = False

    @abstractmethod
    def nterms():
        """
        Return the number of basis functions.
        """
        raise NotImplementedError()

    @abstractmethod
    def nvars(self):
        """
        Return the number of inputs to the basis.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, samples):
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

    def jacobian(self, samples):
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

    def __init__(self, bases_1d, indices=None):
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
        self._jacobian_implemented = True

    def set_hyperbolic_indices(self, nterms, pnorm):
        indices = self._bkd._la_asarray(
            compute_hyperbolic_indices(self.nvars(), nterms, pnorm),
            dtype=int,
        )
        self.set_indices(indices)

    def set_tensor_product_indices(self, nterms):
        if len(nterms) != self.nvars():
            raise ValueError("must specify nterms for each dimension")
        self.set_indices(
            self._bkd._la_cartesian_product(
                [self._bkd._la_arange(nt) for nt in nterms]
            )
        )

    def _update_nterms(self, nterms_per_1d_basis):
        for ii, basis_1d in enumerate(self._bases_1d):
            basis_1d.set_nterms(nterms_per_1d_basis[ii])

    def set_indices(self, indices):
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
        self._indices = self._bkd._la_array(indices, dtype=int)
        self._update_nterms(self._bkd._la_max(self._indices, axis=1)+1)

    def get_indices(self):
        """Return the indices defining the basis terms."""
        return self._indices

    def nterms(self):
        if self._indices is None:
            raise ValueError("indices have not been set")
        return self._indices.shape[1]

    def nvars(self):
        # use bases_1d so do not have to set indices to determine nvars
        # like is done for base class
        return len(self._bases_1d)

    def _basis_vals_1d(self, samples):
        return [
            poly(samples[dd:dd+1, :]) for dd, poly in enumerate(self._bases_1d)
        ]

    def _basis_derivs_1d(self, samples):
        return [
            poly.derivatives(samples[dd:dd+1, :], 1)
            for dd, poly in enumerate(self._bases_1d)
        ]

    def __call__(self, samples):
        if samples.shape[0] != self.nvars():
            raise ValueError("samples must have nrows={0}".format(
                self.nvars()))
        basis_vals_1d = self._basis_vals_1d(samples)
        basis_matrix = basis_vals_1d[0][:, self._indices[0, :]]
        for dd in range(1, self.nvars()):
            basis_matrix *= basis_vals_1d[dd][:, self._indices[dd, :]]
        return basis_matrix

    def jacobian(self, samples):
        basis_vals_1d = self._basis_vals_1d(samples)
        deriv_vals_1d = self._basis_derivs_1d(samples)
        jac = []
        for ii in range(self.nterms()):
            inner_jac = []
            index = self._indices[:, ii]
            for jj in range(self.nvars()):
                # derivative in jj direction
                basis_vals = self._bkd._la_copy(
                    deriv_vals_1d[jj][:, index[jj]]
                )
                # basis values in other directions
                for dd in range(self.nvars()):
                    if dd != jj:
                        basis_vals *= basis_vals_1d[dd][:, index[dd]]
                inner_jac.append(basis_vals)
            jac.append(self._bkd._la_stack(inner_jac, axis=0))
        jac = self._bkd._la_moveaxis(self._bkd._la_stack(jac, axis=0), -1, 0)
        return jac

    def __repr__(self):
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
            self._bkd._la_max(self._indices[poly_id])+1
        )


class TensorProductInterpolatingBasis(MultiIndexBasis):

    def __init__(self, bases_1d):
        for basis in bases_1d:
            if not isinstance(basis, UnivariateInterpolatingBasis):
                raise ValueError("basis must be instance of Basis1D")
        super().__init__(bases_1d)
        self._nodes_1d = None

    def set_hyperbolic_indices(self, nvars, nterms, pnorm):
        raise NotImplementedError(
            "{0} cannot have a hyperbolic index set".format(self)
        )

    def tensor_product_grid(self):
        nodes_1d = []
        for basis in self._bases_1d:
            if basis._nodes is None:
                raise RuntimeError("must call set_1d_nodes")
            nodes_1d.append(basis._nodes[0])
        return self._bkd._la_cartesian_product(
            nodes_1d)

    def set_1d_nodes(self, nodes_1d):
        for basis, nodes in zip(self._bases_1d, nodes_1d):
            basis.set_nodes(nodes)
        self.set_tensor_product_indices(
            [basis.nterms() for basis in self._bases_1d]
        )

    def _update_nterms(self, nterms):
        pass

    def nterms(self):
        return self._bkd._la_prod(
            self._bkd._la_array(
                [basis.nterms() for basis in self._bases_1d],
                dtype=int),
        )

    def quadrature_rule(self):
        samples_1d, weights_1d = [], []
        for basis in self._bases_1d:
            xx, ww = basis.quadrature_rule()
            samples_1d.append(xx[0])
            weights_1d.append(ww[:, 0])
        samples = self._bkd._la_cartesian_product(samples_1d)
        weights = self._bkd._la_outer_product(weights_1d)[:, None]
        return samples, weights

    def _plot_single_basis(
            self, ax, ii, jj, nterms_1d,
            plot_limits, num_pts_1d, surface_cmap, contour_cmap):
        X, Y, pts = get_meshgrid_samples(
            plot_limits, num_pts_1d, bkd=self._bkd
        )
        idx = jj*nterms_1d[0]+ii
        basis_vals = self(pts)
        Z = self._bkd._la_reshape(basis_vals[:, idx], X.shape)
        if surface_cmap is not None:
            plot_surface(X, Y, Z, ax, axis_labels=None, limit_state=None,
                         alpha=0.3, cmap=surface_cmap, zorder=3,
                         plot_axes=False)
        if contour_cmap is not None:
            num_contour_levels = 30
            offset = -(Z.max()-Z.min())/2
            ax.contourf(
                X, Y, Z, zdir='z', offset=offset,
                levels=self._bkd._la_linspace(
                    Z.min(), Z.max(), num_contour_levels
                ),
                cmap=contour_cmap, zorder=-1)
        return offset, idx, nterms_1d, X, Y

    def _plot_nodes(self, ax, offset, X, Y, idx, nterms_1d):
        nodes = self.tensor_product_grid()
        nodes_1d = [basis._nodes for basis in self._bases_1d]
        ax.plot(nodes[0, :], nodes[1, :],
                offset*self._bkd._la_ones(nodes.shape[1]), 'o',
                zorder=100, color='b')

        x = self._bkd._la_linspace(-1, 1, 100)
        y = nodes[1, idx]*self._bkd._la_ones((x.shape[0],))
        z = self(self._bkd._la_vstack((x[None, :], y[None, :])))[:, idx]
        ax.plot(x, Y.max()*self._bkd._la_ones((x.shape[0],)), z, '-r')
        ax.plot(nodes_1d[0][0], Y.max()*self._bkd._la_ones(
            (nterms_1d[0],)), self._bkd._la_zeros(nterms_1d[0],), 'or')

        y = self._bkd._la_linspace(-1, 1, 100)
        x = nodes[0, idx]*self._bkd._la_ones((y.shape[0],))
        z = self(self._bkd._la_vstack((x[None, :], y[None, :])))[:, idx]
        ax.plot(X.min()*self._bkd._la_ones((x.shape[0],)), y, z, '-r')
        ax.plot(X.min()*self._bkd._la_ones(
            (nterms_1d[1],)), nodes_1d[1][0],
                self._bkd._la_zeros(nterms_1d[1],), 'or')

    def plot_single_basis(
            self, ax, ii, jj,
            plot_limits=[-1, 1, -1, 1],
            num_pts_1d=101, surface_cmap="coolwarm",
            contour_cmap="gray",
            plot_nodes=False):
        # evaluate 1D basis functions once to get number of basis functions
        sample = self._bkd._la_reshape(
            self._bkd._la_asarray(plot_limits), (2, 2)
        ).T[:, :1].T
        nterms_1d = [basis(sample).shape[1] for basis in self._bases_1d]
        offset, idx, nterms_1d, X, Y = self._plot_single_basis(
            ax, ii, jj, nterms_1d,
            plot_limits, num_pts_1d, surface_cmap, contour_cmap)
        if plot_nodes is None:
            return
        self._plot_nodes(ax, offset, X, Y, idx, nterms_1d)
