from abc import ABC, abstractmethod

from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.optimize import brenth

from pyapprox.util.linalg import adjust_sign_eig
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.joint import JointVariable
from pyapprox.surrogates.affine.basis import TrigonometricBasis
from pyapprox.surrogates.affine.basisexp import TrigonometricExpansion


def exponential_kle_eigenvalues(
    sigma2: float, corr_len: float, omega: Array
) -> Array:
    return 2 * corr_len * sigma2 / (1.0 + (omega * corr_len) ** 2)


def exponential_kle_basis(
    x: Array, corr_len: float, sigma2: float, dom_len: float, omega: Array
) -> Array:
    r"""
        Basis for the kernel K(x,y)=\sigma^2\exp(-|x-y|/l)
    ab
        Parameters
        ----------
        x : np.ndarray (num_spatial_locations)
            The spatial coordinates of the nodes defining the random field in [0,1]

        corr_len : double
            correlation length l of the covariance kernel

        sigma2 : double
            variance sigma^2 of the covariance kernel

        omega : np.ndarray (nvars)
            The roots of the characteristic equation

        Returns
        -------
        basis_vals : np.ndarray (num_spatial_locations, nvars)
            The values of every basis at each of the spatial locations

        References
        ----------
        https://doi.org/10.1016/j.jcp.2003.09.015
    """
    nvars = omega.shape[0]
    assert x.ndim == 1
    num_spatial_locations = x.shape[0]
    # bn = 1/((corr_len**2*omega[jj]**2+1)*dom_len/2+corr_len)
    # an = corr_len*omega*bn
    # basis_vals = an*np.cos(omega[jj]*x)+bn*np.cos(omega[jj]*x)
    basis_vals = np.empty((num_spatial_locations, nvars), float)
    for jj in range(nvars):
        bn = 1 / (
            (corr_len**2 * omega[jj] ** 2 + 1) * dom_len / 2.0 + corr_len
        )
        bn = np.sqrt(bn)
        an = corr_len * omega[jj] * bn
        basis_vals[:, jj] = an * np.cos(omega[jj] * x) + bn * np.sin(
            omega[jj] * x
        )
    return basis_vals


def compute_roots_of_exponential_kernel_characteristic_equation(
    corr_len: float,
    nvars: int,
    dom_len: float,
    maxw: float = None,
    plot: bool = False,
) -> Array:
    r"""
    Compute roots of characteristic equation of the exponential kernel.

    Parameters
    ----------
    corr_len : double
        Correlation length l of the covariance kernel

    nvars : integer
        The number of roots to compute

    maxw : float
         The maximum range to search for roots

    Returns
    -------
    omega : np.ndarray (nvars)
        The roots of the characteristic equation
    """

    # def func(w): return (1-corr_len*w*np.tan(w/2.))*(corr_len*w+np.tan(w/2.))
    def func(w):
        return (corr_len**2 * w**2 - 1.0) * np.sin(
            w * dom_len
        ) - 2 * corr_len * w * np.cos(w * dom_len)

    omega = np.empty((nvars), float)
    dw = 1e-2
    tol = 1e-5
    if maxw is None:
        maxw = nvars * 5
    w = np.linspace(dw, maxw, int(maxw // dw))
    fw = func(w)
    fw_sign = np.sign(fw)
    signchange = ((np.roll(fw_sign, -1) - fw_sign) != 0).astype(int)
    I = np.where(signchange)[0]
    wI = w[I]
    fail = False
    if I.shape[0] < nvars + 1:
        msg = "Not enough roots extend maxw"
        print(msg)
        fail = True

    if not fail:
        prev_root = 0
        for ii in range(nvars):
            root = brenth(func, wI[ii], wI[ii + 1], maxiter=1000, xtol=tol)
            assert root > 0 and abs(root - prev_root) > tol * 100
            omega[ii] = root
            prev_root = root
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(w, fw, "-ko")
        plt.plot(wI, fw[I], "ro")
        plt.plot(omega, func(omega), "og", label="roots found")
        plt.ylim([-100, 100])
        plt.legend()
    if fail:
        raise Exception(msg)
    return omega


def evaluate_exponential_kle(
    mean_field: Array,
    corr_len: float,
    sigma2: float,
    dom_len: float,
    x: Array,
    z: Array,
    basis_vals: Array = None,
    eig_vals: Array = None,
) -> Array:
    r"""
    Return realizations of a random field with a exponential covariance kernel.

    Parameters
    ----------
    mean_field : vector (num_spatial_locations)
       The mean temperature profile

    corr_len : double
        Correlation length l of the covariance kernel

    sigma2 : double
        The variance \sigma^2  of the random field

    x : np.ndarray (num_spatial_locations)
        The spatial coordinates of the nodes defining the random field in [0,1]

    z : np.ndarray (nvars, num_samples)
        A set of random samples

    basis_vals : np.ndarray (num_spatial_locations, nvars)
        The values of every basis at each of the spatial locations

    Returns
    -------
    vals : vector (num_spatial_locations x num_samples)
        The values of the temperature profile at each of the spatial locations
    """
    if z.ndim == 1:
        z = z.reshape((z.shape[0], 1))

    if np.isscalar(x):
        x = np.asarray([x])

    assert np.all((x >= 0.0) & (x <= 1.0))

    nvars, num_samples = z.shape
    num_spatial_locations = x.shape[0]

    if basis_vals is None:
        omega = compute_roots_of_exponential_kernel_characteristic_equation(
            corr_len, nvars, dom_len
        )
        eig_vals = exponential_kle_eigenvalues(sigma2, corr_len, omega)
        basis_vals = exponential_kle_basis(x, corr_len, sigma2, dom_len, omega)

    assert nvars == basis_vals.shape[1]
    assert basis_vals.shape[0] == x.shape[0]

    if np.isscalar(mean_field):
        mean_field = mean_field * np.ones(num_spatial_locations)
    elif callable(mean_field):
        mean_field = mean_field(x)

    assert mean_field.ndim == 1
    assert mean_field.shape[0] == num_spatial_locations

    vals = mean_field[:, np.newaxis] + np.dot(
        np.sqrt(eig_vals)[:, None] * basis_vals, z
    )
    assert vals.shape[1] == z.shape[1]
    return vals, eig_vals


class KLE1D(object):
    def __init__(self, kle_opts: dict):
        "Defined on [0, L]"
        self.mean_field = kle_opts["mean_field"]
        self.sigma2 = kle_opts["sigma2"]  # \sigma_Y^2 in paper
        self.corr_len = kle_opts["corr_len"]  # \nu in paper
        self.nvars = kle_opts["nvars"]
        self._use_log = kle_opts.get("use_log", True)
        self.dom_len = kle_opts["dom_len"]

        self.basis_vals = None
        self.omega = (
            compute_roots_of_exponential_kernel_characteristic_equation(
                self.corr_len,
                self.nvars,
                self.dom_len,
                maxw=kle_opts.get("maxw", None),
            )
        )
        self.eig_vals = exponential_kle_eigenvalues(
            self.sigma2, self.corr_len, self.omega
        )

    def update_basis_vals(self, mesh: Array):
        if self.basis_vals is None:
            self.basis_vals = exponential_kle_basis(
                mesh, self.corr_len, self.sigma2, self.dom_len, self.omega
            )

    def __call__(self, sample: Array, mesh: Array):
        self.update_basis_vals(mesh)
        vals = evaluate_exponential_kle(
            self.mean_field,
            self.corr_len,
            self.sigma2,
            self.dom_len,
            mesh,
            sample,
            self.basis_vals,
            self.eig_vals,
        )
        if self._use_log:
            return np.exp(vals)
        else:
            return vals


def correlation_function(X: Array, s: Array, corr_type: str):
    assert X.ndim == 2
    # this is an NxD matrix, where N is number of items and D its
    # dimensionalities
    pairwise_dists = squareform(pdist(X.T, "euclidean"))
    if corr_type == "gauss":
        K = np.exp(-(pairwise_dists**2) / s**2)
    elif corr_type == "exp":
        K = np.exp(-np.absolute(pairwise_dists) / s)
    else:
        raise Exception("incorrect corr_type")
    assert K.shape[0] == X.shape[1]
    return K


def compute_nobile_diffusivity_eigenvectors(
    nvars: int, corr_len: float, mesh: Array
) -> Array:
    domain_len = 1
    assert mesh.ndim == 1
    mesh = mesh[:, np.newaxis]
    sqrtpi = np.sqrt(np.pi)
    Lp = max(domain_len, 2 * corr_len)
    L = corr_len / Lp
    sqrtpi = np.sqrt(np.pi)
    nn = np.arange(2, nvars + 1)
    eigenvalues = np.sqrt(sqrtpi * L) * np.exp(
        -(((np.floor(nn / 2) * np.pi * L)) ** 2) / 8
    )
    eigenvectors = np.empty((mesh.shape[0], nvars - 1))
    eigenvectors[:, ::2] = np.sin(
        ((np.floor(nn[::2] / 2) * np.pi * mesh)) / Lp
    )
    eigenvectors[:, 1::2] = np.cos(
        ((np.floor(nn[1::2] / 2) * np.pi * mesh)) / Lp
    )
    eigenvectors *= eigenvalues
    return eigenvectors


def nobile_diffusivity(
    eigenvectors: Array, corr_len: float, samples: Array
) -> Array:
    if samples.ndim == 1:
        samples = samples.reshape((samples.shape[0], 1))
    assert samples.ndim == 2
    assert samples.shape[0] == eigenvectors.shape[1] + 1
    domain_len = 1
    Lp = max(domain_len, 2 * corr_len)
    L = corr_len / Lp
    field = eigenvectors.dot(samples[1:, :])
    field += 1 + samples[0, :] * np.sqrt(np.sqrt(np.pi) * L / 2)
    field = np.exp(field) + 0.5
    return field


class AbstractKLE(ABC):
    def __init__(
        self,
        mean_field: Array = 0,
        use_log: bool = False,
        quad_weights: Array = None,
        nterms: int = None,
        backend: BackendMixin = NumpyMixin,
    ):
        self._bkd = backend
        self._use_log = use_log
        self._quad_weights = quad_weights
        if quad_weights is not None:
            assert quad_weights.ndim == 1
        self._set_mean_field(mean_field)
        self._set_nterms(nterms)
        self._compute_basis()

    def _set_mean_field(self, mean_field: Array):
        self._mean_field = mean_field

    def _set_nterms(self, nterms: int):
        self._nterms = nterms

    def nvars(self) -> int:
        return self._nterms

    @abstractmethod
    def _compute_kernel_matrix(self) -> Array:
        raise NotImplementedError

    def _compute_basis(self):
        """
        Compute the KLE basis

        Parameters
        ----------
        num_nterms : integer
            The number of KLE modes. If None then compute all modes
        """
        K = self._compute_kernel_matrix()
        if self._quad_weights is None:
            # scipy.linalg.eigh is inconsisent across platforms. Found
            # this running test_pde.test_pyapprox_paper_inversion_benchmark
            # always compute eigenvalue decomposition using numpy to
            # ensure results do not vary across platforms. Ordering
            # of eigenvectors for repeated eigenvalues matters to KLE.
            # The downside is that we cannot use autograd on quantities
            # used to construct K but the need for this is unlikely
            eig_vals, eig_vecs = np.linalg.eigh(self._bkd.to_numpy(K))
            eig_vals = eig_vals[-self._nterms :]
            eig_vecs = eig_vecs[:, -self._nterms :]
            eig_vals = self._bkd.asarray(eig_vals)
            eig_vecs = self._bkd.asarray(eig_vecs)
        else:
            # see https://etheses.lse.ac.uk/2950/1/U615901.pdf
            # page 42
            sqrt_weights = self._bkd.sqrt(self._quad_weights)
            sym_eig_vals, sym_eig_vecs = np.linalg.eigh(
                self._bkd.to_numpy(sqrt_weights[:, None] * K * sqrt_weights),
            )
            sym_eig_vals = sym_eig_vals[-self._nterms :]
            sym_eig_vecs = sym_eig_vecs[:, -self._nterms :]
            sym_eig_vals = self._bkd.asarray(sym_eig_vals)
            sym_eig_vecs = self._bkd.asarray(sym_eig_vecs)
            eig_vecs = 1 / sqrt_weights[:, None] * sym_eig_vecs
            eig_vals = sym_eig_vals
        II = self._bkd.flip(self._bkd.argsort(eig_vals))
        # sort eigvals for repeated eigvals up to 1e-12 sort by magnitude
        # of first entry in the eigvec. This avoids some cross platform
        # differences. However running some kle with numpy and torch
        # can still cause differences due to rounding error
        tuples = zip(
            self._bkd.arange(self._nterms, dtype=int),
            self._bkd.asarray(
                np.round(self._bkd.to_numpy(eig_vals), decimals=12)
            ),
            -self._bkd.abs(eig_vecs[0, :]),
        )
        tuples = sorted(tuples, key=lambda tup: (tup[1], tup[2]), reverse=True)
        II = self._bkd.hstack([tup[0] for tup in tuples])
        eig_vecs = adjust_sign_eig(eig_vecs[:, II], self._bkd)
        assert self._bkd.all(eig_vals[II] > 0), (
            eig_vals[II],
            self._bkd.where(eig_vals[II] <= 0)[0],
        )
        self._sqrt_eig_vals = self._bkd.sqrt(eig_vals[II])
        self._eig_vecs = eig_vecs * self._sqrt_eig_vals
        self._unweighted_eig_vecs = eig_vecs

    def __call__(self, coef: Array) -> Array:
        """
        Evaluate the expansion

        Parameters
        ----------
        coef : np.ndarray (nterms, nsamples)
            The coefficients of the KLE basis
        """
        if coef.ndim != 2:
            raise ValueError(f"{coef.ndim=} but should be 2")
        if coef.shape[0] != self._nterms:
            raise ValueError(
                "coef.shape[0] {0} != self._nterms {1}".format(
                    coef.shape[0], self._nterms
                )
            )
        if self._use_log:
            return self._bkd.exp(
                self._mean_field[:, None] + self._eig_vecs @ coef
            )
        return self._mean_field[:, None] + self._eig_vecs @ coef

    def __repr__(self) -> str:
        if self._nterms is None:
            return "{0}()".format(self.__class__.__name__)
        return "{0}(nterms={1})".format(self.__class__.__name__, self._nterms)

    def weighted_eigenvectors(self) -> Array:
        """Return the eigenvectors multiplied by associated eigenvalue"""
        return self._eig_vecs

    def eigenvectors(self) -> Array:
        return self._unweighted_eig_vecs

    def singular_values(self) -> Array:
        """
        Get the singular values from SVD.

        Returns
        -------
        Array
            Singular values.
        """
        return self._sqrt_eig_vals**2


class MeshKLE(AbstractKLE):
    """
    Compute a Karhunen Loeve expansion of a covariance function.

    Parameters
    ----------
    mesh_coords : np.ndarray (nphys_vars, ncoords)
        The coordinates to evaluate the KLE basis

    mean_field : np.ndarray (ncoords)
        The mean field of the KLE

    use_log : boolean
        True - return exp(k(x))
        False - return k(x)
    """

    def __init__(
        self,
        mesh_coords: Array,
        length_scale: float,
        sigma: float = 1.0,
        mean_field: float = 0,
        use_log: bool = False,
        matern_nu: float = np.inf,
        quad_weights: Array = None,
        nterms: int = None,
        backend: BackendMixin = NumpyMixin,
    ):
        self._bkd = backend
        self._set_mesh_coordinates(mesh_coords)
        self._matern_nu = matern_nu
        self._set_lenscale(length_scale)
        self._sigma = sigma
        super().__init__(
            mean_field, use_log, quad_weights, nterms, backend=backend
        )
        # normalize the basis
        self._eig_vecs *= sigma

    def _set_mean_field(self, mean_field: Array):
        if np.isscalar(mean_field):
            mean_field = (
                self._bkd.full((self._mesh_coords.shape[1],), 1) * mean_field
            )
        super()._set_mean_field(mean_field)

    def _set_nterms(self, nterms: int):
        if nterms is None:
            nterms = self._mesh_coords.shape[1]
        assert nterms <= self._mesh_coords.shape[1]
        self._nterms = nterms

    def _set_mesh_coordinates(self, mesh_coords: Array):
        assert mesh_coords.shape[0] <= 2
        self._mesh_coords = mesh_coords

    def _set_lenscale(self, length_scale: Array):
        length_scale = self._bkd.atleast1d(self._bkd.asarray(length_scale))
        if length_scale.shape[0] == 1:
            length_scale = self._bkd.full(
                (self._mesh_coords.shape[0],), length_scale[0]
            )
        assert length_scale.shape[0] == self._mesh_coords.shape[0]
        self._lenscale = length_scale

    def _compute_kernel_matrix(self) -> Array:
        if self._matern_nu == np.inf:
            dists = pdist(
                self._bkd.to_numpy(self._mesh_coords.T / self._lenscale),
                metric="sqeuclidean",
            )
            K = squareform(np.exp(-0.5 * dists))
            np.fill_diagonal(K, 1)
            return self._bkd.asarray(K)

        dists = pdist(
            self._bkd.to_numpy(self._mesh_coords.T / self._lenscale),
            metric="euclidean",
        )
        if self._matern_nu == 0.5:
            K = squareform(np.exp(-dists))
        elif self._matern_nu == 1.5:
            dists = np.sqrt(3) * dists
            K = squareform((1 + dists) * np.exp(-dists))
        elif self._matern_nu == 2.5:
            K = squareform((1 + dists + dists**2 / 3) * np.exp(-dists))
        np.fill_diagonal(K, 1)
        return self._bkd.asarray(K)

    def __repr__(self) -> int:
        if self._nterms is None:
            return "{0}(nterms={1}, mu={2})".format(
                self.___class__.__name__, self._nterms, self._matern_nu
            )
        return "{0}(nu={1}, nterms={2}, lenscale={3}, sigma={4})".format(
            self.__class__.__name__,
            self._matern_nu,
            self._nterms,
            self._lenscale,
            self._sigma,
        )


class DataDrivenKLE(AbstractKLE):
    def __init__(
        self,
        field_samples: Array,
        mean_field: Array = 0,
        use_log: bool = False,
        nterms: int = None,
        quad_weights: Array = None,
        backend: BackendMixin = NumpyMixin,
    ):
        self._field_samples = field_samples
        super().__init__(
            mean_field, use_log, quad_weights, nterms, backend=backend
        )

    def _set_mean_field(self, mean_field: Array):
        if np.isscalar(mean_field):
            mean_field = (
                self._bkd.full((self._field_samples.shape[0],), 1) * mean_field
            )
        super()._set_mean_field(mean_field)

    def _set_nterms(self, nterms: int):
        if nterms is None:
            nterms = self._field_samples.shape[0]
        assert nterms <= self._field_samples.shape[0]
        self._nterms = nterms

    def _set_mesh_coordinates(self, mesh_coords: Array):
        self._mesh_coords = None

    def _compute_kernel_matrix(self) -> Array:
        return self._bkd.cov(self._field_samples, rowvar=True, ddof=1)

    def _compute_basis(self):
        # C = A^T A
        # A = USV^T
        # C = VSU^TUSV = VS^2V^T
        # So eigen value equations are CV = VS^2V^TV = VS^2
        # Thus V are eignevectors of Eig(C)
        # and S^2 are Eigvals
        # Principal components are AV = USV^TV = US

        # Use SVD here because it is more accurate than computing covariance
        # then taking eigdecomp. The latter approach loses precision due to
        # rounding errors
        if self._quad_weights is None:
            field_samples = self._field_samples
        else:
            sqrt_weights = self._bkd.sqrt(self._quad_weights)
            field_samples = sqrt_weights[:, None] * self._field_samples
        U, S, Vh = self._bkd.svd(field_samples)
        eig_vecs = adjust_sign_eig(U[:, : self._nterms], self._bkd)
        if self._quad_weights is not None:
            eig_vecs = 1 / sqrt_weights[:, None] * eig_vecs
        # divide S by sqrt(1/(n-1)) to be consistent with computing covariance
        # of C=A^TA/(n-1) then taking eigdecomp
        self._sqrt_eig_vals = S[: self._nterms] / np.sqrt(
            self._field_samples.shape[1] - 1
        )
        self._eig_vecs = eig_vecs * self._sqrt_eig_vals
        self._unweighted_eig_vecs = eig_vecs


class InterpolatedMeshKLE(MeshKLE):
    def __init__(self, kle_mesh: Array, kle: AbstractKLE, mesh: Array):
        self._bkd = kle._bkd
        self._kle_mesh = kle_mesh
        self._kle = kle
        assert isinstance(self._kle, MeshKLE)
        self._mesh = mesh

        self.matern_nu = self._kle._matern_nu
        self.nterms = self._kle._nterms
        self.lenscale = self._kle._lenscale

        self._basis_mat = self._kle_mesh._get_lagrange_basis_mat(
            self._kle_mesh._canonical_mesh_pts_1d,
            mesh._map_samples_to_canonical_domain(self._mesh.mesh_pts),
        )

    def _fast_interpolate(self, values: Array, xx: Array) -> Array:
        assert xx.shape[1] == self._mesh.mesh_pts.shape[1]
        assert np.allclose(xx, self._mesh.mesh_pts)
        interp_vals = self._bkd.multidot((self._basis_mat, values))
        return interp_vals

    def __call__(self, coef: Array) -> Array:
        use_log = self._kle._use_log
        self._kle._use_log = False
        vals = self._kle(coef)
        interp_vals = self._fast_interpolate(vals, self._mesh.mesh_pts)
        mean_field = self._fast_interpolate(
            self._kle._mean_field[:, None], self._mesh.mesh_pts
        )
        if use_log:
            interp_vals = self._bkd.exp(mean_field + interp_vals)
        self._kle._use_log = use_log
        return interp_vals


class PeriodicReiszGaussianRandomField(JointVariable):
    def __init__(
        self,
        sigma: float,
        tau: float,
        gamma: float,
        neigs: int,
        bounds: Array,
        backend: BackendMixin = NumpyMixin,
    ):
        self._bkd = backend
        self._sigma = sigma
        self._tau = tau
        self._gamma = gamma
        self._bounds = bounds

        self._neigs = None
        self._eigs = None
        self._trig_exp = None

        self.set_neigs(neigs)

    def nvars(self) -> int:
        """The dimension of prediction points that the KLE is evaluated at"""
        if not hasattr(self, "_nvars"):
            raise RuntimeError("Must call set_domain_samples")
        return self._nvars

    def nterms(self) -> int:
        """The number of terms in the KLE"""
        return self._trig_exp.nterms() - 1

    def set_neigs(self, neigs: int):
        self._neigs = neigs
        self._eigs = (
            np.sqrt(2)
            * (
                np.abs(self._sigma)
                * (
                    (2 * np.pi * self._bkd.arange(1, self._neigs + 1)) ** 2
                    + self._tau**2
                )
                ** (-self._gamma / 2)
            )[:, None]
        )
        nterms = self._neigs * 2 + 1
        trig_basis = TrigonometricBasis(self._bounds, backend=self._bkd)
        trig_basis.set_indices(self._bkd.arange(nterms)[None, :])
        self._trig_exp = TrigonometricExpansion(trig_basis)

    def set_domain_samples(self, domain_samples: Array):
        self._domain_samples = domain_samples
        self._nvars = domain_samples.shape[1]

    def values(self, samples: Array) -> Array:
        if (
            samples.shape[0] != self._trig_exp.nterms() - 1
            or samples.ndim != 2
        ):
            raise ValueError(
                "samples has the wrong shape of {0} should be {1}".format(
                    samples.shape, (self._trig_exp.nterms() - 1, 1)
                )
            )
        if not hasattr(self, "_domain_samples"):
            raise RuntimeError("Must call set_domain_samples")
        alpha = self._eigs * samples[: samples.shape[0] // 2]
        beta = self._eigs * samples[samples.shape[0] // 2 :]
        trig_coefs = self._bkd.vstack(
            (self._bkd.zeros((1, samples.shape[1])), alpha, beta)
        )
        self._trig_exp._nqoi = trig_coefs.shape[1]
        self._trig_exp.set_coefficients(trig_coefs)
        shift = (self._bounds[1] - self._bounds[0]) / 2
        return self._trig_exp(self._domain_samples - shift)

    def rvs(self, nsamples: int) -> Array:
        return self.values(
            self._bkd.asarray(
                np.random.normal(0, 1, (2 * self._neigs, nsamples))
            )
        )


class PrincipalComponentAnalysis(DataDrivenKLE):
    """
    Principal Component Analysis (PCA) for dimensionality reduction.

    This class computes a reduced basis using Singular Value Decomposition (SVD)
    and provides methods for reducing states, expanding reduced states, and
    reducing operator matrix-vector products.
    """

    def __init__(
        self,
        snapshots: Array,
        nterms: int,
        backend: BackendMixin,
        quad_weights: Array = None,
    ):
        normalized_snapshots = (
            snapshots - backend.mean(snapshots, axis=1)[:, None]
        ) / backend.max(snapshots, axis=1)[:, None]
        super().__init__(
            normalized_snapshots, 0, False, nterms, quad_weights, backend
        )

    def reduce_state(self, state: Array) -> Array:
        """
        Project a state onto the reduced basis.

        Parameters
        ----------
        state : Array
            Full-order state to be reduced.

        Returns
        -------
        Array
            Reduced-order state.
        """
        return self.eigenvectors().T @ state

    def expand_reduced_state(self, reduced_state: Array) -> Array:
        """
        Expand a reduced-order state back to the full-order state.

        Parameters
        ----------
        reduced_state : Array
            Reduced-order state to be expanded.

        Returns
        -------
        Array
            Full-order state.
        """
        return self.eigenvectors() @ reduced_state

    def snapshots(self) -> Array:
        return self._field_samples
