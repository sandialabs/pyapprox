r"""Tests for SPDEMaternKLE and SPDE factory functions.

NumPy-only tests (eigensolve is scipy-based; skfem is a dependency).
Uses small meshes (10x10 or 20-element 1D) to keep tests fast and
memory-light.

These tests live in pde/field_maps/tests/ (not surrogates/kle/tests/)
because the factory functions depend on skfem, which is a PDE-layer
dependency.

Math Behind the Three Cross-Method Tests
=========================================

All three tests verify a single underlying mathematical object: the
Karhunen-Loeve Expansion (KLE) of a random field.  They approach it
from three completely different discretization strategies and check
that the eigenvalues agree.

Background: The Continuous KLE
------------------------------

A zero-mean Gaussian random field u(x) with covariance kernel C(x,y)
on domain D has the expansion::

    u(x) = sigma * sum_k sqrt(lambda_k) * phi_k(x) * z_k,
    z_k ~ N(0,1)

where (lambda_k, phi_k) are eigenpairs of the Fredholm integral
equation::

    integral_D C(x,y) phi_k(y) dy = lambda_k phi_k(x)

The three methods discretize this integral equation differently.

Test 3: Nystrom vs Galerkin (test_nystrom_vs_galerkin_1d_matern32)
------------------------------------------------------------------

**What it tests:** Two independent kernel-based discretizations of the
same Fredholm equation produce the same eigenvalues.

**Nystrom method** (MeshKLE / ``create_fem_nystrom_nodes_kle``):

Collocates at mesh nodes {x_i} with lumped-mass quadrature weights
{w_i}.  The integral becomes a matrix eigenvalue problem::

    sum_j w_j C(x_i, x_j) phi_k(x_j) = lambda_k phi_k(x_i)

Using the symmetrization trick
:math:`\tilde{C}_{ij} = \sqrt{w_i}\,C(x_i, x_j)\,\sqrt{w_j}`,
this becomes a standard symmetric eigenproblem on
:math:`\tilde{C}`.

**Galerkin method** (GalerkinKLE / ``create_fem_galerkin_kle``):

Projects through the FEM basis.  Let :math:`\Phi_{qj}` be the value
of basis function j at quadrature point q, and :math:`\mathrm{dx}_q`
the quadrature weight.  The projected covariance is::

    C_h = Phi^T diag(dx) K_q diag(dx) Phi

where :math:`K_q` is the kernel evaluated at all quadrature points.
The generalized eigenproblem is::

    C_h v_k = lambda_k M v_k

where M is the FEM mass matrix.

**Why they should agree:** Both discretize the same continuous operator
C on the same mesh.  The Nystrom method with lumped-mass weights is
mathematically equivalent to mass-lumped Galerkin.  The residual
``rtol=5e-3`` comes from:

- Nystrom uses point evaluation at nodes with lumped weights
  (row sums of M)
- Galerkin uses full quadrature and the consistent mass matrix

The test uses gamma=4, delta=1, giving kappa = sqrt(1/4) = 0.5,
nu=3/2, and Matern-3/2 range parameter
rho = sqrt(2*nu)/kappa = sqrt(3)/0.5 = 3.46.

Test 1: SPDE Internal Consistency (test_spde_internal_consistency)
------------------------------------------------------------------

**What it tests:** The closed-form eigenvalue formula
:math:`\lambda_k = \gamma^2 / (\tau^2 \mu_k^2)` is algebraically
correct.

**The SPDE approach:**

A Matern field can be defined as the solution of the stochastic PDE::

    L u = tau^{-1} W(x)

where W is white noise, :math:`L = \kappa^2 - \nabla^2` is the
differential operator, and :math:`\kappa = \sqrt{\delta/\gamma}`.
For the bilaplacian prior (alpha=2), the FEM discretization assembles::

    A = gamma K_stiff + delta M + xi M_boundary

The key relationship is :math:`A = \gamma L_h` where :math:`L_h` is
the discretized SPDE operator (this holds exactly when
:math:`\xi/\gamma = \kappa^2`; for independently chosen :math:`\xi`
the boundary mass is a lower-order correction).  Therefore
:math:`L_h^{-1} = \gamma A^{-1}`.

**Deriving the covariance:**

The SPDE solution covariance is
:math:`C = \tau^{-2} L_h^{-1} M L_h^{-T}` (the M appears because
the white noise inner product in the FEM basis is
:math:`\langle W, \phi_i \rangle \sim N(0, M)`).  Since
:math:`L_h^{-1} = \gamma A^{-1}`::

    Sigma = (gamma^2 / tau^2) A^{-1} M A^{-1}

**The eigenvalue problem:**

The factory solves :math:`A \phi_k = \mu_k M \phi_k`.  This means
:math:`A^{-1} M \phi_k = (1/\mu_k) \phi_k`.  Substituting into
:math:`\Sigma M`::

    Sigma M phi_k = (gamma^2/tau^2) A^{-1} M A^{-1} M phi_k
                  = (gamma^2/tau^2) A^{-1} M (1/mu_k) phi_k
                  = (gamma^2/tau^2) (1/mu_k^2) phi_k

So the eigenvalues of :math:`\Sigma M` are exactly
:math:`\gamma^2 / (\tau^2 \mu_k^2)`.

**What tau^2 is:**

The SPDE-Matern variance formula determines tau so the marginal
variance equals :math:`\sigma^2`::

    sigma^2 = Gamma(nu) / (Gamma(nu + d/2) * (4*pi)^{d/2}
              * kappa^{2*nu} * tau^2)

**The test:** Forms :math:`A^{-1}` explicitly (small mesh, nx=50),
computes :math:`\Sigma = (\gamma^2/\tau^2) A^{-1} M A^{-1}`, gets
eigenvalues of :math:`\Sigma M` via ``numpy.linalg.eigvals``, and
checks they match the formula to ``rtol=1e-10``.  This is a pure
algebraic identity -- no approximation is involved -- so machine
precision is expected.

Test 2: Asymptotic Convergence
(test_spde_asymptotic_convergence_to_stationary)
------------------------------------------------------------------

**What it tests:** The SPDE eigenvalues converge toward the stationary
Matern kernel eigenvalues as the domain grows.

**Why they differ on a bounded domain:**

The SPDE with Robin boundary conditions
:math:`\gamma\,\partial u/\partial n + \xi\,u = 0` on
:math:`\partial D` produces a *modified* Green's function that differs
from the stationary Matern kernel C(x,y).  The SPDE covariance
satisfies the boundary conditions; the stationary kernel does not.
This is not a bug -- it's a fundamental mathematical difference.

However, as :math:`L \to \infty`, the boundary effects become
negligible because:

1. The Matern kernel decays exponentially:
   :math:`C(r) \sim \exp(-\kappa r)` for large r
2. The SPDE Green's function and the stationary kernel differ
   primarily near boundaries
3. The leading eigenfunctions are concentrated in the interior,
   far from boundaries

So on [0, L], the relative eigenvalue error
:math:`|\lambda_k^{\mathrm{SPDE}} - \lambda_k^{\mathrm{Nystrom}}|
/ \lambda_k^{\mathrm{Nystrom}}` decreases as L grows.

**The test:** Runs SPDE and Nystrom on the same domain for
L = 10, 20, 40, 80 with fixed resolution h = 1/10.  Checks:

1. The max relative error over the first 5 modes decreases
   monotonically with L
2. At L = 40, the error is below 5%

This is a *trend test* -- it doesn't require exact agreement, only
convergence toward it.

Summary
-------

=========================  =============================================  ====================  ====================  ===========================================================
Test                       Compares                                       Type                  Tolerance             What it validates
=========================  =============================================  ====================  ====================  ===========================================================
Internal consistency       SPDE formula vs dense covariance eigenvalues   Algebraic identity    1e-10                 Eigenvalue formula gamma^2/(tau^2 mu_k^2) is correct
Asymptotic convergence     SPDE vs Nystrom on growing domains             Trend                 5% at L=40, monotone  SPDE Green's function approaches stationary kernel
Nystrom vs Galerkin        Two kernel discretizations                     Cross-validation      5e-3                  Kernel parameterization and both discretizations are correct
=========================  =============================================  ====================  ====================  ===========================================================
"""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.surrogates.kle.protocols import KLEProtocol
from pyapprox.typing.pde.galerkin.mesh.structured import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.typing.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.typing.pde.field_maps.kle_factory import (
    create_spde_matern_kle,
    create_spde_lognormal_kle_field_map,
    create_fem_nystrom_nodes_kle,
    create_fem_galerkin_kle,
)


def _make_2d_kle(bkd, n_modes=5, nx=10, ny=10, gamma=1.0, delta=1.0,
                 sigma=1.0, xi=None, mean_field=0.0):
    """Create an SPDE Matern KLE on a unit square mesh."""
    mesh = StructuredMesh2D(
        nx=nx, ny=ny, bounds=[[0, 1], [0, 1]], bkd=bkd,
    )
    basis = LagrangeBasis(mesh, degree=1)
    return create_spde_matern_kle(
        basis, n_modes=n_modes, gamma=gamma, delta=delta,
        sigma=sigma, bkd=bkd, xi=xi, mean_field=mean_field,
    ), basis


def _make_1d_kle(bkd, n_modes=5, nx=20, gamma=1.0, delta=1.0,
                 sigma=1.0, xi=None, mean_field=0.0):
    """Create an SPDE Matern KLE on [0, 1]."""
    mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    return create_spde_matern_kle(
        basis, n_modes=n_modes, gamma=gamma, delta=delta,
        sigma=sigma, bkd=bkd, xi=xi, mean_field=mean_field,
    ), basis


class TestSPDEMaternKLE(unittest.TestCase):
    """Tests for SPDEMaternKLE class and factory functions."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def test_eigenvalue_positivity(self) -> None:
        """All scaled eigenvalues are positive."""
        kle, _ = _make_2d_kle(self._bkd)
        self.assertTrue(self._bkd.all_bool(kle.eigenvalues() > 0))

    def test_eigenvalue_descending(self) -> None:
        """Eigenvalues are sorted in descending order (up to roundoff)."""
        kle, _ = _make_2d_kle(self._bkd, n_modes=10)
        eigs = self._bkd.to_numpy(kle.eigenvalues())
        for i in range(len(eigs) - 1):
            # Allow tiny roundoff for degenerate eigenvalues on symmetric domains
            self.assertGreaterEqual(eigs[i] + 1e-14, eigs[i + 1])

    def test_eigenvector_m_orthonormality(self) -> None:
        """Eigenvectors are M-orthonormal: phi^T M phi = I."""
        from skfem import asm
        from skfem.models.poisson import mass

        kle, basis = _make_2d_kle(self._bkd, n_modes=5, nx=8, ny=8)
        M = asm(mass, basis.skfem_basis())
        M_dense = M.toarray()

        vecs = self._bkd.to_numpy(kle.eigenvectors())
        gram = vecs.T @ M_dense @ vecs
        self._bkd.assert_allclose(
            self._bkd.asarray(gram),
            self._bkd.asarray(np.eye(5)),
            atol=1e-10,
        )

    def test_pointwise_variance_matches_target(self) -> None:
        """Interior pointwise variance is approximately sigma^2.

        Truncation and boundary effects cause deviation, so use a
        generous tolerance.
        """
        sigma = 1.0
        kle, basis = _make_2d_kle(
            self._bkd, n_modes=30, nx=10, ny=10,
            gamma=1.0, delta=1.0, sigma=sigma,
        )
        var = self._bkd.to_numpy(kle.pointwise_variance())

        # Get interior nodes (away from boundaries)
        coords = self._bkd.to_numpy(basis.dof_coordinates())
        margin = 0.15
        interior = (
            (coords[0] > margin) & (coords[0] < 1 - margin) &
            (coords[1] > margin) & (coords[1] < 1 - margin)
        )
        interior_var = var[interior]
        mean_interior_var = np.mean(interior_var)

        # Average interior variance should be close to sigma^2
        self.assertAlmostEqual(
            mean_interior_var, sigma ** 2, delta=0.3,
        )

    def test_robin_bc_affects_boundary_variance(self) -> None:
        """Robin BC parameter changes boundary variance profile.

        Different xi values produce different pointwise variance
        distributions, showing the Robin BC is active.
        """
        kle_default, _ = _make_2d_kle(
            self._bkd, n_modes=15, nx=8, ny=8,
        )
        kle_large_xi, _ = _make_2d_kle(
            self._bkd, n_modes=15, nx=8, ny=8, xi=10.0,
        )

        var_default = self._bkd.to_numpy(kle_default.pointwise_variance())
        var_large = self._bkd.to_numpy(kle_large_xi.pointwise_variance())

        # Different xi should produce different variance profiles
        self.assertFalse(
            np.allclose(var_default, var_large, rtol=0.05),
            "Different xi values should produce different variance profiles",
        )

    def test_call_shape(self) -> None:
        """__call__(coef) returns shape (nnodes, nsamples)."""
        n_modes = 5
        kle, basis = _make_2d_kle(self._bkd, n_modes=n_modes, nx=8, ny=8)
        nsamples = 3
        coef = self._bkd.array(np.random.randn(n_modes, nsamples))
        result = kle(coef)
        self.assertEqual(result.shape, (basis.ndofs(), nsamples))

    def test_zero_coef_gives_mean(self) -> None:
        """__call__ with zero coefficients returns mean_field."""
        mean_val = 2.5
        n_modes = 5
        kle, basis = _make_2d_kle(
            self._bkd, n_modes=n_modes, nx=6, ny=6,
            mean_field=mean_val,
        )
        coef = self._bkd.zeros((n_modes, 1))
        result = kle(coef)
        expected = self._bkd.full((basis.ndofs(), 1), mean_val)
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_correlation_length(self) -> None:
        """correlation_length() returns sqrt(gamma/delta)."""
        gamma, delta = 4.0, 1.0
        kle, _ = _make_1d_kle(
            self._bkd, gamma=gamma, delta=delta,
        )
        expected = np.sqrt(gamma / delta)
        self.assertAlmostEqual(kle.correlation_length(), expected, places=12)

    def test_eigenvector_smoothness(self) -> None:
        """Earlier modes are smoother (fewer zero crossings) than later modes."""
        kle, _ = _make_1d_kle(self._bkd, n_modes=10, nx=50)
        vecs = self._bkd.to_numpy(kle.eigenvectors())

        def count_crossings(v):
            return np.sum(np.diff(np.sign(v)) != 0)

        crossings = [count_crossings(vecs[:, k]) for k in range(10)]
        # First mode should have fewer crossings than later modes
        self.assertLess(crossings[0], crossings[-1])

    def test_sample_covariance_decay(self) -> None:
        """Empirical correlation decays with distance.

        Uses a long domain [0, 10] with short correlation length
        (gamma=0.01, delta=1 -> l_c=0.1) so correlation has room
        to decay to near zero.
        """
        n_modes = 20
        bkd = self._bkd
        mesh = StructuredMesh1D(nx=80, bounds=(0.0, 10.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        kle = create_spde_matern_kle(
            basis, n_modes=n_modes, gamma=0.01, delta=1.0,
            sigma=1.0, bkd=bkd,
        )

        # Draw many samples
        np.random.seed(42)
        nsamples = 5000
        coef = bkd.array(np.random.randn(n_modes, nsamples))
        samples = bkd.to_numpy(kle(coef))

        # Compute empirical correlation between center node and all others
        coords = bkd.to_numpy(basis.dof_coordinates())
        mid_idx = len(coords[0]) // 2
        ref_samples = samples[mid_idx, :]
        distances = np.abs(coords[0] - coords[0, mid_idx])

        corr = np.array([
            np.corrcoef(ref_samples, samples[j, :])[0, 1]
            for j in range(samples.shape[0])
        ])

        # Correlation at distance 0 should be ~1
        self.assertGreater(corr[mid_idx], 0.95)
        # Correlation at large distance should be small
        far_idx = np.argmax(distances)
        self.assertLess(abs(corr[far_idx]), 0.5)

    def test_coef_validation(self) -> None:
        """Wrong ndim or shape raises ValueError."""
        n_modes = 5
        kle, _ = _make_1d_kle(self._bkd, n_modes=n_modes)

        # Wrong ndim
        with self.assertRaises(ValueError):
            kle(self._bkd.array(np.random.randn(n_modes)))

        # Wrong shape[0]
        with self.assertRaises(ValueError):
            kle(self._bkd.array(np.random.randn(n_modes + 2, 3)))

    def test_isinstance_kle_protocol(self) -> None:
        """SPDEMaternKLE satisfies KLEProtocol."""
        kle, _ = _make_1d_kle(self._bkd)
        self.assertTrue(isinstance(kle, KLEProtocol))

    def test_lognormal_factory_shapes(self) -> None:
        """create_spde_lognormal_kle_field_map returns correct shapes."""
        bkd = self._bkd
        n_modes = 3
        mesh = StructuredMesh2D(
            nx=8, ny=8, bounds=[[0, 1], [0, 1]], bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)
        mean_log = bkd.zeros((basis.ndofs(),))

        tfm = create_spde_lognormal_kle_field_map(
            basis, mean_log, bkd,
            n_modes=n_modes, gamma=1.0, delta=1.0, sigma=0.3,
        )
        self.assertEqual(tfm.nvars(), n_modes)
        params = bkd.zeros((n_modes,))
        result = tfm(params)
        self.assertEqual(result.shape[0], basis.ndofs())

    def test_lognormal_factory_positive(self) -> None:
        """Lognormal output is always positive."""
        bkd = self._bkd
        n_modes = 3
        mesh = StructuredMesh2D(
            nx=6, ny=6, bounds=[[0, 1], [0, 1]], bkd=bkd,
        )
        basis = LagrangeBasis(mesh, degree=1)
        mean_log = bkd.zeros((basis.ndofs(),))

        tfm = create_spde_lognormal_kle_field_map(
            basis, mean_log, bkd,
            n_modes=n_modes, gamma=1.0, delta=1.0, sigma=0.5,
        )
        np.random.seed(42)
        for _ in range(5):
            params = bkd.array(np.random.randn(n_modes) * 3.0)
            result = tfm(params)
            self.assertGreater(float(bkd.min(result)), 0.0)

    def test_lognormal_zero_params_gives_exp_mean(self) -> None:
        """Lognormal KLE with theta=0 returns exp(mean_log_field)."""
        bkd = self._bkd
        n_modes = 3
        mesh = StructuredMesh1D(nx=15, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        mean_log = bkd.linspace(0.0, 1.0, basis.ndofs())

        tfm = create_spde_lognormal_kle_field_map(
            basis, mean_log, bkd,
            n_modes=n_modes, gamma=1.0, delta=1.0, sigma=0.3,
        )
        params = bkd.zeros((n_modes,))
        result = tfm(params)
        expected = bkd.exp(mean_log)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_nystrom_vs_galerkin_1d_matern32(self) -> None:
        """Nystrom and Galerkin eigenvalues agree for Matern-3/2 kernel.

        Verifies that the Matern-3/2 kernel parameterization is correct
        by checking that kernel-based Nystrom and Galerkin methods
        produce the same eigenvalues on a 1D mesh.
        """
        from pyapprox.typing.surrogates.kernels.matern import Matern32Kernel

        bkd = self._bkd
        n_modes = 5
        gamma, delta = 4.0, 1.0
        kappa = np.sqrt(delta / gamma)
        nu = 1.5

        # Matern-3/2 range: rho = sqrt(2*nu) / kappa
        rho = np.sqrt(2 * nu) / kappa

        mesh = StructuredMesh1D(nx=60, bounds=(0.0, 10.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        skfem_basis = basis.skfem_basis()

        lenscale = bkd.array([rho])
        kernel = Matern32Kernel(lenscale, (0.01, 100.0), 1, bkd)

        kle_nystrom = create_fem_nystrom_nodes_kle(
            skfem_basis, kernel, nterms=n_modes, sigma=1.0, bkd=bkd,
        )
        kle_galerkin = create_fem_galerkin_kle(
            skfem_basis, kernel, nterms=n_modes, sigma=1.0, bkd=bkd,
        )

        bkd.assert_allclose(
            kle_nystrom.eigenvalues(),
            kle_galerkin.eigenvalues(),
            rtol=5e-3,
        )

    def test_spde_internal_consistency(self) -> None:
        r"""SPDE eigenvalue formula matches dense covariance eigenvalues.

        On a small mesh (nx=50), form the dense SPDE nodal covariance.
        The SPDE precision operator is :math:`L_h = A/\gamma` where
        :math:`A = \gamma K + \delta M + \xi M_\partial`, so
        :math:`L_h^{-1} = \gamma A^{-1}` and the covariance is:

        .. math::

            \Sigma = \frac{\gamma^2}{\tau^2}\,A^{-1}\,M\,A^{-1}

        The eigenvalues of :math:`\Sigma M` should match
        :math:`\gamma^2 / (\tau^2 \mu_k^2)` to machine precision, where
        :math:`\mu_k` are from :math:`A \phi_k = \mu_k M \phi_k`.
        """
        from scipy.sparse.linalg import eigsh, spsolve
        from skfem import asm
        from skfem.models.poisson import mass

        from pyapprox.typing.pde.galerkin.bilaplacian import (
            BiLaplacianPrior,
        )
        from pyapprox.typing.pde.field_maps.kle_factory import (
            _compute_spde_tau_squared,
        )

        bkd = self._bkd
        gamma, delta, sigma = 4.0, 1.0, 1.0
        nx = 50
        n_modes = 10

        mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 5.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)

        xi = np.sqrt(gamma * delta)
        prior = BiLaplacianPrior.with_uniform_robin(
            basis, gamma=gamma, delta=delta, bkd=bkd, robin_alpha=xi,
        )
        A = prior.stiffness_matrix()
        M = asm(mass, basis.skfem_basis())

        # Solve generalized eigenproblem A phi = mu M phi
        mu_vals, _ = eigsh(A, k=n_modes, M=M, sigma=0.0, which="LM")

        # Compute tau^2
        d = basis.mesh().ndim()
        tau_sq = _compute_spde_tau_squared(sigma, gamma, delta, d)

        # Formula eigenvalues: lambda_k = gamma^2 / (tau^2 * mu_k^2)
        lambda_formula = gamma ** 2 / (tau_sq * mu_vals ** 2)
        lambda_formula = np.sort(lambda_formula)[::-1]

        # Dense A_inv: solve A @ A_inv = I column by column
        M_dense = M.toarray()
        ndofs = M_dense.shape[0]
        A_inv = np.zeros((ndofs, ndofs))
        for j in range(ndofs):
            e_j = np.zeros(ndofs)
            e_j[j] = 1.0
            A_inv[:, j] = spsolve(A, e_j)

        # Dense covariance: Sigma = (gamma^2/tau^2) * A_inv @ M @ A_inv
        Sigma = (gamma ** 2 / tau_sq) * A_inv @ M_dense @ A_inv

        # Eigenvalues of Sigma @ M
        # Sigma @ M is not symmetric, so use eig not eigvalsh
        SigmaM = Sigma @ M_dense
        eig_dense = np.sort(np.real(np.linalg.eigvals(SigmaM)))[::-1]
        eig_dense = eig_dense[:n_modes]

        bkd.assert_allclose(
            bkd.asarray(lambda_formula),
            bkd.asarray(eig_dense),
            rtol=1e-10,
        )

    def test_spde_asymptotic_convergence_to_stationary(self) -> None:
        r"""SPDE eigenvalues converge to stationary kernel as domain grows.

        The SPDE Green's function on a bounded domain with BCs differs
        from the stationary Matern kernel.  However, as the domain grows,
        boundary effects become negligible and the SPDE eigenvalues
        converge toward Nystrom eigenvalues computed from the stationary
        kernel.

        Tests that the first 5 mode errors decrease monotonically as
        L increases from 10 to 80.
        """
        from pyapprox.typing.surrogates.kernels.matern import Matern32Kernel

        bkd = self._bkd
        gamma, delta, sigma = 1.0, 1.0, 1.0
        kappa = np.sqrt(delta / gamma)
        nu = 1.5
        rho = np.sqrt(2 * nu) / kappa
        n_modes = 5
        nx_per_unit = 10  # fixed resolution

        domain_sizes = [10, 20, 40, 80]
        max_errors = []

        for L in domain_sizes:
            nx = nx_per_unit * L
            mesh = StructuredMesh1D(nx=nx, bounds=(0.0, float(L)), bkd=bkd)
            basis = LagrangeBasis(mesh, degree=1)
            skfem_basis = basis.skfem_basis()

            kle_spde = create_spde_matern_kle(
                basis, n_modes=n_modes, gamma=gamma, delta=delta,
                sigma=sigma, bkd=bkd,
            )

            lenscale = bkd.array([rho])
            kernel = Matern32Kernel(lenscale, (0.01, 500.0), 1, bkd)
            kle_nystrom = create_fem_nystrom_nodes_kle(
                skfem_basis, kernel, nterms=n_modes, sigma=sigma,
                bkd=bkd,
            )

            eig_spde = bkd.to_numpy(kle_spde.eigenvalues())
            eig_nystrom = bkd.to_numpy(kle_nystrom.eigenvalues())
            rel_err = np.abs(eig_spde - eig_nystrom) / eig_nystrom
            max_errors.append(np.max(rel_err[:n_modes]))

        # Errors must decrease monotonically as domain grows
        for i in range(len(max_errors) - 1):
            self.assertLess(
                max_errors[i + 1], max_errors[i],
                f"L={domain_sizes[i+1]}: max_err={max_errors[i+1]:.4f} "
                f">= L={domain_sizes[i]}: max_err={max_errors[i]:.4f}",
            )

        # At L=40, error should be modest (< 5%)
        self.assertLess(
            max_errors[2], 0.05,
            f"L=40: max_err={max_errors[2]:.4f} >= 0.05",
        )


if __name__ == "__main__":
    unittest.main()
