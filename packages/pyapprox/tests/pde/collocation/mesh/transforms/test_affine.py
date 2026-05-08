"""Tests for affine transforms."""


from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)


class TestAffineTransforms:
    """Base test class for affine transforms."""

    def test_transform_1d_identity(self, bkd):
        """Test 1D transform maps [-1, 1] to [-1, 1]."""
        transform = AffineTransform1D((-1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_1d_scaling(self, bkd):
        """Test 1D transform maps [-1, 1] to [0, 2]."""
        transform = AffineTransform1D((0.0, 2.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0, 1.0, 2.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_1d_inverse(self, bkd):
        """Test 1D transform inverse."""
        transform = AffineTransform1D((0.0, 10.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 0.5, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_1d_jacobian(self, bkd):
        """Test 1D Jacobian is constant scale factor."""
        transform = AffineTransform1D((0.0, 4.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factor is (4-0)/2 = 2
        assert jac_mat.shape == (3, 1, 1)
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((3,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_det, bkd.full((3,), 2.0), atol=1e-14)

    def test_transform_2d_identity(self, bkd):
        """Test 2D transform maps [-1, 1]^2 to itself."""
        transform = AffineTransform2D((-1.0, 1.0, -1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_2d_scaling(self, bkd):
        """Test 2D transform maps [-1, 1]^2 to [0, 2] x [0, 3]."""
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)

        # Test corners
        ref_pts = bkd.asarray([[-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 3.0, 3.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_2d_inverse(self, bkd):
        """Test 2D transform inverse."""
        transform = AffineTransform2D((1.0, 5.0, -2.0, 2.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.5, 1.0], [0.0, -0.5, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_2d_jacobian(self, bkd):
        """Test 2D Jacobian is diagonal with scale factors."""
        transform = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factors: (4-0)/2=2, (6-0)/2=3
        assert jac_mat.shape == (3, 2, 2)
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((3,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 1], bkd.full((3,), 3.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 0, 1], bkd.zeros((3,)), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 0], bkd.zeros((3,)), atol=1e-14)
        bkd.assert_allclose(jac_det, bkd.full((3,), 6.0), atol=1e-14)

    def test_transform_3d_identity(self, bkd):
        """Test 3D transform maps [-1, 1]^3 to itself."""
        transform = AffineTransform3D((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[0.0, 0.5], [0.0, -0.5], [0.0, 0.25]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_3d_scaling(self, bkd):
        """Test 3D transform scaling."""
        transform = AffineTransform3D((0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd)

        # Test origin
        ref_pts = bkd.asarray([[-1.0], [-1.0], [-1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0], [0.0], [0.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

        # Test opposite corner
        ref_pts = bkd.asarray([[1.0], [1.0], [1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[2.0], [4.0], [6.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_3d_inverse(self, bkd):
        """Test 3D transform inverse."""
        transform = AffineTransform3D((1.0, 3.0, -1.0, 5.0, 0.0, 10.0), bkd)

        ref_pts = bkd.asarray([[-0.5, 0.5], [0.25, -0.25], [0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_3d_jacobian(self, bkd):
        """Test 3D Jacobian is diagonal with scale factors."""
        transform = AffineTransform3D((0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0, 0.5], [0.0, 0.0], [0.0, -0.5]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factors: 1, 2, 3
        assert jac_mat.shape == (2, 3, 3)
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((2,), 1.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 1], bkd.full((2,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 2, 2], bkd.full((2,), 3.0), atol=1e-14)
        # Determinant = 1*2*3 = 6
        bkd.assert_allclose(jac_det, bkd.full((2,), 6.0), atol=1e-14)

    # Tests for gradient_factors, scale_factors, and unit_curvilinear_basis

    def test_gradient_factors_1d(self, bkd):
        """Test 1D gradient factors are inverse of scale."""
        # Map [-1,1] to [0,4], so scale = 2, gradient factor = 1/2
        transform = AffineTransform1D((0.0, 4.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        grad_factors = transform.gradient_factors(ref_pts)

        assert grad_factors.shape == (3, 1, 1)
        # d/dx_phys = (1/2) * d/d_xi_ref
        bkd.assert_allclose(grad_factors[:, 0, 0], bkd.full((3,), 0.5), atol=1e-14)

    def test_scale_factors_1d(self, bkd):
        """Test 1D scale factors equal Jacobian diagonal."""
        transform = AffineTransform1D((0.0, 4.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        scales = transform.scale_factors(ref_pts)

        assert scales.shape == (3, 1)
        bkd.assert_allclose(scales[:, 0], bkd.full((3,), 2.0), atol=1e-14)

    def test_unit_basis_1d(self, bkd):
        """Test 1D unit basis is identity."""
        transform = AffineTransform1D((0.0, 4.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        basis = transform.unit_curvilinear_basis(ref_pts)

        assert basis.shape == (3, 1, 1)
        bkd.assert_allclose(basis[:, 0, 0], bkd.ones((3,)), atol=1e-14)

    def test_gradient_factors_2d(self, bkd):
        """Test 2D gradient factors are inverse of scales on diagonal."""
        # Map to [0,4] x [0,6], scales = (2, 3), grad factors = (1/2, 1/3)
        transform = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0, 0.5], [0.0, -0.5]])
        grad_factors = transform.gradient_factors(ref_pts)

        assert grad_factors.shape == (2, 2, 2)
        # Diagonal entries
        bkd.assert_allclose(grad_factors[:, 0, 0], bkd.full((2,), 0.5), atol=1e-14)
        bkd.assert_allclose(
            grad_factors[:, 1, 1], bkd.full((2,), 1.0 / 3.0), atol=1e-14
        )
        # Off-diagonal entries should be zero
        bkd.assert_allclose(grad_factors[:, 0, 1], bkd.zeros((2,)), atol=1e-14)
        bkd.assert_allclose(grad_factors[:, 1, 0], bkd.zeros((2,)), atol=1e-14)

    def test_scale_factors_2d(self, bkd):
        """Test 2D scale factors equal Jacobian diagonal entries."""
        transform = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0], [0.0]])
        scales = transform.scale_factors(ref_pts)

        assert scales.shape == (1, 2)
        bkd.assert_allclose(scales[:, 0], bkd.asarray([2.0]), atol=1e-14)
        bkd.assert_allclose(scales[:, 1], bkd.asarray([3.0]), atol=1e-14)

    def test_unit_basis_2d(self, bkd):
        """Test 2D unit basis is identity."""
        transform = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0], [0.0]])
        basis = transform.unit_curvilinear_basis(ref_pts)

        assert basis.shape == (1, 2, 2)
        expected = bkd.asarray([[[1.0, 0.0], [0.0, 1.0]]])
        bkd.assert_allclose(basis, expected, atol=1e-14)

    def test_gradient_factors_3d(self, bkd):
        """Test 3D gradient factors are inverse of scales on diagonal."""
        # Map to [0,2] x [0,4] x [0,6], scales = (1, 2, 3)
        transform = AffineTransform3D((0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0], [0.0], [0.0]])
        grad_factors = transform.gradient_factors(ref_pts)

        assert grad_factors.shape == (1, 3, 3)
        # Diagonal: 1/1, 1/2, 1/3
        bkd.assert_allclose(grad_factors[:, 0, 0], bkd.asarray([1.0]), atol=1e-14)
        bkd.assert_allclose(grad_factors[:, 1, 1], bkd.asarray([0.5]), atol=1e-14)
        bkd.assert_allclose(grad_factors[:, 2, 2], bkd.asarray([1.0 / 3.0]), atol=1e-14)
        # Off-diagonal should be zero
        for i in range(3):
            for j in range(3):
                if i != j:
                    bkd.assert_allclose(
                        grad_factors[:, i, j], bkd.asarray([0.0]), atol=1e-14
                    )

    def test_scale_factors_3d(self, bkd):
        """Test 3D scale factors."""
        transform = AffineTransform3D((0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0], [0.0], [0.0]])
        scales = transform.scale_factors(ref_pts)

        assert scales.shape == (1, 3)
        bkd.assert_allclose(scales[:, 0], bkd.asarray([1.0]), atol=1e-14)
        bkd.assert_allclose(scales[:, 1], bkd.asarray([2.0]), atol=1e-14)
        bkd.assert_allclose(scales[:, 2], bkd.asarray([3.0]), atol=1e-14)

    def test_unit_basis_3d(self, bkd):
        """Test 3D unit basis is identity."""
        transform = AffineTransform3D((0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[0.0], [0.0], [0.0]])
        basis = transform.unit_curvilinear_basis(ref_pts)

        assert basis.shape == (1, 3, 3)
        expected = bkd.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        bkd.assert_allclose(basis, expected, atol=1e-14)

    def test_gradient_factors_identity(self, bkd):
        """Test gradient factors are identity when domain is [-1,1]."""
        transform = AffineTransform1D((-1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[0.0]])
        grad_factors = transform.gradient_factors(ref_pts)

        # Scale = 1, so gradient factor = 1
        bkd.assert_allclose(grad_factors[:, 0, 0], bkd.asarray([1.0]), atol=1e-14)


class TestAffineTransformsNumpy(TestAffineTransforms):
    """NumPy backend tests for affine transforms."""
