"""Tests for differential operators."""


from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
)
from pyapprox.pde.collocation.operators import (
    Gradient,
    Laplacian,
    divergence,
    gradient,
    input_field,
    laplacian,
)


class TestGradient:
    """Base test class for Gradient operator."""

    def test_gradient_1d_linear(self, bkd):
        """Test gradient of linear function f(x) = x."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        # f(x) = x, df/dx = 1
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes)

        grad_op = Gradient(basis, bkd)
        grad_f = grad_op(f)

        assert len(grad_f) == 1
        bkd.assert_allclose(grad_f[0].as_flat(), bkd.ones((npts,)), atol=1e-12)

    def test_gradient_1d_quadratic(self, bkd):
        """Test gradient of quadratic function f(x) = x^2."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # f(x) = x^2, df/dx = 2x
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes**2)

        grad_f = gradient(f, basis, bkd)

        expected = 2.0 * nodes
        bkd.assert_allclose(grad_f[0].as_flat(), expected, atol=1e-11)

    def test_gradient_2d_separable(self, bkd):
        """Test gradient of separable function f(x,y) = x^2 + y^2."""
        mesh = TransformedMesh2D(6, 6, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        # Create mesh coordinates
        x = basis.nodes_x()
        y = basis.nodes_y()
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # f(x,y) = x^2 + y^2
        # df/dx = 2x, df/dy = 2y
        f_values = xx**2 + yy**2
        f = input_field(basis, bkd, f_values)

        grad_f = gradient(f, basis, bkd)

        assert len(grad_f) == 2
        bkd.assert_allclose(grad_f[0].as_flat(), 2.0 * xx, atol=1e-10)
        bkd.assert_allclose(grad_f[1].as_flat(), 2.0 * yy, atol=1e-10)


class TestDivergence:
    """Base test class for Divergence operator."""

    def test_divergence_2d_linear(self, bkd):
        """Test divergence of vector field v = (x, y)."""
        mesh = TransformedMesh2D(6, 6, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Create mesh coordinates
        x = basis.nodes_x()
        y = basis.nodes_y()
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # v = (x, y), div(v) = 1 + 1 = 2
        v_x = input_field(basis, bkd, xx)
        v_y = input_field(basis, bkd, yy)

        div_v = divergence([v_x, v_y], basis, bkd)

        expected = bkd.ones((npts,)) * 2.0
        bkd.assert_allclose(div_v.as_flat(), expected, atol=1e-12)

    def test_divergence_2d_nonlinear(self, bkd):
        """Test divergence of vector field v = (x^2, y^2)."""
        mesh = TransformedMesh2D(6, 6, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        # Create mesh coordinates
        x = basis.nodes_x()
        y = basis.nodes_y()
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # v = (x^2, y^2), div(v) = 2x + 2y
        v_x = input_field(basis, bkd, xx**2)
        v_y = input_field(basis, bkd, yy**2)

        div_v = divergence([v_x, v_y], basis, bkd)

        expected = 2.0 * xx + 2.0 * yy
        bkd.assert_allclose(div_v.as_flat(), expected, atol=1e-10)


class TestLaplacian:
    """Base test class for Laplacian operator."""

    def test_laplacian_1d_quadratic(self, bkd):
        """Test Laplacian of quadratic function f(x) = x^2."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        npts = basis.npts()

        # f(x) = x^2, nabla^2 f = 2
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes**2)

        lap_f = laplacian(f, basis, bkd)

        expected = bkd.ones((npts,)) * 2.0
        bkd.assert_allclose(lap_f.as_flat(), expected, atol=1e-10)

    def test_laplacian_1d_cubic(self, bkd):
        """Test Laplacian of cubic function f(x) = x^3."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)

        # f(x) = x^3, nabla^2 f = 6x
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes**3)

        lap_f = laplacian(f, basis, bkd)

        expected = 6.0 * nodes
        bkd.assert_allclose(lap_f.as_flat(), expected, atol=1e-9)

    def test_laplacian_2d_separable(self, bkd):
        """Test Laplacian of separable function f(x,y) = x^2 + y^2."""
        mesh = TransformedMesh2D(6, 6, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        # Create mesh coordinates
        x = basis.nodes_x()
        y = basis.nodes_y()
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # f(x,y) = x^2 + y^2, nabla^2 f = 2 + 2 = 4
        f_values = xx**2 + yy**2
        f = input_field(basis, bkd, f_values)

        lap_f = laplacian(f, basis, bkd)

        expected = bkd.ones((npts,)) * 4.0
        bkd.assert_allclose(lap_f.as_flat(), expected, atol=1e-10)

    def test_laplacian_2d_mixed(self, bkd):
        """Test Laplacian of mixed function f(x,y) = x^2*y^2."""
        mesh = TransformedMesh2D(8, 8, bkd)

        basis = ChebyshevBasis2D(mesh, bkd)

        # Create mesh coordinates
        x = basis.nodes_x()
        y = basis.nodes_y()
        npts_x = x.shape[0]
        npts_y = y.shape[0]
        xx = bkd.tile(x, (npts_y,))
        yy = bkd.repeat(y, npts_x)

        # f(x,y) = x^2*y^2
        # d^2f/dx^2 = 2*y^2
        # d^2f/dy^2 = 2*x^2
        # nabla^2 f = 2*y^2 + 2*x^2
        f_values = xx**2 * yy**2
        f = input_field(basis, bkd, f_values)

        lap_f = laplacian(f, basis, bkd)

        expected = 2.0 * yy**2 + 2.0 * xx**2
        bkd.assert_allclose(lap_f.as_flat(), expected, atol=1e-9)


class TestOperatorClasses:
    """Test operator class vs function interface."""

    def test_gradient_class_vs_function(self, bkd):
        """Test Gradient class gives same result as gradient function."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes**2)

        # Class interface
        grad_op = Gradient(basis, bkd)
        result_class = grad_op(f)

        # Function interface
        result_func = gradient(f, basis, bkd)

        bkd.assert_allclose(
            result_class[0].as_flat(), result_func[0].as_flat(), atol=1e-14
        )

    def test_laplacian_class_vs_function(self, bkd):
        """Test Laplacian class gives same result as laplacian function."""
        mesh = TransformedMesh1D(10, bkd)

        basis = ChebyshevBasis1D(mesh, bkd)
        nodes = basis.nodes()
        f = input_field(basis, bkd, nodes**3)

        # Class interface
        lap_op = Laplacian(basis, bkd)
        result_class = lap_op(f)

        # Function interface
        result_func = laplacian(f, basis, bkd)

        bkd.assert_allclose(result_class.as_flat(), result_func.as_flat(), atol=1e-14)


class TestGradientNumpy(TestGradient):
    """NumPy backend tests for Gradient."""


class TestDivergenceNumpy(TestDivergence):
    """NumPy backend tests for Divergence."""


class TestLaplacianNumpy(TestLaplacian):
    """NumPy backend tests for Laplacian."""


class TestOperatorClassesNumpy(TestOperatorClasses):
    """NumPy backend tests for operator classes."""
