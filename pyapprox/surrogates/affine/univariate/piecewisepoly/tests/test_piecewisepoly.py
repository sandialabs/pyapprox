"""Tests for piecewise polynomial basis functions."""

import pytest

from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    PiecewiseConstantLeft,
    PiecewiseConstantMidpoint,
    PiecewiseConstantRight,
    PiecewiseCubic,
    PiecewiseLinear,
    PiecewiseQuadratic,
)


class TestPiecewisePolynomialBasis:

    def test_piecewise_linear(self, bkd) -> None:
        """
        Test that the piecewise linear basis interpolates a linear function
        and the quadrature rule exactly integrates the linear function.
        """
        nodes = bkd.asarray([0.0, 0.5, 1.0])
        xx = bkd.linspace(0.0, 1.0, 100)

        def linear_function(x):
            return x

        # Create PiecewiseLinear instance
        piecewise_linear = PiecewiseLinear(nodes, bkd)

        # Evaluate basis functions
        basis_values = piecewise_linear(xx)
        interpolated_values = basis_values @ linear_function(nodes)

        # Check interpolation
        expected_values = linear_function(xx)
        bkd.assert_allclose(interpolated_values, expected_values, atol=1e-7)

        # Check quadrature rule
        quadrature_points, quadrature_weights = piecewise_linear.quadrature_rule()
        integral = quadrature_weights @ linear_function(quadrature_points)
        expected_integral = 0.5  # Integral of f(x) = x over [0, 1]
        assert integral == pytest.approx(expected_integral, abs=1e-7)

    def test_piecewise_quadratic(self, bkd) -> None:
        """
        Test that the piecewise quadratic basis interpolates a quadratic function
        and the quadrature rule exactly integrates the quadratic function.
        """
        nodes = bkd.asarray([0.0, 0.5, 1.0])
        xx = bkd.linspace(0.0, 1.0, 100)

        def quadratic_function(x):
            return x**2

        # Create PiecewiseQuadratic instance
        piecewise_quadratic = PiecewiseQuadratic(nodes, bkd)

        # Evaluate basis functions
        basis_values = piecewise_quadratic(xx)
        interpolated_values = basis_values @ quadratic_function(nodes)

        # Check interpolation
        expected_values = quadratic_function(xx)
        bkd.assert_allclose(interpolated_values, expected_values, atol=1e-7)

        # Check quadrature rule
        quadrature_points, quadrature_weights = piecewise_quadratic.quadrature_rule()
        integral = quadrature_weights @ quadratic_function(quadrature_points)
        expected_integral = 1 / 3  # Integral of f(x) = x^2 over [0, 1]
        assert integral == pytest.approx(expected_integral, abs=1e-7)

    def test_piecewise_cubic(self, bkd) -> None:
        """
        Test that the piecewise cubic basis interpolates a cubic function
        and the quadrature rule exactly integrates the cubic function.
        """
        nodes = bkd.asarray([0.0, 0.25, 0.5, 1.0])
        xx = bkd.linspace(0.0, 1.0, 100)

        def cubic_function(x):
            return x**3

        # Create PiecewiseCubic instance
        piecewise_cubic = PiecewiseCubic(nodes, bkd)

        # Evaluate basis functions
        basis_values = piecewise_cubic(xx)
        interpolated_values = basis_values @ cubic_function(nodes)

        # Check interpolation
        expected_values = cubic_function(xx)
        bkd.assert_allclose(interpolated_values, expected_values, atol=1e-7)

        # Check quadrature rule
        quadrature_points, quadrature_weights = piecewise_cubic.quadrature_rule()
        integral = quadrature_weights @ cubic_function(quadrature_points)
        expected_integral = 0.25  # Integral of f(x) = x^3 over [0, 1]
        assert integral == pytest.approx(expected_integral, abs=1e-7)


class TestPiecewisePolynomialConvergence:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.target_function = lambda x: x**4  # Polynomial degree 4
        self.interval = (0.0, 1.0)  # Interval [0, 1]

    def compute_error(self, bkd, basis_class, nodes, xx) -> float:
        """
        Compute the interpolation error for a given basis class and nodes.

        Parameters
        ----------
        bkd : Backend
            The backend to use.
        basis_class : class
            The basis class to test (e.g., PiecewiseLinear, PiecewiseQuadratic).
        nodes : Array
            The nodes used to define the basis functions.
        xx : Array
            The evaluation points.

        Returns
        -------
        float
            The interpolation error.
        """
        basis = basis_class(nodes, bkd)
        quad_points = basis.quadrature_rule()[0]
        basis_values = basis(xx)
        interpolated_values = basis_values @ self.target_function(quad_points)
        exact_values = self.target_function(xx)
        error = bkd.norm(interpolated_values - exact_values)
        return error.item()

    def test_convergence_piecewise_linear(self, bkd) -> None:
        """
        Test convergence of piecewise linear basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseLinear, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 2.0, (
                "PiecewiseLinear does not converge at the expected rate."
            )

    def test_convergence_piecewise_quadratic(self, bkd) -> None:
        """
        Test convergence of piecewise quadratic basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 11, 21, 41]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseQuadratic, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 4.0, (
                "PiecewiseQuadratic does not converge at the expected rate."
            )

    def test_convergence_piecewise_cubic(self, bkd) -> None:
        """
        Test convergence of piecewise cubic basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [7, 13, 25, 46]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseCubic, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 8.0, (
                "PiecewiseCubic does not converge at the expected rate."
            )

    def test_convergence_piecewise_constant_left(self, bkd) -> None:
        """
        Test convergence of piecewise left constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseConstantLeft, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 1.0, (
                "PiecewiseConstantLeft does not converge at the expected rate."
            )

    def test_convergence_piecewise_constant_right(self, bkd) -> None:
        """
        Test convergence of piecewise right constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseConstantRight, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 1.0, (
                "PiecewiseConstantRight does not converge at the expected rate."
            )

    def test_convergence_piecewise_constant_midpoint(self, bkd) -> None:
        """
        Test convergence of piecewise midpoint constant basis functions to the target
        polynomial function of degree 4.
        """
        errors = []
        node_counts = [5, 10, 20, 40]  # Increasing number of nodes
        for n in node_counts:
            nodes = bkd.linspace(self.interval[0], self.interval[1], n)
            xx = bkd.linspace(self.interval[0], self.interval[1], 1000)
            error = self.compute_error(bkd, PiecewiseConstantMidpoint, nodes, xx)
            errors.append(error)

        # Check convergence rate
        for i in range(1, len(errors)):
            rate = errors[i - 1] / errors[i]
            assert rate > 1.0, (
                "PiecewiseConstantMidpoint does not converge at the expected rate."
            )


from pyapprox.surrogates.affine.univariate.piecewisepoly import (
    DynamicPiecewiseBasis,
    EquidistantNodeGenerator,
)


class TestDynamicPiecewiseBasis:
    """Tests for DynamicPiecewiseBasis and node generators."""

    def test_equidistant_node_generator(self, bkd) -> None:
        """Test that EquidistantNodeGenerator generates correct nodes."""
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))

        # 5 nodes on [-1, 1]
        nodes = node_gen(5)
        expected = bkd.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
        bkd.assert_allclose(nodes, expected, atol=1e-12)

        # 3 nodes on [-1, 1]
        nodes = node_gen(3)
        expected = bkd.asarray([-1.0, 0.0, 1.0])
        bkd.assert_allclose(nodes, expected, atol=1e-12)

    def test_equidistant_node_generator_custom_bounds(self, bkd) -> None:
        """Test EquidistantNodeGenerator with custom bounds."""
        node_gen = EquidistantNodeGenerator(bkd, (0.0, 2.0))
        nodes = node_gen(5)
        expected = bkd.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
        bkd.assert_allclose(nodes, expected, atol=1e-12)

    def test_equidistant_node_generator_bkd(self, bkd) -> None:
        """Test that EquidistantNodeGenerator returns correct backend."""
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        assert node_gen.bkd() is bkd

    def test_dynamic_piecewise_basis_set_nterms(self, bkd) -> None:
        """Test DynamicPiecewiseBasis.set_nterms creates correct basis."""
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)

        # Initially no terms
        assert basis.nterms() == 0

        # Set to 5 terms
        basis.set_nterms(5)
        assert basis.nterms() == 5

        # Check quadrature rule has correct size
        # quadrature_rule returns (1, nterms) points and (nterms, 1) weights
        pts, wts = basis.quadrature_rule()
        assert pts.shape == (1, 5)
        assert wts.shape == (5, 1)

    def test_dynamic_piecewise_basis_evaluation(self, bkd) -> None:
        """Test DynamicPiecewiseBasis evaluation after set_nterms."""
        node_gen = EquidistantNodeGenerator(bkd, (0.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseLinear, node_gen)
        basis.set_nterms(5)

        # Evaluate at test points - samples shape (1, nsamples)
        samples = bkd.reshape(
            bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0]), (1, 5)
        )
        values = basis(samples)

        # Linear basis at nodes should give identity matrix
        # Values shape: (nsamples, nterms)
        expected_diag = bkd.eye(5)
        bkd.assert_allclose(values, expected_diag, atol=1e-12)

    def test_dynamic_piecewise_basis_error_before_set_nterms(self, bkd) -> None:
        """Test DynamicPiecewiseBasis raises error if used before set_nterms."""
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)

        # Should raise ValueError on call - samples shape (1, nsamples)
        with pytest.raises(ValueError):
            basis(bkd.reshape(bkd.asarray([0.0]), (1, 1)))

        # Should raise ValueError on quadrature_rule
        with pytest.raises(ValueError):
            basis.quadrature_rule()

    def test_dynamic_piecewise_basis_interpolation(self, bkd) -> None:
        """Test DynamicPiecewiseBasis interpolates quadratic function exactly."""
        node_gen = EquidistantNodeGenerator(bkd, (0.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)
        basis.set_nterms(5)

        # Target function: f(x) = x^2
        def target(x):
            return x**2

        # Get quadrature points (1, nterms) and evaluate target on 1D points
        pts, _ = basis.quadrature_rule()
        pts_1d = pts[0]
        target_vals = target(pts_1d)

        # Evaluate basis at test points - shape (1, nsamples)
        test_pts_1d = bkd.linspace(0.0, 1.0, 50)
        test_pts = bkd.reshape(test_pts_1d, (1, 50))
        basis_vals = basis(test_pts)

        # Interpolate
        interp_vals = basis_vals @ target_vals

        # Should match target function
        expected = target(test_pts_1d)
        bkd.assert_allclose(interp_vals, expected, atol=1e-10)

    def test_dynamic_piecewise_basis_quadrature_integration(self, bkd) -> None:
        """Test DynamicPiecewiseBasis quadrature integrates correctly."""
        node_gen = EquidistantNodeGenerator(bkd, (0.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)
        basis.set_nterms(5)

        # Quadrature for integral of x^2 on [0, 1]
        # pts shape (1, nterms), wts shape (nterms, 1)
        pts, wts = basis.quadrature_rule()
        pts_1d = pts[0]
        wts_1d = wts[:, 0]

        def target(x):
            return x**2

        integral = wts_1d @ target(pts_1d)

        # Integral of x^2 on [0, 1] = 1/3
        expected = bkd.asarray([1.0 / 3.0])
        bkd.assert_allclose(
            bkd.asarray([float(integral)]), expected, rtol=1e-10
        )

    def test_dynamic_piecewise_basis_bkd(self, bkd) -> None:
        """Test DynamicPiecewiseBasis returns correct backend."""
        node_gen = EquidistantNodeGenerator(bkd, (-1.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseQuadratic, node_gen)
        assert basis.bkd() is bkd

    def test_dynamic_piecewise_basis_reset_nterms(self, bkd) -> None:
        """Test DynamicPiecewiseBasis can be reset to different nterms."""
        node_gen = EquidistantNodeGenerator(bkd, (0.0, 1.0))
        basis = DynamicPiecewiseBasis(bkd, PiecewiseLinear, node_gen)

        # Set to 3 terms
        basis.set_nterms(3)
        assert basis.nterms() == 3
        pts3, _ = basis.quadrature_rule()
        assert pts3.shape == (1, 3)

        # Reset to 7 terms
        basis.set_nterms(7)
        assert basis.nterms() == 7
        pts7, _ = basis.quadrature_rule()
        assert pts7.shape == (1, 7)
