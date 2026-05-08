from pyapprox.surrogates.basis.piecewisepoly import (
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
        bkd.assert_allclose(
            bkd.asarray([float(integral)]),
            bkd.asarray([expected_integral]),
            atol=1e-7,
        )

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
        bkd.assert_allclose(
            bkd.asarray([float(integral)]),
            bkd.asarray([expected_integral]),
            atol=1e-7,
        )

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
        bkd.assert_allclose(
            bkd.asarray([float(integral)]),
            bkd.asarray([expected_integral]),
            atol=1e-7,
        )


class TestPiecewisePolynomialConvergence:
    def _setup_target(self):
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
        self._setup_target()
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
        self._setup_target()
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
        self._setup_target()
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
        self._setup_target()
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
        self._setup_target()
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
        self._setup_target()
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
        self._setup_target()
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
