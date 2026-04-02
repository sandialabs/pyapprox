from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


class TestFunction1D:
    def _setup_functions(self, bkd):
        self.nqoi = 1
        self.nvars = 1
        self.samples = bkd.reshape(
            bkd.linspace(0, 10, 100), (1, -1)
        )  # Shape (nvars, npts)
        self.vec = bkd.ones((self.nvars, 1))  # Vector for Hessian tests

        def example_function(samples):
            return bkd.sin(samples)

        def example_jacobian(sample):
            return bkd.cos(sample)

        def example_hvp(sample, vec):
            return -bkd.sin(sample) * vec

        # Define the function
        self.function = FunctionFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=example_function,
            bkd=bkd,
        )

        # Define the function with Jacobian
        self.function_with_jacobian = FunctionWithJacobianFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=example_function,
            jacobian=example_jacobian,
            bkd=bkd,
        )

        # Define the function with Hessian
        self.function_with_hessian = FunctionWithJacobianAndHVPFromCallable(
            nvars=self.nvars,
            fun=example_function,
            jacobian=example_jacobian,
            hvp=example_hvp,
            bkd=bkd,
        )

    def test_function_call(self, bkd) -> None:
        self._setup_functions(bkd)
        values = self.function(self.samples)
        assert values.shape == (self.nqoi, self.samples.shape[1])
        bkd.assert_allclose(values, bkd.sin(self.samples))

    def test_jacobian(self, bkd) -> None:
        self._setup_functions(bkd)
        sample = self.samples[:, :1]
        jacobian = self.function_with_jacobian.jacobian(sample)
        assert jacobian.shape == (self.nqoi, self.nvars)
        bkd.assert_allclose(jacobian, bkd.cos(sample))

    def test_hvp(self, bkd) -> None:
        self._setup_functions(bkd)
        sample = self.samples[:, :1]
        hvp = self.function_with_hessian.hvp(sample, self.vec)
        assert hvp.shape == (self.nvars, 1)
        bkd.assert_allclose(hvp, -bkd.sin(sample) * self.vec)


class TestFunction3D:
    def _setup_functions(self, bkd):
        self.nqoi = 1
        self.nvars = 3
        self.samples = bkd.stack(
            [
                bkd.linspace(0, 10, 100),
                bkd.linspace(10, 20, 100),
                bkd.linspace(20, 30, 100),
            ]
        )  # Shape (3, npts)
        self.vec = bkd.ones((self.nvars, 1))  # Vector for Hessian tests

        def example_function(samples):
            return bkd.reshape(
                bkd.sum(bkd.sin(samples), axis=0), (1, -1)
            )

        def example_jacobian(sample):
            return bkd.cos(sample).T

        def example_hvp(sample, vec):
            return -bkd.sin(sample) * vec

        self._example_function = example_function
        self._example_jacobian = example_jacobian
        self._example_hvp = example_hvp

        # Define the function
        self.function = FunctionFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=example_function,
            bkd=bkd,
        )

        # Define the function with Jacobian
        self.function_with_jacobian = FunctionWithJacobianFromCallable(
            nqoi=self.nqoi,
            nvars=self.nvars,
            fun=example_function,
            jacobian=example_jacobian,
            bkd=bkd,
        )

        # Define the function with Hessian
        self.function_with_hessian = FunctionWithJacobianAndHVPFromCallable(
            nvars=self.nvars,
            fun=example_function,
            jacobian=example_jacobian,
            hvp=example_hvp,
            bkd=bkd,
        )

    def test_function_call(self, bkd) -> None:
        self._setup_functions(bkd)
        values = self.function(self.samples)
        assert values.shape == (self.nqoi, self.samples.shape[1])
        bkd.assert_allclose(values, self._example_function(self.samples))

    def test_jacobian(self, bkd) -> None:
        self._setup_functions(bkd)
        sample = self.samples[:, :1]
        jacobian = self.function_with_jacobian.jacobian(sample)
        assert jacobian.shape == (self.nqoi, self.nvars)
        bkd.assert_allclose(jacobian, self._example_jacobian(sample))

    def test_hvp(self, bkd) -> None:
        self._setup_functions(bkd)
        sample = self.samples[:, :1]
        hvp = self.function_with_hessian.hvp(sample, self.vec)
        assert hvp.shape == (self.nvars, 1)
        bkd.assert_allclose(hvp, self._example_hvp(sample, self.vec))
