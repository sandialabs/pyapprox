import unittest
import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin, torch


class TestBackend:

    def setUp(self):
        np.random.seed(1)


class TestNumpyBackend(TestBackend, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchBackend(TestBackend, unittest.TestCase):
    def get_backend(self):
        return TorchMixin

    def test_set_default_device(self):
        bkd = self.get_backend()
        with self.assertWarns(UserWarning):
            bkd.set_gpu_as_default()
        assert torch.get_default_device().__repr__() != "device(type='cpu')"
        with self.assertWarns(UserWarning):
            bkd.set_cpu_as_default()
        assert torch.get_default_device().__repr__() == "device(type='cpu')"
        assert torch.get_default_dtype() == torch.double


if __name__ == "__main__":
    unittest.main(verbosity=2)
