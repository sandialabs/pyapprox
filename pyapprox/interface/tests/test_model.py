import unittest
import os
import tempfile
import glob

import numpy as np
import sympy as sp

from pyapprox.interface.model import (
    ModelFromCallable, ScipyModelWrapper, UmbridgeModelWrapper, umbridge,
    IOModel)


class TestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        # sp_lambda returns a single function output
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = np.atleast_2d(sp_lambda(*sample[:, 0]))
        return vals

    def test_scalar_model_from_callable(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s*(ii+1) for ii, s in enumerate(symbs)])**4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec,
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample) @ vec)
        sample = np.random.uniform(0, 1, (nvars, 1))
        model.check_apply_jacobian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when jacobian() is not provided
        assert np.allclose(
           model.jacobian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample))
        model.check_apply_hessian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when hessian() is not provided
        assert np.allclose(model.hessian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_hessian, "numpy"), sample))

    def test_vector_model_from_callable(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = [sum([s*(ii+1) for ii, s in enumerate(symbs)])**4,
                  sum([s*(ii+1) for ii, s in enumerate(symbs)])**5]
        sp_grad = [[fun.diff(x) for x in symbs] for fun in sp_fun]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec)
        sample = np.random.uniform(0, 1, (nvars, 1))
        model.check_apply_jacobian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when jacobian() is not provided
        assert np.allclose(
           model.jacobian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample))

    def test_scipy_wrapper(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s*(ii+1) for ii, s in enumerate(symbs)])**4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec,
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample) @ vec)
        scipy_model = ScipyModelWrapper(model)
        # check scipy model works with 1D sample array
        sample = np.random.uniform(0, 1, (nvars))
        assert np.allclose(
           scipy_model.jac(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample[:, None]))
        assert np.allclose(scipy_model.hess(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_hessian, "numpy"), sample[:, None]))

        # test error is thrown if scipy model does not return a scalar output
        sp_fun = [sum([s*(ii+1) for ii, s in enumerate(symbs)])**4,
                  sum([s*(ii+1) for ii, s in enumerate(symbs)])**5]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample))
        scipy_model = ScipyModelWrapper(model)
        self.assertRaises(ValueError, scipy_model, sample)

    def test_umbridge_model(self):
        import os
        server_dir = os.path.dirname(__file__)
        url = 'http://localhost:4242'
        run_server_string = "python {0}".format(
            os.path.join(server_dir, "genz_umbridge_server.py"))
        process, out = UmbridgeModelWrapper.start_server(
            run_server_string, url=url)
        umb_model = umbridge.HTTPModel(url, "genz")
        config = {"name": "oscillatory", "nvars": 2, "coef_type": "none"}
        model = UmbridgeModelWrapper(umb_model, config)
        sample = np.random.uniform(0, 1, (config["nvars"], 1))
        model.check_apply_jacobian(sample, disp=True)

    def test_io_model(self):
        intmpdir = tempfile.TemporaryDirectory()

        infilenames = [os.path.join(intmpdir.name, "vec.npz")]

        nvars = 2
        vec = np.random.uniform(0., 1., (1, nvars))
        np.savez(infilenames[0], vec=vec)

        class TestModel(IOModel):
            def _run(self, sample, linked_filenames, outdirname):
                vec = np.load(linked_filenames[0])["vec"]
                # save a file to check cleaning of directories works
                np.savez(os.path.join(outdirname, "out.npz"), sample+1)
                return np.atleast_2d(vec.dot(sample))

        nsamples = 10
        samples = np.random.uniform(0., 1., (nvars, nsamples))
        test_values = np.array([vec @ sample for sample in samples.T])

        # test when temp work directories are used
        model = TestModel(infilenames)
        values = model(samples)
        assert np.allclose(values, test_values)

        # test when all contents of work directories are saved
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        model = TestModel(
            infilenames, outdir_basename, save="full",
            datafilename=datafilename)
        values = model(samples)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, '*')):
            filenames = glob.glob(os.path.join(outdirname, "*"))
            filenames = [os.path.basename(fname) for fname in filenames]
            filenames.sort()
            assert len(filenames) == 3
            assert os.path.basename(filenames[0]) == datafilename
            assert os.path.basename(filenames[1]) == "out.npz"
            assert os.path.basename(filenames[2]) == "vec.npz"
        outtmpdir.cleanup()

        # test when save="limited"
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        model = TestModel(infilenames, outdir_basename, save="limited",
                          datafilename=datafilename)
        values = model(samples)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, '*')):
            filenames = glob.glob(os.path.join(outdirname, "*"))
            assert len(filenames) == 1
            assert os.path.basename(filenames[0]) == datafilename
        outtmpdir.cleanup()

        intmpdir.cleanup()


if __name__ == "__main__":
    model_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestModel)
    unittest.TextTestRunner(verbosity=2).run(model_test_suite)
