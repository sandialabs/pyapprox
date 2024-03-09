import unittest
import os
import tempfile
import glob

import numpy as np
import sympy as sp

from pyapprox.interface.model import (
    ModelFromCallable, ScipyModelWrapper, UmbridgeModelWrapper, umbridge,
    IOModel, UmbridgeIOModelWrapper)


class TestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        # sp_lambda returns a single function output
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = np.atleast_2d(sp_lambda(*sample[:, 0]))
        return vals

    def test_scalar_model_from_callable_2D_sample(self):
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

    def test_scalar_model_from_callable_1D_sample(self):
        # check jacobian with 1D samples
        model = ModelFromCallable(
            lambda x: ((x[0] - 1)**2 + (x[1] - 2.5)**2),
            jacobian=lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2.5)]),
            sample_ndim=1, values_ndim=0)
        sample = np.array([2, 0])[:, None]
        errors = model.check_apply_jacobian(sample)
        assert errors.min()/errors.max() < 1e-6

        # check apply_jacobian and apply_hessian with 1D samples
        model = ModelFromCallable(
            lambda x: ((x[0] - 1)**2 + (x[1] - 2.5)**2),
            jacobian=lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2.5)]),
            apply_jacobian=lambda x, v: 2*(x[0] - 1)*v[0]+2*(x[1] - 2.5)*v[1],
            apply_hessian=lambda x, v: np.array(np.diag([2, 2])) @ v,
            sample_ndim=1, values_ndim=0)
        sample = np.array([2, 0])[:, None]
        errors = model.check_apply_jacobian(sample)
        assert errors.min()/errors.max() < 1e-6
        # hessian is constant diagonal matrix so check first
        # finite difference is exact
        errors = model.check_apply_hessian(sample)
        assert errors[0] < 1e-15

        

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
        UmbridgeModelWrapper.kill_server(process, out)

    def test_io_model(self):
        intmpdir = tempfile.TemporaryDirectory()

        infilenames = [os.path.join(intmpdir.name, "vec.npz")]

        nvars = 2
        vec = np.random.uniform(0., 1., (1, nvars))
        np.savez(infilenames[0], vec=vec)

        class TestIOModel(IOModel):
            def _run(self, sample, linked_filenames, outdirname):
                vec = np.load(linked_filenames[0])["vec"]
                # save a file to check cleaning of directories works
                np.savez(os.path.join(outdirname, "out.npz"), sample+1)
                return np.atleast_2d(vec.dot(sample))

        nsamples = 10
        samples = np.random.uniform(0., 1., (nvars, nsamples))
        test_values = np.array([vec @ sample for sample in samples.T])

        # test when temp work directories are used
        model = TestIOModel(infilenames)
        values = model(samples)
        assert np.allclose(values, test_values)

        # test when all contents of work directories are saved
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        model = TestIOModel(
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
        model = TestIOModel(infilenames, outdir_basename, save="limited",
                            datafilename=datafilename)
        values = model(samples)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, '*')):
            filenames = glob.glob(os.path.join(outdirname, "*"))
            assert len(filenames) == 1
            assert os.path.basename(filenames[0]) == datafilename
        outtmpdir.cleanup()

        intmpdir.cleanup()

    def test_umbridge_io_model(self):
        server_string = """
import os
import umbridge
import numpy as np
from pyapprox.interface.model import IOModel


class TestIOModel(IOModel):
    def _run(self, sample, linked_filenames, outdirname):
        vec = np.load(linked_filenames[0])["vec"]
        # save a file to check cleaning of directories works
        np.savez(os.path.join(outdirname, "out.npz"), sample+1)
        return np.atleast_2d(vec.dot(sample))


class UMBModel(umbridge.Model):
    def __init__(self):
        super().__init__("umbmodel")

    def get_input_sizes(self, config):
        return [config.get("nvars", 5)]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config):
        model = TestIOModel(
            config["infilenames"], config["outdir_basename"],
            save=config["save"], datafilename=config["datafilename"])
        return [model(np.array(parameters).T)[0].tolist()]

    def supports_evaluate(self):
        return True


if __name__ == "__main__":
    models = [UMBModel()]
    umbridge.serve_models(models, 4242)
"""
        intmpdir = tempfile.TemporaryDirectory()
        server_filename = os.path.join(intmpdir.name, "server.py")
        out = open(os.path.join(intmpdir.name, "out"), "w")

        with open(server_filename, 'w') as writefile:
            writefile.write(server_string)
        nvars = 2
        vec = np.random.uniform(0., 1., (1, nvars))
        infilenames = [os.path.join(intmpdir.name, "vec.npz")]
        np.savez(infilenames[0], vec=vec)

        url = 'http://localhost:4242'
        run_server_string = "python {0}".format(server_filename)
        process, out = UmbridgeIOModelWrapper.start_server(
           run_server_string, url=url, out=out)
        umb_model = umbridge.HTTPModel(url, "umbmodel")

        nsamples = 10
        samples = np.random.uniform(0, 1, (nvars, nsamples))

        # test limited save
        outtmpdir = tempfile.TemporaryDirectory()
        config = {
            "infilenames": infilenames, "save": "limited", "nvars": nvars,
            "datafilename": "data.npz"}
        outdir_basename = os.path.join(outtmpdir.name, "results")
        model = UmbridgeIOModelWrapper(
            umb_model, config, outdir_basename=outdir_basename)
        values = model(samples)
        test_values = np.array([vec @ sample for sample in samples.T])
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, '*')):
            # UmbridgeIOModelWrapper creates folders
            # join(self._outdir_basename, "wdir-{0}".format(sample_id)
            # but then TestIOModel also creates wdir-0 so look two levels
            # deep from outdir_basename to find files
            filenames = glob.glob(os.path.join(outdirname, "wdir-0/*"))
            assert len(filenames) == 1
            assert os.path.basename(filenames[0]) == config["datafilename"]
        outtmpdir.cleanup()

        # test full save
        outtmpdir = tempfile.TemporaryDirectory()
        config = {
            "infilenames": infilenames, "save": "full", "nvars": nvars,
            "datafilename": "data.npz"}
        outdir_basename = os.path.join(outtmpdir.name, "results")
        model = UmbridgeIOModelWrapper(
            umb_model, config, outdir_basename=outdir_basename)
        values = model(samples)
        test_values = np.array([vec @ sample for sample in samples.T])
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, '*')):
            # UmbridgeIOModelWrapper creates folders
            # join(self._outdir_basename, "wdir-{0}".format(sample_id)
            # but then TestIOModel also creates wdir-0 so look two levels
            # deep from outdir_basename to find files
            filenames = glob.glob(os.path.join(outdirname, "wdir-0/*"))
            assert len(filenames) == 3
        outtmpdir.cleanup()

        intmpdir.cleanup()
        UmbridgeIOModelWrapper.kill_server(process)
        out.close()


if __name__ == "__main__":
    model_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestModel)
    unittest.TextTestRunner(verbosity=2).run(model_test_suite)
