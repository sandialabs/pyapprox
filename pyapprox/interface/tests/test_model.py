import unittest
import os
import tempfile
import glob
from functools import partial

import numpy as np
import sympy as sp

from pyapprox.interface.model import (
    ModelFromSingleSampleCallable,
    ScipyModelWrapper,
    UmbridgeModelWrapper,
    umbridge,
    IOModel,
    UmbridgeIOModelWrapper,
    PoolModelWrapper,
    SerialIOModel,
    AsyncIOModel,
    CenteredFiniteDifference,
    DenseMatrixLinearModel,
    QuadraticMatrixModel,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


def _pickable_function(bkd, sample):
    return bkd.stack(
        (bkd.sum(sample**2, axis=0), bkd.sum(sample**3, axis=0)), axis=1
    )


def raise_exception(condition, msg):
    """
    To allow exception to be raised with a conditional statment on one line
    Useful when calling chain of statemnts from command line with python -c
    """
    if condition:
        raise Exception(msg)


def get_shell_command_for_io_model(
    delay: float = 0.0,
    fault_percentage: float = 0.0,
    backend=NumpyMixin,
):
    """
    Return a FileIOModel that wrapts a call to the 2D target function
    [x[0]**2 + 2*x[1]**3, x[0]**3 + x[0]*x[1]]) with two QoI.
    """
    # vec is only loaded but not used for anything just to test that linked files are
    # linked correctly
    shell_command = (
        """python -c "import numpy as np; np.load('vec.npz')['vec']; target_function = lambda x: np.array([x[0]**2 + 2*x[1]**3, x[0]**3 + x[0]*x[1]]); sample = np.loadtxt('params.in'); u=np.random.uniform(0.,1.); from pyapprox.interface.tests.test_model import raise_exception; raise_exception(u<%f/100., 'fault occurred'); vals = target_function(sample); np.savetxt('results.out',vals); delay=%f; print(delay); import time; time.sleep(delay);" """
        % (
            fault_percentage,
            delay + np.random.uniform(-1.0, 1.0) * delay * 0.1,
        )
    )

    def target_function(x):
        return backend.asarray(
            [x[0] ** 2 + 2 * x[1] ** 3, x[0] ** 3 + x[0] * x[1]]
        )

    target_model = ModelFromSingleSampleCallable(
        2, 2, target_function, sample_ndim=1, values_ndim=1, backend=backend
    )
    return shell_command, target_model


class TestModel:
    def setUp(self):
        np.random.seed(1)

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        # sp_lambda returns a single function output
        bkd = self.get_backend()
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = bkd.asarray(sp_lambda(*bkd.to_numpy(sample[:, 0])))
        return bkd.atleast2d(vals)

    def test_scalar_model_from_callable_2D_sample(self):
        bkd = self.get_backend()
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s * (ii + 1) for ii, s in enumerate(symbs)]) ** 4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]

        # check when jac and hessian are defined
        model = ModelFromSingleSampleCallable(
            1,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            jacobian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
            hessian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample
            )[None, ...],
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        sample = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
        model.check_apply_jacobian(sample, disp=True)
        # check that when jac is used to compute jvp the number of
        # evaluations is updated correctly
        assert model.work_tracker().nevaluations("jac") == 1
        assert model.work_tracker().nevaluations("jvp") == 0
        # check full jacobian is computed correctly from apply_jacobian
        # when jacobian() is not provided
        assert np.allclose(
            model.jacobian(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
        )
        assert model.work_tracker().nevaluations("jac") == 2
        model.check_apply_hessian(sample, disp=True)
        # previous 2 + 14 different finite difference step sizes + 1
        # grad at the nominal point used in the finite difference
        assert model.work_tracker().nevaluations("jac") == 17
        # check that when hess is used to compute hvp the number of
        # evaluations is updated correctly
        assert model.work_tracker().nevaluations("hess") == 1
        assert model.work_tracker().nevaluations("hvp") == 0
        # check full jacobian is computed correctly from apply_jacobian
        # when hessian() is not provided
        assert np.allclose(
            model.hessian(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian, "numpy"), sample
            ),
        )

        # check when apply_jac and apply_hessian are defined
        model = ModelFromSingleSampleCallable(
            1,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            apply_jacobian=lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            )
            @ vec,
            apply_hessian=lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample
            )
            @ vec,
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        model.check_apply_jacobian(sample, disp=True)
        assert model.work_tracker().nevaluations("jac") == 0
        assert model.work_tracker().nevaluations("jvp") == 1
        model.check_apply_hessian(sample, disp=True)
        # previous 1 + 14 different finite difference step sizes + 1
        # grad at the nominal point used in the finite difference.
        # Each of these latter 15 requies 3 jvp (one for each variable)
        # TODO: Can i reduce the number of jvp in cases such as this
        assert model.work_tracker().nevaluations("jvp") == 1 + 14 * 3 + 3
        assert model.work_tracker().nevaluations("hess") == 0
        assert model.work_tracker().nevaluations("hvp") == 1

        # check use of database
        model = ModelFromSingleSampleCallable(
            1,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            jacobian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
            hessian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample
            )[None, ...],
            backend=bkd,
        )
        # check values are updated correctly
        model.activate_model_data_base()
        model.work_tracker().set_active(True)
        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, 3)))
        two_values = model(samples[:, :2])
        three_values = model(samples[:, :3])
        assert bkd.allclose(two_values, three_values[:2])
        assert model.work_tracker().nevaluations("val") == 3

        # check jacobians are updated correctly
        for ii in [0, 2]:
            model.jacobian(samples[:, ii : ii + 1])

        for ii in [0, 1, 2]:
            model.jacobian(samples[:, ii : ii + 1])
        print(model.work_tracker())
        assert model.work_tracker().nevaluations("jac") == 3

        # check hessians are updated correctly
        for ii in [0, 2]:
            model.hessian(samples[:, ii : ii + 1])

        for ii in [0, 1, 2]:
            model.hessian(samples[:, ii : ii + 1])
        print(model.work_tracker())
        assert model.work_tracker().nevaluations("hess") == 3

    def test_scalar_model_from_callable_1D_sample(self):
        # check jacobian with 1D samples
        bkd = self.get_backend()
        model = ModelFromSingleSampleCallable(
            1,
            2,
            lambda x: ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2),
            jacobian=lambda x: bkd.array([[2 * (x[0] - 1), 2 * (x[1] - 2.5)]]),
            sample_ndim=1,
            values_ndim=0,
            backend=bkd,
        )
        sample = bkd.array([2, 0])[:, None]
        errors = model.check_apply_jacobian(sample)
        assert errors.min() / errors.max() < 1e-6

        # check apply_jacobian and apply_hessian with 1D samples
        model = ModelFromSingleSampleCallable(
            1,
            2,
            lambda x: ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2),
            jacobian=lambda x: bkd.array([[2 * (x[0] - 1), 2 * (x[1] - 2.5)]]),
            apply_jacobian=lambda x, v: 2 * (x[0] - 1) * v[0]
            + 2 * (x[1] - 2.5) * v[1],
            hessian=lambda x: bkd.diag(bkd.array([2, 2]))[None, ...],
            sample_ndim=1,
            values_ndim=0,
            backend=bkd,
        )
        sample = bkd.array([2, 0])[:, None]
        errors = model.check_apply_jacobian(sample)
        assert errors.min() / errors.max() < 1e-6
        # hessian is constant diagonal matrix so check first
        # finite difference is exact
        errors = model.check_apply_hessian(sample)
        assert errors[0] < 1e-15

        # test hessian is created corectly from apply_hessian
        assert np.allclose(model.hessian(sample), np.diag([2, 2])[None, :])

    def test_vector_model_from_callable(self):
        bkd = self.get_backend()
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = [
            sum([s * (ii + 1) for ii, s in enumerate(symbs)]) ** 4,
            sum([s * (ii + 1) for ii, s in enumerate(symbs)]) ** 5,
        ]
        sp_grad = [[fun.diff(x) for x in symbs] for fun in sp_fun]
        sp_hess = [
            [[fun.diff(_x).diff(_y) for _x in symbs] for _y in symbs]
            for fun in sp_fun
        ]
        sample = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))

        def hessian(sample):
            return self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hess, "numpy"), sample
            )

        def apply_weighted_hessian(sample, vec, weights):
            hess = hessian(sample)
            return bkd.sum(weights[..., None] * hess, axis=0) @ vec

        model = ModelFromSingleSampleCallable(
            2,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            jacobian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
            apply_jacobian=lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            )
            @ vec,
            hessian=hessian,
            apply_weighted_hessian=apply_weighted_hessian,
            backend=bkd,
        )
        sample = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
        errors = model.check_apply_jacobian(sample)
        # turn off apply_jacobian to check that it can be reconstructed
        # from jacobian
        model.apply_jacobian_implemented = lambda: False
        errors = model.check_apply_jacobian(sample, disp=True)

        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.1
        model.apply_jacobian_implemented = lambda: True

        errors = model.check_apply_hessian(
            sample, weights=bkd.ones((model.nqoi(), 1)), disp=True
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() > 0.3
        # turn off apply_weighted_hessian to check that it can be reconstructed
        # from hessian
        model.apply_weighted_hessian_implemented = lambda: False
        errors = model.check_apply_hessian(
            sample, weights=bkd.ones((model.nqoi(), 1)), disp=True
        )

        # check full hessian is correct
        assert np.allclose(model.hessian(sample), hessian(sample))

    def test_scipy_wrapper(self):
        bkd = self.get_backend()
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s * (ii + 1) for ii, s in enumerate(symbs)]) ** 4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromSingleSampleCallable(
            1,
            3,
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample
            ),
            jacobian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample
            ),
            hessian=lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample
            )[None, ...],
            backend=bkd,
        )
        scipy_model = ScipyModelWrapper(model)
        # check scipy model works with 1D sample array
        sample = bkd.asarray(np.random.uniform(0, 1, (nvars)))
        assert np.allclose(
            scipy_model.jac(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample[:, None]
            ),
        )
        assert np.allclose(
            scipy_model.hess(sample),
            self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian, "numpy"), sample[:, None]
            ),
        )

    def test_umbridge_model(self):
        from pyapprox.interface.tests.genz_umbridge_server import (
            GenzUMBModel,
            GenzIntegral,
        )

        config = {"name": "oscillatory", "nvars": 2, "coef_type": "none"}
        sample = np.random.uniform(0, 1, (config["nvars"], 1))
        model = GenzUMBModel()
        # check model runs
        model(sample.T, config)
        model.gradient(1, config["nvars"], sample.T, None, config)
        model = GenzIntegral()
        model(sample.T, config)

        bkd = self.get_backend()
        server_dir = os.path.dirname(__file__)
        url = "http://localhost:4242"
        run_server_string = "python {0}".format(
            os.path.join(server_dir, "genz_umbridge_server.py")
        )
        process, out = UmbridgeModelWrapper.start_server(
            run_server_string, url=url
        )
        umb_model = umbridge.HTTPModel(url, "genz")
        config = {"name": "oscillatory", "nvars": 2, "coef_type": "none"}
        model = UmbridgeModelWrapper(umb_model, config, backend=bkd)
        sample = bkd.asarray(np.random.uniform(0, 1, (config["nvars"], 1)))
        model.check_apply_jacobian(sample, disp=True)
        UmbridgeModelWrapper.kill_server(process, out)

    def test_io_model(self):
        bkd = self.get_backend()
        intmpdir = tempfile.TemporaryDirectory()

        infilenames = [os.path.join(intmpdir.name, "vec.npz")]

        nvars = 2
        vec = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
        np.savez(infilenames[0], vec=vec)

        class TestIOModel(IOModel):
            def _run(self, sample, linked_filenames, outdirname):
                vec = np.load(linked_filenames[0])["vec"]
                # save a file to check cleaning of directories works
                np.savez(os.path.join(outdirname, "out.npz"), sample + 1)
                return np.atleast_2d(vec.dot(sample))

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = bkd.stack([vec @ sample for sample in samples.T], axis=0)

        # test when temp work directories are used
        model = TestIOModel(1, nvars, infilenames, backend=bkd)
        values = model(samples)
        assert np.allclose(values, test_values)

        # test when all contents of work directories are saved
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        model = TestIOModel(
            1,
            nvars,
            infilenames,
            outdir_basename,
            save="full",
            datafilename=datafilename,
            backend=bkd,
        )
        values = model(samples)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
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
        model = TestIOModel(
            1,
            nvars,
            infilenames,
            outdir_basename,
            save="limited",
            datafilename=datafilename,
            backend=bkd,
        )
        values = model(samples)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
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
            1,
            self.get_input_sizes(config)[0],
            config["infilenames"],
            config["outdir_basename"],
            save=config["save"],
            datafilename=config["datafilename"])
        return [model(np.array(parameters).T)[0].tolist()]

    def supports_evaluate(self):
        return True


if __name__ == "__main__":
    models = [UMBModel()]
    umbridge.serve_models(models, 4242)
"""
        bkd = self.get_backend()
        intmpdir = tempfile.TemporaryDirectory()
        server_filename = os.path.join(intmpdir.name, "server.py")
        out = open(os.path.join(intmpdir.name, "out"), "w")

        with open(server_filename, "w") as writefile:
            writefile.write(server_string)
        nvars = 2
        vec = bkd.array(np.random.uniform(0.0, 1.0, (1, nvars)))
        infilenames = [os.path.join(intmpdir.name, "vec.npz")]
        np.savez(infilenames[0], vec=vec)

        url = "http://localhost:4242"
        run_server_string = "python {0}".format(server_filename)
        process, out = UmbridgeIOModelWrapper.start_server(
            run_server_string, url=url, out=out
        )
        umb_model = umbridge.HTTPModel(url, "umbmodel")

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))

        # test limited save
        outtmpdir = tempfile.TemporaryDirectory()
        config = {
            "infilenames": infilenames,
            "save": "limited",
            "nvars": nvars,
            "datafilename": "data.npz",
        }
        outdir_basename = os.path.join(outtmpdir.name, "results")
        model = UmbridgeIOModelWrapper(
            umb_model, config, outdir_basename=outdir_basename, backend=bkd
        )
        values = model(samples)
        test_values = bkd.stack([vec @ sample for sample in samples.T], axis=0)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
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
            "infilenames": infilenames,
            "save": "full",
            "nvars": nvars,
            "datafilename": "data.npz",
        }
        outdir_basename = os.path.join(outtmpdir.name, "results")
        model = UmbridgeIOModelWrapper(
            umb_model, config, outdir_basename=outdir_basename, backend=bkd
        )
        values = model(samples)
        test_values = bkd.stack([vec @ sample for sample in samples.T], axis=0)
        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
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

    def test_pool_model_wrapper(self):
        bkd = self.get_backend()
        nvars, nsamples = 3, 4
        model = ModelFromSingleSampleCallable(
            2,
            nvars,
            partial(_pickable_function, bkd),
            backend=bkd,
        )
        pool_model = PoolModelWrapper(model, nprocs=2, assert_omp=False)
        pool_model.model().work_tracker().set_active(True)
        pool_model.work_tracker().set_active(True)
        samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))
        values = pool_model(samples)
        assert bkd.allclose(values, _pickable_function(bkd, samples))
        assert pool_model.work_tracker().nevaluations("val") == 4
        assert pool_model.model().work_tracker().nevaluations("val") == 4

    def test_serial_io_model(self):
        bkd = self.get_backend()
        nvars = 2
        intmpdir = tempfile.TemporaryDirectory()
        infilenames = [os.path.join(intmpdir.name, "vec.npz")]
        vec = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
        np.savez(infilenames[0], vec=vec)
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name

        shell_command, target_model = get_shell_command_for_io_model(
            backend=bkd
        )
        model = SerialIOModel(
            2,
            nvars,
            infilenames,
            shell_command,
            outdir_basename=outdir_basename,
            verbosity=1,
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = target_model(samples)
        values = model(samples)
        print(model.work_tracker())
        assert model.work_tracker().nevaluations("val") == 3
        assert np.allclose(values, test_values)
        outtmpdir.cleanup()
        intmpdir.cleanup()

    def test_asynchronous_io_model(self):
        bkd = self.get_backend()
        nvars = 2
        intmpdir = tempfile.TemporaryDirectory()
        infilenames = [os.path.join(intmpdir.name, "vec.npz")]
        vec = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
        np.savez(infilenames[0], vec=vec)
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        shell_command, target_model = get_shell_command_for_io_model(
            backend=bkd
        )

        # test with save="full"
        model = AsyncIOModel(
            2,
            nvars,
            infilenames,
            shell_command,
            outdir_basename=outdir_basename,
            verbosity=0,
            save="full",
            datafilename=datafilename,
            nprocs=2,
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = target_model(samples)
        values = model(samples)
        assert model.work_tracker().nevaluations("val") == 3
        assert np.allclose(values, test_values)

        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
            filenames = glob.glob(os.path.join(outdirname, "*"))
            filenames = [os.path.basename(fname) for fname in filenames]
            filenames.sort()
            assert len(filenames) == 4  # this will be 5 if verbosity > 0
            assert os.path.basename(filenames[0]) == datafilename
            assert os.path.basename(filenames[1]) == "params.in"
            assert os.path.basename(filenames[2]) == "results.out"
            assert os.path.basename(filenames[3]) == "vec.npz"
        outtmpdir.cleanup()

        # test with save="limited"
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        model = AsyncIOModel(
            2,
            nvars,
            infilenames,
            shell_command,
            outdir_basename=outdir_basename,
            verbosity=1,
            save="limited",
            datafilename=datafilename,
            nprocs=2,
            backend=bkd,
        )
        model.work_tracker().set_active(True)

        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = target_model(samples)
        values = model(samples)
        assert model.work_tracker().nevaluations("val") == 3
        assert np.allclose(values, test_values)

        # evalaute another batch of samples to make sure counter is
        # still updated correctly
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = target_model(samples)
        values = model(samples)
        assert model.work_tracker().nevaluations("val") == 6
        assert np.allclose(values, test_values)

        assert np.allclose(values, test_values)
        for outdirname in glob.glob(os.path.join(outdir_basename, "*")):
            filenames = glob.glob(os.path.join(outdirname, "*"))
            assert len(filenames) == 1
            assert os.path.basename(filenames[0]) == datafilename
        outtmpdir.cleanup()

        # test database is used correctly
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        model = AsyncIOModel(
            2,
            nvars,
            infilenames,
            shell_command,
            outdir_basename=outdir_basename,
            verbosity=1,
            save="limited",
            datafilename=datafilename,
            nprocs=2,
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        model.activate_model_data_base()
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = target_model(samples)
        values = model(samples[:, [0, 2]])
        assert model.work_tracker().nevaluations("val") == 2
        assert np.allclose(values, test_values[[0, 2]])
        values = model(samples)
        assert model.work_tracker().nevaluations("val") == 3
        assert np.allclose(values, test_values)
        outtmpdir.cleanup()

        intmpdir.cleanup()

    def test_asynchronous_io_model_with_hard_faults(self):
        # note seed does not control which samples fail. As shell command uses
        # a seed independent of this script to determine if a sample fails.
        #  TODO: stop this being a stochastic test.
        bkd = self.get_backend()
        nvars = 2
        intmpdir = tempfile.TemporaryDirectory()
        infilenames = [os.path.join(intmpdir.name, "vec.npz")]
        vec = bkd.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
        np.savez(infilenames[0], vec=vec)
        outtmpdir = tempfile.TemporaryDirectory()
        outdir_basename = outtmpdir.name
        datafilename = "data.npz"
        shell_command, target_model = get_shell_command_for_io_model(
            backend=bkd
        )
        # test with save="full"
        model = AsyncIOModel(
            2,
            nvars,
            infilenames,
            shell_command,
            outdir_basename=outdir_basename,
            verbosity=0,
            save="full",
            datafilename=datafilename,
            nprocs=2,
            backend=bkd,
        )
        model.work_tracker().set_active(True)
        nsamples = 3
        samples = bkd.asarray(np.random.uniform(0.0, 1.0, (nvars, nsamples)))
        test_values = bkd.asarray(target_model(samples))
        values = model(samples)
        assert model.work_tracker().nevaluations("val") == 3
        # regression test (using np.random.seed(1)) that last value fails and
        # is recorded as a failure
        pass_idx = bkd.asarray(np.isfinite(values), dtype=bool)
        assert np.allclose(values[pass_idx], test_values[pass_idx])
        outtmpdir.cleanup()
        intmpdir.cleanup()

    def _check_finite_differences(self, FD_cls):
        bkd = self.get_backend()
        nvars = 3

        model = ModelFromSingleSampleCallable(
            2,
            nvars,
            lambda x: bkd.hstack(
                [
                    1 * ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2),
                    2 * ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2),
                ],
            ),
            jacobian=lambda x: bkd.stack(
                [
                    1 * bkd.array([2 * (x[0] - 1), 2 * (x[1] - 2.5), 0]),
                    2 * bkd.array([2 * (x[0] - 1), 2 * (x[1] - 2.5), 0]),
                ],
                axis=0,
            ),
            apply_jacobian=lambda x, v: bkd.stack(
                [
                    1 * (2 * (x[0] - 1) * v[0] + 2 * (x[1] - 2.5) * v[1]),
                    2 * (2 * (x[0] - 1) * v[0] + 2 * (x[1] - 2.5) * v[1]),
                ],
                axis=0,
            ),
            hessian=lambda x: bkd.stack(
                [
                    bkd.diag(bkd.array([2, 2, 0])),
                    bkd.diag(bkd.array([4, 4, 0])),
                ]
            ),
            apply_hessian=lambda x, v: bkd.stack(
                [
                    bkd.diag(bkd.array([2, 2, 0])) @ v,
                    bkd.diag(bkd.array([4, 4, 0])) @ v,
                ],
                axis=0,
            ),
            sample_ndim=1,
            values_ndim=1,
            backend=bkd,
        )
        rtol = 1e-6
        fd = FD_cls(model)
        nvecs = 3
        sample = bkd.array(np.random.uniform(0, 1, (nvars, 1)))
        vec = bkd.array(np.random.normal(0, 1, (nvars, nvecs)))
        assert bkd.allclose(
            fd.jacobian(sample), model.jacobian(sample), rtol=rtol
        )
        assert bkd.allclose(
            fd.apply_jacobian(sample, vec),
            model.apply_jacobian(sample, vec),
            rtol=rtol,
        )
        assert bkd.allclose(
            fd.hessian(sample), model.hessian(sample), rtol=rtol
        )

        # model.apply_hessian only works for nqoi == 1
        # so create a new model
        model = ModelFromSingleSampleCallable(
            1,
            nvars,
            lambda x: bkd.hstack(
                [1 * ((x[0] - 1) ** 2 + (x[1] - 2.5) ** 2)],
            ),
            jacobian=lambda x: bkd.stack(
                [1 * bkd.array([2 * (x[0] - 1), 2 * (x[1] - 2.5), 0])],
                axis=0,
            ),
            apply_jacobian=lambda x, v: bkd.asarray(
                [1 * (2 * (x[0] - 1) * v[0] + 2 * (x[1] - 2.5) * v[1])]
            ),
            hessian=lambda x: bkd.stack([bkd.diag(bkd.array([2, 2, 0]))]),
            apply_hessian=lambda x, v: bkd.stack(
                [bkd.diag(bkd.array([2, 2, 0])) @ v]
            ),
            sample_ndim=1,
            values_ndim=1,
            backend=bkd,
        )
        fd = FD_cls(model)
        assert bkd.allclose(
            fd.apply_hessian(sample, vec[:, :1]),
            model.apply_hessian(sample, vec[:, :1]),
            rtol=rtol,
        )

    def test_finite_differences(self):
        test_cases = [
            # ForwardFiniteDifference,
            # BackwardFiniteDifference,
            CenteredFiniteDifference,
        ]
        for test_case in test_cases:
            self._check_finite_differences(test_case)

    def test_linear_matrix_model(self):
        bkd = self.get_backend()
        nrows, ncols = 3, 2
        matrix = bkd.asarray(np.random.normal(0, 1, (nrows, ncols)))
        model = DenseMatrixLinearModel(matrix, backend=bkd)
        sample = bkd.asarray(np.random.uniform(-1, 1, (ncols, 1)))
        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_quadratic_matrix_model(self):
        bkd = self.get_backend()
        nrows, ncols = 3, 2
        matrix = bkd.asarray(np.random.normal(0, 1, (nrows, ncols)))
        model = QuadraticMatrixModel(matrix, backend=bkd)
        sample = bkd.asarray(np.random.uniform(-1, 1, (ncols, 1)))
        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

        errors = model.check_apply_hessian(
            sample, weights=bkd.ones((nrows, 1)), disp=True
        )
        assert errors.min() / errors.max() < 1e-6


class TestNumpyModel(TestModel, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchModel(TestModel, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Note if port is not closed properly look for PID using
# lsof -i -P -n | grep 4242
