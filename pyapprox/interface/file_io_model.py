import numpy as np
import shutil
import os
import subprocess

from pyapprox.interface.model import SingleSampleModel
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class FileIOModel(SingleSampleModel):
    """
    Evaluate a model by writing parameters to file,
    running a shell script and reading the results from the file
    created by the shell script.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        shell_command: str,
        params_filename: str = "params.in",
        results_filename: str = "results.out",
        process_sample_override: callable = None,
        load_results_override: callable = None,
        verbosity: int = 0,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        """
        Parameters
        ----------
        shell_command : string
            The command that executes the model in a shell

        params_filename : string, default=params.in
            The filename of the file that will contain the
            sample at which the model will be evaluated. A copy
            of the params_filename will be made in the current
            directory called params_filename.function_eval_id

        results_filename : string, default=results.out
            The filename of the file that will contain the
            model output at a sample. A copy
            of the params_filename will be made in the current
            directory called results_filename.function_eval_id
        """
        super().__init__(backend)
        self._nqoi = nqoi
        self._nvars = nvars
        self._params_filename = params_filename
        self._results_filename = results_filename
        self._shell_command = shell_command
        self._verbosity = verbosity

        if process_sample_override is not None:
            assert callable(process_sample_override)
            self._process_sample = process_sample_override
        else:
            self._process_sample = None
        if load_results_override is not None:
            assert callable(load_results_override)
            self._load_results = load_results_override
        else:
            self._load_results = None

        self._set_function_evaluation_id(0)

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def set_function_evaluation_id(self, eval_id: int):
        assert type(eval_id) == int
        self._function_eval_id = eval_id

    def _run_shell_command(self):
        if self._verbosity == 0:
            subprocess.check_output(self._shell_command, shell=True, env=env)
        elif self._verbosity == 1:
            filename = "shell_command.out"
            with open(filename, "w") as f:
                subprocess.call(
                    self._shell_command,
                    shell=True,
                    stdout=f,
                    stderr=f,
                    env=None,
                )
        else:
            subprocess.call(self._shell_command, shell=True, env=None)

    def _evaluate(self, sample: Array):
        np.savetxt(self._params_filename, sample)
        if self._process_sample is not None:
            self._process_sample(sample)
        self._run_shell_command()
        if self._load_results is not None:
            vals = self._load_results()
        else:
            vals = np.loadtxt(self._results_filename, usecols=[0])
        assert vals.ndim == 1

        # store a copy of the parameters and return values with
        # a unique filename
        shutil.copy(
            self._params_filename,
            self._params_filename + ".%d" % self._function_eval_id,
        )
        if os.path.exists(self._results_filename):
            shutil.copy(
                self._results_filename,
                self._results_filename + ".%d" % self._function_eval_id,
            )
        else:
            # load_results must have been set but the function provided
            # did not create a file with vals inside. So create for sake
            # of consistency here
            np.savetxt(
                self._results_filename + ".%d" % self._function_eval_id, vals
            )
        return vals

    def get(self, key, assert_exists=True):
        return self._model.get(key, assert_exists)
