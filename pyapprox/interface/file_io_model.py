import numpy as np
import shutil
import os

from pyapprox.interface.wrappers import (
    evaluate_1darray_function_on_2d_array, run_shell_command
)


class FileIOModel(object):
    """
    Evaluate a model by writing parameters to file,
    running a shell script and reading the results from the file
    created by the shell script.
    """

    def __init__(self, shell_command, params_filename='params.in',
                 results_filename='results.out', process_sample_override=None,
                 load_results_override=None):
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
        self.params_filename = params_filename
        self.results_filename = results_filename
        self.shell_command = shell_command

        if process_sample_override is not None:
            assert callable(process_sample_override)
            self.process_sample = process_sample_override
        else:
            self.process_sample = None
        if load_results_override is not None:
            assert callable(load_results_override)
            self.load_results = load_results_override
        else:
            self.load_results = None

        self.set_function_evaluation_id(0)

    def set_function_evaluation_id(self, eval_id):
        assert type(eval_id) == int
        self.function_eval_id = eval_id

    def run(self, sample):
        np.savetxt(self.params_filename, sample)
        if self.process_sample is not None:
            self.process_sample(sample)

        model_output_verbosity = 0
        run_shell_command(
            self.shell_command, {'verbosity': model_output_verbosity})
        if self.load_results is not None:
            vals = self.load_results()
        else:
            vals = np.loadtxt(self.results_filename, usecols=[0])
        assert vals.ndim == 1

        # store a copy of the parameters and return values with
        # a unique filename
        shutil.copy(
            self.params_filename,
            self.params_filename+'.%d' % self.function_eval_id)
        if os.path.exists(self.results_filename):
            shutil.copy(self.results_filename,
                        self.results_filename+'.%d' % self.function_eval_id)
        else:
            # load_results must have been set but the function provided
            # did not create a file with vals inside. So create for sake
            # of consistency here
            np.savetxt(self.results_filename+'.%d' %
                       self.function_eval_id, vals)
        return vals

    def get(self, key, assert_exists=True):
        return self.model.get(key, assert_exists)

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(self.run, samples)
