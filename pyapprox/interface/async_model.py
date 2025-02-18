import os
import tempfile
import subprocess
import shutil
import re
import glob
from typing import List, Dict, Tuple

import numpy as np

from pyapprox.interface.model import Model
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class AsyncModel(Model):
    """
    Evaluate a model in parallel when model instances are invoked by a shell
    script.

    Parameters
    ----------
    process_sample: callable function (default=None)
        Function that overwrites the basic implementation
        which reads the sample in from a file called params_filename.
        This is useful if there are a number of pre-processing steps
        needed by the model before shall command is executed.

    load_results: callable function (default=None)
        Function that overwrites the basic implementation
        which reads the results from a files called results_filename
        This is useful if there are a number of post-processing steps
        needed by the model after the shell command is executed.
        If evaluation fails this function must return None

    workdir_basename: string (default=None):
        The name of the directory to store local copies or soft links
        of files. The shell command will be executed in these directories.
        If None temporary directories with python generated names
        will be created and then deleted once the shell command has been
        run.

    save_workdirs : string (default=True)
        'no'      - do not save workdirs
        'yes'     - save all files in each workdir
        'limited' - save only params and results files
        If workdir_basename is None this variable is ignored
        the tmp directories that are created will always be deleted.

    model_name :  string (default=None)
        A identifier used when printing model evaluation info

    saved_data_basename : string (default=None)
        The basename of the file used to store the output data from the
        model. A new file is created every time __call__ is exceuted.
        a unique identifier is created based upon the value of evaluation id
        when __call__ is started.

    link_filenames : list (default=[])
        List of filenames (strings), including their absolute path that will
        be made available in each workdirectory using soft links

    max_eval_concurrency : integer (default=1)
        How many shell commands to execute a synchronously.

    params_filename : string (default='params.in')
        The name of the file containing the sample realization read in by the
        shell script.

    results_filename : string (default='results.out')
        The name of the file containing the model output written by the
        shell script.

    params_file_header : string (default='')
        A header that will be added to the params_file, which may be
        required by some scripts.
    """

    def __init__(
        self,
        nqoi: int,
        nvars: int,
        shell_command: str,
        max_eval_concurrency: int = 1,
        workdir_basename: str = None,
        link_filenames: List = [],
        params_filename: str = "params.in",
        results_filename: str = "results.out",
        params_file_header: str = "",
        process_sample: callable = None,
        load_results: callable = None,
        saved_data_basename: str = None,
        save_workdirs: str = "yes",
        model_name: "str" = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        self._nqoi = nqoi
        self._nvars = nvars
        self._shell_command = shell_command
        self._max_eval_concurrency = max_eval_concurrency
        self._workdir_basename = workdir_basename
        self._link_filenames = link_filenames
        self._params_filename = params_filename
        self._results_filename = results_filename
        self._params_file_header = params_file_header
        # ensure process sample is not a member function. If it is
        # this model will not pikcle
        self._process_sample = process_sample
        self._load_results = load_results
        self._saved_data_basename = saved_data_basename
        self._save_workdirs = save_workdirs
        if not (
            (save_workdirs == "no")
            or (save_workdirs == "yes")
            or (save_workdirs == "limited")
        ):
            raise Exception('save_workdirs must be ["no","yes","limited"]')
        self._model_name = model_name
        if self._saved_data_basename is not None:
            saved_data_dir = os.path.split(saved_data_basename)[0]
            if not saved_data_dir == "" and not os.path.exists(saved_data_dir):
                os.makedirs(saved_data_dir)

        self._running_procs = []
        self._running_workdirs = []
        self._function_eval_id = 0

        self._current_vals = []

    def nqoi(self) -> int:
        return self._nqoi

    def nvars(self) -> int:
        return self._nvars

    def cleanup_threads(self, opts: Dict):
        verbosity = opts.get("verbosity", 0)
        finished_proc_indices = []
        for i in range(len(self._running_procs)):
            proc = self._running_procs[i]
            if proc.poll() is not None:
                finished_proc_indices.append(i)

        curdir = os.getcwd()
        for i in range(len(finished_proc_indices)):

            workdir = self._running_workdirs[finished_proc_indices[i]]
            os.chdir(workdir)

            function_eval_id = int(
                re.findall(r"[0-9]+", os.path.split(workdir)[1])[-1]
            )

            if self._load_results is None:
                if not os.path.exists(self._results_filename):
                    if verbosity > 0:
                        print(
                            "Eval %d: %s was not found in directory %s"
                            % (
                                function_eval_id,
                                self._results_filename,
                                workdir,
                            )
                        )
                    vals = None
                else:
                    vals = np.loadtxt(self._results_filename, usecols=[0])
                    shutil.copy(
                        self._results_filename,
                        self._results_filename + ".%d" % function_eval_id,
                    )
            else:
                try:
                    vals = self._load_results(opts)
                except ImportError:
                    vals = None
                # load results may not have generated a results file
                # so write one here
                if vals is not None:
                    np.savetxt(
                        self._results_filename + ".%d" % function_eval_id, vals
                    )
            sample = np.loadtxt(
                self._params_filename + ".%d" % function_eval_id
            )

            if (
                self._workdir_basename is not None
                and self._save_workdirs == "limited"
            ):
                filenames_to_delete = glob.glob("*")
                if vals is not None:
                    filenames_to_delete.remove(
                        self._results_filename + ".%d" % function_eval_id
                    )
                filenames_to_delete.remove(
                    self._params_filename + ".%d" % function_eval_id
                )
                if verbosity > 0:
                    filenames_to_delete.remove("stdout.txt")
                for filename in filenames_to_delete:
                    os.remove(filename)

            if verbosity > 0:
                print(
                    "Model %s: completed eval %d"
                    % (self._model_name, function_eval_id)
                )

            self._current_vals.append(vals)
            self._current_samples.append(sample)
            self._completed_function_eval_ids.append(function_eval_id)

            os.chdir(curdir)
            if self._workdir_basename is None or self._save_workdirs == "no":
                shutil.rmtree(workdir)

        self._running_procs = [
            v
            for i, v in enumerate(self._running_procs)
            if i not in finished_proc_indices
        ]
        self._running_workdirs = [
            v
            for i, v in enumerate(self._running_workdirs)
            if i not in finished_proc_indices
        ]

    def create_work_dir(self) -> Tuple[tempfile.TemporaryDirectory, str]:
        if self._workdir_basename is None:
            tmpdir = tempfile.TemporaryDirectory(
                suffix=".%d" % self._function_eval_id
            )
            tmpdirname = tmpdir.name
        else:
            tmpdirname = (
                self._workdir_basename + ".%d" % self._function_eval_id
            )
            if not os.path.exists(tmpdirname):
                os.makedirs(tmpdirname)
            else:
                msg = "work_dir %s already exists. " % (tmpdirname)
                msg += "Exiting so as not to overwrite previous results"
                raise Exception(msg)
            tmpdir = None
        return tmpdir, tmpdirname

    def asynchronous_evaluate_using_shell_command(
        self, sample: Array, opts: Dict
    ):
        verbosity = opts.get("verbosity", 0)

        curdir = os.getcwd()
        workdir, workdirname = self._create_work_dir()
        os.chdir(workdirname)

        for filename in self._link_filenames:
            if not os.path.exists(os.path.split(filename)[1]):
                os.symlink(filename, os.path.split(filename)[1])
            else:
                msg = "%s exists in %s cannot create soft link" % (
                    filename,
                    workdirname,
                )
                raise Exception(msg)
        # default of savetxt is to write header with # at start of line
        # comments='' removes the #
        if self._process_sample is not None:
            self._process_sample(sample)
        else:
            np.savetxt(
                self._params_filename,
                sample,
                header=self._params_file_header,
                comments="",
            )

        if verbosity > 0:
            out = open("stdout.txt", "wb")
        else:
            out = open(os.devnull, "w")
        proc = subprocess.Popen(
            self._shell_command, shell=True, stdout=out, stderr=out
        )

        out.close()

        self._running_procs.append(proc)
        self._running_workdirs.append(workdirname)

        # store a copy of the parameters and return values with
        # a unique filename
        shutil.copy(
            self._params_filename,
            self._params_filename + ".%d" % self._function_eval_id,
        )

        self._function_eval_id += 1
        os.chdir(curdir)
        if workdir is not None:
            workdir.cleanup()

    def __call__(self, samples: Array, opts: Dict = dict()) -> Array:

        self._current_vals = []
        self._current_samples = []
        self._completed_function_eval_ids = []
        nsamples = samples.shape[1]
        for i in range(nsamples):
            while len(self._running_procs) >= self._max_eval_concurrency:
                self._cleanup_threads(opts)
            self._asynchronous_evaluate_using_shell_command(
                samples[:, i], opts
            )

        while len(self._running_procs) > 0:
            self._cleanup_threads(opts)

        if self._saved_data_basename is not None:
            data_filename = self._saved_data_basename + "-%d-%d.npz" % (
                self._function_eval_id - nsamples,
                self._function_eval_id,
            )
        else:
            data_filename = None

        vals = self._prepare_values(
            self._current_samples,
            self._current_vals,
            self._completed_function_eval_ids,
            data_filename,
        )

        return vals

    def prepare_values(
        self,
        samples: Array,
        vals: Array,
        completed_function_eval_ids: Array,
        data_filename: str,
    ) -> Array:
        nsamples = len(vals)
        # get number of QoI
        nqoi = 0
        for i in range(nsamples):
            if vals[i] is not None:
                nqoi = vals[i].shape[0]
                break

        if nqoi == 0 and vals[0] is None:
            raise Exception("All model evaluations failed")

        # return nan for failed function evaluations
        for i in range(nsamples):
            if vals[i] is None:
                vals[i] = np.zeros((nqoi)) + np.nan

        II = np.argsort(np.array(completed_function_eval_ids))
        prepared_vals = np.array(vals)[II, :]
        samples = np.asarray(samples).T[:, II]

        if data_filename is not None:
            np.savez(data_filename, vals=prepared_vals, samples=samples)
        return prepared_vals
