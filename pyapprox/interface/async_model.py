import os
import numpy as np
import tempfile
import subprocess
import shutil
import re
import glob


class AynchModel(object):
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

    def __init__(self, shell_command, max_eval_concurrency=1,
                 workdir_basename=None, link_filenames=[],
                 params_filename='params.in', results_filename='results.out',
                 params_file_header='', process_sample=None,
                 load_results=None, saved_data_basename=None,
                 save_workdirs='yes', model_name=None):

        self.shell_command = shell_command
        self.max_eval_concurrency = max_eval_concurrency
        self.workdir_basename = workdir_basename
        self.link_filenames = link_filenames
        self.params_filename = params_filename
        self.results_filename = results_filename
        self.params_file_header = params_file_header
        # ensure process sample is not a member function. If it is
        # this model will not pikcle
        self.process_sample = process_sample
        self.load_results = load_results
        self.saved_data_basename = saved_data_basename
        self.save_workdirs = save_workdirs
        if (not ((save_workdirs == 'no') or (save_workdirs == 'yes') or
                 (save_workdirs == 'limited'))):
            raise Exception('save_workdirs must be ["no","yes","limited"]')
        self.model_name = model_name
        if self.saved_data_basename is not None:
            saved_data_dir = os.path.split(saved_data_basename)[0]
            if not saved_data_dir == '' and not os.path.exists(saved_data_dir):
                os.makedirs(saved_data_dir)

        self.running_procs = []
        self.running_workdirs = []
        self.function_eval_id = 0
        self.num_qoi = 0

        self.current_vals = []

    def cleanup_threads(self, opts):
        verbosity = opts.get("verbosity", 0)
        finished_proc_indices = []
        for i in range(len(self.running_procs)):
            proc = self.running_procs[i]
            if proc.poll() is not None:
                finished_proc_indices.append(i)

        curdir = os.getcwd()
        for i in range(len(finished_proc_indices)):

            workdir = self.running_workdirs[finished_proc_indices[i]]
            os.chdir(workdir)

            function_eval_id = int(re.findall(
                r'[0-9]+', os.path.split(workdir)[1])[-1])

            if self.load_results is None:
                if not os.path.exists(self.results_filename):
                    if verbosity > 0:
                        print('Eval %d: %s was not found in directory %s' % (
                            function_eval_id, self.results_filename, workdir))
                    vals = None
                else:
                    vals = np.loadtxt(self.results_filename, usecols=[0])
                    shutil.copy(
                        self.results_filename,
                        self.results_filename+'.%d' % function_eval_id)
            else:
                try:
                    vals = self.load_results(opts)
                except ImportError:
                    vals = None
                # load results may not have generated a results file
                # so write one here
                if vals is not None:
                    np.savetxt(
                        self.results_filename+'.%d' % function_eval_id, vals)
            sample = np.loadtxt(self.params_filename+'.%d' % function_eval_id)

            if (self.workdir_basename is not None and
                    self.save_workdirs == 'limited'):
                filenames_to_delete = glob.glob('*')
                if vals is not None:
                    filenames_to_delete.remove(
                        self.results_filename+'.%d' % function_eval_id)
                filenames_to_delete.remove(
                    self.params_filename+'.%d' % function_eval_id)
                if verbosity > 0:
                    filenames_to_delete.remove('stdout.txt')
                for filename in filenames_to_delete:
                    os.remove(filename)

            if verbosity > 0:
                print('Model %s: completed eval %d' % (
                    self.model_name, function_eval_id))

            self.current_vals.append(vals)
            self.current_samples.append(sample)
            self.completed_function_eval_ids.append(function_eval_id)

            os.chdir(curdir)
            if self.workdir_basename is None or self.save_workdirs == 'no':
                shutil.rmtree(workdir)

        self.running_procs = [v for i, v in enumerate(
            self.running_procs) if i not in finished_proc_indices]
        self.running_workdirs = [v for i, v in enumerate(
            self.running_workdirs) if i not in finished_proc_indices]

    def create_work_dir(self):
        if self.workdir_basename is None:
            tmpdir = tempfile.TemporaryDirectory(
                suffix='.%d' % self.function_eval_id)
            tmpdirname = tmpdir.name
        else:
            tmpdirname = self.workdir_basename+'.%d' % self.function_eval_id
            if not os.path.exists(tmpdirname):
                os.makedirs(tmpdirname)
            else:
                msg = 'work_dir %s already exists. ' % (tmpdirname)
                msg += 'Exiting so as not to overwrite previous results'
                raise Exception(msg)
            tmpdir = None
        return tmpdir, tmpdirname

    def asynchronous_evaluate_using_shell_command(self, sample, opts):
        verbosity = opts.get("verbosity", 0)

        curdir = os.getcwd()
        workdir, workdirname = self.create_work_dir()
        os.chdir(workdirname)

        for filename in self.link_filenames:
            if not os.path.exists(os.path.split(filename)[1]):
                os.symlink(
                    filename, os.path.split(filename)[1])
            else:
                msg = '%s exists in %s cannot create soft link' % (
                    filename, workdirname)
                raise Exception(msg)
        # default of savetxt is to write header with # at start of line
        # comments='' removes the #
        if self.process_sample is not None:
            self.process_sample(sample)
        else:
            np.savetxt(self.params_filename, sample,
                       header=self.params_file_header,
                       comments='')

        if verbosity > 0:
            out = open("stdout.txt", "wb")
        else:
            out = open(os.devnull, 'w')
        proc = subprocess.Popen(
            self.shell_command, shell=True, stdout=out, stderr=out)

        out.close()

        self.running_procs.append(proc)
        self.running_workdirs.append(workdirname)

        # store a copy of the parameters and return values with
        # a unique filename
        shutil.copy(
            self.params_filename,
            self.params_filename+'.%d' % self.function_eval_id)

        self.function_eval_id += 1
        os.chdir(curdir)
        if workdir is not None:
            workdir.cleanup()

    def __call__(self, samples, opts=dict()):

        self.current_vals = []
        self.current_samples = []
        self.completed_function_eval_ids = []
        nsamples = samples.shape[1]
        for i in range(nsamples):
            while len(self.running_procs) >= self.max_eval_concurrency:
                self.cleanup_threads(opts)
            self.asynchronous_evaluate_using_shell_command(
                samples[:, i], opts)

        while len(self.running_procs) > 0:
            self.cleanup_threads(opts)

        if self.saved_data_basename is not None:
            data_filename = self.saved_data_basename+'-%d-%d.npz' % (
                self.function_eval_id-nsamples, self.function_eval_id)
        else:
            data_filename = None

        vals = self.prepare_values(
            self.current_samples, self.current_vals,
            self.completed_function_eval_ids, data_filename)

        return vals

    def prepare_values(self, samples, vals, completed_function_eval_ids,
                       data_filename):
        nsamples = len(vals)
        # get number of QoI
        num_qoi = 0
        for i in range(nsamples):
            if vals[i] is not None:
                num_qoi = vals[i].shape[0]
                break

        if num_qoi == 0 and vals[0] is None:
            raise Exception('All model evaluations failed')

        # return nan for failed function evaluations
        for i in range(nsamples):
            if vals[i] is None:
                vals[i] = np.zeros((num_qoi))+np.nan

        II = np.argsort(np.array(completed_function_eval_ids))
        prepared_vals = np.array(vals)[II, :]
        samples = np.asarray(samples).T[:, II]

        if data_filename is not None:
            np.savez(data_filename, vals=prepared_vals, samples=samples)
        return prepared_vals
