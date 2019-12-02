import unittest, os, glob
from pyapprox.models.async_model import *
from pyapprox.models.file_io_model import *

import multiprocessing
max_eval_concurrency = max(2,multiprocessing.cpu_count()-2)

def cleanup_work_directories(basename='work-dir'):
    """
    If a previous tests fails it may leave work directories that
    will cause problems for next test so call this function at begining
    of every test

    Note it test run from wrong directory this can remove directories
    that it should not
    """
    if basename is not None:
        directories = glob.glob(basename+'*')
        for directory in directories:
            shutil.rmtree(directory)

def remove_files(filenames):
    for filename in filenames:
        os.remove(filename)

def cleanup_fileiomodel_files():
    """
    Remove all the params.in and results.out files created by a test. 
    This function is only useful when not using work directories
    """
    filenames = glob.glob('params.in*')
    filenames += glob.glob('results.out*')
    remove_files(filenames)

def check_model_values(model, target_function, num_vars, num_samples,
                       opts={}, ignore_nans=False,saved_data_basename=None):
    """
    Assert the model produces the same set of values for a set of
    samples as evaluating the target function it wraps directly.

    Assumes domain of model samples contains [-1,1]^num_vars
    """
    samples = np.random.uniform(-1.,1.,(num_vars,num_samples))
    vals = model(samples, opts)
    num_qoi = vals.shape[1]
    true_vals = np.empty((num_samples,num_qoi))
    finite_evals_index=[]
    for i in range(num_samples):
        true_vals[i,:] = target_function(samples[:,i])
        if not ignore_nans or not not np.any(np.isfinite(vals[i])):
            finite_evals_index.append(i)

    finite_evals_index = np.array(finite_evals_index)
    assert np.allclose(
        true_vals[finite_evals_index,:],vals[finite_evals_index,:])

    if saved_data_basename is not None:
        # used for async model
        filenames = glob.glob(saved_data_basename+'*.npz')
        # this part of the test assumes model.__call__
        # has only been called once
        assert len(filenames)==1
        data = np.load(filenames[0])
        file_samples = data['samples']
        file_values = data['vals']
        assert np.allclose(
            true_vals[finite_evals_index,:],file_values[finite_evals_index,:])
        assert np.allclose(samples,file_samples)
    
    return finite_evals_index

def check_fileiomodel_files(directories, link_filenames):
    for directory in directories:
        # check each work directories contains params.in.i and
        # results.out.i
        dir_num = int(re.findall(
            r'[0-9]+', os.path.split(directory)[1] )[-1])
        assert os.path.exists(
            os.path.join(directory,'params.in.%d'%dir_num))
        assert os.path.exists(os.path.join(directory,'results.out.%d'%dir_num))
        for link_filename in lionk_filenames:
            assert os.path.exists(os.path.join(directory,link_filename))

def raise_exception(condition, msg):
    """
    To allow exception to be raised with a conditional statment on one line
    Useful when calling chain of statemnts from command line with python -c
    """
    if condition:
        raise Exception(msg)

def get_file_io_model(delay=0.,fault_percentage=0):
    """
    Return a ShellIOModel that wrapts a call to the 2D target function
    [x[0]**2 + 2*x[1]**3, x[0]**3 + x[0]*x[1]]) with two QoI.
    """
    shell_command = """echo "import numpy as np; target_function = lambda x: np.array([x[0]**2 + 2*x[1]**3, x[0]**3 + x[0]*x[1]]); sample = np.loadtxt('params.in'); u=np.random.uniform(0.,1.); from pyapprox.models.test_async_model import raise_exception; raise_exception(u<%f/100., 'fault occurred'); vals = target_function(sample); np.savetxt('results.out',vals); delay=%f; import time; time.sleep(delay);"| python"""%(fault_percentage,delay+np.random.uniform(-1.,1.)*delay*0.1)
    #print shell_command

    target_function = lambda x: np.array(
        [x[0]**2 + 2*x[1]**3, x[0]**3 + x[0]*x[1]])
    model = FileIOModel(shell_command)
    return model, target_function, 2

class TestAsyncModel(unittest.TestCase):
    def test_file_io_model(self):
        """
        Test FileIOModel using current directory to write and read files
        """
        num_samples = 5
        model, target_function, num_vars = get_file_io_model()
        check_model_values(model, target_function, num_vars, num_samples)
        cleanup_fileiomodel_files()
    
    def test_async_model(self):
        workdir_basename='work-dir'
        #workdir_basename=None
        cleanup_work_directories(workdir_basename)

        num_samples = 4*max_eval_concurrency;
        model, target_function, num_vars = get_file_io_model(0.02)
        shell_command = model.shell_command
        model = AsynchronousEvaluationModel(
            shell_command, max_eval_concurrency=max_eval_concurrency,
            workdir_basename=workdir_basename)

        check_model_values(model, target_function, num_vars, num_samples)
        check_model_values(model, target_function, num_vars, num_samples)

    def test_fault_tolerant_async_model_full_save(self):
        """
        Test that async model continues execution when one or model evaluations
        fails and when the entirity of each work directory is preserved
        """
        workdir_basename='work-dir'
        save_workdirs = 'yes'
        cleanup_work_directories(workdir_basename)

        num_samples = 4*max_eval_concurrency;
        verbosity=0
        model, target_function, num_vars = get_file_io_model(
            0.02,fault_percentage=10)
        shell_command = model.shell_command
        saved_data_basename = 'saved-data'
        model = AsynchronousEvaluationModel(
            shell_command, max_eval_concurrency=max_eval_concurrency,
            workdir_basename=workdir_basename,
            save_workdirs=save_workdirs,
            saved_data_basename=saved_data_basename)

        finite_evals_index = check_model_values(
            model, target_function, num_vars, num_samples, ignore_nans=True,
            saved_data_basename=saved_data_basename,
            opts={'verbosity':verbosity})
        os.remove('saved-data-%d-%d.npz'%(0,num_samples))

        workdirs = glob.glob(workdir_basename+'*')
        assert len(workdirs)==num_samples
        for workdir in workdirs:
            function_eval_id = int(re.findall(
                r'[0-9]+',os.path.split(workdir)[1])[-1])
            workdir_filenames = glob.glob(os.path.join(workdir,'*'))
            if function_eval_id in finite_evals_index:
                assert len(workdir_filenames)==4+min(1,verbosity)
                assert os.path.exists(
                    os.path.join(
                        workdir,model.results_filename+'.%d'%function_eval_id))
            else:
                assert len(workdir_filenames)==2+min(1,verbosity)
            assert os.path.exists(
                os.path.join(workdir,
                             model.params_filename+'.%d'%function_eval_id))

            if verbosity>0:
                assert os.path.exists(os.path.join(workdir,'stdout.txt'))

        cleanup_work_directories(workdir_basename)
        
    def test_fault_tolerant_async_model_limited_save(self):
        """
        Test that async model continues execution when one or model evaluations
        fails and when only params and results files are saved.
        """
        workdir_basename='work-dir'
        save_workdirs = 'limited'
        cleanup_work_directories(workdir_basename)

        num_samples = 4*max_eval_concurrency;
        verbosity=0
        model, target_function, num_vars = get_file_io_model(
            0.0,fault_percentage=10)
        shell_command = model.shell_command
        model = AsynchronousEvaluationModel(
            shell_command, max_eval_concurrency=max_eval_concurrency,
            workdir_basename=workdir_basename,
            save_workdirs=save_workdirs)

        finite_evals_index = check_model_values(
            model, target_function, num_vars, num_samples, ignore_nans=True,
            opts={'verbosity':verbosity})

        workdirs = glob.glob(workdir_basename+'*')
        assert len(workdirs)==num_samples
        for workdir in workdirs:
            function_eval_id = int(re.findall(
                r'[0-9]+',os.path.split(workdir)[1])[-1])
            workdir_filenames = glob.glob(os.path.join(workdir,'*'))
            if function_eval_id in finite_evals_index:
                assert len(workdir_filenames)==2+min(1,verbosity)
                assert os.path.exists(
                    os.path.join(
                        workdir,model.results_filename+'.%d'%function_eval_id))
            else:
                assert len(workdir_filenames)==1+min(1,verbosity)
            assert os.path.exists(
                os.path.join(workdir,
                             model.params_filename+'.%d'%function_eval_id))

            if verbosity>0:
                assert os.path.exists(os.path.join(workdir,'stdout.txt'))

        cleanup_work_directories(workdir_basename)

    def test_async_model_backup(self):
        """
        Test that async model saves backup of model output after every call to
        __call__
        """
        import tempfile
        temp_directory = tempfile.TemporaryDirectory()
        temp_dirname = temp_directory.__dict__['name']
        
        workdir_basename='work-dir'
        save_workdirs = 'limited'
        saved_data_basename='backup-data'
        saved_data_basename = os.path.join(
            temp_dirname,saved_data_basename)
        
        cleanup_work_directories(workdir_basename)
        filenames = glob.glob(saved_data_basename+'*.npz')
        remove_files(filenames)

        # Set an alarm to raise an exception after predefined interval
        # def timeout_handler(num, stack):
        #     #print 'function exceeded time limit'
        #     raise Exception('function exceeded time limit')

        # duration=2
        # import signal
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(duration)

        
        num_samples = 3*max_eval_concurrency;
        model, target_function, num_vars = get_file_io_model(
            0.02,fault_percentage=10)
        shell_command = model.shell_command
        model = AsynchronousEvaluationModel(
            shell_command, max_eval_concurrency=max_eval_concurrency,
            workdir_basename=workdir_basename,
            save_workdirs=save_workdirs,
            saved_data_basename=saved_data_basename)

        num_iters=0
        while num_iters<3:
            samples = np.random.uniform(-1.,1.,(num_vars,num_samples))
            finite_evals_index = check_model_values(
                model, target_function, num_vars, num_samples,
                ignore_nans=True)
            num_iters+=1

        backup_files = glob.glob(saved_data_basename+'*')
        assert len(backup_files)==num_iters
        remove_files(backup_files)

if __name__== "__main__":    
    async_model_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestAsyncModel)
    unittest.TextTestRunner(verbosity=2).run(async_model_test_suite)

