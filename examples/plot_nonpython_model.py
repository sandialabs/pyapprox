r"""
Interfacing With Non-Python Models
----------------------------------
This tutorial demonstrats how to evaluate a model at multiple samples in parallel.

The :class:`~pyapprox.interface.async_model.AynchModel` can be used to run models that may not be written in Python, but can be evaluated from the command line via a shell script. The :class:`~pyapprox.interface.async_model.AynchModel` creates a file named params.in (the name can be changed by the user) and assumes that the shell script reads in that file and returns the output of the model in a file called results.out (this name can also be changed). Each evaluation of the model is performed in a separate work directory to ensure that no results are overwritten, which is especially important when running the model in parallel. If a list of filenames needed to run the bash script is provided a soft link to each file is created in each work directory. This is extremely useful when running large finite element simulations that may have input files, such as mesh and topography data, with large memory footprints.
#
# The following creates a model with two inputs and two quantities of interest and evaluates it at three samples. Temporary work directories are created to run the model at each sample. The directories are automatically deleted, however the user can choose to keep each directory.
"""

import tempfile
from scipy import stats

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.async_model import AynchModel
from pyapprox.interface.tests.test_async_model import get_file_io_model

# The :class:`~pyapprox.interface.async_model.AynchModel` can be used to run models that may not be written in Python, but can be evaluated from the command line via a shell script. The :class:`~pyapprox.interface.async_model.AynchModel` creates a file named params.in (the name can be changed by the user) and assumes that the shell script reads in that file and returns the output of the model in a file called results.out (this name can also be changed). Each evaluation of the model is performed in a separate work directory to ensure that no results are overwritten, which is especially important when running the model in parallel. If a list of filenames needed to run the bash script is provided a soft link to each file is created in each work directory. This is extremely useful when running large finite element simulations that may have input files, such as mesh and topography data, with large memory footprints.
#
# The following creates a model with two inputs and two quantities of interest and evaluates it at three samples. Temporary work directories are created to run the model at each sample. The directories are automatically deleted, however the user can choose to keep each directory.
file_io_model = get_file_io_model(0.02)[0]
tmp_dir = tempfile.TemporaryDirectory()
asynch_model = AynchModel(
    file_io_model.shell_command,
    workdir_basename=tmp_dir.name,
    save_workdirs="no",
)
asynch_variable = IndependentMarginalsVariable([stats.uniform(0, 1)] * 2)
samples = asynch_variable.rvs(3)
values = asynch_model(samples)
print(values)
