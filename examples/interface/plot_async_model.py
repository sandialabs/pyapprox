r"""
Calling Non-Python Based Models
-------------------------------
This tutorial demonstrats how to call a external model using a file interface.

The :class:`~pyapprox.interface.model.AyncIOModel` can be used to run models that may not be written in Python, but can be evaluated from the command line via a shell script. The :class:`~pyapprox.interface.model.AynchIOModel` creates a file named params.in (the name can be changed by the user) and assumes that the shell script reads in that file and returns the output of the model in a file called results.out (this name can also be changed). Each evaluation of the model is performed in a separate work directory to ensure that no results are overwritten, which is especially important when running the model in parallel. If a list of filenames needed to run the bash script is provided a soft link to each file is created in each work directory. This is extremely useful when running large finite element simulations that may have input files, such as mesh and topography data, with large memory footprints.

"""

import os
import tempfile
import numpy as np
from scipy import stats

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.model import AsyncIOModel
from pyapprox.interface.tests.test_model import get_shell_command_for_io_model


# %% The following creates a model with two inputs and two quantities of interest and evaluates it at three samples. Temporary work directories are created to run the model at each sample. The directories are automatically deleted, however the user can choose to keep each directory.
nvars = 2
shell_command = get_shell_command_for_io_model(0.02)[0]
intmpdir = tempfile.TemporaryDirectory()
infilenames = [os.path.join(intmpdir.name, "vec.npz")]
vec = np.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
np.savez(infilenames[0], vec=vec)
async_model = AsyncIOModel(
    2,
    nvars,
    infilenames,
    shell_command,
    outdir_basename=None,
    verbosity=0,
    save="no",
    datafilename=None,
    nprocs=1,
)
asynch_variable = IndependentMarginalsVariable([stats.uniform(0, 1)] * nvars)
samples = asynch_variable.rvs(3)
values = async_model(samples)
print(values)

# %%
# This interface can also be used to run multiple samples in parallel by setting nprocs > 1
