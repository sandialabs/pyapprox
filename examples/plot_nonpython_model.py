r"""
Interfacing With Non-Python Models
----------------------------------
This tutorial demonstrats how to evaluate a model at multiple samples in parallel.

The :class:`~pyapprox.interface.model.AsyncIOModel` can be used to run models that may not be written in Python, but can be evaluated from the command line via a shell script. Theclass creates a file named params.in (the name can be changed by the user) and assumes that the shell script reads in that file and returns the output of the model in a file called results.out (this name can also be changed). Each evaluation of the model is performed in a separate work directory to ensure that no results are overwritten, which is especially important when running the model in parallel. If a list of filenames needed to run the bash script is provided a soft link to each file is created in each work directory. This is extremely useful when running large finite element simulations that may have input files, such as mesh and topography data, with large memory footprints.
#
# The following creates a model with two inputs and two quantities of interest and evaluates it at three samples. Temporary work directories are created to run the model at each sample. The directories are automatically deleted, however the user can choose to keep each directory.
"""

import os
import tempfile

from scipy import stats
import numpy as np

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface import AsyncIOModel
from pyapprox.interface.tests.test_model import get_shell_command_for_io_model

# specify the shell command that will run the model for each sample.
# Here we just call an algebraic function, but this could be an MPI
# call to a computationally expensive finite element model, as one example.
shell_command = get_shell_command_for_io_model(0.02)[0]

# the model called by the shell command requires a file vec.npz.
# Store this in a temporary directory, then it will be soft-linked to
# each directory
nvars = 2
intmpdir = tempfile.TemporaryDirectory()
infilenames = [os.path.join(intmpdir.name, "vec.npz")]
vec = np.asarray(np.random.uniform(0.0, 1.0, (1, nvars)))
np.savez(infilenames[0], vec=vec)

# setup the model
tmp_dir = tempfile.TemporaryDirectory()
model = AsyncIOModel(
    2,
    nvars,
    infilenames,
    shell_command,
    outdir_basename=tmp_dir.name,
    verbosity=0,
    save="no",
    nprocs=2,
)

# run the model
variable = IndependentMarginalsVariable([stats.uniform(0, 1)] * nvars)
samples = variable.rvs(3)
values = model(samples)
print(values)

# clean up the temporary directory
intmpdir.cleanup()
