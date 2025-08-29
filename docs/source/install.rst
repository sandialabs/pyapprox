############
Installation
############
Install using pip
-----------------
The latest release of PyApprox can be installed using pip::
  
  pip install pyapprox

Wheels are available for Python versions >= 3.7 on all platforms except macosx M1 architecture which only supports Python versions >= 3.9.


Install from source
-------------------
PyApprox uploads wheels to PyPi for most operating systems and versions of Python. However, if the pip install fails, or you would just like to use the most recent version of PyApprox the following instructions can be used to install PyApprox from source.

Download
^^^^^^^^
Clone the latest version of the software using::
  
  git clone https://github.com/sandialabs/pyapprox.git

or download the source from the `PyApprox github repository <https://github.com/sandialabs/pyapprox>`_.


Conda
^^^^^
Create a new environment or use an existing one. Pyapprox can be installed in an existing environment, however the user will need to take care to ensure that any upgrades or downgrades suggested by conda will not cause issues with the current environment.

To create a new environment use::

    conda env create -f environment.yml

This will create an environment called pyapprox-base. Activate this environment with::

    conda activate pyapprox-base

Once the new environment, or an existing environment, has been activated run the following in the PyApprox root directory to install PyApprox::

    pip install -e .

The -e argument specifies to install softlinks so that any changes made by the user to the source in the source folders are reflected in the install when importing modules.

To use conda to install pyapprox  will all extras create the environment with::

   conda env create -f ~/pyapprox/environment_w_extras.yml

then run::
   pip install -e.[all]


Conda+Mamba
^^^^^^^^^^^
If installing this package with conda is slow due to limitations of Conda (not PyApprox). The speed of install can be improved using Mamba

Before creating an enviornment install Mamba with::

    conda install -c conda-forge mamba

To create a new environment use::

    mamba env create -f environment.yml

This will create an environment called pyapprox-base. Activate this environment with::

    conda activate pyapprox-base

Once the new environment, or an existing environment, has been activated run the following in the PyApprox root directory to install PyApprox::

    pip install -e .

The -e argument specifies to install softlinks so that any changes made by the user to the source in the source folders are reflected in the install when importing modules.

Pip
^^^
To install PyApprox entirely with pip simply run the following in the PyApprox root directory to install PyApprox::

    pip install -e .

Troubleshooting
^^^^^^^^^^^^^^^
Building PyApprox with a mixture of conda and pip can introduce errors. We recomment using one or the other. For example installing pytorch with conda then installing torchvision and torchaudio with pip can cause an error like::

    Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

To use conda to install pyapprox  will all extras create the environment with::

   conda env create -f ~/pyapprox/environment_w_extras.yml

then run::
   pip install -e.[all]
    

Sometimes pip will cause incompatabilities with your currently installed packages and will fail. If so try to reinstall with::

    pip install -e . --no-build-isolation

The above has solved errors such as
"ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject"

A pure pip-based installation on OSX M1 is not recommended. Installing scipy with pip is very difficult on this platform and requires expert intervention. Instead install scipy with Conda and everything should work well.

Sometimes developers need to remove all files generated when installing. To do this use::

  python setup.py clean --all
  find . -name "*.pyc" -exec rm -f {} \;

Test
----
To run all tests run the following in the root directory of PyApprox::
  
  pytest . --disable-warnings

To test docstring examples use::
  
  pytest --doctest-modules
  
or in pyapprox/docs directory run::

  make doctest

To test all docstrings in a file use::
  
  python -m doctest -v file.py

Some tests will be skipped if not all optional packages are installed.


Create Documentation
--------------------
To create the documentation fenics must be installed. Creating documentation also requires Sphinx 1.7 or later, sphinx, numpydoc, and sphinx-automodapi. A local install of latex is also required.


To install these requirements when installing from PyPi first run::

    pip install pyapprox[docs]


When installing from source run::

    pip install -e .[docs]

Html documentation can be generated with::

    cd docs
    make html

A PDF of the documentation can be generated with::

    cd docs
    make latexpdf

Note that sometimes the documentation of functions using numpydoc can render incorrectly when usd with sphinx_rtd_theme (see this `thread <https://github.com/numpy/numpydoc/issues/215>`_). As a workaround find the file::
  
  <path-to-site-packages>/sphinx_rtd_theme/static/css/theme.css

add the following at the end of the file if not already present::

  .classifier:before {
      font-style: normal;
      margin: 0.5em;
      content: ":";
  }

..
  On windows may need to install visual studio. See https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=vs-2019
  On windows not sure how to set proxy for pip so use
  pip install --proxy https://proxy.address <package>


GPU Support
-----------
Pyapprox can be used with both NumPy and Torch. To enable GPU support with Torch
in the base package directory run::
  
    pip install -e .[gpu]

or if installing from PyPi first ::

    pip install pyapprox[gpu]

If using Linux or windows add the following code::
  
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

If using OSX in your script add the following code::

    torch.set_default_device("mps")
    torch.set_default_dtype(torch.float)

PyApprox use torch.double as the default dtype, howver OSX does not support double when using GPU so the first second above overides this default. Warning:
lower precision will and affect the stability and accuracy of PyApprox functions.
  
Enable Developer Environment
----------------------------
Python uses black 8 with max-line length of 79. To enable the linting employed by Pyapprox run::

    pip install -e .[lint]

Note: All dependencies can be installed using::

    pip install -e .[all]
