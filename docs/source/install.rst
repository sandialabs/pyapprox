############
Installation
############

Download
--------
Clone the latest version of the software using::
  
  git clone https://github.com/sandialabs/pyapprox.git

or download the source from the `PyApprox github repository <https://github.com/sandialabs/pyapprox>`_.

Install
-------
To get all the features of PyApprox it is best to install PyApprox dependencies with conda. However if this is not possible or desirable PyApprox can also be installed entirely with pip. If pip is used the user will not have access to the benchmarks based on the finite element modeling package fenics.

Conda
^^^^^
The most robust way to install PyApprox is to create a new environment. Pyapprox can be installed in an existing environment, however the user will need to take care to ensure that any upgrades or downgrades suggested by conda will not cause issues with the current environment.

To create a new environment use::

    conda env create -f environment.yml

This will create an environment called pyapprox-dev. Activate this environment with::

    conda activate pyapprox-dev

Once the new environment, or an existing environment, has been activated run the following in the PyApprox root directory to install PyApprox::

    pip install -e .

The -e argument specifies to install softlinks so that any changes made by the user to the source in the source folders are reflected in the install when importing modules.

Pip
^^^
To install PyApprox entirely with pip simply run the following in the PyApprox root directory to install PyApprox::

    pip install -e .


Note that sometimes developers need to remove all files generated when installing. To do this use::

  python setup.py clean --all
  find . -name "*.pyc" -exec rm -f {} \;


Windows
^^^^^^^

For the visualization methods to work and to support general latex display, a copy of `MikTex <https://miktex.org/download>`_ should be installed.

Windows currently has an issue which breaks the NumPy runtime (`see here for details <https://tinyurl.com/y3dm3h86>`_).
The following instructions provide a workaround.

First, set up an empty `conda` environment

  conda create -n pyapprox-dev python=3.7
  conda activate pyapprox-dev

Pre-install required packages with pip:

  pip install -r requirements.txt

Then perform a local install via `setup.py`

  python setup.py develop

Additional optional packages can be installed via pip...

  pip install -r doc-requirements.txt
  pip install -r optional-requirements.txt

...or by specifying the specific group of optional packages

  pip install pyapprox[docs]
  pip install pyapprox[ode]
  

Test
----
To run all tests use::
  
  python setup.py test

To test docstring examples use::
  
  pytest --doctest-modules
  
or in pyapprox/docs directory run::

  make doctest

To test all docstrings in a file use::
  
  python -m doctest -v file.py

Some tests will be skipped if not all optional packages are installed.


Create Documentation
--------------------
To create the documentation fenics must be installed. Creating documentation also requires Sphinx 1.7 or later, sphinx, numpydoc, and sphinx-automodapi. These are automatically installed with pip install -e . command.

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
  numpy include path is not working with cython on windows. Need to figure out
  how to set it.
