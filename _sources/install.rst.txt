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
To install PyApprox run the following in the PyApprox root directory::

    pip install -e .


To instead install a version for developlement which allows for package source
to be edited and changes activated without needed to reinstall use::

    pip install -r requirements.txt
    python setup.py develop

Note that sometimes developers need to remove all files generated when calling setup.py. To do this use::

  python setup.py clean --all
  find . -name "*.pyc" -exec rm -f {} \;

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


Create Documentation
--------------------
Creating documentation requires Sphinx 1.7 or later, sphinx, numpydoc, and sphinx-automodapi. Html documentation can be generated with::

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
