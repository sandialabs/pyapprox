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

    python setup.py install

To instead install a version for developlement which allows for package source
to be edited and changes activated without needed to reinstall use::
  
    python setup.py develop

Note that sometimes developers need to remove all files generated when calling setup.py. To do this use::

  python setup.py clean --all
  find . -name "*.pyc" -exec rm -f {} \;

Test
----
To run all tests use::
  
  python setup.py test

Create Documentation
--------------------
Creating documentation requires Sphinx 1.7 or later, sphinx, numpydoc, and sphinx-automodapi. Html documentation can be generated with::

    cd docs
    make html

A PDF of the documentation can be generated with::

    cd docs
    make latexpdf

