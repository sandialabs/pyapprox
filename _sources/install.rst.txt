############
Installation
############

Installation
------------
To install PyApprox run::

    python setup.py install

To instead install a version for developlement which allows for package source
to be edited and changes activated without needed to reinstall use::
  
    python setup.py develop

Testing
-------
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

