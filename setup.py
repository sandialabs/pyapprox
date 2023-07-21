import setuptools
from Cython.Build import cythonize
import sys
import os
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


USE_CYTHON = True
print('CYTHONIZING')
extensions = cythonize(
    ["pyapprox/cython/*.pyx"],
    compiler_directives={'language_level': 3},
    annotate=True)

setuptools.setup(
    name="pyapprox",
    version="1.0.2",
    author="John D. Jakeman",
    author_email="29109026+jdjakem@users.noreply.github.com",
    description="High-dimensional function approximation and estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/pyapprox",
    packages=setuptools.find_packages(),
    python_requires='>=3.7', 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[np.get_include()],
    setup_requires=['numpy >= 1.16.4', 'Cython', 'scipy >= 1.0.0'],
    install_requires=[
        'numpy >= 1.16.4',
        'matplotlib',
        'scipy >= 1.0.0',
        'Cython',
        'sympy',
        'torch',
        'scikit-learn',
        'coverage>=6.4',
        'pytest-cov',
        'pytest>=4.6',
        'networkx',
        # 'tqdm',
        'numba',
        'scikit-fem',
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx', 'sphinx_automodapi', 'sphinx_rtd_theme',
                 'sphinx-gallery', 'jupyter']
    },
    ext_modules=extensions,
    license='MIT',
)

#TODO see https://pytest-cov.readthedocs.io/en/latest/config.html
# to add config file for coverage tests to exclude certain files from coverage tests

# to install using specific version of python use, e.g.
# conda env create -f environment.yml python=3.9 --name pyapprox-base-3-9

# to run all tests use
# python -m unittest discover pyapprox

# run a doctest of a single module
# pytest -v --doctest-modules path/to/module.py

# run a doctest of a single function in a module
# pytest -v --doctest-modules path/to/module.py::path.to.module.Class.method

# The loading times of modules that depend on torch can
# increase if torch install from pip

# to install packages needed to compile docs run
# pip install -e .[docs]

# To test pyapprox on test.pypi use the following.
# Note the use of extra-index-url
# python -m pip install --extra-index-url=https://test.pypi.org/simple/ pyapprox==1.0.2

# To build wheel locally use
# python -m build

#to create from empty environment use from directory contatining setup
#pip install twine wheel setuptools numpy scipy cython && python setup.py sdist --format=gztar bdist_wheel

# To test wheel locally create virtual environment and install using
# pip install mypackage-0.31.0-py2.py3-none-any.whl
# with docs use (quotes are important)
# pip install 'mypackage-0.31.0-py2.py3-none-any.whl[docs]'


# When uploading to test.pypi.org if the last wheel is broken, e.g.
# mypackage-0.31.0-py2.py3-none-any.whl
# then add a build number, e.g. -1
# mypackage-0.31.0-1-py2.py3-none-any.whl
# or -2
# mypackage-0.31.0-2-py2.py3-none-any.whl
# and upload
# python -m twine upload --repository testpypi dist/*

# to catch warnings as errors from command, e.g. to run a unittest use
# python -W error -m unittest pyapprox.surrogates.tests.test_approximate.TestApproximate.test_approximate_fixed_pce

# to run a single test with pytest use
# pytest pyapprox/surrogates/tests/test_approximate.py -k test_cross_validate_pce_degree

# To ignore certain warnings use 
# pytest pyapprox/surrogates/tests/test_approximate.py -k test_cross_validate_pce_degree -W ignore:FutureWarning

# However to isolate 3rd party warnings edit pypest.ini, e.g.
# [pytest]
# filterwarnings = ignore::FutureWarning:sklearn.*:
# "Individual warnings filters are specified as a sequence of fields separated by colons:"
# action:message:category:module:line

# # the following can be used to append location of print statement so
# # that errant print statements can be found and removed
# import builtins
# from inspect import getframeinfo, stack
# original_print = print

# def print_wrap(*args, **kwargs):
#     caller = getframeinfo(stack()[1][0])
#     original_print("FN:",caller.filename,"Line:", caller.lineno,"Func:", caller.function,":::", *args, **kwargs)

# builtins.print = print_wrap
