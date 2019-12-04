import sys
import setuptools
from Cython.Build import cythonize
import numpy as np
import os
os.environ["C_INCLUDE_PATH"] = np.get_include()
print (np.get_include())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyapprox",
    version="1.0",
    author="John D. Jakeman",
    author_email="29109026+jdjakem@users.noreply.github.com",
    description="High-dimensional function approximation and estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/pyapprox",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy >= 1.14',
        'matplotlib',
        'scipy >= 1.0.0',
        'cython',
        'cvxopt',
        'numpydoc',
        'sphinx',
        'sphinx_automodapi',
        'sphinx_rtd_theme'
      ],
    ext_modules = cythonize(
        "pyapprox/cython/*.pyx",
        compiler_directives={'language_level' : 3},
        annotate=True),
    test_suite='nose.collector',
    tests_require=['nose'],
    license='MIT',
)

# mshr needed for test_helmholtz consider removing as required dependency
# mshr can only be installed using conda
# conda install -c conda-forge mshr

#conda create -n fenics2017 -c conda-forge fenics=2017 scipy mpi4py matplotlib python=3.5 sympy=1.1.1 cython cvxopt
