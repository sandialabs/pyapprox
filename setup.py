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
    "pyapprox/cython/*.pyx",
    compiler_directives={'language_level': 3},
    annotate=True)

setuptools.setup(
    name="pyapprox",
    version="1.0.0",
    author="John D. Jakeman",
    author_email="29109026+jdjakem@users.noreply.github.com",
    description="High-dimensional function approximation and estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/pyapprox",
    packages=setuptools.find_packages(),
    python_requires='>=3.6', 
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
# conda env create -f environment.yml python=3.6 --name pyapprox-base-3-6

# to run all tests use
# python -m unittest discover pyapprox

# run a doctest of a single module
# pytest -v --doctest-modules path/to/module.py

# run a doctest of a single function in a module
# pytest -v --doctest-modules path/to/module.py::path.to.module.Class.method

# The loading times of modules that depend on torch can
# increase if torch install from pip
