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
    python_requires='>=3.6,<=3.8.99',  # numba only compiles with <=3.8
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
        'seaborn',
        'numba',
        'scikit-learn',
        'pytest-cov',
        'pytest',
        'networkx',
    ],
    extras_require={
        'docs': ['numpydoc', 'sphinx', 'sphinx_automodapi', 'sphinx_rtd_theme',
                 'sphinx-gallery', 'jupyter']
        # 'mfnets':['MFNetsSurrogates @ git+https://github.com/goroda/MFNetsSurrogates@master']
    },
    ext_modules=extensions,
    test_suite='nose.collector',
    tests_require=['nose'],
    # for some reason pytest will not work on github actions.
    # It is discovering a test which causes an error which I cannot reproduce
    # on my machine or find a way which test is causing the issue
    # tests_require=['pytest-cov'],
    license='MIT',
)
