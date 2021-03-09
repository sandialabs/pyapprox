import sys
import setuptools
import os
import numpy as np


setuptools.setup(
    name="pyapprox_dev",
    version="1.0.0",
    author="John D. Jakeman",
    author_email="29109026+jdjakem@users.noreply.github.com",
    description="Additional optional functionality for PyApprox",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/pyapprox",
    packages=setuptools.find_packages(),
    python_requires='>=3.6,<=3.8', # numba only compiles with <=3.8
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[np.get_include()],
    setup_requires=['numpy >= 1.16.4','Cython','scipy >= 1.0.0'],
    install_requires=['pyapprox', 'pymc3'],
    extras_require={'docs':
                    ['numpydoc','sphinx','sphinx_automodapi','sphinx_rtd_theme',
                     'sphinx-gallery','jupyter']
      },
    test_suite='nose.collector',
    tests_require=['nose'],
    license='MIT',
)
