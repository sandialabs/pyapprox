import sys
import setuptools
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

#try:
from Cython.Build import cythonize
USE_CYTHON=True
print('CYTHONIZING')
#except:
#    USE_CYTHON=False
#    print('USING PRECYTHONIZED FILES')

#if USE_CYTHON:
extensions = cythonize(
    "pyapprox/cython/*.pyx",
    compiler_directives={'language_level' : 3},
    annotate=True)
# else:
#     import glob
#     from setuptools import Extension
#     pyx_files = glob.glob("pyapprox/cython/*.pyx")
#     extensions = []
#     for pyx_file in pyx_files:
#         name= pyx_file[:-4].replace('/', '.')
#         ext = Extension(name=name,sources=[pyx_file],include_dirs=[np.get_include()])
#         extensions.append(ext)
#     extensions = no_cythonize(extensions)

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
    setup_requires=['numpy >= 1.16.1','Cython','scipy >= 1.0.0'],
    install_requires=[
        'numpy >= 1.16.1',
        'matplotlib',
        'scipy >= 1.0.0',
        'Cython',
        'sympy',
        'seaborn',
        'pymc3',
        'scikit-learn',
        'pytest-cov',
        'pytest',
        'networkx'
        ],
    extras_require={'docs':['numpydoc','sphinx','sphinx_automodapi','sphinx_rtd_theme',
                            'sphinx-gallery','jupyter']
      },
    ext_modules = extensions,
    test_suite='nose.collector',
    tests_require=['nose'],
    #for some reason pytest will not work on github actions. It is discovering a test which
    #causes an error which I cannot reproduce on my machine or find a way which test
    #is causing the issue
    #tests_require=['pytest-cov'],
    license='MIT',
)
