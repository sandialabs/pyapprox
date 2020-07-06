import sys
import setuptools
import os
try:
    # This is necessary when installing package with pip install -e .
    # and numpy is not yet installed. pip will install numpy then install
    # pyapprox
    import numpy as np
    os.environ["C_INCLUDE_PATH"] = np.get_include()
except:
    pass

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

try:
    from Cython.Build import cythonize
    USE_CYTHON=True
    print('CYTHONIZING')
except:
    USE_CYTHON=False
    print('USING PRECYTHONIZED FILES')

if USE_CYTHON:
    extensions = cythonize(
        "pyapprox/cython/*.pyx",
        compiler_directives={'language_level' : 3},
        annotate=True)
else:
    import glob
    from setuptools import Extension
    pyx_files = glob.glob("pyapprox/cython/*.pyx")
    extensions = []
    for pyx_file in pyx_files:
        name= pyx_file[:-4].replace('/', '.')
        ext = Extension(name=name,sources=[pyx_file])
        extensions.append(ext)
    extensions = no_cythonize(extensions)

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
#        'cython',
        'cvxopt',
        'sympy',
        'seaborn',
        'pymc3',
        #'numpydoc',
        #'sphinx',
        #'sphinx_automodapi',
        #'sphinx_rtd_theme',
        #'sphinx-gallery',
        #'jupyter'
        #'torch',
      ],
    ext_modules = extensions,
    test_suite='nose.collector',
    tests_require=['nose'],
    license='MIT',
)

# mshr needed for test_helmholtz consider removing as required dependency
# mshr can only be installed using conda
# conda install -c conda-forge mshr

#conda create -n default -c conda-forge python=3.6 numpy scipy cython matplotlib  mpi4py fenics=2019 mshr pymc3 mkl-service

# mkl-service needed for pymc3

#replicate previous environ where pymc3 worked
#conda create -n redhat -c conda-forge python=3.7 numpy=1.15 scipy cython matplotlib=3.1  mpi4py fenics=2018 mshr pymc3=3.8 theano=1.0.4

# install tools needed to generate documentation
# conda install -c conda forge sphinx sphinx-gallery numpydoc sphinx-automodapi sphinx_rtd_theme

#install torch with
#conda install pytorch torchvision -c pytorch
