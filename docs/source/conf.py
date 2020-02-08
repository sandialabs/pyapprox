# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath('../../'))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = 'PyApprox'
copyright = '2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.'
author = 'John D. Jakeman'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram']

# wrong siyntax for new versions
# extensions += [
#     'matplotlib.sphinxext.only_directives',
#     'matplotlib.sphinxext.plot_directive']
#     'matplotlib.sphinxext.ipython_directive',
#     'matplotlib.sphinxext.ipython_console_highlighting']
    
extensions +=['matplotlib.sphinxext.plot_directive',
              'IPython.sphinxext.ipython_console_highlighting',
              'IPython.sphinxext.ipython_directive']

extensions += ['sphinx_automodapi.automodapi']

extensions += ['sphinx_gallery.gen_gallery']

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

# numpydoc_show_class_members=False option is ncessary when using automodapi
# to avoid having methods and attributes of classes being shown multiple times.
numpydoc_show_class_members = False

# allow easy reference of headers in other .rst files
extensions += ['sphinx.ext.autosectionlabel']
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['control_variate_monte_carlo.rst','adaptive_leja_sequences.rst','examples.rst','cantilever_beam.rst','parameter_sweeps.rst','tensor_product_lagrange_interpolation.rst','polynomial_chaos_interpolation.rst','isotropic_sparse_grid_quadrature_example.rst'] # temporarily do not create function documentation

exclude_patterns += ['modules.rst']
# use above to temporarily disable automod build. Also need to remove source/api directory and (possibly) build/


# only add documented functions to manual. If not used then the api of functions
# without a docstring will be added with no information.
autodoc_default_flags = ['members']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'sphinxdoc'
#html_theme = 'scipy-sphinx-theme'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'navigation_depth': 2
}
html_logo='./figures/pyapprox-logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# used when building latex and pdf versions
latex_elements = {
     'preamble': r'''
\newcommand\sphinxbackoftitlepage{%
     Copyright 2019 National Technology \& Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.\par
}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\DeclareMathOperator*{\argmin}{argmin}
\newcommand{\V}[1]{{\boldsymbol{V}}}
''',
}

# used when building html version
mathjax_config = {                  
    "TeX": {                        
        "Macros": {                 
            "V": [r'{\boldsymbol{#1}}',1],
            "mean": [r'{\mathbb{E}\left[#1\right]}',1],
            "var": [r'{\mathbb{V}\left[#1\right]}',1],
            "argmin": r'{\mathrm{argmin}}',
            "rv":r'z',
            "reals":r'\mathbb{R}',
            "pdf":r'\rho',
            "rvdom":r'\Gamma',
            "coloneqq":r'\colon=',
            "norm":[r'\lVert #1 \rVert',1],
            "argmax":[r'\operatorname{argmax}']
            }                       
        }                           
    } 
