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
import pkg_resources
from sphinx_gallery.sorting import _SortKey  # , ExampleTitleSortKey
import warnings


sys.path.append(os.path.abspath('../../'))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = 'PyApprox'
copyright = '2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.'
author = 'John D. Jakeman'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


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

extensions += ['matplotlib.sphinxext.plot_directive',
               'IPython.sphinxext.ipython_console_highlighting',
               'IPython.sphinxext.ipython_directive']

extensions += ['sphinx_automodapi.automodapi']

extensions += ['sphinx_gallery.gen_gallery']

example_filenames_in_order = [
    'plot_monte_carlo.py',
    'plot_push_forward_based_inference.py',
    'plot_tensor_product_interpolation.py',
    'plot_sensitivity_analysis.py',
    'plot_bayesian_networks.py',
    'plot_adaptive_leja_interpolation.py',
    'plot_gaussian_processes.py',
    'plot_integrated_surrogates.py',
    'plot_control_variate_monte_carlo.py',
    'plot_approximate_control_variate_monte_carlo.py',
    'plot_multi_level_monte_carlo.py',
    'plot_multi_fidelity_monte_carlo.py',
    'plot_many_model_approximate_control_variate_monte_carlo.py',
    'plot_gaussian_mfnets.py'
]

example_dirs = ['../../tutorials', '../../examples']
gallery_dirs = ['auto_tutorials', 'auto_examples']


installed_pkgs = {pkg.key for pkg in pkg_resources.working_set}
if 'pyapprox-dev' in installed_pkgs:
    # add documentation for modules in pyapprox_dev
    example_dirs += ['../../pyapprox_dev/tutorials']
    gallery_dirs += ['auto_dev_tutorials']
    example_filenames_in_order.insert(1, 'plot_advection_diffusion_model.py')
    example_filenames_in_order.insert(3, 'plot_bayesian_inference.py')
    example_filenames_in_order.insert(14, 'plot_multi_index_collocation.py')

example_filenames_in_order += [
    'plot_variables.py', 'plot_setup_model.py', 'plot_parameter_sweeps.py',
    'plot_low_discrepancy_quadrature.py', 'plot_gaussian_quadrature.py',
    'plot_barycentric_interpolation.py',
    'plot_multivariate_piecewise_polynomial_interpolation.py',
    'plot_sparse_grid_uq.py']

# print(installed_pkgs)
# print(example_filenames_in_order)
# print(example_dirs)


class ExamplesExplicitOrder(_SortKey):

    def __call__(self, filename):
        return example_filenames_in_order.index(filename)


# Note sphink-gallery only runs examples in files that start with plot_
# To add subfolders in examples must add README.rst to that subfolder in
# addition to .py files
# gallery_dirs and example_dirs must be the same length
sphinx_gallery_conf = {
    # path to your example scripts
    'examples_dirs': example_dirs,
    # path to where to save gallery generated output
    'gallery_dirs': gallery_dirs,
    # 'first_notebook_cell' : "%matplotlib inline",
    'within_subsection_order': ExamplesExplicitOrder,
    'ignore_pattern': r'plot_design_under_uncertainty\.py',
}
# If want to specify user latex macrors to jupyter using sphinx-gallery go to
# /miniconda3/envs/pyapprox/lib/python3.6/site-packages/sphinx_gallery/notebook.py
# in function jupyter_notebook replace
#     add_code_cell(work_notebook, first_cell)
# with
#     add_markdown_cell(work_notebook, first_cell)#jdj
#     add_code_cell(work_notebook,"%matplotlib inline")#jdj
# then add user defs like so
sphinx_gallery_conf['first_notebook_cell'] = r"Add latex macros$$\newcommand{\V}[1]{{\boldsymbol{#1}}}\newcommand{mean}[1]{{\mathbb{E}\left[#1\right]}}\newcommand{var}[1]{{\mathbb{V}\left[#1\right]}}\newcommand{covar}[2]{\mathbb{C}\text{ov}\left[#1,#2\right]}\newcommand{corr}[2]{\mathbb{C}\text{or}\left[#1,#2\right]}\newcommand{argmin}{\mathrm{argmin}}\def\rv{z}\def\reals{\mathbb{R}}\def\pdf{\rho}\def\rvdom{\Gamma}\def\coloneqq{\colon=}\newcommand{norm}{\lVert #1 \rVert}\def\argmax{\operatorname{argmax}}\def\ai{\alpha}\def\bi{\beta}\newcommand{\dx}[1]{\;\mathrm{d}#1}$$"

# if change conf make sure to remove source/auto_examples, using make clean
# Note sphinx can use align with single line, e.g. a=1 & & b=1 if \\ is added to the end of the line, i.e  a=1 & & b=1\\

# silence warning created by sphinx-gallery
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

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
exclude_patterns = ['control_variate_monte_carlo.rst','adaptive_leja_sequences.rst','examples.rst','cantilever_beam.rst','parameter_sweeps.rst','tensor_product_lagrange_interpolation.rst','polynomial_chaos_interpolation.rst','isotropic_sparse_grid_quadrature_example.rst','plot_design_under_uncertainty.rst'] # temporarily do not create function documentation

#exclude_patterns += ['modules.rst']

# use above to temporarily disable automod build. Also need to remove source/api directory and (possibly) build/


# only add documented functions to manual. If not used then the api of functions
# without a docstring will be added with no information.
autodoc_default_flags = ['members']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinxdoc'
# html_theme = 'scipy-sphinx-theme'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
#    'navigation_depth': 5 # if this is set it seems to overide maxdepth set in .rst files such as index.rst
}
html_logo = './figures/pyapprox-logo.png'

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
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\V}[1]{{\boldsymbol{#1}}}
\newcommand{\mean}[1]{\mathbb{E}\left[#1\right]}
\newcommand{\var}[1]{\mathbb{V}\left[#1\right]}
\newcommand{\covar}[2]{\mathbb{C}\text{ov}\left[#1,#2\right]}
\newcommand{\corr}[2]{\mathbb{C}\text{or}\left[#1,#2\right]}
%\def\argmin{\mathrm{argmin}}
\def\rv{z}
\def\reals{\mathbb{R}}
\def\pdf{\rho}
\def\rvdom{\Gamma}
\def\coloneqq{\colon=}
\newcommand{\norm}[1]{\lVert #1 \rVert}
%\def\argmax{\operatorname{argmax}}
\def\ai{\alpha}
\def\bi{\beta}
\newcommand{\dx}[1]{\;\mathrm{d}#1}
''',
}

# rst_prolog = r"""
# .. only:: html

#     .. math::
#         \DeclareMathOperator{\argmin}{argmin}
# """

# used when building html version
# mathjax 2
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# mathjax2_config = {
#     "TeX": {
#         "Macros": {
#             "V": [r'{\boldsymbol{#1}}', 1],
#             "mean": [r'{\mathbb{E}\left[#1\right]}', 1],
#             "var": [r'{\mathbb{V}\left[#1\right]}', 1],
#             "rv": r'{z}',
#             "reals": r'\mathbb{R}',
#             "pdf": r'\rho',
#             "rvdom": r'\Gamma',
#             "coloneqq": r'\colon=',
#             "norm": [r'{\lVert #1 \rVert}', 1],
#             "argmax": [r'\operatorname{argmax}'],
#             "argmin": [r'\operatorname{argmin}'],
#             "covar": [r'\mathbb{C}\text{ov}\left[#1,#2\right]', 2],
#             "corr": [r'\mathbb{C}\text{or}\left[#1,#2\right]', 2],
#             "ai": r'\alpha',
#             "bi": r'\beta',
#             "dx": [r'\;\mathrm{d}#1', 1],
#             },
#         'extensions': ["AMSmath.js", "AMSsymbols.js"]
#         }
#     }
# mathjax 3
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
  "tex": {
    "macros": {
      "V": [r'{\boldsymbol{#1}}', 1],
      "mean": [r'{\mathbb{E}\left[#1\right]}', 1],
      "var": [r'{\mathbb{V}\left[#1\right]}', 1],
      "rv": r'{z}',
      "reals": r'\mathbb{R}',
      "pdf": r'\rho',
      "rvdom": r'\Gamma',
      "coloneqq": r'\colon=',
      "norm": [r'{\lVert #1 \rVert}', 1],
      "argmax": [r'\operatorname{argmax}'],
      "argmin": [r'\operatorname{argmin}'],
      "covar": [r'\mathbb{C}\text{ov}\left[#1,#2\right]', 2],
      "corr": [r'\mathbb{C}\text{or}\left[#1,#2\right]', 2],
      "ai": r'\alpha',
      "bi": r'\beta',
      "dx": [r'\;\mathrm{d}#1', 1]
    },
  },
}

# Supress all warnings so they do not appear in the documentation
warnings.filterwarnings("ignore")


#-----
# only document locally defined functios and classes and not imports
def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm
    from sphinx_automodapi.utils import find_mod_objs
    automodsumm.find_mod_objs = lambda *args: find_mod_objs(
        args[0], onlylocals=True)


def setup(app):
    app.connect("builder-inited", patch_automodapi)
#------
