# PyApprox Tutorials

[![PyPI](https://img.shields.io/pypi/v/pyapprox-tutorials.svg)](https://pypi.org/project/pyapprox-tutorials/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sandialabs/pyapprox/blob/main/LICENSE)

Tutorials and tutorial figure helpers for the
[PyApprox](https://pypi.org/project/pyapprox/) library.

The tutorial sources are Quarto documents; this package ships the Python
helpers they import, so the published tutorials stay readable while the
plotting code lives in a real, importable, tested module.

**[Read the tutorials](https://sandialabs.github.io/pyapprox/)** |
**[Repository](https://github.com/sandialabs/pyapprox)**

## What is in this package

- `pyapprox_tutorials.figures` — figure generators used by the tutorial
  library. Each takes already-constructed PyApprox objects and returns a
  matplotlib figure, keeping plotting boilerplate out of the tutorial text.

Most readers do not need to install this package. Install it if you want to
build the tutorial site locally, or to reuse a tutorial figure in your own
work.

## Installation

```bash
pip install pyapprox-tutorials
```

Installing this package pulls in `pyapprox` and `pyapprox-benchmarks`.

## Building the tutorial site

The site is built with [Quarto](https://quarto.org/). From a clone of the
repository:

```bash
cd packages/pyapprox-tutorials/tutorials
./build.sh -j auto            # parallel build
./build.sh --notebooks        # also generate downloadable .ipynb files
./build.sh --serve            # serve locally after building
```

Output is written to
`packages/pyapprox-tutorials/tutorials/library/_site/`.

## Related packages

Developed in a monorepo alongside
[`pyapprox`](https://pypi.org/project/pyapprox/) (the core library) and
[`pyapprox-benchmarks`](https://pypi.org/project/pyapprox-benchmarks/)
(benchmark problems). See the
[repository](https://github.com/sandialabs/pyapprox) for development setup.

## License

MIT. See [LICENSE](https://github.com/sandialabs/pyapprox/blob/main/LICENSE).
