#!/usr/bin/env python3
"""Inject a pip install cell into generated Jupyter notebooks.

Notebooks downloaded from the tutorial site need an install cell so
students can run them in Google Colab or other environments where
pyapprox is not pre-installed.  The source .qmd files intentionally
omit install commands because pyapprox is already available in the
build environment.
"""

import json
import sys
from pathlib import Path

INSTALL_SOURCE = [
    "# Run this cell if using Google Colab or if pyapprox is not installed locally\n",
    "# Installs the latest version from GitHub main branch.\n",
    "try:\n",
    "    import pyapprox\n",
    "    import pyapprox_benchmarks\n",
    "    import pyapprox_tutorials\n",
    "except ImportError:\n",
    "    !pip install -q \\\n",
    '        "pyapprox[runtime-extras] @ git+https://github.com/sandialabs/pyapprox.git#subdirectory=packages/pyapprox" \\\n',
    '        "pyapprox-benchmarks @ git+https://github.com/sandialabs/pyapprox.git#subdirectory=packages/pyapprox-benchmarks" \\\n',
    '        "pyapprox-tutorials @ git+https://github.com/sandialabs/pyapprox.git#subdirectory=packages/pyapprox-tutorials"\n',
]


def has_install_cell(notebook: dict) -> bool:
    """Check whether the notebook already has a pip install cell."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        if "pip install" in text and "pyapprox" in text:
            return True
    return False


def inject_install_cell(notebook_path: Path) -> bool:
    """Insert an install cell at position 0.

    Returns True if the notebook was modified, False if skipped.
    """
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    if has_install_cell(notebook):
        return False

    install_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": INSTALL_SOURCE,
    }

    notebook.setdefault("cells", []).insert(0, install_cell)

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: inject_notebook_install.py <notebooks-dir>")
        sys.exit(1)

    notebook_dir = Path(sys.argv[1])
    if not notebook_dir.exists():
        print(f"Directory not found: {notebook_dir}")
        sys.exit(1)

    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        sys.exit(1)

    injected = 0
    for nb_path in notebooks:
        if inject_install_cell(nb_path):
            print(f"  Injected install cell: {nb_path.name}")
            injected += 1
        else:
            print(f"  Already has install cell: {nb_path.name}")

    print(f"\nProcessed {len(notebooks)} notebooks, injected {injected}")


if __name__ == "__main__":
    main()
