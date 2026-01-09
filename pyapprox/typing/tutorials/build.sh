#!/bin/bash
# Build script for PyApprox UQ Tutorials
# Usage: ./build.sh [options]
#
# Options:
#   --no-execute    Skip code execution (use cached results)
#   --execute       Force re-execute all code
#   --notebooks     Generate downloadable notebooks
#   --serve         Start local server after build
#   --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
NO_EXECUTE=""
FORCE_EXECUTE=""
GEN_NOTEBOOKS=""
SERVE=""

for arg in "$@"; do
    case $arg in
        --no-execute)
            NO_EXECUTE="--no-execute"
            ;;
        --execute)
            FORCE_EXECUTE="--execute"
            ;;
        --notebooks)
            GEN_NOTEBOOKS="yes"
            ;;
        --serve)
            SERVE="yes"
            ;;
        --help)
            head -12 "$0" | tail -11
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "=== Building PyApprox UQ Tutorials ==="
echo "Working directory: $SCRIPT_DIR"
echo ""

# Clean old build
if [ -d "_site" ]; then
    echo "Cleaning old build..."
    rm -rf _site
fi

# Build site
echo "Rendering Quarto site..."
if [ -n "$NO_EXECUTE" ]; then
    echo "  (skipping code execution)"
    quarto render --no-execute
elif [ -n "$FORCE_EXECUTE" ]; then
    echo "  (forcing code execution)"
    quarto render --execute
else
    echo "  (using freeze cache)"
    quarto render
fi

# Generate notebooks
if [ -n "$GEN_NOTEBOOKS" ]; then
    echo ""
    echo "Generating Jupyter notebooks..."
    mkdir -p _site/library/notebooks

    for f in library/*.qmd; do
        if [ "$(basename "$f")" != "index.qmd" ]; then
            name=$(basename "${f%.qmd}")
            echo "  Converting: $name.qmd -> $name.ipynb"
            quarto convert "$f" --output "_site/library/notebooks/${name}.ipynb" 2>/dev/null || true
        fi
    done

    # Expand LaTeX macros to standard LaTeX
    echo "Expanding LaTeX macros..."
    python scripts/inject_notebook_macros.py _site/library/notebooks/

    echo "Notebooks saved to: _site/library/notebooks/"
fi

echo ""
echo "=== Build complete ==="
echo "Output: $SCRIPT_DIR/_site/"
echo ""
echo "To view locally:"
echo "  open _site/index.html"
echo ""

# Start server if requested
if [ -n "$SERVE" ]; then
    echo "Starting local server at http://localhost:8080"
    echo "Press Ctrl+C to stop"
    cd _site
    python -m http.server 8080
fi
