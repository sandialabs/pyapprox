#!/bin/bash
# Build script for PyApprox UQ Tutorials
# Usage: ./build.sh [site] [options]
#
# Sites:
#   library         Build the validated tutorial library (default)
#   in_progress     Build the in-progress tutorial site
#
# Options:
#   --no-execute    Skip code execution (use cached results)
#   --execute       Force re-execute all code
#   --notebooks     Generate downloadable notebooks (library only)
#   --serve         Start local server after build
#   --skip=NAME     Skip rendering a tutorial (e.g. --skip=pacv_usage)
#                   Can be repeated: --skip=pacv_usage --skip=foo
#   --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse site argument
SITE="library"
if [ "$1" = "library" ] || [ "$1" = "in_progress" ]; then
    SITE="$1"
    shift
fi

# Parse options
NO_EXECUTE=""
FORCE_EXECUTE=""
GEN_NOTEBOOKS=""
SERVE=""
SKIP_FILES=()

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
        --skip=*)
            SKIP_FILES+=("${arg#--skip=}")
            ;;
        --help)
            head -17 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

if [ "$SITE" = "library" ]; then
    BUILD_DIR="$SCRIPT_DIR/library"
    SITE_LABEL="Tutorial Library"
else
    BUILD_DIR="$SCRIPT_DIR/in_progress"
    SITE_LABEL="In-Progress Tutorials"
fi

echo "=== Building PyApprox $SITE_LABEL ==="
echo "Working directory: $BUILD_DIR"
echo ""

cd "$BUILD_DIR"

# Clean old build
if [ -d "_site" ]; then
    echo "Cleaning old build..."
    rm -rf _site
fi

# Temporarily hide skipped files
SKIPPED_PATHS=()
for name in "${SKIP_FILES[@]}"; do
    qmd="$BUILD_DIR/${name}.qmd"
    if [ -f "$qmd" ]; then
        echo "  Skipping: ${name}.qmd"
        mv "$qmd" "$qmd.skip"
        SKIPPED_PATHS+=("$qmd")
    else
        echo "  Warning: ${name}.qmd not found, ignoring --skip=${name}"
    fi
done

# Restore skipped files on exit (even if build fails)
restore_skipped() {
    for qmd in "${SKIPPED_PATHS[@]}"; do
        if [ -f "$qmd.skip" ]; then
            mv "$qmd.skip" "$qmd"
        fi
    done
}
trap restore_skipped EXIT

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

# Generate notebooks (library only)
if [ -n "$GEN_NOTEBOOKS" ] && [ "$SITE" = "library" ]; then
    echo ""
    echo "Generating Jupyter notebooks..."
    mkdir -p _site/notebooks

    for f in *.qmd; do
        if [ "$(basename "$f")" != "index.qmd" ]; then
            name=$(basename "${f%.qmd}")
            echo "  Converting: $name.qmd -> $name.ipynb"
            quarto convert "$f" --output "_site/notebooks/${name}.ipynb" 2>/dev/null || true
        fi
    done

    # Expand LaTeX macros to standard LaTeX
    echo "Expanding LaTeX macros..."
    python "$SCRIPT_DIR/scripts/inject_notebook_macros.py" _site/notebooks/

    echo "Notebooks saved to: $BUILD_DIR/_site/notebooks/"
fi

echo ""
echo "=== Build complete ==="
echo "Output: $BUILD_DIR/_site/"
echo ""
echo "To view locally:"
echo "  open $BUILD_DIR/_site/index.html"
echo ""

# Start server if requested
if [ -n "$SERVE" ]; then
    echo "Starting local server at http://localhost:8080"
    echo "Press Ctrl+C to stop"
    cd _site
    python -m http.server 8080
fi
