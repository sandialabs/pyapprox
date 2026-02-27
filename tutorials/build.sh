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
#   -j N            Parallel execution: render N tutorials concurrently (default: 1)
#                   Use -j auto to detect CPU count automatically
#   --notebooks     Generate downloadable notebooks (library only)
#   --serve         Start local server after build
#   --timings       Render each tutorial individually, clear its freeze cache
#                   first for accurate execution times, and log results to
#                   render_timings.csv with a sorted summary at the end
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
TIMINGS=""
NJOBS=1
SKIP_FILES=()

for arg in "$@"; do
    case $arg in
        --no-execute)
            NO_EXECUTE="--no-execute"
            ;;
        --execute)
            FORCE_EXECUTE="--execute"
            ;;
        -j)
            # Next arg is the job count; handled below
            ;;
        -j[0-9a-z]*)
            NJOBS="${arg#-j}"
            ;;
        --notebooks)
            GEN_NOTEBOOKS="yes"
            ;;
        --timings)
            TIMINGS="yes"
            ;;
        --serve)
            SERVE="yes"
            ;;
        --skip=*)
            SKIP_FILES+=("${arg#--skip=}")
            ;;
        --help)
            head -21 "$0" | tail -20
            exit 0
            ;;
        *)
            # Handle "-j N" as two separate args
            if [ "$prev_arg" = "-j" ]; then
                NJOBS="$arg"
            else
                echo "Unknown option: $arg"
                exit 1
            fi
            ;;
    esac
    prev_arg="$arg"
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

# Resolve -j auto to CPU count
if [ "$NJOBS" = "auto" ]; then
    NJOBS=$(python3 -c "import os; print(os.cpu_count() or 1)")
    echo "Detected $NJOBS CPUs"
fi

# Build site
echo "Rendering Quarto site..."
if [ -n "$NO_EXECUTE" ]; then
    echo "  (skipping code execution)"
    quarto render --no-execute
elif [ "$NJOBS" -gt 1 ] 2>/dev/null || [ -n "$TIMINGS" ]; then
    # Per-tutorial mode: render each .qmd independently, then assemble
    # Used for parallel builds (-j N) and timing builds (--timings)
    if [ "$NJOBS" -gt 1 ] 2>/dev/null; then
        echo "  (parallel execution with $NJOBS jobs)"
    else
        NJOBS=1
        echo "  (sequential execution with per-tutorial timing)"
    fi

    # Collect .qmd files to render (excluding index.qmd and skipped files)
    QMD_FILES=()
    for f in *.qmd; do
        [ "$(basename "$f")" = "index.qmd" ] && continue
        QMD_FILES+=("$f")
    done

    echo "  ${#QMD_FILES[@]} tutorials to render"

    # Create log directory
    LOG_DIR="$BUILD_DIR/_build_logs"
    mkdir -p "$LOG_DIR"

    # Initialize timings file
    TIMINGS_FILE="$BUILD_DIR/render_timings.csv"
    if [ -n "$TIMINGS" ]; then
        echo "tutorial,seconds,status" > "$TIMINGS_FILE"
    fi

    # Render individual files using xargs
    EXECUTE_FLAG=""
    [ -n "$FORCE_EXECUTE" ] && EXECUTE_FLAG="--execute"

    FAIL_COUNT=0
    printf '%s\n' "${QMD_FILES[@]}" | xargs -P "$NJOBS" -I {} bash -c '
        name="${1%.qmd}"
        log_dir="$2"
        execute_flag="$3"
        timings_file="$4"
        # Clear freeze cache when timing for accurate execution times
        if [ -n "$timings_file" ] && [ -d "_freeze/$name" ]; then
            rm -rf "_freeze/$name"
        fi
        echo "  [START] $name"
        start=$(date +%s)
        if quarto render "$1" $execute_flag > "$log_dir/${name}.log" 2>&1; then
            elapsed=$(( $(date +%s) - start ))
            echo "  [DONE]  $name (${elapsed}s)"
            [ -n "$timings_file" ] && echo "$name,$elapsed,ok" >> "$timings_file"
        else
            elapsed=$(( $(date +%s) - start ))
            echo "  [FAIL]  $name (${elapsed}s) — see $log_dir/${name}.log"
            [ -n "$timings_file" ] && echo "$name,$elapsed,FAIL" >> "$timings_file"
            exit 1
        fi
    ' _ {} "$LOG_DIR" "$EXECUTE_FLAG" "${TIMINGS:+$TIMINGS_FILE}" || FAIL_COUNT=$?

    if [ "$FAIL_COUNT" -ne 0 ]; then
        echo ""
        echo "ERROR: Some tutorials failed to render. Check logs in $LOG_DIR/"
        echo "Failed logs:"
        grep -l "ERROR\|error\|Error" "$LOG_DIR"/*.log 2>/dev/null | while read -r logf; do
            echo "  $logf"
        done
        if [ -n "$TIMINGS" ]; then
            echo ""
            echo "Partial timings saved to: $TIMINGS_FILE"
        fi
        exit 1
    fi

    # Assemble full site (no execution needed, _freeze/ is populated)
    echo ""
    echo "  Assembling site..."
    quarto render --no-execute

    if [ -n "$TIMINGS" ]; then
        echo ""
        echo "Timings saved to: $TIMINGS_FILE"
        echo ""
        # Print summary sorted by time (descending)
        echo "=== Render Times (slowest first) ==="
        tail -n +2 "$TIMINGS_FILE" | sort -t, -k2 -rn | while IFS=, read -r name secs status; do
            printf "  %6ss  %-5s  %s\n" "$secs" "$status" "$name"
        done
    fi
else
    if [ -n "$FORCE_EXECUTE" ]; then
        echo "  (forcing code execution)"
        quarto render --execute
    else
        echo "  (using freeze cache)"
        quarto render
    fi
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
