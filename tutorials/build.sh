#!/bin/bash
# Build script for PyApprox UQ Tutorials
# Usage: ./build.sh [site] [options]
#
# Sites:
#   library         Build the validated tutorial library (default)
#   in_progress     Build the in-progress tutorial site
#
# Options:
#   --no-execute    Skip code execution AND freeze cache (text only, no outputs)
#   --execute       Force re-execute all code (ignores freeze cache)
#   -j N            Parallel execution: render N tutorials concurrently (default: 1)
#                   Use -j auto to detect CPU count automatically
#                   NOTE: Quarto has race conditions in parallel mode — a few
#                   tutorials may fail with "No such file or directory" errors.
#                   Simply re-run the same command; previously succeeded tutorials
#                   are cached in _freeze/ and skipped, so only failures re-execute.
#   --html-fast     Build HTML site from freeze cache only (skips all execution)
#   --pdf           Generate PDF user manual (uses freeze cache, executes if needed)
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
GEN_PDF=""
HTML_FAST=""
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
        --html-fast)
            HTML_FAST="yes"
            ;;
        --notebooks)
            GEN_NOTEBOOKS="yes"
            ;;
        --pdf)
            GEN_PDF="yes"
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
            head -23 "$0" | tail -21
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

# Determine if we need the HTML site build
# PDF-only mode: skip HTML render, _site cleanup, and notebook generation
PDF_ONLY=""
if [ -n "$GEN_PDF" ] && [ -z "$GEN_NOTEBOOKS" ] && [ -z "$SERVE" ]; then
    PDF_ONLY="yes"
fi

# Clean old build (only when building HTML site)
if [ -z "$PDF_ONLY" ] && [ -d "_site" ]; then
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

# Build HTML site (skip when only generating PDF)
if [ -n "$PDF_ONLY" ]; then
    echo "Skipping HTML site build (PDF only)"
else
echo "Rendering Quarto site..."
if [ -n "$HTML_FAST" ]; then
    echo "  (using frozen outputs only)"
    quarto render --use-freezer
elif [ -n "$NO_EXECUTE" ]; then
    echo "  (skipping code execution and freeze cache)"
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

    # Bootstrap Quarto project scaffolding (site_libs/, etc.) before
    # parallel rendering.  A no-execute render of a single small file
    # creates the shared directories that parallel workers expect.
    echo "  Bootstrapping project scaffolding..."
    quarto render index.qmd --no-execute > /dev/null 2>&1 || true
    # Ensure shared directories exist even if bootstrap render failed
    mkdir -p site_libs
    mkdir -p .quarto

    # Pre-create *_files/ stubs so Quarto's project-level glob
    # resolution doesn't fail when a tutorial is rendered before another
    # tutorial that produces outputs.
    for f in "${QMD_FILES[@]}"; do
        name="${f%.qmd}"
        mkdir -p "${name}_files/mediabag"
        mkdir -p "${name}_files/execute-results"
        mkdir -p "${name}_files/figure-html"
        mkdir -p "${name}_files/figure-pdf"
    done

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
        grep -l "^.*ERROR:" "$LOG_DIR"/*.log 2>/dev/null | while read -r logf; do
            echo "  $logf"
        done
        if [ -n "$TIMINGS" ]; then
            echo ""
            echo "Partial timings saved to: $TIMINGS_FILE"
        fi
        echo ""
        echo "TIP: Parallel builds can fail due to Quarto race conditions."
        echo "     Re-run the same command — cached tutorials are skipped."
        exit 1
    fi

    # Clean up empty stubs created for parallel safety
    for f in "${QMD_FILES[@]}"; do
        name="${f%.qmd}"
        rmdir "${name}_files/mediabag" 2>/dev/null || true
        rmdir "${name}_files/execute-results" 2>/dev/null || true
        rmdir "${name}_files/figure-html" 2>/dev/null || true
        rmdir "${name}_files/figure-pdf" 2>/dev/null || true
        rmdir "${name}_files" 2>/dev/null || true
    done

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
fi  # end PDF_ONLY check

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

    # Add pip install cell for Colab/standalone use
    echo "Injecting install cells..."
    python "$SCRIPT_DIR/scripts/inject_notebook_install.py" _site/notebooks/

    echo "Notebooks saved to: $BUILD_DIR/_site/notebooks/"
fi

# Generate PDF user manual (library only)
if [ -n "$GEN_PDF" ] && [ "$SITE" = "library" ]; then
    echo ""
    echo "Generating PDF user manual..."
    # Note: --use-freezer is broken with Quarto book project type
    # (renameSync error). We rely on freeze: auto instead, which
    # skips execution for any tutorial with cached tex.json.
    echo "  (using freeze cache, executing if needed)"
    # Clean stale index.html from website build (causes renameSync error)
    rm -f "$BUILD_DIR/index.html"
    quarto render --profile pdf
    echo "PDF saved to: $BUILD_DIR/_book/pyapprox-user-manual.pdf"
fi

echo ""
echo "=== Build complete ==="
if [ -n "$PDF_ONLY" ]; then
    echo "Output: $BUILD_DIR/_book/"
else
    echo "Output: $BUILD_DIR/_site/"
    echo ""
    echo "To view locally:"
    echo "  open $BUILD_DIR/_site/index.html"
fi
echo ""

# Start server if requested
if [ -n "$SERVE" ]; then
    echo "Starting local server at http://localhost:8080"
    echo "Press Ctrl+C to stop"
    cd _site
    python -m http.server 8080
fi
