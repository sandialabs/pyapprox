# Building PyApprox Typing Module Documentation

This guide covers how to build and deploy the Quarto-based documentation for the `pyapprox.typing` module tutorials.

## Prerequisites

### 1. Install Quarto

Download and install Quarto from https://quarto.org/docs/get-started/

```bash
# macOS (Homebrew)
brew install quarto

# Verify installation
quarto --version
```

### 2. Python Environment

Ensure you have the PyApprox environment with all dependencies:

```bash
# Using conda
conda activate linalg  # or your PyApprox environment

# Install pyapprox in development mode
pip install -e /path/to/pyapprox

# Required packages for tutorials
pip install matplotlib scipy torch  # torch is optional
```

### 3. Jupyter Kernel

Quarto needs a Jupyter kernel to execute Python code:

```bash
# Install ipykernel if not present
pip install ipykernel

# Register the kernel (optional, if using non-default env)
python -m ipykernel install --user --name=linalg --display-name="Python (linalg)"
```

---

## Local Development

### Build Script (Recommended)

Use the `build.sh` script for common build tasks:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials

# Full build with notebooks
./build.sh --notebooks

# Quick build (skip code execution)
./build.sh --no-execute --notebooks

# Build and start local server
./build.sh --notebooks --serve

# Show all options
./build.sh --help
```

### Quick Preview (Single Tutorial)

Preview a single tutorial with live reload:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto preview library/monte_carlo_basics.qmd
```

This opens a browser with live updates as you edit.

### Build Full Site (Library + Workshops)

Build the complete tutorial website (library and workshops together):

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto render
```

Output is written to `_site/` directory.

### Quick Build (Skip Execution)

For content changes without re-running code:

```bash
quarto render --no-execute
```

---

## Tutorial Time Estimation

Before finalizing tutorials, check that they meet time and size guidelines from `CONVENTIONS.md`.

### Estimate Time for a Single Tutorial

```bash
python scripts/estimate_tutorial_time.py library/my_tutorial.qmd
```

### Estimate Time for All Tutorials

```bash
python scripts/estimate_tutorial_time.py library/
```

### Verbose Output (Show Breakdown)

```bash
python scripts/estimate_tutorial_time.py library/my_tutorial.qmd --verbose
```

### Check Against Limits (CI/Pre-commit)

```bash
# Exit code 1 if any tutorial exceeds limits
python scripts/estimate_tutorial_time.py library/ --check
```

### JSON Output (Programmatic Use)

```bash
python scripts/estimate_tutorial_time.py library/ --json
```

### Check Multiple Specific Tutorials

```bash
# For workshop planning
python scripts/estimate_tutorial_time.py \
    library/sobol_indices_concept.qmd \
    library/sobol_indices_usage.qmd \
    library/sobol_estimator_analysis.qmd \
    --summary-only
```

### Example Output

```
============================================================
Tutorial: sobol_indices_concept.qmd
Title: Sobol Sensitivity Indices
============================================================
Type: concept
Difficulty: intermediate

Estimated presentation time: 8.5 min
Declared time in YAML: 10 min

Limits (concept):
  Target time: 10 min
  Suggested limit: 15 min
  Max lines: 150
  Max visible code: 2
  Max equations: 6
  Max figures: 6
```

### Understanding Code Block Counts

The script distinguishes:
- **Visible code blocks**: Code shown to the reader (counts toward limit)
- **Hidden code blocks**: Code with `#| echo: false` (doesn't count for concept tutorials)

The summary table shows `Code` as `visible+hidden`, e.g., `2+5` means 2 visible, 5 hidden.

---

## Workshop Planning

### Verify Workshop Timing

When composing a workshop from tutorials, verify total time fits the session:

```bash
python scripts/estimate_tutorial_time.py \
    library/variance_decomposition_concept.qmd \
    library/sobol_indices_concept.qmd \
    library/sobol_indices_usage.qmd \
    --summary-only
```

### Workshop Session Guidelines

From `CONVENTIONS.md`, tutorials target 5-10 minutes, allowing 6-8 per 90-minute block:

```
 0:00 - 0:10   Concept 1                     10 min
 0:10 - 0:20   Concept 2                     10 min
 0:20 - 0:30   Usage 1                       10 min
 0:30 - 0:40   Hands-on exercises            10 min
 0:40 - 0:50   Break                         10 min
 0:50 - 1:00   Usage 2                       10 min
 1:00 - 1:20   Analysis (extended)           20 min
 1:20 - 1:30   Wrap-up / Q&A                 10 min
```

Analysis tutorials may exceed the 10-minute target when derivations cannot be meaningfully split.

---

## Caching and Incremental Builds

### How Freeze Works

The `_quarto.yml` has `freeze: auto` which caches executed code:

1. **First render**: Executes all Python code, saves results to `_freeze/`
2. **Subsequent renders**: Only re-executes if the `.qmd` source changed
3. **Unchanged files**: Uses cached results (very fast)

### Development Workflow

| Scenario | Command |
|----------|---------|
| Content edits (text/math only) | `quarto render --no-execute` |
| Single tutorial update | `quarto render library/monte_carlo_basics.qmd` |
| Force re-execute all code | `quarto render --execute` |
| Live preview (auto-rebuild) | `quarto preview` |

---

## GitHub Pages Deployment

### Option 1: Manual Deployment

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto render
quarto publish gh-pages --no-browser
```

### Option 2: GitHub Actions

Create `.github/workflows/publish-tutorials.yml`:

```yaml
name: Publish Tutorials

on:
  push:
    branches: [master]
    paths:
      - 'pyapprox/typing/tutorials/**'
  workflow_dispatch:

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install matplotlib scipy jupyter nbformat

      - name: Setup Quarto
        uses: quarto-dev/quarto-actions/setup@v2

      - name: Check tutorial time limits
        run: |
          cd pyapprox/typing/tutorials
          python scripts/estimate_tutorial_time.py library/ --check

      - name: Render tutorials
        run: |
          cd pyapprox/typing/tutorials
          quarto render

      - name: Generate downloadable notebooks
        run: |
          cd pyapprox/typing/tutorials
          mkdir -p _site/library/notebooks
          for f in library/*.qmd; do
            name=$(basename "${f%.qmd}")
            quarto convert "$f" --output "_site/library/notebooks/${name}.ipynb"
          done

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: pyapprox/typing/tutorials/_site
          destination_dir: tutorials
```

---

## Directory Structure

```
pyapprox/typing/tutorials/
├── _quarto.yml                 # Quarto config
├── index.qmd                   # Main landing page
├── build.sh                    # Build script
├── BUILD.md                    # This file
├── CONVENTIONS.md              # Style and structure conventions
├── _macros.tex                 # LaTeX macros (PDF)
├── _macros_html.tex            # MathJax macros (HTML)
├── _site/                      # Build output (git-ignored)
├── _freeze/                    # Cached execution results
│
├── scripts/
│   ├── estimate_tutorial_time.py   # Time estimation tool
│   ├── generate_workshop_index.py
│   └── inject_notebook_macros.py
│
├── library/                    # Tutorial library
│   ├── index.qmd               # Library catalog
│   ├── styles.css              # Custom CSS
│   ├── figures/                # Static images
│   └── *.qmd                   # Tutorial files
│
└── workshops/                  # Workshop landing pages
    ├── index.qmd               # Workshop listing
    └── */                      # Workshop directories
```

---

## Troubleshooting

### "Jupyter kernel not found"

```bash
jupyter kernelspec list
python -m ipykernel install --user --name=linalg
```

### "Module not found: pyapprox"

```bash
conda activate linalg
pip install -e /path/to/pyapprox
```

### Tutorial exceeds time limits

1. Run `python scripts/estimate_tutorial_time.py file.qmd --verbose` to see breakdown
2. If > 10 min, consider splitting per `CONVENTIONS.md` guidelines
3. Move derivations to Analysis tutorial
4. Convert visible code to hidden code for illustrative figures
5. If splitting isn't feasible (e.g., indivisible derivation), add `extended_time_reason` to YAML

### Slow builds

```bash
# Use caching
quarto render --execute  # First time
quarto render            # Uses cache

# Or skip execution for content-only changes
quarto render --no-execute
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Preview single tutorial | `quarto preview library/file.qmd` |
| Preview full site | `quarto preview` |
| Build full site | `quarto render` |
| Build without execution | `quarto render --no-execute` |
| Estimate tutorial time | `python scripts/estimate_tutorial_time.py library/file.qmd` |
| Verbose time breakdown | `python scripts/estimate_tutorial_time.py library/file.qmd -v` |
| Check all time limits | `python scripts/estimate_tutorial_time.py library/ --check` |
| Publish to gh-pages | `quarto publish gh-pages` |

---

## Related Documents

- `CONVENTIONS.md` - Tutorial structure, notation, and time guidelines
- `scripts/estimate_tutorial_time.py` - Time estimation script (run with `--help` for options)
