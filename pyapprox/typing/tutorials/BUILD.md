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

Output is written to `_site/` directory with this structure:
```
_site/
├── index.html              # Main landing page
├── library/                # All tutorials
│   ├── index.html
│   ├── monte_carlo_basics.html
│   └── ...
└── workshops/              # Workshop landing pages
    ├── index.html
    └── intro_to_uq_2025/
        └── index.html
```

### Preview Full Site

Preview the complete site locally:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto preview
```

Opens at http://localhost:4848 (or similar port).

### Quick Build (Skip Execution)

For content changes without re-running code:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto render --no-execute
```

### Serve Built Site

After building, serve the site locally:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials/_site
python -m http.server 8080
```

Then open http://localhost:8080 in your browser.

Or open HTML files directly in Safari:

```bash
open /Users/jdjakem/pyapprox/pyapprox/typing/tutorials/_site/index.html
```

### Generate Downloadable Notebooks (Local)

Each tutorial has a "Download Notebook" link. To make these work locally, generate the notebooks after building:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
mkdir -p _site/library/notebooks
for f in library/*.qmd; do
  name=$(basename "${f%.qmd}")
  quarto convert "$f" --output "_site/library/notebooks/${name}.ipynb"
done
```

On GitHub Pages, notebooks are generated automatically by the deployment workflow.

---

## Caching and Incremental Builds

### How Freeze Works

The `_quarto.yml` has `freeze: auto` which caches executed code:

1. **First render**: Executes all Python code, saves results to `_freeze/`
2. **Subsequent renders**: Only re-executes if the `.qmd` source changed
3. **Unchanged files**: Uses cached results (very fast)

### The `_freeze/` Directory

After building, cached execution results are stored in:

```
tutorials/
├── _freeze/              # Cached execution results
│   └── library/
│       └── monte_carlo_basics/
│           └── execute-results/
└── _site/                # Rendered HTML output
```

You can optionally commit `_freeze/` to git to share cached results with collaborators.

### Development Workflow

| Scenario | Command |
|----------|---------|
| Content edits (text/math only) | `quarto render --no-execute` |
| Single tutorial update | `quarto render library/monte_carlo_basics.qmd` |
| Force re-execute all code | `quarto render --execute` |
| Live preview (auto-rebuild) | `quarto preview` |

### Recommended Workflow

1. **For text/math edits**: Use `quarto render --no-execute` (seconds)
2. **For code changes**: Render single file `quarto render library/file.qmd`
3. **For live editing**: Use `quarto preview` (watches files, auto-rebuilds)
4. **Before deployment**: Full `quarto render` to ensure everything works

---

## Build Options

### Skip Code Execution (Fast Build)

For quick iteration on content without re-running code:

```bash
quarto render --no-execute
```

### Freeze Execution Results

Cache computation results between builds:

```bash
# First build: execute and freeze
quarto render --execute

# Subsequent builds: use frozen results
quarto render
```

The `freeze: auto` setting in `_quarto.yml` handles this automatically.

### Build Specific Format

```bash
# HTML only (default)
quarto render --to html

# PDF (requires LaTeX)
quarto render --to pdf
```

---

## GitHub Pages Deployment

### Architecture Overview

The tutorial system uses a **unified build** where:
- Tutorials are authored ONCE in `library/`
- Workshops are landing pages that LINK to library tutorials (no duplication)
- Single Quarto project from `tutorials/` renders everything together

### Option 1: Manual Deployment (Recommended for Testing)

#### Step 1: Create a deployment branch

```bash
# From repository root
git checkout --orphan gh-pages
git reset --hard
git commit --allow-empty -m "Initialize gh-pages branch"
git push origin gh-pages
git checkout linalg-refactor  # Return to your working branch
```

#### Step 2: Build documentation

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto render
```

#### Step 3: Deploy to gh-pages

```bash
# From tutorials directory
quarto publish gh-pages --no-browser
```

Or manually:

```bash
# Copy _site contents to gh-pages branch
git worktree add ../gh-pages-deploy gh-pages
cp -r _site/* ../gh-pages-deploy/
cd ../gh-pages-deploy
git add .
git commit -m "Update documentation"
git push origin gh-pages
git worktree remove ../gh-pages-deploy
```

#### Step 4: Configure GitHub Pages

1. Go to repository Settings → Pages
2. Source: "Deploy from a branch"
3. Branch: `gh-pages` / `/ (root)`
4. Save

Site will be available at: `https://sandialabs.github.io/pyapprox/tutorials/`

### Option 2: Automated GitHub Actions (Recommended)

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

      - name: Inject LaTeX macros into notebooks
        run: |
          cd pyapprox/typing/tutorials
          python scripts/inject_notebook_macros.py _site/library/notebooks/

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: pyapprox/typing/tutorials/_site
          destination_dir: tutorials
```

This deploys to `https://sandialabs.github.io/pyapprox/tutorials/`

Notebooks will be available at `https://sandialabs.github.io/pyapprox/tutorials/library/notebooks/`

### Option 3: Separate Repository for Typing Docs

For complete isolation:

```bash
# Create new repo: pyapprox-typing-docs

# Build locally
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
quarto render

# Push _site contents to new repo
cd _site
git init
git add .
git commit -m "Deploy typing module tutorials"
git remote add origin https://github.com/YOUR_USERNAME/pyapprox-typing-docs.git
git push -u origin main

# Enable GitHub Pages on the new repo (Settings → Pages → main branch)
```

---

## Directory Structure

```
pyapprox/typing/tutorials/
├── _quarto.yml                 # ROOT: Unified Quarto config
├── index.qmd                   # ROOT: Main landing page
├── build.sh                    # Build script (run with --help)
├── BUILD.md                    # This file
├── CONVENTIONS.md              # Mathematical notation standards
├── _macros.tex                 # LaTeX macros (PDF)
├── _macros_html.tex            # MathJax macros (HTML)
├── _site/                      # Build output (git-ignored)
│
├── library/                    # Tutorial library
│   ├── index.qmd               # Library catalog
│   ├── styles.css              # Custom CSS
│   └── *.qmd                   # 18 tutorial files
│
├── workshops/                  # Workshop landing pages
│   ├── index.qmd               # Workshop listing
│   └── intro_to_uq_2025/       # Specific workshop
│       ├── index.qmd           # Links to library tutorials
│       └── workshop.yml        # Workshop metadata
│
└── scripts/
    └── generate_workshop_index.py
```

**Key Points:**
- Root `_quarto.yml` controls the unified build
- Library tutorials are the single source of truth
- Workshops are landing pages that LINK to library tutorials (no content duplication)
- Build output goes to `tutorials/_site/` (not library/_site/)

---

## Interactive Development & Debugging

When developing tutorials, you often want to run and debug Python code without rendering the full document.

### Option 1: VS Code + Quarto Extension (Recommended)

The best interactive experience for editing `.qmd` files:

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the [Quarto extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
3. Open any `.qmd` file
4. Run cells interactively with `Shift+Enter`

This gives you a notebook-like experience directly in the `.qmd` file.

### Option 2: Convert to Jupyter Notebook

Convert a tutorial to a notebook for interactive exploration:

```bash
# Convert .qmd to .ipynb
quarto convert tutorial.qmd --output tutorial.ipynb

# Open in Jupyter
jupyter lab tutorial.ipynb

# When done, convert back (if needed)
quarto convert tutorial.ipynb --output tutorial.qmd
```

### Option 3: JupyterLab with Jupytext

JupyterLab can open `.qmd` files directly with the jupytext extension:

```bash
pip install jupytext

# Open .qmd files directly in JupyterLab
jupyter lab tutorial.qmd
```

### Option 4: Execute Without Rendering

Run all code cells without generating HTML output:

```bash
quarto run tutorial.qmd
```

This executes the Python code but skips rendering, useful for checking if code runs correctly.

### Option 5: Extract and Run Code Blocks

For quick debugging, extract Python code to a standalone file:

```bash
# Extract Python code blocks
grep -Pzo '```\{python\}[\s\S]*?```' tutorial.qmd | \
    sed 's/```{python}//g' | sed 's/```//g' > debug.py

# Run in Python
python debug.py
```

Or manually copy code blocks into a Python REPL or script.

### Option 6: IPython/Python REPL

Start an interactive session with the same setup as tutorials:

```bash
cd /Users/jdjakem/pyapprox/pyapprox/typing/tutorials
conda activate linalg
ipython
```

Then copy-paste code blocks for testing.

### Workflow Tips

- **Rapid iteration**: Use VS Code + Quarto extension for cell-by-cell execution
- **Full testing**: Use `quarto run tutorial.qmd` to verify all code executes
- **Content changes**: Use `quarto render --no-execute` to preview text/math changes quickly
- **Debugging errors**: Convert to `.ipynb` for full debugging with variable inspection

---

## Troubleshooting

### "Jupyter kernel not found"

```bash
# List available kernels
jupyter kernelspec list

# Install kernel for your environment
python -m ipykernel install --user --name=linalg
```

Then update `_quarto.yml`:
```yaml
jupyter: linalg  # Use your kernel name
```

### "Module not found: pyapprox"

Ensure PyApprox is installed in the active environment:

```bash
conda activate linalg
pip install -e /path/to/pyapprox
```

### Slow builds

Use execution caching:

```bash
# Build once with execution
quarto render --execute

# Subsequent builds use cache (freeze: auto in _quarto.yml)
quarto render
```

Or skip execution entirely for content-only changes:

```bash
quarto render --no-execute
```

### LaTeX/PDF errors

For PDF output, install TinyTeX:

```bash
quarto install tinytex
```

### Port already in use

```bash
quarto preview --port 5555
```

---

## Quick Reference

All commands run from `pyapprox/typing/tutorials/`:

| Task | Command |
|------|---------|
| Preview single tutorial | `quarto preview library/file.qmd` |
| Preview full site | `quarto preview` |
| Build full site | `quarto render` |
| Build without execution | `quarto render --no-execute` |
| Execute without rendering | `quarto run library/file.qmd` |
| Convert to notebook | `quarto convert library/file.qmd --output file.ipynb` |
| Publish to gh-pages | `quarto publish gh-pages` |
| Serve built site | `cd _site && python -m http.server 8080` |
| Check Quarto version | `quarto --version` |
| List Jupyter kernels | `jupyter kernelspec list` |

---

## Deployment Checklist

- [ ] All tutorials render without errors
- [ ] Code cells execute successfully
- [ ] Navigation links work
- [ ] Images and plots display correctly
- [ ] MathJax equations render properly
- [ ] gh-pages branch exists
- [ ] GitHub Pages enabled in repository settings
- [ ] Deployed site accessible at expected URL
