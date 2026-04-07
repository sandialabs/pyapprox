#!/usr/bin/env python3
"""Expand LaTeX macros in Jupyter notebooks.

Quarto macros don't carry over when converting .qmd to .ipynb.
This script expands all custom macros to standard LaTeX so notebooks
render correctly without needing macro definitions.
"""

import json
import re
import sys
from pathlib import Path

# Macro definitions: name -> (num_args, replacement)
# For macros with arguments, use {0}, {1}, etc. as placeholders
MACROS = {
    # Vectors/Matrices
    r'\vc': (1, r'\mathbf{{{0}}}'),
    r'\mt': (1, r'\mathbf{{{0}}}'),
    r'\ts': (1, r'\mathcal{{{0}}}'),

    # Common Vectors
    r'\params': (0, r'\boldsymbol{\theta}'),
    r'\inputs': (0, r'\mathbf{x}'),
    r'\outputs': (0, r'\mathbf{y}'),
    r'\data': (0, r'\mathbf{y}'),
    r'\state': (0, r'\mathbf{u}'),

    # Dimensions
    r'\dparams': (0, r'd_\theta'),
    r'\dinputs': (0, r'd_x'),
    r'\doutputs': (0, r'd_y'),
    r'\dstates': (0, r'd_u'),
    r'\Reals': (1, r'\mathbb{{R}}^{{{0}}}'),

    # Samples
    r'\nsamples': (0, r'N'),
    r'\sample': (1, r'^{{({0})}}'),
    r'\samplemat': (1, r'\mathbf{{{0}}}'),

    # Random Variables
    r'\rv': (1, r'\tilde{{{0}}}'),

    # Functions and Models
    r'\model': (0, r'f'),
    r'\qoifunc': (0, r'q'),
    r'\qoifunctional': (0, r'g'),
    r'\surrogate': (1, r'f_{{{0}}}'),
    r'\surr': (0, r'f_N'),

    # Probability
    r'\E': (0, r'\mathbb{E}'),
    r'\Var': (0, r'\mathbb{V}'),
    r'\Cov': (0, r'\mathrm{Cov}'),
    r'\pdf': (0, r'p'),
    r'\normal': (0, r'\mathcal{N}'),
    r'\uniform': (0, r'\mathcal{U}'),

    # Sensitivity Analysis
    r'\Sobol': (1, r'S_{{{0}}}'),
    r'\SobolT': (1, r'S_{{{0}}}^T'),
    r'\maineff': (1, r'V_{{{0}}}'),
    r'\totaleff': (1, r'V_{{{0}}}^T'),

    # Operators
    r'\argmin': (0, r'\operatorname*{arg\,min}'),
    r'\argmax': (0, r'\operatorname*{arg\,max}'),
    r'\dd': (0, r'\mathrm{d}'),
    r'\pd': (2, r'\frac{{\partial {0}}}{{\partial {1}}}'),

    # Common Matrices
    r'\CovMat': (0, r'\boldsymbol{\Sigma}'),
    r'\IdentMat': (0, r'\mathbf{I}'),
    r'\JacMat': (0, r'\mathbf{J}'),
    r'\HessMat': (0, r'\mathbf{H}'),

    # Time-dependent Problems
    r'\tinit': (0, r't_0'),
    r'\tfinal': (0, r't_f'),
    r'\deltat': (0, r'\Delta t'),

    # Integration/Quadrature
    r'\integral': (0, r'\mathcal{I}'),
    r'\domain': (0, r'\Omega'),
}


def extract_brace_arg(text: str, start: int) -> tuple[str, int]:
    """Extract content within braces starting at position start.

    Returns (content, end_position) where end_position is after the closing brace.
    """
    if start >= len(text) or text[start] != '{':
        return '', start

    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start+1:i], i + 1
        i += 1
    return text[start+1:], len(text)


def expand_macro(text: str, macro: str, num_args: int, replacement: str) -> str:
    """Expand a single macro in the text."""
    # Escape the backslash for regex
    pattern = re.escape(macro)

    if num_args == 0:
        # Simple replacement - but be careful not to match partial macro names
        # e.g., \E should not match inside \Error
        # Match macro followed by non-letter or end of string
        pattern = pattern + r'(?![a-zA-Z])'
        return re.sub(pattern, replacement.replace('\\', r'\\'), text)

    # For macros with arguments, we need to find and extract them
    result = []
    i = 0
    macro_len = len(macro)

    while i < len(text):
        # Check if we're at the start of this macro
        if text[i:i+macro_len] == macro:
            # Check it's not part of a longer macro name
            next_pos = i + macro_len
            if next_pos < len(text) and text[next_pos].isalpha():
                result.append(text[i])
                i += 1
                continue

            # Extract arguments
            args = []
            pos = next_pos
            for _ in range(num_args):
                # Skip whitespace
                while pos < len(text) and text[pos] in ' \t\n':
                    pos += 1
                if pos < len(text) and text[pos] == '{':
                    arg, pos = extract_brace_arg(text, pos)
                    args.append(arg)
                else:
                    # No brace found, take single character
                    if pos < len(text):
                        args.append(text[pos])
                        pos += 1

            if len(args) == num_args:
                # Apply replacement
                expanded = replacement.format(*args)
                result.append(expanded)
                i = pos
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)


def fix_subscript_braces(text: str) -> str:
    """Fix subscripts that need braces around multi-char content.

    e.g., _\boldsymbol{\theta} -> _{\boldsymbol{\theta}}
    """
    import re
    # Pattern: underscore followed by backslash command (not in braces)
    # Match _\command{...} and wrap in braces
    pattern = r'_\\([a-zA-Z]+)(\{[^}]*\})'
    replacement = r'_{\\\1\2}'
    return re.sub(pattern, replacement, text)


def fix_qmd_links(text: str) -> str:
    """Convert .qmd links to .ipynb for notebooks."""
    import re
    # Convert internal .qmd links to .ipynb
    # Pattern: [text](filename.qmd) -> [text](filename.ipynb)
    text = re.sub(r'\]\(([^)]+)\.qmd\)', r'](\1.ipynb)', text)
    # Remove notebook download links (they point to themselves)
    text = re.sub(r'::: \{\.callout-tip collapse="true"\}\s*\n## Download Notebook\n\[Download as Jupyter Notebook\]\([^)]+\)\s*\n:::', '', text)
    return text


def expand_all_macros(text: str) -> str:
    """Expand all macros in the text."""
    # Sort macros by length (longest first) to avoid partial matches
    sorted_macros = sorted(MACROS.items(), key=lambda x: -len(x[0]))

    for macro, (num_args, replacement) in sorted_macros:
        text = expand_macro(text, macro, num_args, replacement)

    # Fix subscript bracing issues
    text = fix_subscript_braces(text)

    # Fix .qmd links
    text = fix_qmd_links(text)

    return text


def _strip_yaml_front_matter(cells: list) -> tuple[list, bool]:
    """Strip YAML front matter from the first markdown cell.

    Quarto's ``quarto convert`` puts the YAML header (title, tags,
    execute options, etc.) at the start of the first markdown cell.
    This metadata is meaningless in a standalone notebook.
    """
    modified = False
    for cell in cells:
        if cell.get('cell_type') != 'markdown':
            continue
        src = cell.get('source', [])
        text = ''.join(src) if isinstance(src, list) else src
        if not text.lstrip().startswith('---'):
            break
        # Find the closing --- of the front matter
        # Skip the opening ---
        first_fence = text.index('---')
        rest = text[first_fence + 3:]
        second_fence = rest.find('\n---')
        if second_fence == -1:
            break
        # Everything after the closing --- line
        after_fence = rest[second_fence + 4:]  # skip \n---
        # Skip the newline after closing ---
        if after_fence.startswith('\n'):
            after_fence = after_fence[1:]
        cleaned = after_fence.lstrip('\n')
        if cleaned:
            cell['source'] = cleaned.split('\n')
            cell['source'] = [line + '\n' for line in cell['source']]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
        else:
            # Entire cell was front matter — mark for removal
            cell['source'] = []
        modified = True
        break  # only process the first markdown cell
    # Remove empty cells
    cells = [c for c in cells if c.get('source', [''])]
    return cells, modified


def process_notebook(notebook_path: Path) -> bool:
    """Expand macros in a notebook.

    Returns True if notebook was modified, False otherwise.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = False
    cells = notebook.get('cells', [])

    # Strip YAML front matter cells (Quarto metadata not useful in notebooks)
    cells, front_matter_removed = _strip_yaml_front_matter(cells)
    if front_matter_removed:
        notebook['cells'] = cells
        modified = True

    for cell in cells:
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                original = ''.join(source)
            else:
                original = source

            expanded = expand_all_macros(original)

            if expanded != original:
                modified = True
                # Convert back to list format
                cell['source'] = expanded.split('\n')
                cell['source'] = [line + '\n' for line in cell['source']]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')

    if modified:
        # Remove any previously injected macro cell
        cells = [c for c in cells if 'PyApprox Tutorial Macros' not in ''.join(c.get('source', []))]
        notebook['cells'] = cells

        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

    return modified


def main():
    """Process all notebooks in the given directory."""
    if len(sys.argv) < 2:
        notebook_dir = Path(__file__).parent.parent / '_site' / 'library' / 'notebooks'
    else:
        notebook_dir = Path(sys.argv[1])

    if not notebook_dir.exists():
        print(f"Directory not found: {notebook_dir}")
        sys.exit(1)

    notebooks = list(notebook_dir.glob('*.ipynb'))
    if not notebooks:
        print(f"No notebooks found in {notebook_dir}")
        sys.exit(1)

    modified = 0
    for nb_path in notebooks:
        if process_notebook(nb_path):
            print(f"  Expanded macros: {nb_path.name}")
            modified += 1
        else:
            print(f"  No macros found: {nb_path.name}")

    print(f"\nProcessed {len(notebooks)} notebooks, modified {modified}")


if __name__ == '__main__':
    main()
