#!/usr/bin/env python3
"""Generate an index.html landing page for the notebooks directory.

GitHub Pages does not serve directory listings, so navigating to
``/notebooks/`` returns a 404.  This script generates a simple HTML
page that lists all ``.ipynb`` files with download links.

Usage:
    python generate_notebook_index.py <notebooks_dir>

Example:
    python generate_notebook_index.py _site/notebooks/
"""

import sys
from pathlib import Path

# Non-tutorial files that should never appear in the listing.
EXCLUDE_STEMS = {"404", "tutorials", "index"}


def humanize(name: str) -> str:
    """Convert a filename stem to a human-readable title."""
    return name.replace("_", " ").replace("-", " ").title()


def _list_notebooks(notebooks_dir: Path) -> list[Path]:
    """Return sorted tutorial notebooks, excluding non-tutorial files."""
    return sorted(
        nb for nb in notebooks_dir.glob("*.ipynb")
        if nb.stem not in EXCLUDE_STEMS
    )


def generate_index(notebooks_dir: Path) -> str:
    """Generate index.html content listing all notebooks."""
    notebooks = _list_notebooks(notebooks_dir)

    rows = []
    for nb in notebooks:
        title = humanize(nb.stem)
        rows.append(
            f'      <tr>\n'
            f'        <td>{title}</td>\n'
            f'        <td><a href="{nb.name}" download="{nb.name}" '
            f'class="download-btn">Download</a></td>\n'
            f'      </tr>'
        )

    table_rows = "\n".join(rows)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PyApprox Tutorial Notebooks</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      max-width: 900px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #333;
      line-height: 1.6;
    }}
    h1 {{ color: #2c3e50; margin-bottom: 0.3rem; }}
    .subtitle {{ color: #7f8c8d; margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      text-align: left;
      padding: 0.75rem;
      border-bottom: 2px solid #2c3e50;
      color: #2c3e50;
    }}
    td {{
      padding: 0.5rem 0.75rem;
      border-bottom: 1px solid #ecf0f1;
    }}
    tr:hover {{ background-color: #f8f9fa; }}
    a {{ color: #2471a3; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .download-btn {{
      display: inline-block;
      padding: 0.25rem 0.75rem;
      background: #2471a3;
      color: white !important;
      border-radius: 4px;
      font-size: 0.85rem;
      text-decoration: none !important;
    }}
    .download-btn:hover {{
      background: #1a5276;
    }}
    .info {{
      background: #eaf2f8;
      border-left: 4px solid #2471a3;
      padding: 0.75rem 1rem;
      margin-bottom: 1.5rem;
      border-radius: 0 4px 4px 0;
    }}
    .back {{ margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <p class="back"><a href="../">&larr; Back to tutorials</a></p>
  <h1>Tutorial Notebooks</h1>
  <p class="subtitle">{len(notebooks)} notebooks available for download</p>

  <div class="info">
    Each notebook includes an auto-install cell so you can run it
    directly in <a href="https://colab.research.google.com">Google Colab</a>
    or any Jupyter environment.
  </div>

  <table>
    <thead>
      <tr>
        <th>Tutorial</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
{table_rows}
    </tbody>
  </table>

</body>
</html>
"""


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <notebooks_dir>", file=sys.stderr)
        sys.exit(1)

    notebooks_dir = Path(sys.argv[1])
    if not notebooks_dir.is_dir():
        print(f"Error: {notebooks_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    notebooks = _list_notebooks(notebooks_dir)
    html = generate_index(notebooks_dir)
    index_path = notebooks_dir / "index.html"
    index_path.write_text(html)
    print(f"  Generated {index_path} ({len(notebooks)} notebooks)")


if __name__ == "__main__":
    main()
