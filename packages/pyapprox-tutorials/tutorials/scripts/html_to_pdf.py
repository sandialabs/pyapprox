#!/usr/bin/env python3
"""Convert the entire HTML tutorial site to a single PDF using Chrome headless.

Usage:
    python html_to_pdf.py <site_dir> <output_pdf> [--quarto-yml <path>] [--tmpdir <path>] [--skip N]

Reads the sidebar order from _quarto.yml to produce pages in tutorial order.
Uses Chrome headless --print-to-pdf for each HTML file, then merges with pypdf.
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path

import yaml
from pypdf import PdfWriter


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def get_ordered_pages(quarto_yml: Path) -> list[str]:
    """Extract ordered page list from _quarto.yml sidebar."""
    with open(quarto_yml) as f:
        cfg = yaml.safe_load(f)

    pages = ["index.html"]

    def _extract(items):
        for item in items:
            if isinstance(item, str):
                pages.append(item.replace(".qmd", ".html"))
            elif isinstance(item, dict):
                if "href" in item:
                    pages.append(item["href"].replace(".qmd", ".html"))
                if "contents" in item:
                    _extract(item["contents"])
                if "chapters" in item:
                    _extract(item["chapters"])

    sidebar = cfg.get("website", {}).get("sidebar", [])
    if isinstance(sidebar, list):
        for section in sidebar:
            if isinstance(section, dict) and "contents" in section:
                _extract(section["contents"])

    return pages


def html_to_pdf_chrome(html_path: Path, pdf_path: Path):
    """Use Chrome headless to print an HTML file to PDF."""
    url = f"file://{html_path.resolve()}"
    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        "--disable-background-networking",
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert HTML site to PDF")
    parser.add_argument("site_dir", type=Path, help="Path to _site directory")
    parser.add_argument("output_pdf", type=Path, help="Output PDF path")
    parser.add_argument("--quarto-yml", type=Path, default=None,
                        help="Path to _quarto.yml for page ordering")
    parser.add_argument("--tmpdir", type=Path, default=None,
                        help="Temp dir with pre-existing per-page PDFs (for resume)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N pages (already converted)")
    args = parser.parse_args()

    site = args.site_dir
    if not site.is_dir():
        print(f"Error: {site} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Get ordered page list
    if args.quarto_yml and args.quarto_yml.exists():
        pages = get_ordered_pages(args.quarto_yml)
    else:
        # Fall back to alphabetical
        pages = sorted(p.name for p in site.glob("*.html") if p.name != "404.html")

    # Filter to existing files
    pages = [p for p in pages if (site / p).exists()]
    print(f"Total pages: {len(pages)}")

    # Use provided tmpdir or create a new one
    if args.tmpdir:
        tmpdir = args.tmpdir
        tmpdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        import tempfile as _tmp
        _td = _tmp.mkdtemp()
        tmpdir = Path(_td)
        cleanup = True

    print(f"Working dir: {tmpdir}")

    merger = PdfWriter()
    ok_count = 0
    fail_count = 0

    for i, page in enumerate(pages):
        pdf_path = tmpdir / f"{i:03d}_{page.replace('.html', '.pdf')}"

        # Skip already-converted pages
        if i < args.skip and pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"  [{i+1}/{len(pages)}] {page} CACHED")
            merger.append(str(pdf_path))
            ok_count += 1
            continue

        html_path = site / page
        print(f"  [{i+1}/{len(pages)}] {page}", end="", flush=True)

        if html_to_pdf_chrome(html_path, pdf_path):
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                merger.append(str(pdf_path))
                size_kb = pdf_path.stat().st_size / 1024
                print(f" OK ({size_kb:.0f} KB)")
                ok_count += 1
            else:
                print(" EMPTY (skipped)")
                fail_count += 1
        else:
            print(" FAILED (skipped)")
            fail_count += 1

    print(f"\nMerging {ok_count} pages into {args.output_pdf}...")
    if ok_count > 0:
        merger.write(str(args.output_pdf))
    merger.close()

    if ok_count > 0:
        size_mb = args.output_pdf.stat().st_size / (1024 * 1024)
        print(f"Done: {args.output_pdf} ({size_mb:.1f} MB)")
        if fail_count > 0:
            print(f"Warning: {fail_count} pages failed to convert")
    else:
        print("Error: No pages were successfully converted", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
