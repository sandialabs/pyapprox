#!/usr/bin/env python3
"""
Estimate presentation time for PyApprox tutorials.

This script analyzes Quarto (.qmd) tutorial files and estimates the time
required to present them based on content analysis.

Usage:
    # Single tutorial
    python estimate_tutorial_time.py library/my_tutorial.qmd

    # All tutorials in a directory
    python estimate_tutorial_time.py library/

    # With verbose output
    python estimate_tutorial_time.py library/my_tutorial.qmd --verbose

    # Check against limits (exit code 1 if any exceed limits)
    python estimate_tutorial_time.py library/ --check

    # JSON output for programmatic use
    python estimate_tutorial_time.py library/ --json

See CONVENTIONS.md for time guidelines and limits.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Time weights (minutes per element)
TIME_WEIGHTS = {
    "code_block_visible": 1.75,   # Visible code block with discussion
    "code_block_hidden": 0.5,     # Hidden code (figure generation)
    "figure": 1.25,               # Figure interpretation
    "prose_paragraph": 0.3,       # Paragraph of text
    "heading": 0.2,               # Section heading (transition time)
    "list_item": 0.1,             # Bullet point
    "table": 0.75,                # Table interpretation
}

# Equation difficulty tiers (minutes per equation block)
# Difficulty is scored from features: length, notation complexity, structure
EQUATION_DIFFICULTY = {
    "simple": 0.5,      # Short definition, e.g. $\delta = u_y(L, H/2)$
    "standard": 1.0,    # Single-line with subscripts/Greek, ~80 chars
    "moderate": 1.5,    # Fractions, sums, or multi-term, ~100-200 chars
    "complex": 2.0,     # Multi-line, matrices, cases, align, >200 chars
}

# Limits by tutorial type (extensible)
# suggested_time is a soft limit; tutorials can exceed with justification
TYPE_LIMITS = {
    "concept": {
        "target_time": 10,
        "suggested_time": 15,
        "max_lines": 150,
        "max_code_blocks_visible": 2,
        "max_code_blocks_hidden": 999,  # unlimited
        "max_equations": 6,
        "max_figures": 6,
    },
    "usage": {
        "target_time": 10,
        "suggested_time": 15,
        "max_lines": 200,
        "max_code_blocks_visible": 8,
        "max_code_blocks_hidden": 3,
        "max_equations": 3,
        "max_figures": 6,
    },
    "analysis": {
        "target_time": 20,
        "suggested_time": 30,
        "max_lines": 350,
        "max_code_blocks_visible": 6,
        "max_code_blocks_hidden": 3,
        "max_equations": 15,
        "max_figures": 8,
    },
}

# Default limits for unknown types
DEFAULT_LIMITS = {
    "target_time": 10,
    "suggested_time": 15,
    "max_lines": 175,
    "max_code_blocks_visible": 5,
    "max_code_blocks_hidden": 5,
    "max_equations": 8,
    "max_figures": 6,
}


@dataclass
class TutorialMetrics:
    """Metrics extracted from a tutorial file."""

    filepath: Path
    title: str = ""
    tutorial_type: str = ""
    declared_time: Optional[int] = None
    extended_time_reason: str = ""
    difficulty: str = ""
    topic: str = ""

    # Content counts
    total_lines: int = 0
    code_blocks_visible: int = 0
    code_blocks_hidden: int = 0
    equations: int = 0
    equation_blocks: int = 0
    equation_difficulties: list = field(default_factory=list)
    figures_generated: int = 0
    figures_static: int = 0
    prose_paragraphs: int = 0
    headings: int = 0
    list_items: int = 0
    tables: int = 0

    # Computed
    estimated_time: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def total_figures(self) -> int:
        return self.figures_generated + self.figures_static

    @property
    def total_equations(self) -> int:
        return self.equations + self.equation_blocks

    def _equation_time(self) -> float:
        """Compute total equation time using per-equation difficulty."""
        return sum(
            EQUATION_DIFFICULTY[d] for d in self.equation_difficulties
        )

    def compute_time(self) -> float:
        """Compute estimated presentation time in minutes."""
        time = 0.0
        time += self._equation_time()
        time += self.code_blocks_visible * TIME_WEIGHTS["code_block_visible"]
        time += self.code_blocks_hidden * TIME_WEIGHTS["code_block_hidden"]
        time += self.total_figures * TIME_WEIGHTS["figure"]
        time += self.prose_paragraphs * TIME_WEIGHTS["prose_paragraph"]
        time += self.headings * TIME_WEIGHTS["heading"]
        time += self.list_items * TIME_WEIGHTS["list_item"]
        time += self.tables * TIME_WEIGHTS["table"]
        self.estimated_time = round(time, 1)
        return self.estimated_time

    def get_limits(self) -> dict:
        """Get applicable limits based on tutorial type."""
        return TYPE_LIMITS.get(self.tutorial_type, DEFAULT_LIMITS)

    def check_limits(self) -> list:
        """Check if tutorial exceeds any limits. Returns list of violations."""
        limits = self.get_limits()
        violations = []
        type_name = self.tutorial_type or "default"

        # Time check - suggested limit, not hard limit
        if self.estimated_time > limits["suggested_time"]:
            if self.extended_time_reason:
                # Has justification - warning only, not violation
                pass
            else:
                violations.append(
                    f"Time {self.estimated_time:.1f} min exceeds suggested "
                    f"{limits['suggested_time']} min for {type_name} "
                    f"(add extended_time_reason if justified)"
                )
        elif self.estimated_time > limits["target_time"]:
            # Between target and suggested - just a note, not a violation
            self.warnings.append(
                f"Time {self.estimated_time:.1f} min exceeds target "
                f"{limits['target_time']} min for {type_name}"
            )

        if self.total_lines > limits["max_lines"]:
            violations.append(
                f"Lines {self.total_lines} exceeds "
                f"{limits['max_lines']} limit for {type_name}"
            )

        if self.code_blocks_visible > limits["max_code_blocks_visible"]:
            violations.append(
                f"Visible code blocks {self.code_blocks_visible} exceeds "
                f"{limits['max_code_blocks_visible']} limit for {type_name}"
            )

        if self.code_blocks_hidden > limits["max_code_blocks_hidden"]:
            violations.append(
                f"Hidden code blocks {self.code_blocks_hidden} exceeds "
                f"{limits['max_code_blocks_hidden']} limit for {type_name}"
            )

        if self.total_equations > limits["max_equations"]:
            violations.append(
                f"Equations {self.total_equations} exceeds "
                f"{limits['max_equations']} limit for {type_name}"
            )

        if self.total_figures > limits["max_figures"]:
            violations.append(
                f"Figures {self.total_figures} exceeds "
                f"{limits['max_figures']} limit for {type_name}"
            )

        return violations

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "filepath": str(self.filepath),
            "title": self.title,
            "tutorial_type": self.tutorial_type,
            "declared_time": self.declared_time,
            "extended_time_reason": self.extended_time_reason,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "metrics": {
                "total_lines": self.total_lines,
                "code_blocks_visible": self.code_blocks_visible,
                "code_blocks_hidden": self.code_blocks_hidden,
                "equations": self.equations,
                "equation_blocks": self.equation_blocks,
                "equation_difficulties": self.equation_difficulties,
                "figures_generated": self.figures_generated,
                "figures_static": self.figures_static,
                "prose_paragraphs": self.prose_paragraphs,
                "headings": self.headings,
                "list_items": self.list_items,
                "tables": self.tables,
            },
            "estimated_time": self.estimated_time,
            "limits": self.get_limits(),
            "violations": self.check_limits(),
            "warnings": self.warnings,
        }


def parse_yaml_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from Quarto document."""
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    yaml_content = match.group(1)
    metadata = {}

    # Simple YAML parsing for common fields
    patterns = {
        "title": r'^title:\s*["\']?([^"\'\n]+)["\']?\s*$',
        "tutorial_type": r'^tutorial_type:\s*(\w+)\s*$',
        "estimated_time": r'^estimated_time:\s*(\d+)\s*$',
        "difficulty": r'^difficulty:\s*(\w+)\s*$',
        "topic": r'^topic:\s*([\w_-]+)\s*$',
        "extended_time_reason": r'^extended_time_reason:\s*["\']?([^"\'\n]+)["\']?\s*$',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, yaml_content, re.MULTILINE)
        if match:
            value = match.group(1)
            if key == "estimated_time":
                metadata[key] = int(value)
            else:
                metadata[key] = value

    return metadata


def _remove_hidden_code_blocks(content: str) -> str:
    """Remove hidden code blocks from content for line counting.

    Hidden code blocks (those with ``#| echo: false``) generate figures
    but do not contribute to presentation content or cognitive load.
    """
    def _is_hidden(block_content: str) -> bool:
        return bool(re.search(r"#\|\s*echo:\s*false", block_content))

    pattern = r"```\{python[^}]*\}\s*\n(.*?)```"
    result = content
    for match in reversed(list(re.finditer(pattern, content, re.DOTALL))):
        block_body = match.group(1)
        if _is_hidden(block_body):
            result = result[:match.start()] + result[match.end():]
    return result


def count_code_blocks(content: str) -> tuple:
    """Count visible and hidden code blocks.

    Returns (visible_count, hidden_count, figures_from_hidden).
    """
    # Match code blocks: ```{python} ... ```
    pattern = r"```\{python[^}]*\}\s*\n(.*?)```"
    blocks = re.findall(pattern, content, re.DOTALL)

    visible = 0
    hidden = 0
    figures_from_hidden = 0

    for block in blocks:
        # Check if block has #| echo: false
        if re.search(r"#\|\s*echo:\s*false", block):
            hidden += 1
            # Check if it generates a figure
            if re.search(r"(plt\.show|fig\.show|display|#\|\s*fig-cap)", block):
                figures_from_hidden += 1
        else:
            visible += 1

    return visible, hidden, figures_from_hidden


def _classify_equation_difficulty(eq_text: str) -> str:
    """Classify a display equation block into a difficulty tier.

    Uses heuristics based on character length, notation complexity
    (fractions, sums/integrals, matrices), multi-line structure, and
    term count to estimate how long a presenter needs to explain it.

    Parameters
    ----------
    eq_text : str
        Raw LaTeX content between $$ delimiters (or \\begin/\\end).

    Returns
    -------
    str
        One of "simple", "standard", "moderate", "complex".
    """
    text = eq_text.strip()
    char_len = len(text)
    nlines = len([l for l in text.splitlines() if l.strip()])

    # Feature detection
    has_sum_int = bool(re.search(r"\\(sum|prod|int|iint|oint)", text))
    has_frac = bool(re.search(r"\\frac", text))
    has_matrix = bool(re.search(
        r"\\begin\{[bp]?matrix\}|\\begin\{cases\}", text
    ))
    n_align_breaks = text.count("\\\\")
    n_terms = text.count("+") + text.count("-")

    # Score: accumulate complexity points
    score = 0

    # Length contribution
    if char_len > 200:
        score += 3
    elif char_len > 120:
        score += 2
    elif char_len > 60:
        score += 1

    # Notation complexity
    if has_matrix:
        score += 3
    if has_sum_int:
        score += 2
    if has_frac:
        score += 1

    # Multi-line / alignment
    if n_align_breaks >= 3:
        score += 3
    elif n_align_breaks >= 1:
        score += 1

    # Many terms
    if n_terms > 8:
        score += 2
    elif n_terms > 4:
        score += 1

    # Multi-line source
    if nlines > 4:
        score += 1

    # Map score to tier
    if score >= 6:
        return "complex"
    elif score >= 3:
        return "moderate"
    elif score >= 1:
        return "standard"
    else:
        return "simple"


def count_equations(content: str) -> tuple:
    """Count inline equations and equation blocks with difficulty.

    Returns (inline_count, block_difficulties) where block_difficulties
    is a list of difficulty tier strings for each display equation block.
    """
    # Remove code blocks first to avoid false positives
    content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Equation blocks: $$ ... $$ or \begin{equation/align/gather}
    block_patterns = [
        (r"\$\$(.*?)\$\$", re.DOTALL),
        (r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", re.DOTALL),
        (r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", re.DOTALL),
        (r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", re.DOTALL),
    ]
    block_difficulties = []
    matched_spans = []
    for pattern, flags in block_patterns:
        for m in re.finditer(pattern, content_no_code, flags):
            block_difficulties.append(
                _classify_equation_difficulty(m.group(1))
            )
            matched_spans.append(m.span())

    # Remove blocks before counting inline
    for pattern, flags in block_patterns:
        content_no_code = re.sub(pattern, "", content_no_code, flags=flags)

    # Inline equations: $ ... $ (not $$)
    inline_pattern = r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)"
    inline_count = len(re.findall(inline_pattern, content_no_code))

    return inline_count, block_difficulties


def count_static_figures(content: str) -> int:
    """Count static images (markdown image syntax)."""
    # ![caption](path) or ![](path)
    return len(re.findall(r"!\[[^\]]*\]\([^)]+\)", content))


def count_prose_paragraphs(content: str) -> int:
    """Count prose paragraphs (excluding code, math, lists, callouts).

    Only counts substantial prose blocks that require presentation time.
    Short lead-in sentences before equations or figures are excluded.
    """
    # Remove YAML frontmatter
    content = re.sub(r"^---.*?---", "", content, flags=re.DOTALL, count=1)

    # Remove code blocks
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Remove equation blocks
    content = re.sub(r"\$\$.*?\$\$", "", content, flags=re.DOTALL)

    # Remove Quarto callout/div blocks (::: ... :::)
    content = re.sub(r":::.*?:::", "", content, flags=re.DOTALL)

    # Remove cross-reference lines (just "@fig-..." starting a paragraph)
    content = re.sub(r"^@\w+-\S+\s*$", "", content, flags=re.MULTILINE)

    # Split by blank lines and count non-empty, non-list paragraphs
    paragraphs = re.split(r"\n\s*\n", content)
    count = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Skip headings
        if para.startswith("#"):
            continue
        # Skip lists
        if para.startswith("-") or para.startswith("*") or re.match(r"^\d+\.", para):
            continue
        # Skip short fragments: single-line lead-ins to equations/figures
        # A real paragraph has at least ~80 chars of actual prose
        if len(para) < 80:
            continue
        count += 1

    return count


def count_headings(content: str) -> int:
    """Count section headings (excluding those inside code blocks)."""
    content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    return len(re.findall(r"^#{1,6}\s+.+$", content_no_code, re.MULTILINE))


def count_list_items(content: str) -> int:
    """Count bullet points and numbered list items (excluding code blocks)."""
    content_no_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    bullets = len(re.findall(r"^\s*[-*]\s+.+$", content_no_code, re.MULTILINE))
    numbered = len(re.findall(r"^\s*\d+\.\s+.+$", content_no_code, re.MULTILINE))
    return bullets + numbered


def count_tables(content: str) -> int:
    """Count Markdown tables."""
    # Tables have | separators and a header row with dashes
    return len(re.findall(r"^\|.*\|.*\n\|[-:\s|]+\|", content, re.MULTILINE))


def analyze_tutorial(filepath: Path) -> TutorialMetrics:
    """Analyze a tutorial file and return metrics."""
    metrics = TutorialMetrics(filepath=filepath)

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        metrics.warnings.append(f"Could not read file: {e}")
        return metrics

    # Parse metadata
    metadata = parse_yaml_frontmatter(content)
    metrics.title = metadata.get("title", filepath.stem)
    metrics.tutorial_type = metadata.get("tutorial_type", "")
    metrics.declared_time = metadata.get("estimated_time")
    metrics.extended_time_reason = metadata.get("extended_time_reason", "")
    metrics.difficulty = metadata.get("difficulty", "")
    metrics.topic = metadata.get("topic", "")

    # Count content elements
    # Exclude hidden code blocks from line count (they don't add to
    # presentation time or cognitive load)
    content_no_hidden = _remove_hidden_code_blocks(content)
    metrics.total_lines = len(content_no_hidden.splitlines())

    visible, hidden, figs_from_hidden = count_code_blocks(content)
    metrics.code_blocks_visible = visible
    metrics.code_blocks_hidden = hidden
    metrics.figures_generated = figs_from_hidden

    _inline_eq, block_difficulties = count_equations(content)
    # Only count display equation blocks, not inline math references
    # like $E_1$ or $\delta_{\text{tip}}$ which are variable names
    metrics.equations = 0
    metrics.equation_blocks = len(block_difficulties)
    metrics.equation_difficulties = block_difficulties

    metrics.figures_static = count_static_figures(content)
    metrics.prose_paragraphs = count_prose_paragraphs(content)
    metrics.headings = count_headings(content)
    metrics.list_items = count_list_items(content)
    metrics.tables = count_tables(content)

    # Compute estimated time
    metrics.compute_time()

    # Add warnings
    if not metrics.tutorial_type:
        metrics.warnings.append("No tutorial_type specified in frontmatter")

    if metrics.declared_time and abs(metrics.declared_time - metrics.estimated_time) > 5:
        metrics.warnings.append(
            f"Declared time ({metrics.declared_time} min) differs significantly "
            f"from estimated ({metrics.estimated_time:.1f} min)"
        )

    return metrics


def format_report(metrics: TutorialMetrics, verbose: bool = False) -> str:
    """Format a human-readable report for a tutorial."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Tutorial: {metrics.filepath.name}")
    lines.append(f"Title: {metrics.title}")
    lines.append(f"{'='*60}")

    if metrics.tutorial_type:
        lines.append(f"Type: {metrics.tutorial_type}")
    if metrics.difficulty:
        lines.append(f"Difficulty: {metrics.difficulty}")

    lines.append(f"\nEstimated presentation time: {metrics.estimated_time:.1f} min")
    if metrics.declared_time:
        lines.append(f"Declared time in YAML: {metrics.declared_time} min")

    if verbose:
        lines.append("\nContent breakdown:")
        lines.append(f"  Total lines (excl. hidden code): {metrics.total_lines}")
        lines.append(f"  Code blocks (visible): {metrics.code_blocks_visible}")
        lines.append(f"  Code blocks (hidden): {metrics.code_blocks_hidden}")
        lines.append(f"  Equation blocks: {metrics.equation_blocks}")
        lines.append(f"  Figures (generated): {metrics.figures_generated}")
        lines.append(f"  Figures (static): {metrics.figures_static}")
        lines.append(f"  Prose paragraphs: {metrics.prose_paragraphs}")
        lines.append(f"  Headings: {metrics.headings}")
        lines.append(f"  List items: {metrics.list_items}")
        lines.append(f"  Tables: {metrics.tables}")

        lines.append("\nTime calculation:")
        eq_time = metrics._equation_time()
        if metrics.equation_difficulties:
            # Show per-equation difficulty breakdown
            diff_summary = ", ".join(
                f"{d}({EQUATION_DIFFICULTY[d]})"
                for d in metrics.equation_difficulties
            )
            lines.append(
                f"  Equations: {metrics.equation_blocks} blocks "
                f"[{diff_summary}] = {eq_time:.1f} min"
            )
        else:
            lines.append(f"  Equations: 0 blocks = 0.0 min")
        lines.append(
            f"  Visible code: {metrics.code_blocks_visible} × "
            f"{TIME_WEIGHTS['code_block_visible']} = "
            f"{metrics.code_blocks_visible * TIME_WEIGHTS['code_block_visible']:.1f} min"
        )
        lines.append(
            f"  Hidden code: {metrics.code_blocks_hidden} × "
            f"{TIME_WEIGHTS['code_block_hidden']} = "
            f"{metrics.code_blocks_hidden * TIME_WEIGHTS['code_block_hidden']:.1f} min"
        )
        lines.append(
            f"  Figures: {metrics.total_figures} × "
            f"{TIME_WEIGHTS['figure']} = "
            f"{metrics.total_figures * TIME_WEIGHTS['figure']:.1f} min"
        )
        lines.append(
            f"  Paragraphs: {metrics.prose_paragraphs} × "
            f"{TIME_WEIGHTS['prose_paragraph']} = "
            f"{metrics.prose_paragraphs * TIME_WEIGHTS['prose_paragraph']:.1f} min"
        )

    # Check limits
    violations = metrics.check_limits()
    if violations:
        lines.append("\n⚠️  LIMIT VIOLATIONS:")
        for v in violations:
            lines.append(f"  - {v}")

    if metrics.warnings:
        lines.append("\n⚠️  WARNINGS:")
        for w in metrics.warnings:
            lines.append(f"  - {w}")

    limits = metrics.get_limits()
    type_name = metrics.tutorial_type or "default"
    lines.append(f"\nLimits ({type_name}):")
    lines.append(f"  Target time: {limits['target_time']} min")
    lines.append(f"  Suggested limit: {limits['suggested_time']} min")
    lines.append(f"  Max lines: {limits['max_lines']}")
    lines.append(f"  Max visible code: {limits['max_code_blocks_visible']}")
    lines.append(f"  Max equations: {limits['max_equations']}")
    lines.append(f"  Max figures: {limits['max_figures']}")

    if metrics.extended_time_reason:
        lines.append(f"\nExtended time justification: {metrics.extended_time_reason}")

    return "\n".join(lines)


def format_summary_table(all_metrics: list) -> str:
    """Format a summary table of all tutorials."""
    lines = []
    lines.append("\n" + "=" * 85)
    lines.append("SUMMARY")
    lines.append("=" * 85)

    # Header
    header = f"{'Tutorial':<35} {'Type':<10} {'Est.':<6} {'Decl.':<6} {'Code':<6} {'Eq':<4} {'Status'}"
    lines.append(header)
    lines.append("-" * 85)

    total_time = 0
    violations_count = 0

    for m in sorted(all_metrics, key=lambda x: x.filepath.name):
        name = m.filepath.stem[:33]
        ttype = (m.tutorial_type[:8] if m.tutorial_type else "-")
        est = f"{m.estimated_time:.0f}"
        decl = str(m.declared_time) if m.declared_time else "-"
        code = f"{m.code_blocks_visible}+{m.code_blocks_hidden}"
        eq = str(m.total_equations)
        violations = m.check_limits()
        status = "⚠️ " + str(len(violations)) if violations else "✓"
        if violations:
            violations_count += 1

        lines.append(f"{name:<35} {ttype:<10} {est:<6} {decl:<6} {code:<6} {eq:<4} {status}")
        total_time += m.estimated_time

    lines.append("-" * 85)
    lines.append(f"Total tutorials: {len(all_metrics)}")
    lines.append(f"Total estimated time: {total_time:.0f} min ({total_time/60:.1f} hours)")
    lines.append(f"Tutorials with violations: {violations_count}")
    lines.append("\nCode column shows: visible+hidden blocks")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate presentation time for PyApprox tutorials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="+",
        help="Tutorial file(s) (.qmd) or directory containing tutorials",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed breakdown of time calculation",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any tutorial exceeds limits",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary table, not individual reports",
    )

    args = parser.parse_args()

    # Collect tutorial files
    files = []
    for path in args.path:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            dir_files = sorted(path.glob("*.qmd"))
            # Exclude index files
            dir_files = [f for f in dir_files if not f.name.startswith("index")]
            files.extend(dir_files)
        else:
            print(f"Warning: {path} is not a file or directory", file=sys.stderr)

    if not files:
        print("No .qmd files found", file=sys.stderr)
        sys.exit(1)

    # Analyze all tutorials
    all_metrics = [analyze_tutorial(f) for f in files]

    # Output results
    if args.json:
        output = {
            "tutorials": [m.to_dict() for m in all_metrics],
            "summary": {
                "total_tutorials": len(all_metrics),
                "total_time": sum(m.estimated_time for m in all_metrics),
                "violations_count": sum(1 for m in all_metrics if m.check_limits()),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        if not args.summary_only:
            for m in all_metrics:
                print(format_report(m, verbose=args.verbose))

        if len(all_metrics) > 1 or args.summary_only:
            print(format_summary_table(all_metrics))

    # Check mode: exit with error if violations found
    if args.check:
        violations_count = sum(1 for m in all_metrics if m.check_limits())
        if violations_count > 0:
            print(f"\n❌ {violations_count} tutorial(s) exceed limits", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n✓ All tutorials within limits")


if __name__ == "__main__":
    main()
