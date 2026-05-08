#!/usr/bin/env python3
"""Generate workshop index.qmd and _quarto.yml from workshop.yml configuration.

Usage:
    python generate_workshop_index.py <workshop_name>

Example:
    python generate_workshop_index.py intro_to_uq_2025

This reads workshops/<workshop_name>/workshop.yml and generates:
    - workshops/<workshop_name>/index.qmd
    - workshops/<workshop_name>/_quarto.yml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_workshop_config(workshop_dir: Path) -> Dict[str, Any]:
    """Load workshop.yml configuration."""
    config_path = workshop_dir / "workshop.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Workshop config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_index_qmd(config: Dict[str, Any], workshop_dir: Path) -> str:
    """Generate index.qmd content from workshop config."""
    name = config.get("name", "Untitled Workshop")
    date = config.get("date", "TBD")
    duration = config.get("duration", "TBD")
    description = config.get("description", "")
    prerequisites = config.get("attendee_prerequisites", [])
    tutorials = config.get("tutorials", [])
    optional_tutorials = config.get("optional_tutorials", [])
    resources = config.get("resources", [])

    lines = [
        "---",
        f'title: "{name}"',
        'subtitle: "PyApprox Workshop"',
        "---",
        "",
        "## Workshop Overview",
        "",
        f"**Date:** {date}",
        f"**Duration:** {duration}",
        "",
        description,
        "",
        "## Prerequisites",
        "",
    ]

    for prereq in prerequisites:
        lines.append(f"- {prereq}")

    lines.extend([
        "",
        "## Schedule",
        "",
        "| Time | Topic | Duration |",
        "|------|-------|----------|",
    ])

    # Track cumulative time for schedule
    cumulative_minutes = 0
    for item in tutorials:
        duration_min = item.get("duration_minutes", 0)

        if item.get("type") == "break":
            topic = "**Break**"
            link = ""
        else:
            tutorial_id = item.get("id", "unknown")
            topic = f"[{_format_tutorial_name(tutorial_id)}](../../library/{tutorial_id}.qmd)"

        time_str = _format_time(cumulative_minutes)
        lines.append(f"| {time_str} | {topic} | {duration_min} min |")
        cumulative_minutes += duration_min

    # Add total duration
    lines.append(f"| | **Total** | **{cumulative_minutes} min** |")

    # Optional tutorials
    if optional_tutorials:
        lines.extend([
            "",
            "## Optional Extensions",
            "",
            "If time permits:",
            "",
        ])
        for item in optional_tutorials:
            tutorial_id = item.get("id", "unknown")
            notes = item.get("notes", "")
            lines.append(
                f"- [{_format_tutorial_name(tutorial_id)}](../../library/{tutorial_id}.qmd)"
                + (f" - {notes}" if notes else "")
            )

    # Resources
    if resources:
        lines.extend([
            "",
            "## Resources",
            "",
        ])
        for resource in resources:
            rtype = resource.get("type", "link")
            url = resource.get("url", "")
            if rtype == "github_repo":
                lines.append(f"- [GitHub Repository]({url})")
            elif rtype == "documentation":
                lines.append(f"- [Documentation]({url})")
            else:
                lines.append(f"- [{rtype}]({url})")

    # Instructor notes section
    lines.extend([
        "",
        "## Instructor Notes",
        "",
        "Detailed notes for each tutorial section:",
        "",
    ])

    for item in tutorials:
        if item.get("type") == "break":
            continue
        tutorial_id = item.get("id", "unknown")
        notes = item.get("notes", "No specific notes.")
        lines.append(f"### {_format_tutorial_name(tutorial_id)}")
        lines.append("")
        lines.append(notes)
        lines.append("")

    return "\n".join(lines)


def generate_quarto_yml(config: Dict[str, Any]) -> str:
    """Generate _quarto.yml content from workshop config."""
    name = config.get("name", "Workshop")
    tutorials = config.get("tutorials", [])

    # Build sidebar contents
    sidebar_contents = []
    for item in tutorials:
        if item.get("type") != "break":
            tutorial_id = item.get("id", "unknown")
            sidebar_contents.append(f"../../library/{tutorial_id}.qmd")

    yml_content = {
        "project": {
            "type": "website",
            "output-dir": "_site"
        },
        "website": {
            "title": name,
            "navbar": {
                "left": [
                    {"text": "Schedule", "href": "index.qmd"},
                    {"text": "Tutorial Library", "href": "../../library/index.qmd"}
                ]
            },
            "sidebar": {
                "style": "docked",
                "contents": ["index.qmd"] + sidebar_contents
            }
        },
        "format": {
            "html": {
                "theme": "cosmo",
                "toc": True,
                "include-in-header": [
                    {"file": "../../_macros_html.tex"}
                ]
            }
        },
        "execute": {
            "echo": True,
            "warning": False
        }
    }

    return yaml.dump(yml_content, default_flow_style=False, sort_keys=False)


def _format_tutorial_name(tutorial_id: str) -> str:
    """Convert tutorial_id to display name."""
    return tutorial_id.replace("_", " ").title()


def _format_time(minutes: int) -> str:
    """Format cumulative minutes as time offset."""
    hours = minutes // 60
    mins = minutes % 60
    if hours > 0:
        return f"+{hours}:{mins:02d}"
    return f"+{mins} min"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate workshop index.qmd and _quarto.yml from workshop.yml"
    )
    parser.add_argument(
        "workshop_name",
        help="Name of workshop directory (e.g., intro_to_uq_2025)"
    )
    parser.add_argument(
        "--workshops-dir",
        type=Path,
        default=None,
        help="Path to workshops directory (default: auto-detect)"
    )

    args = parser.parse_args()

    # Find workshops directory
    if args.workshops_dir:
        workshops_dir = args.workshops_dir
    else:
        # Try to find relative to script location
        script_dir = Path(__file__).parent
        workshops_dir = script_dir.parent / "workshops"

    workshop_dir = workshops_dir / args.workshop_name

    if not workshop_dir.exists():
        print(f"Error: Workshop directory not found: {workshop_dir}")
        return 1

    try:
        config = load_workshop_config(workshop_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"Error parsing workshop.yml: {e}")
        return 1

    # Generate index.qmd
    index_content = generate_index_qmd(config, workshop_dir)
    index_path = workshop_dir / "index.qmd"
    with open(index_path, "w") as f:
        f.write(index_content)
    print(f"Generated: {index_path}")

    # Generate _quarto.yml
    quarto_content = generate_quarto_yml(config)
    quarto_path = workshop_dir / "_quarto.yml"
    with open(quarto_path, "w") as f:
        f.write(quarto_content)
    print(f"Generated: {quarto_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
