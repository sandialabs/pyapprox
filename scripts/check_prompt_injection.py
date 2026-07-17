#!/usr/bin/env python3
"""Pre-commit scanner for hidden-Unicode / prompt-injection payloads.

Detects (ERROR — blocks the commit):
  * Zero-width / invisible characters (ZWSP, ZWNJ, ZWJ, word joiner, soft
    hyphen, etc.) that can hide text from human reviewers.
  * Bidirectional control characters ("Trojan Source", CVE-2021-42574) that
    make displayed code differ from what compilers/LLMs actually read.
  * Unicode "tag" characters (U+E0000-U+E007F) — invisible characters used to
    smuggle ASCII payloads to LLMs. The hidden payload is decoded and shown.
  * A byte-order mark (U+FEFF) anywhere other than the start of the file.

Detects (WARNING — reported, only blocks with --strict):
  * Phrases characteristic of instructions aimed at an AI agent
    ("ignore previous ...", "do not tell ...", and similar).
  * Long runs of variation selectors (a known data-smuggling channel).

Usage:
  python check_prompt_injection.py FILE [FILE ...]   # scan given files
  python check_prompt_injection.py --staged          # scan git staged files
  python check_prompt_injection.py --all             # whole repo (tracked +
                                                     #  untracked, not ignored)
  python check_prompt_injection.py --all ~/other     # any other repo or dir
  python check_prompt_injection.py --stdin LABEL     # scan piped text (PR
                                                     #  bodies, commit msgs)
  python check_prompt_injection.py --strict ...      # warnings also fail
  python check_prompt_injection.py --format github   # PR annotations in CI

How this scanner gets invoked (it is the detection engine only —
enforcement lives elsewhere):

  * Locally: by local_guard.py (personal hooks + the bundle command),
    installed per-machine in ~/.githooks/ — see INSTALL.md. Git hooks
    are deliberately NOT versioned in this repo: a repo-tracked
    .githooks/ directory is itself a blocked path under the security
    policy, and hooks are per-machine self-protection only (skippable
    with --no-verify; the bundle step and CI re-verify regardless).
  * Server-side: by .github/workflows/prompt-injection-scan.yml on
    every PR (changed files, PR title/body, commit messages) and in a
    weekly full-repo sweep.
  * Manually: `--all` / `--all --no-ignore` for a peace-of-mind sweep
    of this or any other repo.

To intentionally keep a flagged line (e.g., test fixtures), add the marker
"prompt-injection-check: allow" in a comment on that line, or skip a whole
file with --exclude GLOB.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ALLOW_MARKER = "prompt-injection-check: allow"
MAX_FINDINGS_PER_FILE = 25
MAX_FILE_BYTES = 5 * 1024 * 1024  # skip files larger than 5 MB

# --- character classes -------------------------------------------------------

INVISIBLE_CHARS = {
    0x00AD: "SOFT HYPHEN",
    0x034F: "COMBINING GRAPHEME JOINER",
    0x115F: "HANGUL CHOSEONG FILLER",
    0x1160: "HANGUL JUNGSEONG FILLER",
    0x17B4: "KHMER VOWEL INHERENT AQ",
    0x17B5: "KHMER VOWEL INHERENT AA",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x2060: "WORD JOINER",
    0x2061: "FUNCTION APPLICATION",
    0x2062: "INVISIBLE TIMES",
    0x2063: "INVISIBLE SEPARATOR",
    0x2064: "INVISIBLE PLUS",
    0x3164: "HANGUL FILLER",
    0xFFA0: "HALFWIDTH HANGUL FILLER",
}

BIDI_CHARS = {
    0x061C: "ARABIC LETTER MARK",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
}

TAG_RANGE = range(0xE0000, 0xE0080)          # invisible "tag" characters
VARIATION_SELECTORS = (
    set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))
)
VS_RUN_THRESHOLD = 3  # >= this many consecutive VS chars -> warning

# --- AI-directed phrase heuristics (case-insensitive) ------------------------

PHRASE_PATTERNS = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in [
        (r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all)\b"
         r".{0,20}\binstruction", "override-instructions phrase"),
        (r"\byou\s+are\s+(now\s+)?(an?\s+)?(ai|llm|assistant|agent|claude|"
         r"copilot|chatbot)\b", "AI role-assignment phrase"),
        (r"\b(do\s+not|don'?t|never)\s+(tell|inform|reveal\s+to|mention\s+to|"
         r"alert|warn)\s+(the\s+)?(user|human|developer|owner)\b",
         "conceal-from-user phrase"),
        (r"\bsystem\s*prompt\b", "system-prompt reference"),
        (r"<\s*/?\s*(system|assistant_instructions?|hidden_instructions?)\s*>",
         "fake system tag"),
        (r"\b(dear|attention|note\s+to|important\s+for|instructions?\s+for)\s+"
         r"(ai|llm|assistant|agent|claude|copilot|gpt|model)s?\b",
         "AI-addressed preamble"),
        (r"\bwhen\s+(an?\s+)?(ai|llm|assistant|agent)\s+(reads?|processes?|"
         r"sees?)\s+this\b", "AI-conditional phrase"),
        (r"\bcurl\b[^\n]{0,120}\|\s*(sh|bash|zsh|python3?|node)\b",
         "pipe-to-shell pattern"),
    ]
]


def is_probably_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def decode_tag_run(chars: list[int]) -> str:
    """Decode a run of tag characters back to the ASCII they smuggle."""
    out = []
    for cp in chars:
        base = cp - 0xE0000
        if 0x20 <= base <= 0x7E:
            out.append(chr(base))
    return "".join(out)


def scan_text(text: str, path: str) -> list[tuple[str, int, str]]:
    """Return findings as (severity, line_number, message)."""
    findings: list[tuple[str, int, str]] = []
    lines = text.split("\n")
    offset = 0  # absolute char offset of line start (for BOM check)

    for lineno, line in enumerate(lines, start=1):
        if ALLOW_MARKER in line:
            offset += len(line) + 1
            continue

        tag_run: list[int] = []
        vs_run = 0
        for col, ch in enumerate(line, start=1):
            cp = ord(ch)

            if cp in TAG_RANGE:
                tag_run.append(cp)
                continue
            if tag_run:
                payload = decode_tag_run(tag_run)
                findings.append(("ERROR", lineno,
                                 f"col {col - len(tag_run)}: {len(tag_run)} invisible "
                                 f"Unicode TAG characters — hidden payload: "
                                 f"{payload!r}"))
                tag_run = []

            if cp in VARIATION_SELECTORS:
                vs_run += 1
                continue
            if vs_run:
                if vs_run >= VS_RUN_THRESHOLD:
                    findings.append(("WARNING", lineno,
                                     f"col {col - vs_run}: run of {vs_run} "
                                     f"variation selectors (possible data "
                                     f"smuggling)"))
                vs_run = 0

            if cp in INVISIBLE_CHARS:
                findings.append(("ERROR", lineno,
                                 f"col {col}: invisible character U+{cp:04X} "
                                 f"({INVISIBLE_CHARS[cp]})"))
            elif cp in BIDI_CHARS:
                findings.append(("ERROR", lineno,
                                 f"col {col}: bidi control U+{cp:04X} "
                                 f"({BIDI_CHARS[cp]}) — Trojan Source risk"))
            elif cp == 0xFEFF and (offset + col - 1) != 0:
                findings.append(("ERROR", lineno,
                                 f"col {col}: U+FEFF (BOM/zero-width no-break "
                                 f"space) not at start of file"))

        # flush runs that end at end-of-line
        if tag_run:
            payload = decode_tag_run(tag_run)
            findings.append(("ERROR", lineno,
                             f"{len(tag_run)} invisible Unicode TAG characters "
                             f"at end of line — hidden payload: {payload!r}"))
        if vs_run >= VS_RUN_THRESHOLD:
            findings.append(("WARNING", lineno,
                             f"run of {vs_run} variation selectors at end of "
                             f"line (possible data smuggling)"))

        for pattern, label in PHRASE_PATTERNS:
            m = pattern.search(line)
            if m:
                snippet = m.group(0)
                if len(snippet) > 80:
                    snippet = snippet[:77] + "..."
                findings.append(("WARNING", lineno,
                                 f"suspicious {label}: {snippet!r}"))

        offset += len(line) + 1
        if len(findings) >= MAX_FINDINGS_PER_FILE:
            findings.append(("WARNING", lineno,
                             f"more than {MAX_FINDINGS_PER_FILE} findings; "
                             f"output truncated"))
            break

    return findings


def emit(severity: str, path: str, lineno: int, msg: str, fmt: str) -> None:
    if fmt == "github":
        level = "error" if severity == "ERROR" else "warning"
        gh_msg = (msg.replace("%", "%25")
                     .replace("\r", "%0D")
                     .replace("\n", "%0A"))
        print(f"::{level} file={path},line={lineno},"
              f"title=prompt-injection scan::{gh_msg}")
    else:
        print(f"{severity}: {path}:{lineno}: {msg}")


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.split("\0") if f]


SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__",
             ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache"}


def repo_files(root: str, include_ignored: bool = False) -> list[str]:
    """All files worth scanning under root.

    In a git repo: all tracked files (including any that were force-added
    with `git add -f` despite .gitignore) plus untracked files. Untracked
    files matched by .gitignore are skipped unless include_ignored is True
    (tracked files are always scanned — gitignore can't hide them).
    Elsewhere: a plain directory walk, skipping VCS/venv/cache directories.
    """
    try:
        cmd = ["git", "-C", root, "ls-files", "-z", "--cached", "--others"]
        if not include_ignored:
            cmd.append("--exclude-standard")
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        ).stdout
        return [str(Path(root) / f) for f in out.split("\0") if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            str(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and not (SKIP_DIRS & set(p.parts))
        ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="files to scan")
    ap.add_argument("--staged", action="store_true",
                    help="scan files staged in git instead of listed files")
    ap.add_argument("--all", nargs="?", const=".", default=None,
                    metavar="DIR", dest="all_dir",
                    help="scan an entire repo/directory (default: current "
                         "repo); e.g. --all or --all ~/some-other-repo")
    ap.add_argument("--stdin", metavar="LABEL", default=None,
                    help="scan text piped on stdin, reporting findings under "
                         "LABEL (for PR descriptions, commit messages, etc.)")
    ap.add_argument("--no-ignore", action="store_true",
                    help="with --all: also scan untracked files matched by "
                         ".gitignore (downloads, data, build output that an "
                         "agent might still read)")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors")
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="GLOB", help="glob(s) of paths to skip")
    ap.add_argument("--format", choices=["text", "github"], default="text",
                    help="'github' emits ::error/::warning workflow commands "
                         "that annotate the PR diff in GitHub Actions")
    args = ap.parse_args()

    n_errors = n_warnings = 0

    if args.stdin is not None:
        for severity, lineno, msg in scan_text(sys.stdin.read(), args.stdin):
            emit(severity, args.stdin, lineno, msg, args.format)
            if severity == "ERROR":
                n_errors += 1
            else:
                n_warnings += 1

    if args.staged:
        paths = staged_files()
    elif args.all_dir is not None:
        paths = repo_files(args.all_dir, include_ignored=args.no_ignore)
    else:
        paths = args.files

    for path in paths:
        if any(fnmatch.fnmatch(path, g) for g in args.exclude):
            continue
        p = Path(path)
        if not p.is_file():
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            data = p.read_bytes()
        except OSError as e:
            print(f"{path}: unreadable ({e}); skipping", file=sys.stderr)
            continue
        if is_probably_binary(data):
            continue
        text = data.decode("utf-8", errors="replace")

        for severity, lineno, msg in scan_text(text, path):
            emit(severity, path, lineno, msg, args.format)
            if severity == "ERROR":
                n_errors += 1
            else:
                n_warnings += 1

    if n_errors or n_warnings:
        print(f"\nprompt-injection scan: {n_errors} error(s), "
              f"{n_warnings} warning(s)")
        if n_errors or (args.strict and n_warnings):
            print("Check failed. If a finding is intentional, add "
                  f"'{ALLOW_MARKER}' in a comment on that line.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
