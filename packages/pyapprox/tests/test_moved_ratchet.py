"""The moved-module ratchet and added-lines check: references to old
prefixes are counted as whole dotted names, compared with
ratchet_baselines.txt, and refused on staged added lines."""

import shutil
import subprocess  # nosec B404: runs git with fixed arguments in a tmp repo
from pathlib import Path

import pytest

import pyapprox
from pyapprox._moved import MOVED
from pyapprox._moved_ratchet import (
    BASELINE_FILE,
    added_line_problems,
    count_references,
    main,
    ratchet_problems,
    read_baseline,
)

_TABLE = {"pkg.old": "pkg.new", "pkg.old.deep": "pkg.deeper"}


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_counts_whole_dotted_names_in_scanned_files(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/a.py",
        "from pkg.old import x\n"
        "import pkg.old.mod as m\n"
        "from pkg.old.deep.sub import y\n"
        "patch('pkg.old.f')\n"
        "from pkg.older import z\n"  # another name
        "from mypkg.old import w\n"  # another package
        "from pkg.new import v\n",
    )
    _write(tmp_path, "docs/page.qmd", "See `pkg.old.thing`.\n")
    _write(tmp_path, "docs/notes.md", "pkg.old moved  # moved: allow\n")
    _write(tmp_path, "build/gen.py", "import pkg.old\n")  # skipped directory
    _write(tmp_path, "data/blob.bin", "pkg.old\n")  # not a scanned suffix
    _write(tmp_path, "src/pyapprox/_moved.py", "MOVED = {'pkg.old': 'pkg.new'}\n")
    _write(tmp_path, BASELINE_FILE, "moved:pkg.old=4\n")
    counts = count_references(_TABLE, tmp_path)
    # pkg.old.deep is counted under the more specific prefix.
    assert counts == {"pkg.old": 4, "pkg.old.deep": 1}
    assert count_references({}, tmp_path) == {}


def test_ratchet_holds_counts_at_their_baselines(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASELINE_FILE,
        "# comment\nmypy_src=362\nmoved:pkg.old=4\nmoved:gone.old=2\n",
    )
    baseline = read_baseline(tmp_path / BASELINE_FILE)
    assert baseline == {"pkg.old": 4, "gone.old": 2}
    table = {"pkg.old": "pkg.new", "pkg.other": "pkg.elsewhere"}
    problems = ratchet_problems(table, {"pkg.old": 5, "pkg.other": 3}, baseline)
    assert problems == [
        "pkg.old: 5 references, baseline 4; import the new path instead",
        "no baseline for pkg.other: add moved:pkg.other=3",
        "moved:gone.old is stale: gone.old is not moved",
    ]
    fell = ratchet_problems({"pkg.old": "pkg.new"}, {"pkg.old": 1}, {"pkg.old": 4})
    assert fell == ["pkg.old: down to 1; set moved:pkg.old=1"]
    assert (
        ratchet_problems({"pkg.old": "pkg.new"}, {"pkg.old": 4}, {"pkg.old": 4}) == []
    )


_DIFF = """\
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -3,0 +4,3 @@ def f():
+from pkg.new import x
+from pkg.old import y
+text = "pkg.old"  # moved: allow
@@ -9 +12 @@
-from pkg.old import z
+from pkg.new import z
diff --git a/src/pyapprox/_moved.py b/src/pyapprox/_moved.py
--- a/src/pyapprox/_moved.py
+++ b/src/pyapprox/_moved.py
@@ -1,0 +2 @@
+MOVED = {"pkg.old": "pkg.new"}
diff --git a/gone.py b/gone.py
--- a/gone.py
+++ /dev/null
@@ -1 +0,0 @@
-import pkg.old
"""


def test_added_lines_naming_old_paths_are_reported() -> None:
    # Only the added old import, at its new-file line; removed lines, the
    # allow marker and the table itself pass.
    assert added_line_problems(_TABLE, _DIFF) == ["src/a.py:5: from pkg.old import y"]
    assert added_line_problems({}, _DIFF) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_staged_check_reads_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(  # nosec B603 B607: fixed git arguments
            ["git", *args], cwd=tmp_path, check=True, capture_output=True
        )

    git("init", "-q")
    _write(tmp_path, "a.py", "x = 1\n")
    git("add", "a.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(MOVED, "pkg.old", "pkg.new")
    _write(tmp_path, "a.py", "x = 1\nfrom pkg.old import y\n")
    assert main(["--staged"]) == 0  # not staged yet
    git("add", "a.py")
    assert main(["--staged"]) == 1
    _write(tmp_path, "a.py", "x = 1\nfrom pkg.new import y\n")
    git("add", "a.py")
    assert main(["--staged"]) == 0


def test_repository_holds_its_baselines() -> None:
    # src/pyapprox/__init__.py -> the repository root holding the baselines.
    root = Path(pyapprox.__file__).resolve().parents[4]
    if not (root / BASELINE_FILE).exists():
        pytest.skip("not a source checkout")
    counts = count_references(MOVED, root)
    assert ratchet_problems(MOVED, counts, read_baseline(root / BASELINE_FILE)) == []
