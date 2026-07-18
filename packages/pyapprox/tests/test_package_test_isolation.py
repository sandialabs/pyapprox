"""Meta-test enforcing the package-tier test boundary."""

from pathlib import Path


def test_no_benchmarks_imports_in_package_tests():
    """packages/pyapprox/tests must not import pyapprox_benchmarks.

    Cross-package tests belong in tests/integration/. This boundary is
    what breaks the pyapprox <-> pyapprox-benchmarks install cycle, so
    ``pip install pyapprox[test]`` never needs the benchmarks package.
    (Replaces the formerly unwired scripts/check_core_test_imports.sh.)
    """
    tests_root = Path(__file__).parent
    offenders = []
    for path in sorted(tests_root.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text().splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith(
                ("import pyapprox_benchmarks", "from pyapprox_benchmarks")
            ):
                offenders.append(
                    f"{path.relative_to(tests_root)}:{lineno}: {stripped}"
                )
    assert not offenders, (
        "pyapprox package tests must not import pyapprox_benchmarks; "
        "move these tests to tests/integration/:\n" + "\n".join(offenders)
    )
