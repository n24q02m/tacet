"""Pre-commit guard: keep the README's "# N tests" comment in sync with reality.

The comment has gone stale three times (README says one count, `pytest`
collects a different one). This script is the root-cause fix: it collects
the real test count and fails the commit when the README disagrees with it,
instead of relying on someone remembering to update the comment by hand.

Run directly:

    uv run python scripts/check_readme_test_count.py
"""

import re
import subprocess
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
README_PATTERN = re.compile(r"#\s*(\d+)\s+tests")
COLLECTED_PATTERN = re.compile(r"(\d+)\s+tests? collected")


def readme_count() -> int:
    text = README.read_text(encoding="utf-8")
    match = README_PATTERN.search(text)
    if not match:
        print(f"ERROR: could not find a '# N tests' comment in {README}")
        sys.exit(1)
    return int(match.group(1))


def collected_count() -> int:
    result = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
    )
    match = None
    for line in reversed(result.stdout.splitlines()):
        match = COLLECTED_PATTERN.search(line)
        if match:
            break
    if not match:
        print("ERROR: could not parse a test count from pytest --collect-only -q output")
        print(result.stdout[-2000:])
        sys.exit(1)
    return int(match.group(1))


def main() -> None:
    readme_n = readme_count()
    collected_n = collected_count()
    if readme_n != collected_n:
        print(
            f"ERROR: README.md says '# {readme_n} tests' but pytest collects "
            f"{collected_n} tests. Update the comment in README.md to match."
        )
        sys.exit(1)
    print(f"README test count matches: {collected_n} tests")


if __name__ == "__main__":
    main()
