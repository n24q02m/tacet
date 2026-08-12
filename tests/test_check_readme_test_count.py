"""Regression tests for the README test-count guard."""

import importlib.util
from pathlib import Path
from subprocess import CompletedProcess

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_readme_test_count.py"
SPEC = importlib.util.spec_from_file_location("check_readme_test_count", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_readme_test_count = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_readme_test_count)


def test_collected_count_uses_the_canonical_all_extras_environment(monkeypatch) -> None:
    seen_command: list[str] = []

    def fake_run(command: list[str], **_kwargs) -> CompletedProcess[str]:
        seen_command.extend(command)
        return CompletedProcess(command, 0, stdout="508 tests collected\n", stderr="")

    monkeypatch.setattr(check_readme_test_count.subprocess, "run", fake_run)

    assert check_readme_test_count.collected_count() == 508
    assert seen_command == [
        "uv",
        "run",
        "--all-extras",
        "pytest",
        "--collect-only",
        "-q",
    ]


def test_collected_count_rejects_a_failed_collection_even_if_stdout_has_a_count(
    monkeypatch,
) -> None:
    def fake_run(command: list[str], **_kwargs) -> CompletedProcess[str]:
        return CompletedProcess(
            command,
            2,
            stdout="503 tests collected, 3 errors\n",
            stderr="collection failed\n",
        )

    monkeypatch.setattr(check_readme_test_count.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as raised:
        check_readme_test_count.collected_count()

    assert raised.value.code == 2
