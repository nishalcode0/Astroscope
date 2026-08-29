"""Test command-line interface behavior."""

import pytest

from astroscope.cli.main import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI --help option outputs header and exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Astroscope" in captured.out
    assert "Open-source computational astronomical observatory" in captured.out


def test_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI subcommands output clear not-implemented messaging."""
    subcommands = ["discover", "ingest", "process", "analyze", "candidates"]
    for cmd in subcommands:
        exit_code = main([cmd])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert f"Command '{cmd}' is not implemented yet in Session 1." in captured.out
