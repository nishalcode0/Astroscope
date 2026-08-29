"""Test package imports and module accessibility."""

import importlib


def test_package_import() -> None:
    """Verify top-level astroscope package imports."""
    import astroscope

    assert hasattr(astroscope, "__version__")
    assert astroscope.__version__ == "0.1.0"
    assert hasattr(astroscope, "Observation")


def test_submodules_import() -> None:
    """Verify all submodules can be imported without errors."""
    submodules = [
        "astroscope.archive",
        "astroscope.archive.base",
        "astroscope.archive.hubble",
        "astroscope.archive.jwst",
        "astroscope.ingestion",
        "astroscope.processing",
        "astroscope.sources",
        "astroscope.timeseries",
        "astroscope.detection",
        "astroscope.crossmatch",
        "astroscope.catalog",
        "astroscope.cli",
        "astroscope.cli.main",
    ]
    for mod in submodules:
        imported = importlib.import_module(mod)
        assert imported is not None
