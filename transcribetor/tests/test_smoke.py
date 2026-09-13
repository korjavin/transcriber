"""Smoke test: the package imports from the repo root (pytest path wiring works)."""

import transcribetor


def test_package_importable():
    assert transcribetor is not None
