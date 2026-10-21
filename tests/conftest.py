"""Shared fixtures. Tests that need a .cosmo file use the demo directory."""
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


@pytest.fixture(scope="session")
def demo_cosmo():
    """Path to one demo .cosmo file, or skip when the demo data is absent."""
    files = sorted(DEMO.glob("*.cosmo"))
    if not files:
        pytest.skip("no .cosmo files in data/demo")
    return files[0]
