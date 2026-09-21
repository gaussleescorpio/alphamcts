import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alphamcts.data import generate_synthetic_panel  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    return generate_synthetic_panel(n_stocks=30, n_days=300, seed=7)
