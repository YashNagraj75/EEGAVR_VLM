import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegavr.eval.mockset import make_mock_set  # noqa: E402


@pytest.fixture(scope="session")
def mockset(tmp_path_factory):
    return make_mock_set(tmp_path_factory.mktemp("set"), n_per_rule=4, seed=1)
