import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def config():
    from beer_sentiment.config import load_config

    return load_config()


@pytest.fixture
def mock_judge(config):
    from beer_sentiment.llm.mock import MockJudge

    return MockJudge(config)
