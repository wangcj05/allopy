import os

import pandas as pd
import pytest


@pytest.fixture
def outlook():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data/scenario.csv"))
