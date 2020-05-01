import os
import sys
import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "nlper")
sys.path += [ROOT_DIR, SRC_DIR]


@pytest.fixture(scope='module')
def paths():
    PATH = '/example_path'
    return dict(PATH=PATH)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_runtest_setup(item):
    if not item.config.getoption("--runslow"):
        pytest.skip("need --runslow option to run")
