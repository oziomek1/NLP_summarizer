import os
import sys
import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path += [ROOT_DIR, SRC_DIR]


@pytest.fixture(scope='module')
def paths():
    PATH = '/example_path'
    return dict(PATH=PATH)


def pytest_addoption(parser):
    parser.addoption(
        '--runslow',
        action="store_true",
        help="run slow",
    )
    parser.addoption(
        '--runrandom',
        action="store_true",
        help="run random",
    )
