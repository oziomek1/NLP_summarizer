from nlper.main import clean_data
from tests.tools_for_testing import click_integration_test_for_app


def test__data_cleaner__works_without_errors():
    click_integration_test_for_app(app=clean_data, options=["tests/assets/test_dataframe_cleaner_config.yaml"])
