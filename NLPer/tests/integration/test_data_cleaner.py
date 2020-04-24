from nlper.main import clean_data
from tests.tools_for_testing import click_integration_test_for_app


def test__data_cleaner__works_without_errors():
    click_integration_test_for_app(
        app=clean_data,
        options=["tests/assets/config_files/test_dataframe_cleaner_config.yaml"],
    )


def test__data_cleaner_no_data_trim__works_without_errors():
    click_integration_test_for_app(
        app=clean_data,
        options=["tests/assets/config_files/test_dataframe_cleaner_config_no_trim.yaml"],
    )
