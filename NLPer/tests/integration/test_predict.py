from nlper.main import predict
from tests.tools_for_testing import click_integration_test_for_app


def test__predict__works_without_errors():
    click_integration_test_for_app(app=predict, options={'predict_text'})
