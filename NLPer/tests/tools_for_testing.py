import os

from click.testing import CliRunner


# slow = pytest.mark.skipif(
#     not pytest.config.getoption("--runslow"),
#     reason="need --runslow option to run",
# )
#
#
# random = pytest.mark.skipif(
#     not pytest.config.getoption("--runrandom"),
#     reason="need --runrandom option to run",
# )


def click_integration_test_for_app(app, options, expected_exit_code=0):
    result = CliRunner().invoke(app, options, catch_exceptions=False)
    assert expected_exit_code == result.exit_code, result.output


def check_lists_equal(list1, list2):
    return len(list1) == len(list2) and sorted(list1) == sorted(list2)


def remove_extension_from_file_names(file_names):
    return [
        file_name.split('.')[0] for file_name in file_names
    ]


def remove_path_from_file_names(file_paths):
    return [
        os.path.basename(file_path) for file_path in file_paths
    ]
