import os

from glob import glob

from nlper.file_io.dataframe_reader import FileReader
from tests.tools_for_testing import check_lists_equal
from tests.tools_for_testing import remove_extension_from_file_names
from tests.tools_for_testing import remove_path_from_file_names


path = 'tests/assets/data_files/'
file_extension = '.jsonl'


def test__dataframe_reader__get_correct_files_and_files_names():
    file_reader = FileReader(path=path, allowed_extensions=file_extension)
    file_names = remove_path_from_file_names(glob(path + '*' + file_extension))

    file_paths = [
        os.path.abspath(os.path.join(path + file_name))
        for file_name in file_names
    ]

    assert check_lists_equal(file_reader.file_paths, file_paths)
    assert check_lists_equal(
        file_reader.file_names, remove_extension_from_file_names(file_names))


def test__dataframe_reader__get_correct_data():
    file_reader = FileReader(path=path, allowed_extensions=file_extension)
    file_names = remove_path_from_file_names(glob(path + '*' + file_extension))
    required_columns_in_dataframe = ['url', 'title', 'lead', 'text']

    data = file_reader.read_json_lines_files()
    assert check_lists_equal(list(data.keys()), remove_extension_from_file_names(file_names))
    assert bool(data)

    for file_name, dataframe in data.items():
        assert all(elem in dataframe.columns.values for elem in required_columns_in_dataframe)
