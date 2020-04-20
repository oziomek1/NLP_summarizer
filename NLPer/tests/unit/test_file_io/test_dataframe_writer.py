import os
import pandas as pd
import pytest

from nlper.file_io.dataframe_writer import FileWriter


data_frame = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})
dataframes_dictionary = {
    'first': pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]}),
    'second': pd.DataFrame({'a': [6, 7, 8], 'b': [9, 10, 11]}),
}


@pytest.mark.parametrize(
    "dataframe, output_type", [
        (data_frame, {'name': 'csv', 'extension': 'csv'}),
        (data_frame, {'name': 'pickle', 'extension': 'pkl'}),
        pytest.param(data_frame, {'name': 'yaml', 'extension': 'yaml'}, marks=pytest.mark.xfail)
    ]
)
def test__dataframe_writer__saves_single_dummy_dataframe(tmpdir, dataframe, output_type):
    name = 'dummy_data'
    file_path = os.path.join(tmpdir, name + '.' + output_type['extension'])

    file_writer = FileWriter(tmpdir, output_type=output_type['name'])
    file_writer.save_file(
        data=dataframe,
        name=name,
        merge_data=False
    )
    assert file_writer.saving_path == file_path

    assert os.path.exists(file_path)
    if output_type['name'] == 'csv':
        assert pd.read_csv(file_path).equals(data_frame)
    elif output_type['name'] == 'pickle':
        assert pd.read_pickle(file_path).equals(data_frame)
    else:
        assert False, 'Not supported type'


@pytest.mark.parametrize(
    "dataframes, output_type", [
        (dataframes_dictionary, {'name': 'csv', 'extension': 'csv'}),
        (dataframes_dictionary, {'name': 'pickle', 'extension': 'pkl'}),
        pytest.param(dataframes_dictionary, {'name': 'yaml', 'extension': 'yaml'}, marks=pytest.mark.xfail)
    ]
)
def test__dataframe_writer__saves_multiple_dummy_dataframes(tmpdir, dataframes, output_type):
    name = 'dummy_data'
    dataframe_keys = list(dataframes.keys())
    file_paths = [
        os.path.join(tmpdir, name + '_' + key + '.' + output_type['extension'])
        for key in dataframe_keys
    ]

    file_writer = FileWriter(tmpdir, output_type=output_type['name'])
    file_writer.save_file(
        data=dataframes,
        name=name,
        merge_data=False
    )

    assert file_writer.saving_path == file_paths[-1]

    assert all(
        os.path.exists(file_path)
        for file_path in file_paths
    )

    if output_type['name'] == 'csv':
        assert all(
            pd.read_csv(file_path).equals(dataframes[dataframe_keys[idx]])
            for idx, file_path in enumerate(file_paths)
        )
    elif output_type['name'] == 'pickle':
        assert all(
            pd.read_pickle(file_path).equals(dataframes[dataframe_keys[idx]])
            for idx, file_path in enumerate(file_paths)
        )
    else:
        assert False, 'Not supported type'


@pytest.mark.parametrize(
    "dataframes, output_type", [
        (dataframes_dictionary, {'name': 'csv', 'extension': 'csv'}),
        (dataframes_dictionary, {'name': 'pickle', 'extension': 'pkl'}),
        pytest.param(dataframes_dictionary, {'name': 'yaml', 'extension': 'yaml'}, marks=pytest.mark.xfail)
    ]
)
def test__dataframe_writer__merge_and_save_multiple_dummy_dataframes(tmpdir, dataframes, output_type):
    name = 'dummy_data'
    file_path = os.path.join(tmpdir, name + '.' + output_type['extension'])

    file_writer = FileWriter(tmpdir, output_type=output_type['name'])
    file_writer.save_file(
        data=dataframes,
        name=name,
        merge_data=True
    )

    merged = []
    for key in dataframes.keys():
        df = dataframes[key].copy(deep=True)
        df['site'] = key
        merged.append(df)
    merged = pd.concat(merged, ignore_index=True)

    assert os.path.exists(file_path)

    if output_type['name'] == 'csv':
        assert pd.read_csv(file_path).equals(merged)
    elif output_type['name'] == 'pickle':
        assert pd.read_pickle(file_path).equals(merged)
    else:
        assert False, 'Not supported type'


@pytest.mark.parametrize(
    "dataframes", [dataframes_dictionary]
)
def test__dataframe_writer__merges_dataframes_correctly(tmpdir, dataframes):
    file_writer = FileWriter(tmpdir)
    file_writer.data = dataframes
    output_dataframe = file_writer.merge_dataframes()

    assert isinstance(output_dataframe, pd.DataFrame)

    merged = []
    for key in dataframes.keys():
        df = dataframes[key].copy(deep=True)
        df['site'] = key
        merged.append(df)
    merged = pd.concat(merged, ignore_index=True)

    assert merged.equals(output_dataframe)
