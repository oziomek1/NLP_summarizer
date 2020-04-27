import logging
import pandas as pd
import os

from typing import Any

from nlper.exceptions import UnsupportedFileTypeException
from nlper.file_io.writer import CsvWriter
from nlper.file_io.writer import PickleWriter


class FileWriter:
    def __init__(self, path: str, output_type: str = 'pickle'):
        self.path = path
        self.data = None
        self.prefix = None
        self.output_type = output_type
        self.logger = logging.getLogger(FileWriter.__name__)
        self.csv_writer = CsvWriter()
        self.pickle_writer = PickleWriter()
        self.saving_path = None

    def resolve_output_format_type_and_save(self, data: pd.DataFrame, name: str) -> None:
        if self.output_type == 'pickle':
            self.saving_path = os.path.join(self.path, name + '.pkl')
            self.pickle_writer.write(path=self.saving_path, file=data)
            # data.to_pickle(self.saving_path)
        elif self.output_type == 'csv':
            self.saving_path = os.path.join(self.path, name + '.csv')
            self.csv_writer.write(path=self.saving_path, file=data)
            # data.to_csv(self.saving_path, index=False)
        else:
            raise UnsupportedFileTypeException(self.output_type)

    def save_dataframe(self, name) -> None:
        self.resolve_output_format_type_and_save(self.data, name)

    def save_dataframes(self, name) -> None:
        for key in self.data.keys():
            name_with_key = name + '_' + key
            self.resolve_output_format_type_and_save(self.data[key], name_with_key)

    def save_file(self, data: Any, name: str, merge_data: Any = None, output_type: str = None) -> str:
        self.data = data
        if output_type:
            self.output_type = output_type

        if isinstance(self.data, dict):
            if merge_data:
                self.data = self.merge_dataframes()
            else:
                self.save_dataframes(name=name)
        if isinstance(self.data, pd.DataFrame):
            self.save_dataframe(name=name)
        return self.saving_path

    def merge_dataframes(self) -> pd.DataFrame:
        dataframes = []
        for key in self.data.keys():
            dataframe = self.data[key].copy(deep=True)
            dataframe['site'] = key
            dataframes.append(dataframe)
        return pd.concat(dataframes, ignore_index=True)
