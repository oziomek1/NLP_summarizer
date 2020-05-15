import logging
import pandas as pd
import os

from typing import Any

from nlper.exceptions import UnsupportedFileTypeException
from nlper.file_io.writer import CsvWriter
from nlper.file_io.writer import PickleWriter


class FileWriter:
    """
    Saving the data into pandas data frames.
    Currently supports saving files in CSV and Pickle format.

    :param path: Path to folder to save files
    :type path: str
    :param output_type: Format of saved files
    :type output_type: str
    """
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
        """
        Resolved output format type and saves a single data frame.
        Currently supports saving only Pickle and CSV file types using python.

        :param data: Data frame to save.
        :type data: pd.DataFrame
        :param name: Name under which save data frame to
        :type name: str
        """
        if self.output_type == 'pickle':
            self.saving_path = os.path.join(self.path, name + '.pkl')
            self.pickle_writer.write(path=self.saving_path, file=data)
        elif self.output_type == 'csv':
            self.saving_path = os.path.join(self.path, name + '.csv')
            self.csv_writer.write(path=self.saving_path, file=data)
        else:
            raise UnsupportedFileTypeException(self.output_type)

    def save_dataframe(self, name: str) -> None:
        """
        Calls method to resolve output format and save single data frame.

        :param name: Name of data frame to save
        :type name: str
        """
        self.resolve_output_format_type_and_save(self.data, name)

    def save_dataframes(self, name: str) -> None:
        """
        Calls method to resolve output format and save multiple data frame.

        :param name: Name of data frame to save
        :type name: str
        """
        for key in self.data.keys():
            name_with_key = name + '_' + key
            self.resolve_output_format_type_and_save(self.data[key], name_with_key)

    def save_file(self, data: Any, name: str, merge_data: Any = None, output_type: str = None) -> str:
        """
        Resolves how to process saving all data frames into files regarding the passed arguments.
        If ``merge_data`` is set to ``True``, all data frames are merged into single one.

        :param data: Dictionary with file names as key and data frames as values, or single data frame.
        :type data: dict, pd.DataFrame
        :param name: Name of output file(s)
        :type name: str
        :param merge_data: Flag to merge of not multiple data frames into one, optional
        :type merge_data: bool, optional
        :param output_type: Format to save data frame(s), if not specified using one from ``__init__`` method.
        :type output_type: str, optional
        :return: File saving location
        :rtype: str
        """
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
        """
        Merges multiple data frames into single.

        :return: Merged data frames
        :rtype: pd.DataFrame
        """
        dataframes = []
        for key in self.data.keys():
            dataframe = self.data[key].copy(deep=True)
            dataframe['site'] = key
            dataframes.append(dataframe)
        return pd.concat(dataframes, ignore_index=True)
