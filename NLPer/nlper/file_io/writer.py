import json
import logging
import os

from abc import ABC
from abc import abstractmethod

from typing import Any


class Writer(ABC):
    def __init__(self):
        self.file = None
        self.logger = logging.getLogger(Writer.__name__)

    def write(self, path: str, file: Any) -> None:
        """
        Safely writes file to specified location.

        :param path: Path to save file
        :type path: str
        :param file: File to save
        :type file: any
        """
        directory = os.path.dirname(path)
        self.create_dir(directory=directory)
        try:
            self._write_file(path=path, file=file)
            self.logger.info(f'Saved file {path}')
        except Exception as e:
            self.logger.error(f'Cannot write file : {e}')
            raise Exception

    @staticmethod
    def create_dir(directory: str) -> None:
        """
        Creates a directory for split data frame parts if not exists.

        :param directory: Directory to create
        :type directory: str
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @abstractmethod
    def _write_file(self, path: str, file: Any) -> None:
        """
        Writes particular format file to path.

        :param path: Path to save file
        :type path: str
        :param file: File to save
        :type file: any
        """


class CsvWriter(Writer):
    def __init__(self):
        super(CsvWriter).__init__()
        self.logger = logging.getLogger(CsvWriter.__name__)

    def _write_file(self, path: str, file: Any) -> None:
        """
        Writes CSV file to path using pandas.

        :param path: Path to save CSV file
        :type path: str
        :param file: CSV file to save
        :type file: any
        """
        file.to_csv(path, index=False)


class JsonWriter(Writer):
    def __init__(self):
        super(JsonWriter).__init__()
        self.logger = logging.getLogger(JsonWriter.__name__)

    def _write_file(self, path: str, file: Any) -> None:
        """
        Writes JSON file to path using json.

        :param path: Path to save JSON file
        :type path: str
        :param file: JSON file to save
        :type file: any
        """
        with open(path, 'w') as save_file:
            json.dump(file, save_file)


class PickleWriter(Writer):
    def __init__(self):
        super(PickleWriter).__init__()
        self.logger = logging.getLogger(PickleWriter.__name__)

    def _write_file(self, path: str, file: Any) -> None:
        """
        Writes Pickle file to path using pandas.

        :param path: Path to save Pickle file
        :type path: str
        :param file: Pickle file to save
        :type file: any
        """
        file.to_pickle(path)
