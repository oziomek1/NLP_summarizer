import json
import logging
import pandas as pd

from abc import ABC
from abc import abstractmethod
from bs4 import BeautifulSoup

from typing import Any


class Reader(ABC):
    def __init__(self):
        self.file = None
        self.logger = logging.getLogger(Reader.__name__)

    def open_file(self, filepath: str) -> Any:
        """
        Safely opens and returns file specified in file path.

        :param filepath: File path to open
        :type filepath: str
        :return: Opened file or FileNotFoundError
        :rtype: any
        """
        try:
            self._read_file(filepath=filepath)
        except FileNotFoundError as e:
            self.logger.error(f'File not available : {e}')
            raise FileNotFoundError
        return self.file

    @abstractmethod
    def _read_file(self, filepath: str) -> None:
        """
        Reads particular format file from file path.

        :param filepath: File path to read
        :type filepath: str
        """


class CsvReader(Reader):
    def __init__(self):
        super(CsvReader).__init__()
        self.logger = logging.getLogger(CsvReader.__name__)

    def _read_file(self, filepath: str) -> None:
        """
        Reads CSV file from file path using pandas.

        :param filepath: CSV file path
        :type filepath: str
        """
        self.file = pd.read_csv(filepath, sep=',')


class HtmlReader(Reader):
    def __init__(self):
        super(HtmlReader).__init__()
        self.logger = logging.getLogger(HtmlReader.__name__)

    def _read_file(self, filepath: str) -> None:
        """
        Reads Html file from file path using BeautifulSoup.

        :param filepath: Html file path
        :type filepath: str
        """
        with open(filepath, 'r') as file:
            file_content = file.read()
            soup = BeautifulSoup(file_content, 'html.parser')
            self.file = soup.findAll('p')


class JsonReader(Reader):
    def __init__(self):
        super(JsonReader).__init__()
        self.logger = logging.getLogger(JsonReader.__name__)

    def _read_file(self, filepath: str) -> None:
        """
        Reads JSON file from file path using json.

        :param filepath: JSON file path
        :type filepath: str
        """
        with open(filepath, 'r') as file:
            self.file = json.load(file)


class TextReader(Reader):
    def __init__(self):
        super(TextReader).__init__()
        self.logger = logging.getLogger(TextReader.__name__)

    def _read_file(self, filepath: str) -> None:
        """
        Reads Text file from file path.

        :param filepath: Text file path
        :type filepath: str
        """
        with open(filepath, 'r') as file:
            self.file = file.read()
