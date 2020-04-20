import json
import pandas as pd

from abc import ABC
from abc import abstractmethod
from bs4 import BeautifulSoup

import logging


class Reader(ABC):
    def __init__(self):
        self.file = None
        self.logger = logging.getLogger(Reader.__name__)

    def open_file(self, filepath):
        try:
            self.read_file(filepath=filepath)
        except FileNotFoundError as e:
            self.logger.error(f'File not available : {e}')
        return self.file

    @abstractmethod
    def read_file(self, filepath):
        ...


class CsvReader(Reader):
    def __init__(self):
        super(CsvReader).__init__()
        self.logger = logging.getLogger(CsvReader.__name__)

    def read_file(self, filepath):
        self.file = pd.read_csv(filepath, sep='\t', header=None)


class HtmlReader(Reader):
    def __init__(self):
        super(HtmlReader).__init__()
        self.logger = logging.getLogger(HtmlReader.__name__)

    def read_file(self, filepath):
        with open(filepath, 'r') as file:
            file_content = file.read()
            soup = BeautifulSoup(file_content, 'html.parser')
            self.file = soup.findAll('p')


class TextReader(Reader):
    def __init__(self):
        super(TextReader).__init__()
        self.logger = logging.getLogger(TextReader.__name__)

    def read_file(self, filepath):
        with open(filepath, 'r') as file:
            self.file = file.read()


class JsonReader(Reader):
    def __init__(self):
        super(JsonReader).__init__()
        self.logger = logging.getLogger(JsonReader.__name__)

    def read_file(self, filepath):
        with open(filepath, 'r') as file:
            self.file = json.load(file)
