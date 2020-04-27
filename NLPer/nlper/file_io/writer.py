import json
import logging
import os

from abc import ABC
from abc import abstractmethod


class Writer(ABC):
    def __init__(self):
        self.file = None
        self.logger = logging.getLogger(Writer.__name__)

    def write(self, path, file):
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
        if not os.path.exists(directory):
            os.makedirs(directory)

    @abstractmethod
    def _write_file(self, path, file):
        ...


class JsonWriter(Writer):
    def __init__(self):
        super(JsonWriter).__init__()
        self.logger = logging.getLogger(JsonWriter.__name__)

    def _write_file(self, path, file):
        with open(path, 'w') as save_file:
            json.dump(file, save_file)


class CsvWriter(Writer):
    def __init__(self):
        super(CsvWriter).__init__()
        self.logger = logging.getLogger(CsvWriter.__name__)

    def _write_file(self, path, file):
        file.to_csv(path, index=False)


class PickleWriter(Writer):
    def __init__(self):
        super(PickleWriter).__init__()
        self.logger = logging.getLogger(PickleWriter.__name__)

    def _write_file(self, path, file):
        file.to_pickle(path)
