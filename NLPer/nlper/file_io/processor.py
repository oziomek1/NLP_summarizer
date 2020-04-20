import os
import logging

from nlper.file_io.file_type_resolver import FileTypesResolver


# TODO: refactor


class Processor:
    def __init__(self, filepath, output):
        self.filepath = filepath
        self.output = output
        self.logger = logging.getLogger(Processor.__name__)
        self.input_file = None
        self.file_extension = None

    def process(self):
        self._get_file_extension()
        self._open_file()

    def _get_file_extension(self):
        self.file_extension = os.path.basename(self.filepath).split('.')[-1]

    def _open_file(self):
        file_reader = FileTypesResolver.resolve(file_extension=self.file_extension)
        self.input_file = file_reader.open_file(filepath=self.filepath)

        self.logger.info(f'Opened file : {self.filepath}\nContent : {self.input_file}')
