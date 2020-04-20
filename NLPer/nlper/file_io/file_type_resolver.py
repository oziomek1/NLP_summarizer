from enum import Enum

from nlper.exceptions import UnsupportedFileTypeException
from nlper.file_io.reader import CsvReader
from nlper.file_io.reader import HtmlReader
from nlper.file_io.reader import TextReader
from nlper.file_io.reader import JsonReader


class FileTypesResolver(Enum):
    txt = TextReader()
    html = HtmlReader()
    csv = CsvReader()
    json = JsonReader()

    @staticmethod
    def resolve(file_extension):
        for fileType in FileTypesResolver:
            if file_extension.endswith(fileType.name):
                return fileType.value
        raise UnsupportedFileTypeException(file_extension)
