import os

from enum import Enum
from typing import Any

from nlper.exceptions import UnsupportedFileTypeException
from nlper.file_io.reader import CsvReader
from nlper.file_io.reader import HtmlReader
from nlper.file_io.reader import TextReader
from nlper.file_io.reader import JsonReader


class FileTypesResolver(Enum):
    """
    Supported file types readers.
    """
    txt = TextReader()
    html = HtmlReader()
    csv = CsvReader()
    json = JsonReader()

    @staticmethod
    def resolve(file_extension: str) -> Any:
        """
        Resolves file type reader by file extension.

        :param file_extension: File extension to resolve reader
        :type file_extension: str
        :return: Reader class or UnsupportedFileTypeException
        :rtype: any
        """
        for fileType in FileTypesResolver:
            if file_extension.endswith(fileType.name):
                return fileType.value
        raise UnsupportedFileTypeException(file_extension)

    @staticmethod
    def resolve_from_filepath(file_path: str) -> Any:
        """
        Resolves file type reader by file path.

        :param file_path: File path to resolve reader
        :type file_path: str
        :return: Reader class or UnsupportedFileTypeException
        :rtype: any
        """
        file_extension = str(os.path.basename(file_path).split('.')[1])
        return FileTypesResolver.resolve(file_extension)
