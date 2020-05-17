import json
import logging
import pandas as pd
import os

from glob import glob
from typing import Dict
from typing import List
from typing import Sequence


PROJECT_BASE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '../../'
))


class FileReader:
    """
    Extraction of the raw data files into pandas data frames.
    Starts with fetching file paths and names.

    :param path: Path to folder with raw data files
    :type path: str
    :param allowed_extensions: Types of allowed files extension
    :type allowed_extensions: sequence
    """
    def __init__(self, path: str, allowed_extensions: Sequence = ('.jsonl', '.jl')):
        self.allowed_extensions = allowed_extensions
        self.logger = logging.getLogger(FileReader.__name__)
        self.file_paths = self._get_files(path=path)
        self.file_names = self._get_file_names()

    def _get_files(self, path: str) -> List[str]:
        """
        Takes files with allowed extensions in path.

        :param path: Path to folder with raw data files
        :type path: str
        :return: List of files names
        :rtype: list
        """
        files = glob(os.path.normpath(os.path.join(PROJECT_BASE_PATH, path + '*')))
        return [
            file for file in files
            if file.endswith(self.allowed_extensions)
        ]

    def _get_file_names(self) -> List[str]:
        """
        Takes files names from files paths.

        :return: List of files names
        :rtype: list
        """
        return [
            str(os.path.basename(file).split('.')[0])
            for file in self.file_paths
        ]

    def read_json_lines_files(self) -> Dict[str, pd.DataFrame]:
        """
        Reads json lines raw files to pandas data frames and stores it inside dict with name of file as key.

        Example output:

        ``{ 'BBC' : pd.DataFrame(...), 'CNN' : pd.DataFrame(...) }``

        :return: Dictionary with file names and data frames
        :rtype: dict
        """
        return dict(zip(
            self.file_names,
            (pd.DataFrame(self._read_json_lines_file(file)) for file in self.file_paths),
        ))

    @staticmethod
    def _read_json_lines_file(file: str) -> List:
        """
        Reads json lines raw data files and stores as lists of rows.

        :param file: Path to raw data file
        :rtype file: str
        :return: List of converted data files
        :rtype: list
        """
        list_of_lines = []
        with open(file, 'r', encoding='utf-8') as opened_file:
            for line in opened_file:
                list_of_lines.append(json.loads(line.rstrip('\n|\r')))
        return list_of_lines
