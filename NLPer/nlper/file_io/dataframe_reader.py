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
    def __init__(self, path: str, allowed_extensions: Sequence = ('.jsonl', '.jl')):
        self.allowed_extensions = allowed_extensions
        self.logger = logging.getLogger(FileReader.__name__)
        self.file_paths = self._get_files(path=path)
        self.file_names = self._get_file_names()

    def _get_files(self, path) -> List[str]:
        files = glob(os.path.normpath(os.path.join(PROJECT_BASE_PATH, path + '*')))
        return [
            file for file in files
            if file.endswith(self.allowed_extensions)
        ]

    def _get_file_names(self) -> List[str]:
        return [
            str(os.path.basename(file).split('.')[0])
            for file in self.file_paths
        ]

    def read_json_lines_files(self) -> Dict[str, pd.DataFrame]:
        return dict(zip(
            self.file_names,
            (pd.DataFrame(self._read_json_lines_file(file)) for file in self.file_paths),
        ))

    @staticmethod
    def _read_json_lines_file(file: str) -> List:
        list_of_lines = []
        with open(file, 'r', encoding='utf-8') as opened_file:
            for line in opened_file:
                list_of_lines.append(json.loads(line.rstrip('\n|\r')))
        return list_of_lines
