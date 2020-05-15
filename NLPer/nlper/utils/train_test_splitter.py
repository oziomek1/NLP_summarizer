import logging
import numpy as np
import os
import pandas as pd

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from nlper.file_io.file_type_resolver import FileTypesResolver
from nlper.file_io.writer import CsvWriter
from nlper.utils.config_utils import read_config


TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TRAIN_TEXT_COLUMNS = ('text', 'summary')


logging.basicConfig(
    format=f"%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s",
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


class TrainTestSplitter:
    """
    Splits data frame into train, test and valid parts.

    :param config: Path to yaml config file
    :type config: str, optional
    :param filepath: Path to pandas data frame
    :type filepath: str
    :param valid: Flag whether to include split for valid part
    :type valid: bool
    """
    def __init__(self, config: str = None, filepath: str = None, valid: bool = True):
        self.logger = logging.getLogger(TrainTestSplitter.__name__)
        self.filepath = filepath
        self.valid = valid
        self.config = self._read_from_config(config=config)
        self.csv_writer = CsvWriter()
        self.data = None

    def run(self):
        """
        Executes train, test, valid data frame split
        """
        if not self.config:
            self.set_config_from_filepath()
        self.read_file()
        self.create_dirs()
        train, test, val = self.split_data()
        self.save_data(train=train, test=test, val=val)

    def _read_from_config(self, config: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Reads config if specified path.

        :param config: Specified config path
        :type config: str, optional
        :return: Loaded config if specified else return nothing
        :rtype: dict, optional
        """
        if config:
            config_dict = read_config(config, self.logger)
            self.filepath = config_dict['input_file']
            return config_dict
        return None

    def build_paths(self):
        """
        Builds paths required for saving the split data frames
        """
        self.filepath = self.config['input_file']
        self.config['sub_dir'] = os.path.basename(self.filepath).split('.')[0]
        path = os.path.normpath(os.path.join(
            self.config['output_dir'],
            self.config['sub_dir'],
        ))
        self.config['path'] = path
        for file_type in ['train', 'test', 'val']:
            self.config[f'{file_type}_file'] = os.path.join(path, f'{file_type}.csv')

    def create_dirs(self) -> None:
        """
        Creates directories for split data frames to save
        :return:
        """
        self.build_paths()
        self.csv_writer.create_dir(directory=self.config['path'])

    def read_file(self) -> None:
        """
        Reads pandas data frame from path
        """
        file_type_resolver = FileTypesResolver.resolve_from_filepath(self.filepath)
        self.data = file_type_resolver.open_file(filepath=self.filepath)

    def save_data(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            val: Optional[pd.DataFrame] = None,
            columns: Tuple[str] = TRAIN_TEXT_COLUMNS,
    ) -> None:
        """
        Saves specified columns of train, test and valid data frames into csv files.

        :param train: Train part data frame
        :param train: pd.DataFrame
        :param test: Test part data frame
        :param test: pd.DataFrame
        :param val: Valid part data frame
        :param val: pd.DataFrame
        :param columns: Columns to save
        :param columns: tuple
        """
        columns = list(columns)
        self.csv_writer.write(path=self.config['train_file'], file=train[columns])
        self.csv_writer.write(path=self.config['test_file'], file=test[columns])
        self.csv_writer.write(path=self.config['val_file'], file=val[columns])

    def set_config_from_filepath(self) -> None:
        """
        Specifies directory name to write split data frame parts
        """
        self.config['output_dir'] = os.path.dirname(self.filepath)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data frame into train, test and valid parts.

        :return: Train, test and valid data frames
        :rtype: tuple
        """
        def calculate_last_indices(total_length: int) -> Tuple[int, int]:
            """
            Calculates last indices of data frame rows for train, test required for split.

            :param total_length: Number of rows in data frame
            :type total_length: int
            :return: Number of last indices for train and test parts
            :rtype: tuple
            """
            return int(TRAIN_PERCENTAGE * total_length), int((TRAIN_PERCENTAGE + VAL_PERCENTAGE) * total_length)

        def permute_rows() -> Tuple[np.array, np.array, np.array]:
            """
            Permutes rows of data frame for split.

            :return: Permuted rows
            :rtype: tuple
            """
            train_indices = permuted_indices[:last_train_index]
            test_indices = permuted_indices[last_train_index:last_test_index]
            val_indices = permuted_indices[last_test_index:]
            return train_indices, test_indices, val_indices

        np.random.seed(None)
        permuted_indices = np.random.permutation(self.data.index)
        last_train_index, last_test_index = calculate_last_indices(len(self.data))
        train_indices, test_indices, val_indices = permute_rows()
        return self.data.iloc[train_indices].reset_index(drop=True), \
            self.data.iloc[test_indices].reset_index(drop=True), \
            self.data.iloc[val_indices].reset_index(drop=True)
