import logging
import numpy as np
import os
import pandas as pd

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from nlper.utils.config_utils import read_config
from nlper.file_io.file_type_resolver import FileTypesResolver


TRAIN_PERCENTAGE = 0.8
VAL_PERCENTAGE = 0.1
TRAIN_TEXT_COLUMNS = ('text', 'summary')


logging.basicConfig(
    format=f"%(asctime)s [%(levelname)s] | %(name)s | %(funcName)s: %(message)s",
    level=logging.INFO,
    datefmt='%I:%M:%S',
)


class TrainTestSplitter:
    def __init__(self, config: str = None, filepath: str = None, valid: str = True):
        self.logger = logging.getLogger(TrainTestSplitter.__name__)
        self.filepath = filepath
        self.valid = valid
        self.config = self._read_from_config(config=config)
        self.data = None

    def run(self):
        if not self.config:
            self.set_config_from_filepath()
        self.read_file()
        self.create_dirs()
        train, test, val = self.split_data()
        self.save_data(train=train, test=test, val=val)

    def _read_from_config(self, config: Optional[str]) -> Optional[Dict[str, Any]]:
        if config:
            config_dict = read_config(config, self.logger)
            self.filepath = config_dict['input_file']
            return config_dict
        return None

    def build_paths(self):
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
        self.build_paths()
        self._create_dir(path=self.config['path'])

    def read_file(self) -> None:
        file_type_resolver = FileTypesResolver.resolve_from_filepath(self.filepath)
        self.data = file_type_resolver.open_file(filepath=self.filepath)

    def save_data(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            val: Optional[pd.DataFrame] = None,
            columns: Tuple[str] = TRAIN_TEXT_COLUMNS,
    ) -> None:
        columns = list(columns)
        train[columns].to_csv(self.config['train_file'], index=False)
        test[columns].to_csv(self.config['test_file'], index=False)
        val[columns].to_csv(self.config['val_file'], index=False)

    def set_config_from_filepath(self) -> None:
        self.config['output_dir'] = os.path.dirname(self.filepath)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        def calculate_last_indices(total_length):
            return int(TRAIN_PERCENTAGE * total_length), int((TRAIN_PERCENTAGE + VAL_PERCENTAGE) * total_length)

        def permute_rows():
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

    @staticmethod
    def _create_dir(path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
