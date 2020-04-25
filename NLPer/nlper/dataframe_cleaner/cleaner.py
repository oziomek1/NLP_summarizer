import logging
import numpy as np
import pandas as pd

from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from typing import Any
from typing import Dict

from nlper.utils.clean_utils import CleanUtils
from nlper.utils.time_utils import timeit


tqdm.pandas(desc="Cleaning")


class Cleaner:
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Cleaner.__name__)
        self.config = config
        self.data = data
        self.n_cores = cpu_count() // 2
        self.clean_utils = CleanUtils()

    @timeit
    def clean_dataframe(self) -> pd.DataFrame:
        self.convert_list_to_text_in_dataframe()
        self.remove_characters_for_dataframe()
        if self.config['hide_numbers']:
            self.hide_numbers()
        if self.config['lemmatize']:
            self.lemmatize_text()
        return self.data

    def convert_list_to_text_in_dataframe(self) -> None:
        for column_name in self.data:
            self.data[column_name] = [
                self.clean_utils.convert_list_to_text(text_as_list=single_cell)
                for single_cell in self.data[column_name]
            ]

    def hide_numbers(self) -> None:
        for column_name in self.data:
            self.data[column_name] = self.hide_numbers_for_column(self.data[column_name])

    def lemmatize_text(self) -> None:
        self.clean_utils.lang_model = self.config['language_model']

        dataframe_splits = np.array_split(self.data, self.n_cores)
        pool = Pool(self.n_cores)
        self.data = pd.concat(pool.map(self.lemmatize_text_for_dataframe, dataframe_splits))
        pool.close()
        pool.join()

    def lemmatize_text_for_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column_name in dataframe:
            dataframe[column_name] = self.lemmatize_text_for_column(
                column_data=dataframe[column_name],
                clean_utils=self.clean_utils,
            )
        return dataframe

    def remove_characters_for_dataframe(self) -> None:
        for column_name in self.data:
            self.data[column_name] = self.remove_characters_for_column(self.data[column_name])

    @staticmethod
    def hide_numbers_for_column(column_data: pd.Series) -> pd.Series:
        return pd.Series([
            CleanUtils.hide_numbers(text=text)
            for text in column_data
        ])

    @staticmethod
    def lemmatize_text_for_column(column_data: pd.Series, clean_utils: CleanUtils) -> pd.Series:
        return column_data.progress_map(lambda text: clean_utils.lemmatize(text=text))

    @staticmethod
    def remove_characters_for_column(column_data: pd.Series) -> pd.Series:
        return pd.Series([
            CleanUtils.remove_characters_for_text(text=text)
            for text in column_data
        ])
