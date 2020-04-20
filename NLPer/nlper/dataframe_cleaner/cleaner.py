import logging
import pandas as pd

from tqdm import tqdm
from typing import Any
from typing import Dict

from nlper.utils.clean_utils import CleanUtils


class Cleaner:
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Cleaner.__name__)
        self.config = config
        self.data = data

    def clean_dataframe(self) -> pd.DataFrame:
        self.convert_list_to_text_in_dataframe()
        self.remove_characters_for_dataframe()
        if self.config['hide_numbers']:
            self.hide_numbers_for_dataframe()
        if self.config['lemmatize']:
            self.lemmatize_text_for_dataframe()
        return self.data

    def convert_list_to_text_in_dataframe(self) -> None:
        for column_name in self.data:
            self.data[column_name] = [
                CleanUtils.convert_list_to_text(text_as_list=single_cell)
                for single_cell in self.data[column_name]
            ]

    def hide_numbers_for_dataframe(self) -> None:
        for column_name in self.data:
            self.data[column_name] = self.hide_numbers_for_column(self.data[column_name])

    def lemmatize_text_for_dataframe(self) -> None:
        clean_utils = CleanUtils()
        clean_utils.lang_model = self.config['language_model']
        for column_name in self.data:
            self.data[column_name] = self.lemmatize_text_for_column(self.data[column_name], clean_utils)

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
        return pd.Series([
            clean_utils.lemmatize(text=text)
            for text in tqdm(column_data, desc='Lemmatizing')
        ])

    @staticmethod
    def remove_characters_for_column(column_data: pd.Series) -> pd.Series:
        return pd.Series([
            CleanUtils.remove_characters_for_text(text=text)
            for text in column_data
        ])
