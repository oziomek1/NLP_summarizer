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
    """
    Cleans raw text data frame obtaining data in unified format.

    * Removes unwanted characters
    * Hides numbers, date and time in different formats
    * Lemmatizes text

    :param config: Configuration dictionary
    :type config: dict
    :param data: Raw text data frame to clean
    :type data: pd.DataFrame
    """
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Cleaner.__name__)
        self.config = config
        self.data = data
        self.n_cores = cpu_count() // 2
        self.clean_utils = CleanUtils()

    @timeit
    def clean_dataframe(self) -> pd.DataFrame:
        """
        Executes data frame cleaning process.
        :return: Cleaned data frame
        :rtype: pd.DataFrame
        """
        self.convert_list_to_text_in_dataframe()
        self.remove_characters_for_dataframe()
        if self.config['hide_numbers']:
            self.hide_numbers()
        if self.config['lemmatize']:
            self.lemmatize_text()
        return self.data

    def convert_list_to_text_in_dataframe(self) -> None:
        """
        Calls conversion of data in list of sentences format to single, multi sentenced text using cleaning utils.
        """
        for column_name in self.data:
            self.data[column_name] = [
                self.clean_utils.convert_list_to_text(text_as_list=single_cell)
                for single_cell in self.data[column_name]
            ]

    def hide_numbers(self) -> None:
        """
        Splits hiding numbers in data frame to separate columns.
        """
        for column_name in self.data:
            self.data[column_name] = self.hide_numbers_for_column(self.data[column_name])

    def lemmatize_text(self) -> None:
        """
        Applies parallelization of text lemmatization for data frame using python multiprocessing.
        Lemmatization process is computationally expensive and thus parallelization greatly reduces the required time.
        """
        self.clean_utils.lang_model = self.config['language_model']

        dataframe_splits = np.array_split(self.data, self.n_cores)
        pool = Pool(self.n_cores)
        self.data = pd.concat(pool.map(self.lemmatize_text_for_dataframe, dataframe_splits))
        pool.close()
        pool.join()

    def lemmatize_text_for_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Splits text lemmatization in data frame into separated columns.

        :param dataframe: Data frame to lemmatize text in.
        :type dataframe: pd.DataFrame
        :return: Data frame with lemmatized text
        :rtype: pd.DataFrame
        """
        for column_name in dataframe:
            dataframe[column_name] = self.lemmatize_text_for_column(
                column_data=dataframe[column_name],
                clean_utils=self.clean_utils,
            )
        return dataframe

    def remove_characters_for_dataframe(self) -> None:
        """
        Splits unwanted characters removal from data frame to separate columns.
        """
        for column_name in self.data:
            self.data[column_name] = self.remove_characters_for_column(self.data[column_name])

    @staticmethod
    def hide_numbers_for_column(column_data: pd.Series) -> pd.Series:
        """
        Calls hiding numbers from clean utils for single column in data frame using cleaning utils.

        :param column_data: Column in data frame to clean.
        :type column_data: pd.Series
        :return: Cleaned data frame column
        :rtype: pd.Series
        """
        return pd.Series([
            CleanUtils.hide_numbers(text=text)
            for text in column_data
        ])

    @staticmethod
    def lemmatize_text_for_column(column_data: pd.Series, clean_utils: CleanUtils) -> pd.Series:
        """
        Calls text lemmatization on single data frame column using cleaning utils.
        Method uses ``progress_map`` to visualize progress of lemmatization using tqdm.

        :param column_data: Column in data frame to lemmatize.
        :type column_data: pd.Series
        :param clean_utils: Cleaning utility class
        :type clean_utils: object
        :return: Column in data frame with lemmatized text
        :rtype: pd.Series
        """
        return column_data.progress_map(lambda text: clean_utils.lemmatize(text=text))

    @staticmethod
    def remove_characters_for_column(column_data: pd.Series) -> pd.Series:
        """
        Calls removal of unwanted characters in data frame column using cleaning utils.

        :param column_data: Column in data frame to remove characters from.
        :type column_data: pd.Series
        :return: Data frame column with removed characters
        :rtype: pd.Series
        """
        return pd.Series([
            CleanUtils.remove_characters_for_text(text=text)
            for text in column_data
        ])
