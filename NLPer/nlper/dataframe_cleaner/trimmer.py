import logging
import numpy as np
import pandas as pd

from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from typing import Any
from typing import Dict

from nlper.utils.time_utils import timeit
from nlper.utils.trim_utils import TrimUtils


tqdm.pandas(desc="Trimming")


class Trimmer:
    """
    Trims texts by length in data frame.

    For texts with length below minimum threshold, whole text row is removed

    :param config: Configuration dictionary
    :type config: dict
    :param data: Data frame to trim
    :type data: pd.DataFrame
    """
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Trimmer.__name__)
        self.config = config
        self.data = data
        self.n_cores = cpu_count() // 2
        self.trim_utils = TrimUtils()

    @timeit
    def trim_dataframe(self) -> pd.DataFrame:
        """
        Executes data frame text length trimming process.
        :return: Trimmed length data frame
        :rtype: pd.DataFrame
        """
        self.remove_below_lower_length_limit()
        self.trim_to_upper_length_limit()
        return self.data

    def remove_below_lower_length_limit(self):
        """
        Calls removal of rows where text if its length is below set threshold value.
        For each column we apply different minimum text length threshold values.

        The index of data frame is reset after all removal operations.
        """
        for column_name in self.data:
            threshold_executor = TrimUtils.remove_text_below_lower_length_threshold(
                self.config[f'{column_name}_lower_length_limit']
            )
            self.data = self.data[self.data[column_name].map(threshold_executor)]
        self.data.reset_index(drop=True, inplace=True)

    def trim_to_upper_length_limit(self):
        """
        Applies parallelization of text length trimming for data frame using python multiprocessing.
        Trimming process is computationally expensive and thus parallelization greatly reduces the required time.
        """
        self.trim_utils.lang_model = self.config['language_model']

        dataframe_splits = np.array_split(self.data, self.n_cores)
        pool = Pool(self.n_cores)
        self.data = pd.concat(pool.map(self.trim_text_for_dataframe, dataframe_splits))
        pool.close()
        pool.join()

    def trim_text_for_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Splits text length trimming in data frame into separated columns.

        :param data: Data frame to trim length
        :type data: pd.DataFrame
        :return: Data frame with trimmed text
        :rtype: pd.DataFrame
        """
        for column_name in data:
            data[column_name] = self.trim_text_for_column(
                column_data=data[column_name],
                threshold=self.config[f'{column_name}_upper_length_limit'],
                trim_utils=self.trim_utils,
            )
        return data

    @staticmethod
    def trim_text_for_column(column_data: pd.Series, threshold: int, trim_utils: TrimUtils) -> pd.Series:
        """
        Calls text length trimming on single data frame column using trimming utils.
        Method uses ``progress_map`` to visualize progress of length trimming using tqdm.

        :param column_data:
        :param threshold:
        :param trim_utils:
        :return:
        """
        return column_data.progress_map(lambda x: trim_utils.trim_text_to_upper_length_threshold(x, threshold))
