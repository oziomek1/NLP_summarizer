import logging
import pandas as pd

from tqdm import tqdm
from typing import Any
from typing import Dict

from nlper.utils.trim_utils import TrimUtils


tqdm.pandas(desc="Trimming")


class Trimmer:
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Trimmer.__name__)
        self.config = config
        self.data = data

    def trim_dataframe(self) -> pd.DataFrame:
        self.apply_lower_length_limit()
        self.apply_upper_length_limit()
        return self.data

    def apply_lower_length_limit(self):
        for column_name in self.data:
            threshold_executor = TrimUtils.remove_text_below_lower_length_threshold(
                self.config[f'{column_name}_lower_length_limit']
            )
            self.data = self.data[self.data[column_name].map(threshold_executor)]
        self.data.reset_index(drop=True, inplace=True)

    def apply_upper_length_limit(self):
        trim_utils = TrimUtils()
        trim_utils.lang_model = self.config['language_model']
        for column_name in self.data:
            self.data[column_name] = self.trim_text_for_column(
                column_data=self.data[column_name],
                threshold=self.config[f'{column_name}_upper_length_limit'],
                trim_utils=trim_utils,
            )

    @staticmethod
    def trim_text_for_column(column_data: pd.Series, threshold: int, trim_utils: TrimUtils) -> pd.Series:
        return column_data.progress_map(lambda x: trim_utils.trim_text_to_upper_length_threshold(x, threshold))
