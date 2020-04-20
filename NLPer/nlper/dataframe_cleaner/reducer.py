import logging
import pandas as pd

from typing import List
from typing import Optional

from nlper.utils.dataframe_utils import DataFrameUtils
from nlper.utils.dataframe_utils import ColumnsWithDuplicates
from nlper.utils.dataframe_utils import OutputColumns


class Reducer:
    def __init__(self, config, data):
        self.logger = logging.getLogger(Reducer.__name__)
        self.config = config
        self.data = data

    def reduce_dataframe(self) -> pd.DataFrame:
        self.data = DataFrameUtils.drop_columns(self.data, self.config['columns_to_skip'], inplace=False)
        self.unify_dataframe_content()
        self.organize_columns()
        self.data = DataFrameUtils.remove_empty_rows(self.data)
        return self.data

    def merge_or_create_column(self, column, series) -> List:
        if series is None:
            series = self.data[column]
        else:
            series = series + self.data[column]
        return series

    def merge_columns_as_text_or_summary(self, columns_to_merge_on) -> Optional[pd.Series]:
        merged = None
        for column in self.data:
            if columns_to_merge_on and column in columns_to_merge_on:
                merged = self.merge_or_create_column(column, merged)
        return merged

    def merge_columns(self):
        data_text = self.merge_columns_as_text_or_summary(self.config['columns_to_merge_as_text'])
        data_summary = self.merge_columns_as_text_or_summary(self.config['columns_to_merge_as_summary'])
        self.data = pd.concat([data_text, data_summary], keys=OutputColumns.list(), axis=1)

    def organize_columns(self) -> None:
        self.remove_duplicates_in_lead_and_text_columns()
        self.merge_columns()

    def remove_duplicates_in_lead_and_text_columns(self):
        for i, row in self.data[ColumnsWithDuplicates.list()].iterrows():
            if bool(set(row[ColumnsWithDuplicates.Lead.value]) & set(row[ColumnsWithDuplicates.Text.value])):
                self._remove_lead_from_text(row)

    def unify_dataframe_content(self) -> None:
        for column in self.data:
            self.data[column] = self._unify_column_content_to_list(column_data=self.data[column])

    @staticmethod
    def _remove_lead_from_text(row):
        text_to_remove = [
            text_index
            for text_index, text_value in enumerate(row.text)
            if text_value in row[ColumnsWithDuplicates.Lead.value]
        ]
        for text_index in reversed(text_to_remove):
            del row[ColumnsWithDuplicates.Text.value][text_index]
        return row[ColumnsWithDuplicates.Text.value]

    @staticmethod
    def _unify_column_content_to_list(column_data) -> pd.Series:
        column_data.fillna("", inplace=True)
        if isinstance(column_data[0], str):
            column_data = column_data.apply(lambda x: [x])
        return column_data
