import logging
import pandas as pd

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from nlper.utils.dataframe_utils import DataFrameUtils
from nlper.utils.dataframe_utils import ColumnsWithDuplicates
from nlper.utils.dataframe_utils import OutputColumns


class Reducer:
    """
    Reduces data frame by removing unnecessary columns and rows.
    :param config: Configuration dictionary
    :type config: dict
    :param data: Raw data frame to reduce
    :type data: pd.DataFrame
    """
    def __init__(self, config: Dict[str, Any], data: pd.DataFrame):
        self.logger = logging.getLogger(Reducer.__name__)
        self.config = config
        self.data = data

    def reduce_dataframe(self) -> pd.DataFrame:
        """
        Executes data frame reducing process.
        Drops columns specified by config file.
        :return: Cleaned data frame
        :rtype: pd.DataFrame
        """
        self.data = DataFrameUtils.drop_columns(self.data, self.config['columns_to_skip'], inplace=False)
        self.unify_dataframe_content()
        self.organize_columns()
        self.data = DataFrameUtils.remove_empty_rows(self.data)
        return self.data

    def merge_or_create_column(self, column: pd.Series, series: Optional[pd.Series] = None) -> pd.Series:
        """
        Merges columns into single column

        :param column: Column to be merged
        :param column: pd.Series
        :param series: Existing column to merge on, if None then just assigns pd.Series
        :type series: pd.Series, optional
        :return: New merged column
        :rtype: pd.Series
        """
        if series is None:
            series = self.data[column]
        else:
            series = series + self.data[column]
        return series

    def merge_columns_as_text_or_summary(self, columns_to_merge_on: List[str]) -> Optional[pd.Series]:
        """
        Finds columns to merge by names and calls merging method.

        :param columns_to_merge_on: Names of columns to merge
        :type columns_to_merge_on: list
        :return: Merged column
        :rtype: pd.Series, optional
        """
        merged = None
        for column in self.data:
            if columns_to_merge_on and column in columns_to_merge_on:
                merged = self.merge_or_create_column(column, merged)
        return merged

    def merge_columns(self) -> None:
        """
        Calls merging the particular columns and concatenates them to new data frame.
        """
        data_text = self.merge_columns_as_text_or_summary(self.config['columns_to_merge_as_text'])
        data_summary = self.merge_columns_as_text_or_summary(self.config['columns_to_merge_as_summary'])
        self.data = pd.concat([data_text, data_summary], keys=OutputColumns.list(), axis=1)

    def organize_columns(self) -> None:
        """
        Calls removal of duplicated text and merge columns.
        """
        self.remove_duplicates_in_lead_and_text_columns()
        self.merge_columns()

    def remove_duplicates_in_lead_and_text_columns(self) -> None:
        """
        Checks if lead and text columns contain the same text, calls the removal method if so.
        """
        for i, row in self.data[ColumnsWithDuplicates.list()].iterrows():
            if bool(set(row[ColumnsWithDuplicates.Lead.value]) & set(row[ColumnsWithDuplicates.Text.value])):
                self._remove_lead_from_text(row)

    def unify_dataframe_content(self) -> None:
        """
        Splits data frame content unification to separate columns.
        """
        for column in self.data:
            self.data[column] = self._unify_column_content_to_list(column_data=self.data[column])

    @staticmethod
    def _remove_lead_from_text(row: pd.DataFrame) -> pd.DataFrame:
        """
        Obtains the duplicated text part in both lead and text columns. Removes the duplicated text from text column.

        :param row: Row with duplicated text
        :type row: pd.DataFrame
        :return: Row with removed text from text column
        :rtype: pd.DataFrame
        """
        text_to_remove = [
            text_index
            for text_index, text_value in enumerate(row.text)
            if text_value in row[ColumnsWithDuplicates.Lead.value]
        ]
        for text_index in reversed(text_to_remove):
            del row[ColumnsWithDuplicates.Text.value][text_index]
        return row[ColumnsWithDuplicates.Text.value]

    @staticmethod
    def _unify_column_content_to_list(column_data: pd.Series) -> pd.Series:
        """
        Unifies data format in column by converting to list.

        :param column_data: Column in data frame to unify.
        :type column_data: pd.Series
        :return: Unified data frame column
        :rtype: pd.Series
        """
        column_data.fillna("", inplace=True)
        if isinstance(column_data[0], str):
            column_data = column_data.apply(lambda x: [x])
        return column_data
