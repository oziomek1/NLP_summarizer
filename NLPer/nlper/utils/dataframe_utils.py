import pandas as pd

from enum import Enum
from typing import Optional


class EnumWithListing(Enum):
    """
    Enum supporting listing of elements value
    """
    @classmethod
    def list(cls):
        """
        List enum elements value
        :return: list of values
        :rtype: list
        """
        return list(map(lambda c: c.value, cls))


class OutputColumns(EnumWithListing):
    Text = 'text'
    Summary = 'summary'


class ColumnsWithDuplicates(EnumWithListing):
    Lead = 'lead'
    Text = 'text'


class DataFrameUtils:
    """
    Utils for pandas data frame
    """
    @staticmethod
    def drop_columns(dataframe: pd.DataFrame, columns: list, inplace=True) -> Optional[pd.DataFrame]:
        """
        Drops particular columns
        :param dataframe: Data frame to drops columns from
        :type dataframe: pd.DataFrame
        :param columns: list of columns to drop
        :type columns: list
        :param inplace: boolean flag whether to drop inplace, default True
        :type inplace: bool
        :return: Data frame with dropped columns
        :rtype: pd.DataFrame, optional
        """
        return dataframe.drop(columns=columns, inplace=inplace)

    @staticmethod
    def remove_empty_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Removes empty rows from data frame
        :param dataframe: Data frame to remove empty rows
        :type dataframe: pd.DataFrame
        :return: Data frame with dropped rows
        :rtype: pd.DataFrame
        """
        for column in dataframe:
            dataframe = dataframe[dataframe[column].map(lambda cell: len(cell) > 0)]
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe
