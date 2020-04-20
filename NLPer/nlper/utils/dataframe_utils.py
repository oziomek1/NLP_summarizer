import pandas as pd
from enum import Enum


class EnumWithListing(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class OutputColumns(EnumWithListing):
    Text = 'text'
    Summary = 'summary'


class ColumnsWithDuplicates(EnumWithListing):
    Lead = 'lead'
    Text = 'text'


class DataFrameUtils:
    @staticmethod
    def drop_columns(dataframe, columns, inplace=True) -> None:
        return dataframe.drop(columns=columns, inplace=inplace)

    @staticmethod
    def remove_empty_rows(dataframe) -> pd.DataFrame:
        for column in dataframe:
            dataframe = dataframe[dataframe[column].map(lambda cell: len(cell) > 0)]
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe
