import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.impute import SimpleImputer
from collections import Counter
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame, cols: list) -> pd.DataFrame:
        pass

class TestDataCleaningStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, dropped_columns: list) -> pd.DataFrame:
        try:
            data.drop(columns=dropped_columns, inplace = True)
            return data 
        except Exception as e:
            logging.error(f'Error in dropping null columns in test data (src): {e}')
            raise e

class TestDataCleaningStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, dropped_columns: list) -> pd.DataFrame:
        try:
            data.drop(columns=dropped_columns, inplace = True)
            return data 
        except Exception as e:
            logging.error(f'Error in dropping null columns in test data (src): {e}')
            raise e    

class TestDataDroppingRedundantColumns(DataStrategy):
    def handle_data(self, data: pd.DataFrame, redundant_columns: list) -> pd.DataFrame:
        try:
            data.drop(columns=redundant_columns, inplace = True)
            return data 
        except Exception as e:
            logging.error(f'Error in dropping redundant columns in test data (src): {e}')
            raise e