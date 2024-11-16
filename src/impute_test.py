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

class Impute(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame, cols_imputed_vals_dict: dict) -> pd.DataFrame:
        pass

class ImputeMissingVals(Impute):
    def handle_data(self, data: pd.DataFrame, cols_imputed_vals_dict: dict):
        try:
            # imputing missing values in test set columns from parameters obtained from training set
            for col in cols_imputed_vals_dict.keys():
                data[col].fillna(cols_imputed_vals_dict[col], inplace=True)
            return data
        except Exception as e:
            logging.error(f'Error removing correlated features (src): {e}')
            raise e
