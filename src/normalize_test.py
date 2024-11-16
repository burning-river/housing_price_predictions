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

class Model(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class Normalize(Model):
    def handle_data(self, data: pd.DataFrame, normalizer):
        try:
            columns = data.columns
            data_arr = normalizer.transform(data)
            data_df = pd.DataFrame(data = data_arr, columns = columns)
            return data_df
        except Exception as e:
            logging.error(f'Error normalizing test data (src): {e}')
            raise e
