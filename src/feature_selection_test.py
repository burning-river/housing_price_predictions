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
    def handle_data(self, data: pd.DataFrame, correlated_features: list) -> pd.DataFrame:
        pass

class DropCorrelatedFeatures(Model):
    def handle_data(self, data: pd.DataFrame, correlated_features: list):
        try:
            data.drop(columns=correlated_features, inplace=True)
            return data
        except Exception as e:
            logging.error(f'Error removing correlated features (src): {e}')
            raise e
