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

class EncodingStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame, transformer, 
    ordinal_variables: list, ohe_transformed_feat_names: list) -> pd.DataFrame:
        pass

class TestDataOneHotEncoding(EncodingStrategy):
    def handle_data(self, data: pd.DataFrame, ohe_transformer, 
    ordinal_variables: list, feat_names: list):
        try:
            def ordinal_encoding(df, ordinal_variables):
                rating_to_numerical_dict = {
                  'Ex': 5,
                      'Gd': 4,
                      'TA': 3,
                      'Fa': 2,
                      'Po': 1,
                      'NA': 0,
                      'GLQ':4,
                      'ALQ':3,
                      'BLQ':2,
                      'Rec':3,
                      'LwQ':2,
                      'Unf':1,
                      'Av': 3,
                      'Mn': 2,
                      'No': 1,
                      'Fin': 3,
                      'RFn': 2,
                }
                # iterating over ordinal columns
                for col in ordinal_variables:
                    col_values = df[col].values
                    # converting ordered categorical columns to numerical columns.
                    ordinal_values = [rating_to_numerical_dict[val] if val in rating_to_numerical_dict.keys() else val for val in col_values]
                    # naming the new columns by adding '_ordinal' suffix to original column names
                    df[col + '_ordinal'] = ordinal_values
                
                df.drop(columns = ordinal_variables, inplace = True)
                return df

            data = ordinal_encoding(data, ordinal_variables)
            ohe_data_arr = ohe_transformer.transform(data)
            transformed_data = pd.DataFrame(data = ohe_data_arr, columns = feat_names)
            logging.info('shape of input data after one hot encoding: ', transformed_data.shape)

            return transformed_data

        except Exception as e:
            logging.error(f'Error performing one hot encoding (src): {e}')
            raise e
