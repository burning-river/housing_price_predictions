import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.impute import SimpleImputer
from collections import Counter
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline 

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        pass

class DataCleaningStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, list]:
       try:
           data.drop(columns=['Unnamed: 0', 'Id'], inplace = True)

           categorical_columns = ['OverallQual', 'OverallCond', 'BsmtFullBath', 
                   'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                   'GarageCars', 'MoSold', 'YrSold']
           categorical_columns += list(data.select_dtypes(include='object').columns)
           numerical_columns = [col for col in data.columns if col not in categorical_columns]

           dropped_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
           data.drop(columns=dropped_columns, inplace=True)
           dropped_columns.append('Unnamed: 0')
           dropped_columns.append('Id')
           
           return data, dropped_columns
       except Exception as e:
           logging.error(f'Error in dropping columns (src): {e}')
           raise e

class ImputingMissingValues(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
       try:
           def missing_vals(df):
              columns_with_missing_vals = []

              # Iterate over each column in the housing_train_data dataframe
              for col in df.columns.values:
                  # Count the number of missing values in the column
                  num_missing_vals = df[col].isna().sum()
                  # Check if the column has any missing values
                  if num_missing_vals > 0:
                        # Add the column name to the list of columns with missing values
                      columns_with_missing_vals.append(col)
              return columns_with_missing_vals

           def fill_missing_values(df, columns_with_missing_vals):
              # Looping through the columns with missing values
              for ind, col in enumerate(columns_with_missing_vals):
                    # Imputer converts column values to string. Converting string to float for relevant columns
                    if col in ['MasVnrArea', 'GarageYrBlt', 'LotFrontage']:
                        # Converting the values to float16 and assigning them to the reduced dataset
                        df[col] = arr[:, ind].astype(np.float16)
                    else:
                        # Assigning the values to the reduced dataset
                        df[col] = arr[:, ind]
              return df

           columns_with_missing_vals = missing_vals(data)
           data = data.replace(np.nan, None)
           imp = SimpleImputer(missing_values=None, strategy='most_frequent').fit(data[columns_with_missing_vals])
           arr = imp.transform(data[columns_with_missing_vals])

           data_imputed = data.drop(columns = columns_with_missing_vals)
           data_imputed = fill_missing_values(data_imputed, columns_with_missing_vals)
           imp_val_missing_col = {col:imp.statistics_[i] for i, col in enumerate(columns_with_missing_vals)}
           # Asserting the shape of the imputed dataset is same as original dataset
           assert data_imputed.shape == data.shape

           return data_imputed, imp_val_missing_col

       except Exception as e:
           logging.error(f'Error in imputing data (src): {e}')
           raise e

class DroppingRedundantColumns(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, list]:
        try:
            def find_redundant_columns(df):
                '''
                The function uses the Counter class to count the occurrences of each value
                in the input array. It then iterates over the keys of the col_values
                dictionary, which represent the unique values in the array.
                                    
                For each value, the function calculates the ratio of its occurrence to the
                total number of rows in the housing_train_data dataset. If this ratio is
                greater than 0.95 (indicating that the value is repeated more than 95% of
                the time), the function returns True.
                '''
                redundant_columns = []
                for col in df.columns.values:
                    arr = df[col].values
                    col_values = Counter(arr)
                    for val in col_values.keys():
                        if col_values[val]/df.shape[0] > 0.95:
                            redundant_columns.append(col)
                        
                return redundant_columns
            
            redundant_columns = find_redundant_columns(data) 
            data.drop(columns=redundant_columns, inplace=True)

            return data, redundant_columns

        except Exception as e:
            logging.error(f'Error in dropping redundant columns (src): {e}')
            raise e

class RemovingOutliers(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Remove records with extreme values
            feature = 'LotArea'
            # Set the threshold value
            threshold = 100000
            drop_indices = data[data[feature] > threshold].index.values
            data.drop(drop_indices, axis = 0, inplace = True)

            return data
        except Exception as e:
            logging.error(f'Error removing outliers (src): {e}')
            raise e

class ExtractingTarget(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        try:
            target = data['SalePrice'].values
            data.drop(columns = ['SalePrice'], inplace = True) 

            return data, target
        except Exception as e:
            logging.error(f'Error extracting target variable (src): {e}')
            raise e

class OneHotEncoding(DataStrategy):
    def handle_data(self, data: pd.DataFrame):
        try:
            def find_ordinal_columns(df):
                # Initialize an empty list to store the ordinal variables
                ordinal_variables = []
                # Iterate through each column in the string_columns list
                string_columns = df.select_dtypes(include='O').columns.values
                for col in string_columns:
                  # Check if the column contains 'Ex' or 'Gd' which suggest that columns are ordinal
                  if 'Ex' in df[col].values or 'Gd' in df[col].values:
                      # addig column to the list of ordinal columns
                      ordinal_variables.append(col)

                return ordinal_variables

            def ordinal_encoding(df):
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
                ordinal_variables = find_ordinal_columns(df)
                # iterating over ordinal columns
                for col in ordinal_variables:
                    col_values = df[col].values
                    # converting ordered categorical columns to numerical columns.
                    ordinal_values = [rating_to_numerical_dict[val] if val in rating_to_numerical_dict.keys() else val for val in col_values]
                    # naming the new columns by adding '_ordinal' suffix to original column names
                    df[col + '_ordinal'] = ordinal_values
                
                df.drop(columns = ordinal_variables, inplace = True)
                return df, ordinal_variables

            def ohe_transform_func(df):
                '''
                The function takes an input df, which represents the data to be transformed
                
                The function returns three values: 
                df_transform: which represents the transformed data, 
                df_transformer: which represents the fitted transformer object, and 
                feature_names: which returns the feature names generated by the transformer.
                '''
                # making a list of numerical columns that don't need any transformation
                numerical_columns = df.select_dtypes(include = [np.number]).columns.values
                # making a list of nominal columns that need transformation using one hot encoding technique
                categorical_columns = [col for col in df.columns.values if col not in numerical_columns]
                
                # using OHE to convert nominal values to numerical
                # dropping the first OHE column for each categorical feature because it is redundant
                ohe = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')
                # creating a transformer object that performs transformations on specified columns
                features = ColumnTransformer([('categorical', ohe, categorical_columns), ('passthrough','passthrough',numerical_columns)],verbose_feature_names_out=False)
                # creating a Pipeline object that takes a transformer object as input 
                transformer = Pipeline([('features',features)])
                # fitting the transformer on the data
                ohe_transformer = transformer.fit(df)   
                # transforming the input data
                ohe_transform_arr = ohe_transformer.transform(df)
                # generating new featrue names after OHE
                feature_names = transformer.named_steps['features'].get_feature_names_out()
                
                # print(transformer.get_feature_names_out)
                return ohe_transform_arr, ohe_transformer, feature_names

            # making list of columns containing strings/categories 
            # and numerical values
            data, ordinal_variables = ordinal_encoding(data)
            ohe_transform_arr, ohe_transformer, feat_names = ohe_transform_func(data)
            transformed_data = pd.DataFrame(data = ohe_transform_arr, columns = feat_names)
            logging.info('shape of input data after one hot encoding: ', transformed_data.shape)

            return transformed_data, ohe_transformer, ordinal_variables, feat_names

        except Exception as e:
            logging.error(f'Error performing one hot encoding (src): {e}')
            raise e

class FeatureSelection(DataStrategy):
    def handle_data(self, X_train: pd.DataFrame) -> Union[pd.DataFrame, list]:
        try:
            def find_correlated_pairs(X):
                # Initialize an empty list to store correlated feature pairs
                correlated_pairs = []
                # Calculate the correlation matrix for the normalized training data
                train_set_corr_mat = X.corr()
                # print('correlated pairs of features: \n')
                # Iterate over the columns of the correlation matrix
                for row in train_set_corr_mat.columns.values:
                    # Iterate over the columns of the correlation matrix
                    for col in train_set_corr_mat.columns.values:
                          # Check if the absolute value of the correlation coefficient is greater than 0.9 and the features are not the same
                        if abs(train_set_corr_mat.loc[row, col] ) > 0.9 and row != col:
                            # print(row, col, round(train_set_corr_mat.loc[row, col], 3))
                            # Add the correlated feature pair to the list
                            correlated_pairs.append((row, col))
                correlated_pairs = [correlated_pairs[i] for i in range(0,len(correlated_pairs),2)]

                return correlated_pairs

            correlated_pairs = find_correlated_pairs(X_train)
            correlated_cols = [ent[0] for ent in correlated_pairs]
            X_train.drop(columns=correlated_cols, inplace=True)

            return X_train, correlated_cols
        except Exception as e:
            logging.error(f'Error Selecting Features (src): {e}')
            raise e

class FeatureScaling(DataStrategy):
    def handle_data(self, X_train: pd.DataFrame):
        try:
            def normalization(X):
                '''
                The function takes an input X, which represents the data to be scaled
                
                The function returns two values: 
                X_transform: which represents the scaled data, 
                X_transformer: which represents the fitted max absolute scaling transformer object, and 
                '''    
                # Define the columns that will be passed through for scaling
                passthrough_columns = X.columns.values
                features = ColumnTransformer([('passthrough','passthrough',passthrough_columns)])
                # Create a Pipeline object to chain multiple transformers together
                transformer = Pipeline([('features',features), ('scaling', MaxAbsScaler())])
                # fitting the scaling transformer on the dataset
                X_transformer = transformer.fit(X)  
                # applying the scaling on the dataset
                X_transform = X_transformer.transform(X)
                  
                return X_transform, X_transformer

            X_train_norm, X_train_normalizer = normalization(X_train)
            norm_X_train_data_df = pd.DataFrame(data = X_train_norm, columns = list(X_train.columns))

            return norm_X_train_data_df, X_train_normalizer 

        except Exception as e:
            logging.info(f'Error Scaling features (src): {e}')
            raise e

class DataPreProcessing:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> pd.DataFrame:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f'Error in handling data (src): {e}')
            raise e