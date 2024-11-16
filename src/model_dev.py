import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

class Model(ABC):
    @abstractmethod
    def train_data(self, X_train: pd.DataFrame, y_train: np.ndarray):
        pass

class FeatureSelection(Model):
    def train_data(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Union[pd.DataFrame, np.ndarray, list]:
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

            return X_train, y_train, correlated_cols
        except Exception as e:
            logging.error(f'Error Selecting Features (src): {e}')
            raise e

class FeatureScaling(Model):
    def train_data(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Union[pd.DataFrame, np.ndarray]:
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

            return norm_X_train_data_df, y_train, X_train_normalizer 

        except Exception as e:
            logging.info(f'Error Scaling features (src): {e}')
            raise e

class ModelTraining(Model):
    def train_data(self, X_train: pd.DataFrame, y_train: np.ndarray):
        try:
            param_grid = {'max_depth': [1,3,5,7],
                        'n_estimators': [50, 100, 150],
                        'learning_rate': [0.01, 0.1, 1.]} 

            gbr = GradientBoostingRegressor()
            grid_search = GridSearchCV(gbr, param_grid, cv = 5, scoring = 'neg_root_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            gb_best_est = grid_search.best_estimator_
            gb_best_est = gb_best_est.fit(X_train, y_train)
            
            return gb_best_est

        except Exception as e:
            logging.error(f'Error Training model (src): {e}')
            raise e
