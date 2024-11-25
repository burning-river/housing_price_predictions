import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

class Model(ABC):
    @abstractmethod
    def train_data(self, X_train: pd.DataFrame, y_train: np.ndarray):
        pass

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
