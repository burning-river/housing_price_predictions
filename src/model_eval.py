import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, model: RegressorMixin):
        pass

class cross_validation(Evaluation):
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, model: RegressorMixin):
        try:
            r2_scores = cross_val_score(model, X, y, cv=5, scoring = 'r2')
            rmse_scores = cross_val_score(model, X, y, cv=5, scoring = 'neg_root_mean_squared_error')

            return r2_scores, rmse_scores
        except Exception as e:
            logging.error(f'Error evaluating model (src): {e}')
            raise e