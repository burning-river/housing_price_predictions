import logging
import pandas as pd
import numpy as np
import mlflow
from zenml import step
from src.model_dev import ModelTraining
from typing_extensions import Annotated
from typing import Tuple 
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker 

@step(experiment_tracker = experiment_tracker.name)
def train_model(X_train: pd.DataFrame, 
y_train: np.ndarray,
config: ModelNameConfig,
) -> Annotated[RegressorMixin, 'trained_model']:
    try:
        model = None
        if config.model_name == 'GradientBoostingRegressor':
        # print('final shape of training data: ', X_train_scaled.shape)
            mlflow.sklearn.autolog()
            model = ModelTraining()
            trained_model = model.train_data(X_train, y_train)     

            return trained_model

    except Exception as e:
        logging.error(f'Error in model training (steps): {e}')
        raise e      
