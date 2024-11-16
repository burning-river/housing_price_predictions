import logging
import pandas as pd
import numpy as np
import mlflow
from zenml import step
from src.model_dev import FeatureSelection, FeatureScaling, ModelTraining
from typing_extensions import Annotated
from typing import Tuple 
from sklearn.base import RegressorMixin
# from .config import ModelNameConfig
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker 

@step(experiment_tracker = experiment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[
Annotated[pd.DataFrame, 'X_train_scaled'],
Annotated[RegressorMixin, 'trained_model'],
]:
    try:
        # logging.info(f'Shape of initial dataset: {X_train.shape}')
        X_train, y_train, correlated_features = FeatureSelection().train_data(X_train, y_train)
        # logging.info(f'Shape of final dataset: {X_train.shape}')
        # print('shape after dropping correlated features: ', X_train.shape)
        # for col in correlated_features:
        #     print(col)

        X_train_scaled, y_train, X_train_normalizer = FeatureScaling().train_data(X_train, y_train)
        # logging.info(list(X_train_scaled.columns))
        # model = None
        # if config.model_name == 'GradientBoostingRegressor':
        # print('final shape of training data: ', X_train_scaled.shape)
        mlflow.sklearn.autolog()
        model = ModelTraining()
        trained_model = model.train_data(X_train_scaled, y_train)     

        return X_train_scaled, trained_model

    except Exception as e:
        logging.error(f'Error in model development (src): {e}')
        raise e      
