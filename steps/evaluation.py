import logging
from zenml import step
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from src.model_eval import cross_validation
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker 

@step(experiment_tracker = experiment_tracker.name)
def evaluate_model(X: pd.DataFrame, y: np.ndarray, best_model: RegressorMixin
) -> Tuple[
Annotated[float, 'mean_r2'],
Annotated[float, 'std_r2'],
Annotated[float, 'mean_rmse'],
Annotated[float, 'std_rmse'],
]:
    try:
        r2_scores, rmse_scores = cross_validation().evaluate(X, y, best_model)
        mean_r2, std_r2 = np.mean(r2_scores), np.std(r2_scores)
        mean_rmse, std_rmse = np.mean(abs(rmse_scores)), np.std(rmse_scores)
        logging.info('Mean R2 = ', np.round(mean_r2, 2), ', Std Dev R2 = ', np.round(std_r2, 2))
        logging.info('Mean RMSE = ', np.round(mean_rmse, 2), ', Std Dev RMSE = ', np.round(std_rmse, 2))
        # mlflow.log_metric('mean R2', mean_r2)
        # mlflow.log_metric('std R2', std_r2)
        # mlflow.log_metric('mean RMSE', mean_rmse)
        # mlflow.log_metric('std RMSE', std_rmse)

        return mean_r2, std_r2, mean_rmse, std_rmse
    except Exception as e:
        logging.error(f'Error evaluating model (steps): {e}')
        raise e
