import logging
import pandas as pd
import numpy as np
from zenml import step
from src.data_preprocessing import DataPreProcessing, DataCleaningStrategy, \
ImputingMissingValues, DroppingRedundantColumns, RemovingOutliers, \
OneHotEncoding, ExtractingTarget, FeatureSelection, FeatureScaling
from typing_extensions import Annotated
from typing import Tuple 

@step
def preprocess_df(df: pd.DataFrame) -> Tuple[
  Annotated[pd.DataFrame, 'processed_data'],
  Annotated[np.ndarray, 'target'],
  ]:
  try:
    # print('Original shape: ', df.shape)
    clean_strategy = DataCleaningStrategy()
    data_cleaning = DataPreProcessing(df, clean_strategy)
    processed_data, _ = data_cleaning.handle_data()
    # print('shape after dropping Nan columns: ', processed_data.shape)

    impute_strategy = ImputingMissingValues()
    data_imputing = DataPreProcessing(processed_data, impute_strategy)
    processed_data, _ = data_imputing.handle_data()  
    # print('shape after imputing missing vals: ', processed_data.shape)

    drop_redundancy_strategy = DroppingRedundantColumns()
    data_redundancy = DataPreProcessing(processed_data, drop_redundancy_strategy)
    processed_data, _ = data_redundancy.handle_data() 
    # print('shape after dropping redundant columns: ', processed_data.shape)

    drop_outliers_strategy = RemovingOutliers()
    data_outliers = DataPreProcessing(processed_data, drop_outliers_strategy)
    processed_data = data_outliers.handle_data() 
    # print('shape after dropping outliers: ', processed_data.shape)

    extract_target_strategy = ExtractingTarget()
    data_target = DataPreProcessing(processed_data, extract_target_strategy)
    processed_data, target = data_target.handle_data() 
    # print('shape after dropping target: ', processed_data.shape)

    ohe_strategy = OneHotEncoding()
    data_ohe = DataPreProcessing(processed_data, ohe_strategy)
    processed_data, _, _, _ = data_ohe.handle_data() 
    # print('shape after OHE: ', processed_data.shape)

    feature_selection_strategy = FeatureSelection()
    data_uncorrelated = DataPreProcessing(processed_data, feature_selection_strategy)
    processed_data, _ = data_uncorrelated.handle_data()
    # logging.info(f'Shape of pruned dataset: {X_train.shape}')

    feature_scaling_strategy = FeatureScaling()
    data_scaled = DataPreProcessing(processed_data, feature_scaling_strategy)
    processed_data, _ = data_scaled.handle_data()

    # print('final shape of training data: ', processed_data.shape)

    return processed_data, target
  except Exception as e:
    logging.error(f'Error in preprocessing data (steps): {e}')
    raise e
