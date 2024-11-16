import logging
import pandas as pd
import numpy as np
from zenml import step
from src.data_preprocessing import DataPreProcessing, DataCleaningStrategy, \
ImputingMissingValues, DroppingRedundantColumns, RemovingOutliers, \
OneHotEncoding, ExtractingTarget
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
    processed_data, dropped_columns = data_cleaning.handle_data()
    # print('shape after dropping Nan columns: ', processed_data.shape)

    impute_strategy = ImputingMissingValues()
    data_imputing = DataPreProcessing(processed_data, impute_strategy)
    processed_data, cols_imputed_vals_dict = data_imputing.handle_data()  
    # print('shape after imputing missing vals: ', processed_data.shape)

    drop_redundancy_strategy = DroppingRedundantColumns()
    data_redundancy = DataPreProcessing(processed_data, drop_redundancy_strategy)
    processed_data, redundant_columns = data_redundancy.handle_data() 
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
    processed_data, ohe_transformer, ordinal_variables, feat_names = data_ohe.handle_data() 
    # print('shape after OHE: ', processed_data.shape)

    # print('final shape of training data: ', processed_data.shape)

    return processed_data, target
  except Exception as e:
    logging.error(f'Error in preprocessing data (steps): {e}')
    raise e
