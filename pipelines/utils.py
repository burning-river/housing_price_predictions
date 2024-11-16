import logging
import pandas as pd
from zenml import step
from src.data_preprocessing import DataPreProcessing, DataCleaningStrategy, \
ImputingMissingValues, DroppingRedundantColumns, OneHotEncoding, RemovingOutliers
from src.preprocess_test import TestDataCleaningStrategy, \
TestDataDroppingRedundantColumns
from src.impute_test import ImputeMissingVals
from src.model_dev import FeatureSelection, FeatureScaling
from src.ohe_test import TestDataOneHotEncoding
from src.feature_selection_test import DropCorrelatedFeatures
from src.normalize_test import Normalize

@step
def get_data_for_test():
    try:
        train_df = pd.read_csv('/content/drive/MyDrive/housing_price_predictions/data/Housing_Data_Train.csv')
        test_df = pd.read_csv("/content/drive/MyDrive/housing_price_predictions/data/Housing_Data_Test.csv")
        # test_df = test_df.sample(n=100)
        # print('Shape of original train data: ', train_df.shape)
        # print('Shape of original test data: ', test_df.shape)

        clean_strategy = DataCleaningStrategy()
        data_cleaning = DataPreProcessing(train_df, clean_strategy)
        train_data, dropped_columns = data_cleaning.handle_data()

        test_data = TestDataCleaningStrategy().handle_data(test_df, dropped_columns)
        # print('Train shape after dropping Nan columns: ', train_data.shape)
        # print('Test shape after dropping Nan columns: ', test_data.shape)
        
        impute_strategy = ImputingMissingValues()
        data_imputing = DataPreProcessing(train_data, impute_strategy)
        train_data, cols_imputed_vals_dict = data_imputing.handle_data() 

        test_data = ImputeMissingVals().handle_data(test_data, cols_imputed_vals_dict)

        drop_redundancy_strategy = DroppingRedundantColumns()
        data_redundancy = DataPreProcessing(train_data, drop_redundancy_strategy)
        train_data, redundant_columns = data_redundancy.handle_data() 

        test_data = TestDataDroppingRedundantColumns().handle_data(test_data, redundant_columns)
        # print('Train shape after dropping redundant columns: ', train_data.shape)
        # print('Test shape after dropping redundant columns: ', test_data.shape)

        drop_outliers_strategy = RemovingOutliers()
        data_outliers = DataPreProcessing(train_data, drop_outliers_strategy)
        train_data = data_outliers.handle_data() 
        # print('shape after dropping outliers: ', train_data.shape)
    
        train_data.drop(columns=['SalePrice'], inplace = True)
        if 'SalePrice' in test_data.columns:
            test_data.drop(columns=['SalePrice'], inplace = True)
        # print('Shape of train data after dropping target: ', train_data.shape)
        # print('Shape of test data after dropping target: ', test_data.shape)

        ohe_strategy = OneHotEncoding()
        data_ohe = DataPreProcessing(train_data, ohe_strategy)
        train_data, ohe_transformer, ordinal_variables, feat_names = data_ohe.handle_data() 

        test_data = TestDataOneHotEncoding().handle_data(test_data, ohe_transformer, ordinal_variables, feat_names)
        # print('Shape of train data after OHE: ', train_data.shape)
        # print('Shape of test data after OHE: ', test_data.shape)

        train_data, [], correlated_features = FeatureSelection().train_data(train_data, [])
        test_data = DropCorrelatedFeatures().handle_data(test_data, correlated_features)
        # for col in correlated_features:
        #     print(col)
        # print('Shape of train data after dropping correlated features: ', train_data.shape)
        # print('Shape of test data after dropping correlated features: ', test_data.shape)
            
        train_data, [], normalizer = FeatureScaling().train_data(train_data, [])
        test_data = Normalize().handle_data(test_data, normalizer)
        # print('Shape of final train data: ', train_data.shape)
        # print('Shape of final test data: ', test_data.shape)

        # print('NAN values in train data', train_data.isnull().values.any())
        # print('NAN values in test data', test_data.isnull().values.any())

        result = test_data.to_json(orient="split")
        return result

    except Exception as e:
        logging.error(e)
        raise e
