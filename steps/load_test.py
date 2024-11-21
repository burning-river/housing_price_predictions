import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self) -> None:
        pass        
    def get_data(self):
        data_path = '/content/drive/MyDrive/housing_price_predictions/data/Housing_Data_Test.csv'
        df = pd.read_csv(data_path)
        return df
@step
def load_test_df() -> pd.DataFrame:
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f'Error while loading data: {e}')
        raise e

