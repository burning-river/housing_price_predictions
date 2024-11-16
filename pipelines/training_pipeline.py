from zenml import pipeline
from steps.load_data import load_df
from steps.preprocessing import preprocess_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache = True)
def train_pipeline(data_path: str):
    df = load_df(data_path)
    clean_df, target = preprocess_df(df)
    data_scaled, model = train_model(clean_df, target)
    mean_r2, std_r2, mean_rmse, std_rmse = evaluate_model(data_scaled, target, model)