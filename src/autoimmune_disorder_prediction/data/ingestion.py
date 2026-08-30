import pandas as pd

from ..config import RAW_DATASET_PATH, CLEANED_DATASET_PATH

def get_dataset(dataset : str):

    if dataset.lower() == "raw":
        df = pd.read_csv(RAW_DATASET_PATH)

    elif dataset.lower() == "clean":
        df = pd.read_csv(CLEANED_DATASET_PATH)

    return df

def save_dataset(df : pd.DataFrame):
    df.to_parquet(CLEANED_DATASET_PATH, index=False)