from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset specific
RAW_DATASET_PATH = RAW_DATA_DIR / "Autoimmune_Disorder_10k_with_All_Disorders.csv"
CLEANED_DATASET_PATH = PROCESSED_DATA_DIR / "autoimmune_cleaned.parquet"

# Target column
TARGET_COL = "Diagnosis"

# Random seed
RANDOM_STATE = 42