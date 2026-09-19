import kagglehub
from ..config import RAW_DATASET_PATH

def download():

    if RAW_DATASET_PATH.exists():
        print(f"Already exists: {RAW_DATASET_PATH}")
        return

    kagglehub.dataset_download(
        "abdullahragheb/all-autoimmune-disorder-10k",
        path="Autoimmune_Disorder_10k_with_All_Disorders.csv",
        output_dir="data/raw",
    )