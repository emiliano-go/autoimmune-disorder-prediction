import kagglehub
from pathlib import Path


def download():
    output = Path("data/Autoimmune_Disorder_10k_with_All_Disorders.csv")

    if output.exists():
        print(f"Already exists: {output}")
        return

    kagglehub.dataset_download(
        "abdullahragheb/all-autoimmune-disorder-10k",
        path="Autoimmune_Disorder_10k_with_All_Disorders.csv",
        output_dir="data",
    )