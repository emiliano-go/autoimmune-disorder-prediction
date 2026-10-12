from .ingestion import get_dataset, save_dataset

def main():

    df = get_dataset(dataset="raw")

    antibody_cols = ["Anti-dsDNA", "Anti-Sm", "Rheumatoid factor", "ACPA",
                     "Anti-TPO", "Anti-Tg", "Anti-SMA"]
    df[antibody_cols] = df[antibody_cols].fillna(0)

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols + ["Patient_ID"])

    save_dataset(df)

if __name__ == "__main__":
    main()