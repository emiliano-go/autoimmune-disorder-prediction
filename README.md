# Autoimmune Disorder Prediction

A machine-learning exploration project built around the
[All Autoimmune Disorder 10k](https://www.kaggle.com/datasets/abdullahragheb/all-autoimmune-disorder-10k)
dataset from Kaggle. It provides a small Python package for downloading, ingesting and
cleaning the data, plus Jupyter notebooks for exploratory data analysis and feature
engineering.

## Project structure

```text
.
├── data/
│   ├── raw/                  # Original CSV downloaded from Kaggle
│   └── processed/            # Cleaned datasets (Parquet)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_feature.ipynb      # Feature engineering experiments
│   ├── 03_eda_visual.ipynb   # Visual EDA: distributions, groups and correlations
│   ├── 04_model_evaluation.ipynb # Baseline vs ensemble model comparison
│   └── 05_unsupervised_learning.ipynb # PCA, K-Means and DBSCAN exploration
├── docs/
│   ├── 01_problem_framing.md # ML opportunity, data, target and bias analysis
│   └── 02_unsupervised_analysis.md # Clustering, PCA, business and ethics case study
├── src/autoimmune_disorder_prediction/
│   ├── config.py             # Paths, column names and constants
│   ├── clustering.py         # Reusable PCA, K-Means and DBSCAN helpers
│   ├── evaluation.py         # Reusable binary evaluation pipeline
│   └── data/
│       ├── download_data.py  # Download the Kaggle dataset
│       ├── ingestion.py      # Load raw/cleaned data and save Parquet files
│       └── cleaning.py       # Cleaning pipeline entry point
├── pyproject.toml            # Project metadata and dependencies
└── uv.lock                   # Locked dependency tree
```

## Requirements

- Python 3.12.7 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and environment management

## Installation

Clone the repository and install the project with uv:

```bash
uv sync
```

## Usage

### Download the dataset

```bash
uv run download
```

This downloads `Autoimmune_Disorder_10k_with_All_Disorders.csv` from Kaggle into
`data/raw/`.

### Clean the dataset

```bash
uv run python -m autoimmune_disorder_prediction.data.cleaning
```

The cleaned data is written to `data/processed/autoimmune_cleaned.parquet`.

### Run the notebooks

```bash
uv run jupyter lab notebooks/
```

`03_eda_visual.ipynb` reúne el análisis visual del dataset limpio: distribuciones,
comparación `Normal` vs. `Autoinmune`, correlaciones y desbalance de clases.

## Problem framing

The project is framed as a triage support tool: first `Normal` vs. `Autoimmune`, then
multiclass routing among the 115 diseases. The objective, required variables, learning
type, leakage and bias risks are documented in
[`docs/01_problem_framing.md`](docs/01_problem_framing.md).

## Model evaluation

`notebooks/04_model_evaluation.ipynb` compares Logistic Regression and Random Forest
for the binary triage task. It reports confusion matrices, ROC/PR curves, threshold
selection and cross-validation metrics. The reusable pipeline lives in
`src/autoimmune_disorder_prediction/evaluation.py`.

## Unsupervised exploration

`notebooks/05_unsupervised_learning.ipynb` explores the dataset without labels: PCA for
dimensionality reduction, K-Means for patient segmentation and DBSCAN for anomaly
detection. The case-study write-up is in
[`docs/02_unsupervised_analysis.md`](docs/02_unsupervised_analysis.md) and the reusable
helpers live in `src/autoimmune_disorder_prediction/clustering.py`.

## Development

Add new dependencies with uv:

```bash
uv add <package>
```

Run linting or tests as the project grows by extending the scripts section in
`pyproject.toml`.

## Data

The raw dataset is excluded from version control. Only `.gitkeep` files and the
cleaned pipeline code are tracked. Large artifacts such as `.csv`, `.parquet`,
`.pkl` and `.joblib` files are ignored.
