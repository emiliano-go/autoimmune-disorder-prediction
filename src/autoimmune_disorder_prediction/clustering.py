"""Unsupervised exploration utilities for the autoimmune dataset.

The module provides the pieces used by ``notebooks/05_unsupervised_learning.ipynb``:
feature preparation, scaling, PCA and K-Means/DBSCAN helpers. Labels are never
used to fit the models; ``Diagnosis`` is only read afterwards for interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLEANED_DATASET_PATH, RANDOM_STATE, TARGET_COL

LEAKY_COLUMNS = ("Sickness_Duration_Months",)


def load_features(
    path: Path = CLEANED_DATASET_PATH,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Load the cleaned dataset without the target (and optionally the leakage column)."""
    df = pd.read_parquet(path)
    drop = [TARGET_COL]
    if drop_leakage:
        drop += [c for c in LEAKY_COLUMNS if c in df.columns]
    return df.drop(columns=drop)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric features and one-hot encode categorical features."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """Fit the preprocessor and return the scaled feature matrix."""
    return build_preprocessor(X).fit_transform(X)


def pca_variance(
    X_scaled: np.ndarray,
    n_components: int | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the explained variance ratio and its cumulative sum."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_scaled)
    ratio = pca.explained_variance_ratio_
    return ratio, np.cumsum(ratio)


def reduce_dimensions(X_scaled: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Project the scaled matrix onto its first ``n_components`` principal components."""
    return PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X_scaled)


def sweep_kmeans(
    X: np.ndarray,
    k_range: range = range(2, 11),
    sample_size: int = 2000,
) -> pd.DataFrame:
    """Fit K-Means for each k and report inertia and silhouette."""
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        silhouette = silhouette_score(
            X,
            model.labels_,
            sample_size=min(sample_size, len(X)),
            random_state=RANDOM_STATE,
        )
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, n_clusters: int = 5) -> KMeans:
    """Fit K-Means with the project random state."""
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE).fit(X)


def fit_dbscan(X: np.ndarray, eps: float = 2.0, min_samples: int = 20) -> DBSCAN:
    """Fit DBSCAN and return the estimator (label ``-1`` means noise)."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)


def cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Average each numeric column by cluster to build an interpretable profile."""
    profile = df[columns].copy()
    profile["cluster"] = labels
    return profile.groupby("cluster")[columns].mean().round(2)
