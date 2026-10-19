"""Binary model evaluation for the autoimmune triage problem.

The positive class is ``Autoinmune`` (any diagnosis other than ``Normal``).
``Sickness_Duration_Months`` is excluded because it separates the two groups by
construction and would leak the label into the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLEANED_DATASET_PATH, RANDOM_STATE, TARGET_COL

NORMAL = "#2A6F97"
AUTO = "#E76F51"
ACCENT = "#2A9D8F"
GRAY = "#5C677D"


@dataclass
class ModelResult:
    """Metrics and predictions produced by one fitted model."""

    name: str
    threshold: float
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    cm: np.ndarray
    model: Pipeline


def load_binary_data(
    path: Path = CLEANED_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and build the ``Normal`` vs. ``Autoinmune`` target."""
    df = pd.read_parquet(path)
    y = (df[TARGET_COL] != "Normal").astype(int)
    drop = [c for c in (TARGET_COL, "Sickness_Duration_Months") if c in df.columns]
    return df.drop(columns=drop), y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric variables and one-hot encode categorical variables."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return a simple baseline and an ensemble model with balanced class weights."""
    return {
        "Regresión Logística": Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Choose the probability threshold that maximizes F1 on out-of-fold scores."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    cv_splits: int = 5,
) -> dict[str, ModelResult]:
    """Train, tune and evaluate the baseline and ensemble models on a hold-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, ModelResult] = {}

    for name, model in build_models(X).items():
        oof_proba = cross_val_predict(
            clone(model), X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold = tune_threshold(y_train.to_numpy(), oof_proba)
        oof_pred = (oof_proba >= threshold).astype(int)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
            "cv_f1": float(f1_score(y_train, oof_pred, zero_division=0)),
            "cv_roc_auc": float(roc_auc_score(y_train, oof_proba)),
        }
        results[name] = ModelResult(
            name=name,
            threshold=threshold,
            y_true=y_test.to_numpy(),
            y_proba=proba,
            y_pred=pred,
            metrics=metrics,
            cm=cm,
            model=model,
        )
    return results


def results_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    """Return a tidy metrics table ordered by descending F1."""
    rows = []
    for result in results.values():
        rows.append(
            {
                "Modelo": result.name,
                "Umbral": result.threshold,
                "Accuracy": result.metrics["accuracy"],
                "Precision": result.metrics["precision"],
                "Recall": result.metrics["recall"],
                "F1": result.metrics["f1"],
                "ROC-AUC": result.metrics["roc_auc"],
                "PR-AUC": result.metrics["pr_auc"],
                "CV F1": result.metrics["cv_f1"],
                "CV ROC-AUC": result.metrics["cv_roc_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def confusion_matrix_figure(
    results: dict[str, ModelResult],
    figsize: tuple[float, float] = (10, 4.2),
) -> plt.Figure:
    """Plot one confusion matrix per model side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, result in zip(axes, results.values()):
        sns.heatmap(
            result.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=1,
            linecolor="white",
            xticklabels=["Normal", "Autoinmune"],
            yticklabels=["Normal", "Autoinmune"],
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            ax=ax,
        )
        ax.set_title(
            f"{result.name}\nUmbral = {result.threshold:.3f}",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    return fig


def roc_curve_figure(
    results: dict[str, ModelResult],
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """Plot both ROC curves on the same axes with their AUC values."""
    fig, ax = plt.subplots(figsize=figsize)
    for color, result in zip((NORMAL, AUTO), results.values()):
        fpr, tpr, _ = roc_curve(result.y_true, result.y_proba)
        ax.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.2,
            label=f"{result.name} (AUC = {result.metrics['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8A96A3", linewidth=1.2,
            label="Azar (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_title("Curvas ROC comparativas", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def precision_recall_curve_figure(
    results: dict[str, ModelResult],
    figsize: tuple[float, float] = (7, 5),
) -> plt.Figure:
    """Plot both precision-recall curves with their average precision."""
    fig, ax = plt.subplots(figsize=figsize)
    for color, result in zip((NORMAL, AUTO), results.values()):
        precision, recall, _ = precision_recall_curve(result.y_true, result.y_proba)
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=2.2,
            label=f"{result.name} (AP = {result.metrics['pr_auc']:.3f})",
        )
    baseline = float(np.mean(next(iter(results.values())).y_true))
    ax.axhline(baseline, linestyle="--", color="#8A96A3", linewidth=1.2,
               label=f"Línea base (positivos = {baseline:.1%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall (cobertura de autoinmunes)")
    ax.set_ylabel("Precision (aciertos entre los positivos predichos)")
    ax.set_title("Curvas Precision-Recall", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def top_features(
    model: Pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return the strongest coefficients or importances from a fitted pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        kind = "coeficiente"
    else:
        values = classifier.feature_importances_
        kind = "importancia"
    order = np.argsort(np.abs(values))[::-1][:top_n]
    return pd.DataFrame(
        {
            "variable": [names[i].split("__", 1)[1] for i in order],
            kind: values[order],
        }
    )
