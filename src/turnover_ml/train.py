import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from turnover_ml.data_prep import build_xy, clean_dataset, load_and_merge_data
from turnover_ml.features import add_engineered_features


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dirs(models_dir: Path, reports_dir: Path):
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    On choisit le seuil qui maximise le F1 
    à partir de la courbe precision-recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))

    if best_idx >= len(thresholds):
        return 0.5
    return float(thresholds[best_idx])


def main():
    print("TRAIN BEST MODEL (LogReg + UnderSampling) ✅")

    root = project_root()
    data_dir = root / "data"
    models_dir = root / "models"
    reports_dir = root / "reports"

    sirh = data_dir / "extrait_sirh.csv"
    evalf = data_dir / "extrait_eval.csv"
    sondage = data_dir / "extrait_sondage.csv"

    ensure_dirs(models_dir, reports_dir)

    # Data prep
    df = load_and_merge_data(sirh, evalf, sondage)
    df = clean_dataset(df)
    X, y = build_xy(df)
    X = add_engineered_features(X)

    print(f"Dataset ready. X shape={X.shape} | y shape={y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline imblearn : preprocess -> undersample -> logreg
    preprocessor = build_preprocessor(X_train)

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("under", RandomUnderSampler(random_state=42)),
        ("logreg", LogisticRegression(max_iter=5000, random_state=42)),
    ])

    # RandomizedSearchCV
    param_grid = {
        "logreg__C": [0.01, 0.1, 1, 5, 10],
        "logreg__penalty": ["l2"],
        "logreg__solver": ["lbfgs", "liblinear"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42, 
        verbose=1,
    )

    print("Running RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_score = float(search.best_score_)

    # Eval test
    y_proba = best_model.predict_proba(X_test)[:, 1]
    ap_test = float(average_precision_score(y_test, y_proba))
    best_threshold = find_best_threshold(y_test.to_numpy(), y_proba)

    # Save pipeline
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(best_model, model_path)

    # Save metrics + metadata
    metrics = {
        "model": "LogisticRegression + RandomUnderSampler",
        "cv_best_average_precision": best_cv_score,
        "test_average_precision": ap_test,
        "best_threshold_max_f1": best_threshold,
        "best_params": best_params,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "n_features_raw": int(X.shape[1]),
    }

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Best CV avg_precision: {best_cv_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Test avg_precision: {ap_test:.4f}")
    print(f"Best threshold (max F1): {best_threshold:.4f}")
    print(f"Saved pipeline to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()