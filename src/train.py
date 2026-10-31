"""Train, evaluate and compare models, with every run tracked in MLflow.

Usage:  python -m src.train
View experiments afterwards:  mlflow ui --backend-store-uri sqlite:///mlflow.db

What it does:
  1. Loads the data and makes a stratified 80/20 train/test split.
  2. For each model (Logistic Regression, Random Forest) runs a small
     5-fold cross-validated grid search on the TRAINING set only.
  3. Evaluates the best version of each model on the held-out test set
     (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
  4. Logs params, metrics, plots and the model to MLflow (one run per model).
  5. Selects the final model by cross-validated F1 score (balances precision
     and recall -- accuracy alone is misleading with 35% positives), registers
     it in the MLflow Model Registry and saves it to models/diabetes_model.pkl.
"""
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURES, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE, build_preprocessor, load_data, split_data,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODELS_DIR / "diabetes_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "diabetes-prediction"
REGISTERED_MODEL_NAME = "diabetes-prediction-model"
SELECTION_METRIC = "cv_f1"

CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced"],
        },
    ),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def save_plots(name, model, X_test, y_test):
    out = REPORTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cm_path = out / f"confusion_matrix_{name}.png"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Negative", "Positive"], cmap="Blues", ax=ax
    )
    ax.set_title(f"{name} - confusion matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    roc_path = out / f"roc_curve_{name}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{name} - ROC curve")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    return cm_path, roc_path


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, positive rate train={y_train.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, (estimator, grid) in CANDIDATES.items():
        print(f"\n===== {name} =====")
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipeline, grid, cv=cv, n_jobs=-1, refit="f1",
            scoring={"f1": "f1", "roc_auc": "roc_auc", "recall": "recall",
                     "precision": "precision", "accuracy": "accuracy"},
        )

        with mlflow.start_run(run_name=name) as run:
            search.fit(X_train, y_train)
            best = search.best_estimator_
            i = search.best_index_
            cv_metrics = {
                f"cv_{m}": float(search.cv_results_[f"mean_test_{m}"][i])
                for m in ["f1", "roc_auc", "recall", "precision", "accuracy"]
            }
            test_metrics = evaluate(best, X_test, y_test)
            tn, fp, fn, tp = confusion_matrix(y_test, best.predict(X_test)).ravel()

            mlflow.log_params({
                "model_type": name,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_folds": 5,
                "imputation": "median (zeros treated as missing)",
                "scaling": "StandardScaler",
                **{k.replace("model__", ""): v for k, v in search.best_params_.items()},
            })
            mlflow.log_metrics({**cv_metrics, **test_metrics,
                                "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp})
            mlflow.set_tags({"stage": "phase-1", "dataset": "pima-indians-diabetes"})

            for p in save_plots(name, best, X_test, y_test):
                mlflow.log_artifact(str(p), artifact_path="plots")
            report = classification_report(y_test, best.predict(X_test),
                                           target_names=["Negative", "Positive"])
            mlflow.log_text(report, "classification_report.txt")

            signature = infer_signature(X_train, best.predict_proba(X_train))
            mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                     input_example=X_train.head(3))

            print(f"Best params: {search.best_params_}")
            print("CV  :", {k: round(v, 4) for k, v in cv_metrics.items()})
            print("Test:", {k: round(v, 4) for k, v in test_metrics.items()})
            print(f"Confusion matrix [[TN FP] [FN TP]]: [[{tn} {fp}] [{fn} {tp}]]")
            print(report)

            results.append({"name": name, "run_id": run.info.run_id, "model": best,
                            "params": search.best_params_, **cv_metrics, **test_metrics})

    # ---- Model selection: cross-validated F1 (not test-set accuracy) ----
    print("\n===== Comparison =====")
    cols = ["cv_f1", "cv_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    print(f"{'model':<20}" + "".join(f"{c:>15}" for c in cols))
    for r in results:
        print(f"{r['name']:<20}" + "".join(f"{r[c]:>15.4f}" for c in cols))

    winner = max(results, key=lambda r: r[SELECTION_METRIC])
    print(f"\nSelected model: {winner['name']} (highest {SELECTION_METRIC} = {winner[SELECTION_METRIC]:.4f})")

    # ---- Register the winner in the MLflow Model Registry (versioning) ----
    mv = mlflow.register_model(f"runs:/{winner['run_id']}/model", REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
    client.set_tag(winner["run_id"], "selected", "true")
    print(f"Registered '{REGISTERED_MODEL_NAME}' version {mv.version} with alias 'champion'")

    # ---- Save the deployable artifact used by the app / Docker image ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["model"], MODEL_PATH)
    metadata = {
        "model_name": winner["name"],
        "best_params": {k.replace("model__", ""): v for k, v in winner["params"].items()},
        "selection_metric": SELECTION_METRIC,
        "metrics": {k: round(float(winner[k]), 4) for k in cols + ["cv_recall", "cv_precision", "cv_accuracy"]},
        "comparison": {r["name"]: {k: round(float(r[k]), 4) for k in cols} for r in results},
        "features": FEATURES,
        "mlflow_run_id": winner["run_id"],
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": int(mv.version),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    main()
