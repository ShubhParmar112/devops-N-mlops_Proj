# 🩺 Diabetes Prediction System — MLOps & DevOps

An end-to-end machine learning project that predicts whether a person is likely to be
**diabetes-positive** or **diabetes-negative** from eight clinical measurements, built to
demonstrate the complete ML lifecycle: experiment tracking, automated testing, CI/CD,
containerisation, orchestration, monitoring and visualisation.

> ⚠️ **Educational use only.** This project was built for an MLOps/DevOps college practical.
> Its predictions come from a statistical model trained on a small public dataset and are
> **not a medical diagnosis**. Always consult a qualified healthcare professional.

---

## Table of contents
1. [Architecture](#architecture)
2. [Tech stack](#tech-stack)
3. [Project structure](#project-structure)
4. [Quick start (Windows PowerShell)](#quick-start-windows-powershell)
5. [Machine learning pipeline](#machine-learning-pipeline)
6. [MLflow experiment tracking](#mlflow-experiment-tracking)
7. [Streamlit application](#streamlit-application)
8. [Testing](#testing)
9. [Git workflow](#git-workflow)
10. [Roadmap / project phases](#roadmap--project-phases)
11. [Troubleshooting](#troubleshooting)
12. [Dataset & licence](#dataset--licence)

---

## Architecture

```
 data/diabetes.csv ─► src/preprocessing.py ─► src/train.py ──► MLflow (runs, metrics, model registry)
                                                   │
                                                   ▼
                                      models/diabetes_model.pkl  (preprocessing + classifier)
                                                   │
                     app/streamlit_app.py  (UI + /metrics endpoint for Prometheus)
                                                   │
 git push ─► GitHub Actions: train ─► pytest ─► docker build ─► Docker Hub
                                                   │
                         Kubernetes (Minikube): Deployment + Service
                                                   │
                         Prometheus scrapes metrics ─► Grafana dashboards
```

A single serialized pipeline (`models/diabetes_model.pkl`) is the contract between every
stage: the tests, the Streamlit app, the Docker image and the Kubernetes pods all load the
same artifact.

## Tech stack

| Area | Tools |
|---|---|
| Machine learning | Python 3.11, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| MLOps | MLflow (tracking + model registry), Joblib serialization |
| Application | Streamlit, Plotly |
| Testing | pytest, Streamlit `AppTest` |
| DevOps | Git, GitHub, GitHub Actions, Docker, Docker Hub, Kubernetes (Minikube) |
| Monitoring | Prometheus, Grafana |

## Project structure

```
Diabetes-Prediction-MLOps/
├── app/
│   └── streamlit_app.py        # Web UI: prediction, model info, dataset statistics
├── data/
│   └── diabetes.csv            # Pima Indians Diabetes Dataset (768 rows)
├── models/
│   ├── diabetes_model.pkl      # Final pipeline (imputer + scaler + classifier)
│   └── model_metadata.json     # Metrics, params, MLflow run ID / registry version
├── reports/figures/            # EDA plots, confusion matrices, ROC curves
├── src/
│   ├── download_data.py        # Downloads + validates the dataset
│   ├── preprocessing.py        # Zero→NaN handling, imputation, scaling, train/test split
│   ├── eda.py                  # Exploratory data analysis
│   ├── train.py                # Grid search, evaluation, MLflow logging, model registry
│   └── predict.py              # Input validation + prediction
├── tests/                      # pytest suite (19 tests)
├── .streamlit/config.toml      # App theme and server settings
├── monitoring/                 # Prometheus & Grafana config        (Phase 7–8)
├── kubernetes/                 # deployment.yaml, service.yaml      (Phase 6)
├── .github/workflows/          # ci-cd.yml                          (Phase 5)
├── Dockerfile                  #                                    (Phase 4)
├── requirements.txt
├── pytest.ini
├── .gitignore / .gitattributes
└── README.md
```

## Quick start (Windows PowerShell)

Prerequisites: **Python 3.11**, **Git**, VS Code. (Docker Desktop and Minikube are needed from Phase 4.)

```powershell
# 1. Clone
git clone https://github.com/ShubhParmar112/Diabetes-Prediction-MLOps.git
cd Diabetes-Prediction-MLOps

# 2. Virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Data, EDA and training
python src\download_data.py      # skips download if data\diabetes.csv already exists
python -m src.eda                # prints summary + saves plots to reports\figures
python -m src.train              # trains, logs to MLflow, saves models\diabetes_model.pkl

# 4. Tests
pytest

# 5. Run the app  ->  http://localhost:8501
streamlit run app/streamlit_app.py

# 6. MLflow UI     ->  http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

> The repository already contains a trained model, so step 3 is optional if you only want
> to run the app. MLflow history (`mlflow.db`, `mlruns/`) is local and not committed —
> run `python -m src.train` once to create it.

## Machine learning pipeline

**Dataset:** Pima Indians Diabetes Dataset — 768 female patients of Pima Indian heritage,
8 features, binary `Outcome` (500 negative / 268 positive, 34.9 % positive).

**Data quality issue:** zeros in `Glucose` (5), `BloodPressure` (35), `SkinThickness` (227),
`Insulin` (374) and `BMI` (11) are physiologically impossible and really mean *not measured*.

**Preprocessing** (`src/preprocessing.py`, all inside one scikit-learn `Pipeline`):
1. `ZeroToNaN` — converts those zeros to missing values
2. `SimpleImputer(strategy="median")` — fills them with the training-set median
3. `StandardScaler` — standardises features

Because the preprocessing lives inside the saved pipeline, it is fitted only on training
data (no leakage) and is applied identically at prediction time.

**Training** (`src/train.py`):
- Stratified 80/20 train/test split (614 / 154 rows), `random_state=42`
- 5-fold stratified cross-validated grid search for each model
- Models: **Logistic Regression** and **Random Forest** (incl. `class_weight="balanced"` option)
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix

**Model selection:** by **cross-validated F1** on the training data. F1 balances precision
and recall; accuracy alone is misleading because a model that always predicts "negative"
already scores ~65 %. Selecting on CV (not the test set) keeps the test set unbiased.

**Results (held-out test set, 154 patients):**

| Model | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.682 | 0.708 | 0.571 | 0.667 | 0.615 | 0.810 |
| **Random Forest** ✅ | **0.702** | 0.740 | 0.609 | 0.722 | 0.661 | 0.823 |

Selected: **Random Forest** (`n_estimators=100, max_depth=5, min_samples_leaf=5, class_weight="balanced"`).
Test confusion matrix: TN 75 · FP 25 · FN 15 · TP 39.

## MLflow experiment tracking

Every training run is recorded in a local SQLite-backed MLflow store (`mlflow.db`):

- **Experiment:** `diabetes-prediction` — one run per model
- **Params:** model type, best hyperparameters, split/CV settings, preprocessing choices
- **Metrics:** `cv_*` (cross-validation) and `test_*` (hold-out), plus confusion-matrix counts
- **Artifacts:** confusion matrix and ROC curve plots, classification report, the model with signature and input example
- **Model registry:** the winner is registered as `diabetes-prediction-model`; each training run creates a new **version** and moves the `champion` alias to it

Compare experiments: open the MLflow UI → experiment **diabetes-prediction** → tick both runs → **Compare**.
Registered versions are under the **Models** tab. Load the current champion in Python with:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/diabetes-prediction-model@champion")
```

## Streamlit application

`app/streamlit_app.py` has three tabs:

- **🔍 Prediction** — eight validated inputs with tooltips, *Predict* button, result card, probability gauge, and a chart comparing the input with dataset averages
- **🤖 Model information** — test metrics, model comparison, hyperparameters, MLflow version/run ID, feature importance, confusion matrix
- **📊 Dataset statistics** — headline numbers, outcome split, missing-value chart, per-feature histograms, summary table

The educational disclaimer is shown at the top and with every result.

## Testing

```powershell
pytest
```

| File | What it checks |
|---|---|
| `tests/test_preprocessing.py` | dataset shape, zero→NaN only on the right columns, no NaN after preprocessing, stratified split, clear error for missing data |
| `tests/test_model.py` | pipeline structure, output format, high/low-risk sanity cases, input validation, **quality gate** (ROC-AUC ≥ 0.75, recall ≥ 0.60), metadata consistency |
| `tests/test_app.py` | the Streamlit app renders headlessly and *Predict* shows a result |

The quality gate means CI fails — and no Docker image is built — if a change makes the model worse.

## Git workflow

| Branch | Purpose |
|---|---|
| `main` | Stable, demo-ready code. CI/CD builds and publishes images from here. |
| `develop` | Integration branch for ongoing work. |
| `feature/<name>` | Short-lived branches for a single change, merged into `develop` via pull request. |

```powershell
git checkout develop
git checkout -b feature/my-change
# ...edit, then:
git add .
git commit -m "feat(app): describe the change"
git push -u origin feature/my-change
# open a Pull Request feature/my-change -> develop on GitHub; later develop -> main
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `build:`.

## Roadmap / project phases

- [x] **Phase 1** — Dataset, EDA, preprocessing, model training, MLflow tracking & registry
- [x] **Phase 2** — Streamlit application and pytest suite
- [x] **Phase 3** — Git repository, branching strategy, documentation
- [ ] **Phase 4** — Docker image and Docker Hub
- [ ] **Phase 5** — GitHub Actions CI/CD
- [ ] **Phase 6** — Kubernetes deployment on Minikube
- [ ] **Phase 7** — Prometheus metrics
- [ ] **Phase 8** — Grafana dashboards

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named src` | Run from the project root and use `python -m src.train` (not `python src\train.py`) |
| `cannot import name 'FallbackAsyncAdaptedQueuePool'` | SQLAlchemy 2.1 is incompatible with MLflow 2.17 — `pip install sqlalchemy==2.0.36` (already pinned) |
| `models\diabetes_model.pkl not found` | `python -m src.train` |
| `streamlit` / `mlflow` not recognised | Activate the venv: `.\.venv\Scripts\Activate.ps1` |
| `Port 8501 is already in use` | Stop the other instance or add `--server.port 8502` |
| `git push` asks for a password | GitHub needs a **Personal Access Token** (Settings → Developer settings → Tokens), or sign in through the Git Credential Manager browser window |

## Dataset & licence

Dataset: *Pima Indians Diabetes Database*, originally from the National Institute of
Diabetes and Digestive and Kidney Diseases (via the UCI Machine Learning Repository / Kaggle).
Downloaded from the `plotly/datasets` GitHub mirror by `src/download_data.py`.

Code is provided for educational purposes as part of a college practical.
