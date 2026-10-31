# 🩺 Diabetes Prediction System — MLOps & DevOps

[![CI/CD](https://github.com/ShubhParmar112/devops-N-mlops_Proj/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ShubhParmar112/devops-N-mlops_Proj/actions/workflows/ci-cd.yml)

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
10. [Docker](#docker)
11. [CI/CD with GitHub Actions](#cicd-with-github-actions)
12. [Roadmap / project phases](#roadmap--project-phases)
13. [Troubleshooting](#troubleshooting)
14. [Dataset & licence](#dataset--licence)

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
devops-N-mlops_Proj/
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
├── .github/workflows/ci-cd.yml # CI/CD: train → test → build → smoke-test → push
├── Dockerfile                  # Streamlit app image (python:3.11-slim, non-root, healthcheck)
├── .dockerignore               # Keeps .venv, .git, MLflow state and secrets out of the image
├── requirements.txt            # Full dev/training dependencies
├── requirements-app.txt        # Runtime-only dependencies used by the Docker image
├── pytest.ini
├── .gitignore / .gitattributes
└── README.md
```

## Quick start (Windows PowerShell)

Prerequisites: **Python 3.11**, **Git**, VS Code. (Docker Desktop and Minikube are needed from Phase 4.)

```powershell
# 1. Clone
git clone https://github.com/ShubhParmar112/devops-N-mlops_Proj.git
cd devops-N-mlops_Proj

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

## Docker

The image contains only what the app needs at runtime: `src/`, `app/`, `models/`, `data/`,
`reports/figures/` and the dependencies in `requirements-app.txt` (no MLflow, pytest or
Matplotlib). It runs as a non-root user and has a `HEALTHCHECK` on Streamlit's
`/_stcore/health` endpoint.

> `requirements-app.txt` must pin the **same** `scikit-learn` and `numpy` versions as
> `requirements.txt` — a pickled model only loads reliably with the versions it was trained with.

**Build and run locally** (Docker Desktop must be running):

```powershell
docker build -t diabetes-prediction:latest .
docker run -d --name diabetes-app -p 8501:8501 diabetes-prediction:latest
# open http://localhost:8501

docker ps                                   # STATUS should become "(healthy)" after ~30 s
docker logs -f diabetes-app                 # follow logs (Ctrl+C to stop following)
docker exec diabetes-app python -m src.predict   # run a prediction inside the container
docker stop diabetes-app; docker rm diabetes-app # stop and remove
```

**Tag and push to Docker Hub** (replace `<dockerhub-username>` with your Docker Hub username):

```powershell
docker login                                # sign in (use an access token as the password)
docker tag diabetes-prediction:latest <dockerhub-username>/diabetes-prediction:latest
docker tag diabetes-prediction:latest <dockerhub-username>/diabetes-prediction:v1.0.0
docker push <dockerhub-username>/diabetes-prediction:latest
docker push <dockerhub-username>/diabetes-prediction:v1.0.0
```

Anyone can then run it with
`docker run -p 8501:8501 <dockerhub-username>/diabetes-prediction:latest`.
GitHub Actions also builds and pushes this image automatically on every push to `main` (see below).

## CI/CD with GitHub Actions

Workflow: [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml). It runs on every push
and pull request to `main` or `develop`, and can be started manually from the **Actions** tab
(*Run workflow*).

```
 push / PR ──► job 1: Train & test ───────────────► job 2: Build, smoke-test & push
               • install requirements.txt            • download the tested model artifact
               • verify dataset                      • docker build (layer cache)
               • python -m src.train (MLflow)        • run container, wait for /_stcore/health
               • pytest (incl. quality gate)         • predict inside the container
               • upload model + reports artifacts    • push to Docker Hub (main only)
```

- **Continuous integration** — every change is retrained and tested on a clean Ubuntu runner.
  If the model drops below the quality gate (ROC-AUC ≥ 0.75, recall ≥ 0.60) or any test fails,
  job 2 never runs.
- **Continuous delivery** — the image is built with *the exact model that passed the tests*
  (passed between jobs as an artifact), smoke-tested, and pushed to Docker Hub as
  `<user>/diabetes-prediction:latest` and `:sha-<commit>`. The SHA tag makes every image
  traceable back to its Git commit.
- **Artifacts** — each run keeps the trained model, test report (`test-results.xml`), figures
  and MLflow runs for 14 days (run page → *Artifacts*).
- **Pull requests and `develop`** build and test the image but never push it.

### Setting up the Docker Hub secrets

Credentials are **never** written in the code — they are stored as encrypted GitHub secrets.

1. Docker Hub → *Account settings* → *Personal access tokens* → **Generate new token**
   (access: *Read & Write*). Copy the token.
2. GitHub repo → *Settings* → *Secrets and variables* → *Actions* → **New repository secret**:
   - `DOCKERHUB_USERNAME` = your Docker Hub username
   - `DOCKERHUB_TOKEN` = the token from step 1
3. Re-run the workflow (*Actions* → latest run → *Re-run all jobs*) or push to `main`.

Without the secrets the pipeline still passes — it simply skips the push step and says so
in the run summary.

## Roadmap / project phases

- [x] **Phase 1** — Dataset, EDA, preprocessing, model training, MLflow tracking & registry
- [x] **Phase 2** — Streamlit application and pytest suite
- [x] **Phase 3** — Git repository, branching strategy, documentation
- [x] **Phase 4** — Docker image and Docker Hub
- [x] **Phase 5** — GitHub Actions CI/CD
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
| `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` | Docker Desktop is not running — start it and wait until it says *Engine running* |
| `Bind for 0.0.0.0:8501 failed: port is already allocated` | Something else uses 8501 (a local `streamlit run` or an old container): `docker rm -f diabetes-app`, or map another port: `-p 8502:8501` |
| `Conflict. The container name "/diabetes-app" is already in use` | `docker rm -f diabetes-app` |
| `InconsistentVersionWarning` / model fails to load in the container | Versions in `requirements-app.txt` differ from `requirements.txt` — align them and rebuild |
| `denied: requested access to the resource is denied` on `docker push` | Run `docker login`, and make sure the tag starts with *your* Docker Hub username |
| GitHub Actions: *Push image to Docker Hub* skipped | Expected on `develop`/PRs; on `main` add the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets |
| GitHub Actions: `unauthorized: incorrect username or password` | The token is wrong/expired or lacks *Write* scope — regenerate it and update `DOCKERHUB_TOKEN` |
| GitHub Actions: a test fails only in CI | Reproduce in a fresh clone: `python -m src.train` then `pytest`; check that `requirements.txt` pins every package you use |
| `git push` asks for a password | GitHub needs a **Personal Access Token** (Settings → Developer settings → Tokens), or sign in through the Git Credential Manager browser window |

## Dataset & licence

Dataset: *Pima Indians Diabetes Database*, originally from the National Institute of
Diabetes and Digestive and Kidney Diseases (via the UCI Machine Learning Repository / Kaggle).
Downloaded from the `plotly/datasets` GitHub mirror by `src/download_data.py`.

Code is provided for educational purposes as part of a college practical.
