# PMLDL Assignment 1 — Wine Quality MLOps Pipeline

Automated MLOps pipeline for wine-quality prediction using the UCI Wine Quality datasets.

The project implements all **3 required assignment stages**:

1. **Data Engineering** — data loading, cleaning, outlier removal, and train/test split.
2. **Model Engineering** — feature engineering, model training, evaluation, and MLflow tracking.
3. **Deployment** — FastAPI model API and Streamlit web application running in separate Docker containers.

The complete pipeline is orchestrated by **Apache Airflow** and runs automatically every **5 minutes**.

---

## Dataset

Two UCI Wine Quality datasets are used:

* `winequality-red.csv`
* `winequality-white.csv`

They are merged into one dataset with a categorical `wine_type` feature (`red` / `white`).

The target is binary:

```text
good_quality = 1 if quality > 5
good_quality = 0 otherwise
```

The datasets are not stored in Git. They are downloaded automatically with:

```powershell
.venv\Scripts\python.exe code\datasets\download.py
```

---

## 1. Data Engineering

Implemented in:

```text
code/datasets/
├── download.py
├── explore.py
└── preprocess.py
```

The preprocessing stage:

* loads and merges the raw datasets;
* removes exact duplicates;
* removes IQR-based outliers;
* creates the binary target;
* performs an 80/20 stratified train/test split;
* prepares model features without data leakage.

Generated artifacts:

```text
data/processed/train.csv
data/processed/test.csv
models/preprocessor.joblib
```

---

## 2. Model Engineering

Implemented in:

```text
code/models/
├── train.py
└── evaluate.py
```

Model:

```text
RandomForestClassifier
n_estimators=300
random_state=42
```

The model is stored as a single sklearn pipeline containing preprocessing and the classifier.

Output artifacts:

```text
models/model.joblib
models/metrics.json
mlflow.db
```

MLflow tracks the training parameters, metrics, and model artifact.

Example test-set metrics:

```text
Accuracy : 0.7815
Precision: 0.8025
Recall   : 0.8643
F1       : 0.8322
ROC-AUC  : 0.8433
```

---

## 3. Deployment

### FastAPI

The model API is implemented in:

```text
code/deployment/api/
```

Endpoints:

```text
GET  /health
POST /predict
GET  /docs
```

### Streamlit

The web application is implemented in:

```text
code/deployment/app/
```

The application provides input fields, a prediction button, and displays the model prediction and probability.

### Docker

The API and application run in **separate Docker containers**:

```text
wine-api
wine-app
```

Start the deployment manually with:

```powershell
docker compose up -d --build
```

Access:

```text
API:       http://localhost:8000
Swagger:   http://localhost:8000/docs
Streamlit: http://localhost:8501
```

---

## Automated Pipeline

Apache Airflow runs the complete pipeline:

```text
verify_raw_data
       ↓
preprocess
       ↓
train_model
       ↓
evaluate_model
       ↓
deploy
```

The DAG is:

```text
wine_quality_pipeline
```

Schedule:

```text
*/5 * * * *
```

The final `deploy` task automatically:

1. rebuilds the API image using the newly trained model;
2. starts/restarts the API and Streamlit containers;
3. checks API health;
4. sends a prediction request;
5. verifies that the deployed model matches the freshly trained artifact;
6. checks Streamlit health.

DAG configuration:

```text
catchup = false
retries = 2
max_active_runs = 1
```

Start Airflow with:

```powershell
docker compose -f services\airflow\docker-compose.airflow.yml up -d --build
```

Airflow UI:

```text
http://localhost:8080
```

---

## Repository Structure

```text
PMLDL-Assignment-1/
├── code/
│   ├── datasets/
│   ├── models/
│   └── deployment/
│       ├── api/
│       └── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
└── services/
    └── airflow/
        ├── dags/
        └── scripts/
```

Generated data, models, MLflow database, and Airflow logs are not tracked by Git.

---

## Quick Start

### 1. Create environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download and process data

```powershell
.venv\Scripts\python.exe code\datasets\download.py
.venv\Scripts\python.exe code\datasets\preprocess.py
```

### 4. Train and evaluate

```powershell
.venv\Scripts\python.exe code\models\train.py
.venv\Scripts\python.exe code\models\evaluate.py
```

### 5. Start deployment

```powershell
docker compose up -d --build
```

### 6. Start the automated Airflow pipeline

```powershell
docker compose -f services\airflow\docker-compose.airflow.yml up -d --build
```

After Airflow starts, the DAG runs automatically every 5 minutes.

---

## Assignment Requirements

| Assignment requirement | Implementation                    |
| ---------------------- | --------------------------------- |
| Data Engineering       | `download.py`, `preprocess.py`    |
| Model Engineering      | `train.py`, `evaluate.py`, MLflow |
| Model API              | FastAPI                           |
| Web application        | Streamlit                         |
| Separate containers    | Docker Compose                    |
| Automated pipeline     | Apache Airflow                    |
| Automatic deployment   | Airflow `deploy` task             |
| Schedule               | Every 5 minutes                   |

The repository is ready for demonstration of the complete automated pipeline and prediction through the web application.

---

## Documentation

Detailed stage reports and the final project report are available in:

```text
notebooks/
```
