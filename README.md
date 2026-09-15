# Wine Quality Prediction MLOps Pipeline

> **Status:** 🚧 The project is at an early stage of development — only the project skeleton is prepared. No datasets are downloaded, no model is trained, and no metrics (accuracy, F1, etc.) are reported yet.

An automated MLOps pipeline for predicting wine quality based on the two UCI
Wine Quality datasets (`winequality-red.csv` and `winequality-white.csv`).
The datasets will be merged into a single dataset with a categorical
`wine_type` feature (`red` / `white`), cleaned, and used to train a model that
predicts a binary target `good_quality`.

## Planned Technology Stack

| Area | Technology |
|------|------------|
| Language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Orchestration | Apache Airflow |
| Experiment tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Version control | Git / GitHub |

## Pipeline Stages (planned)

1. **Data Engineering** — download both UCI Wine Quality datasets, add the
   categorical feature `wine_type` (`red` / `white`), merge them, clean the
   data (missing values, duplicates, outliers), create the binary target
   `good_quality`, and perform the train/test split.
2. **Model Engineering** — train and evaluate models on the prepared data,
   track experiments, parameters and artifacts with MLflow.
3. **Deployment** — serve the trained model behind a FastAPI endpoint with a
   Streamlit frontend, packaged and orchestrated with Docker / Docker Compose.

## Project Structure

```text
PMLDL-Assignment-1/
│
├── code/                  # Source code of the pipeline
│   ├── datasets/          # Data loading / preparation code
│   ├── models/            # Model training / evaluation code
│   └── deployment/        # Deployment code
│       ├── api/           # FastAPI service
│       └── app/           # Streamlit application
│
├── data/
│   ├── raw/               # Raw datasets (not tracked by Git)
│   └── processed/         # Cleaned / transformed datasets (not tracked by Git)
│
├── notebooks/             # Exploratory Jupyter notebooks
│
├── models/                # Trained model artifacts (not tracked by Git)
│
├── services/
│   └── airflow/
│       ├── dags/          # Airflow DAGs
│       └── logs/          # Airflow logs (not tracked by Git)
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Create and activate the virtual environment

From the project root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

Verify that `python` points to the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
# Expected output: ...\PMLDL-Assignment-1\.venv\Scripts\python.exe
```

### 2. Install dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

At this stage `requirements.txt` contains only the core data-preparation
dependencies: `pandas`, `numpy`, `scikit-learn`. Airflow, MLflow, FastAPI,
Streamlit and Docker-related packages will be added later, when the
corresponding pipeline stages are implemented.

## Roadmap

- [ ] Download and merge the UCI Wine Quality datasets
- [ ] Data cleaning: missing values, duplicates, outliers
- [ ] Binary target `good_quality` and train/test split
- [ ] Model training and evaluation
- [ ] Airflow DAG for the automated pipeline
- [ ] MLflow experiment tracking
- [ ] FastAPI model-serving endpoint
- [ ] Streamlit UI
- [ ] Docker / Docker Compose deployment
