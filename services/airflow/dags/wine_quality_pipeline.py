"""Stage 7 - Airflow DAG: automated wine-quality MLOps pipeline.

Orchestrates the full chain every 5 minutes with BashOperator steps executed
by the project venv inside the container (/opt/venv/bin/python - exact
training library versions):

    verify_raw_data >> preprocess >> train_model >> evaluate_model

The whole repository is bind-mounted at /opt/airflow/project, so every task
operates on the same code and artifacts as local runs (data/, models/,
mlflow.db persist on the host through the mount).

DAG parameters follow the assignment: schedule `*/5 * * * *`, catchup=False,
2 retries per task, no parallel overlapping runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
PYTHON = "/opt/venv/bin/python"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="wine_quality_pipeline",
    description="Wine quality: verify raw -> preprocess -> train (MLflow) -> evaluate",
    default_args=default_args,
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "wine-quality", "assignment-1"],
) as dag:
    verify_raw = BashOperator(
        task_id="verify_raw_data",
        bash_command=f"{PYTHON} {PROJECT}/services/airflow/scripts/verify_raw.py",
    )
    preprocess = BashOperator(
        task_id="preprocess",
        bash_command=f"{PYTHON} {PROJECT}/code/datasets/preprocess.py",
    )
    train = BashOperator(
        task_id="train_model",
        bash_command=f"{PYTHON} {PROJECT}/code/models/train.py",
    )
    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command=f"{PYTHON} {PROJECT}/code/models/evaluate.py",
    )

    verify_raw >> preprocess >> train >> evaluate
