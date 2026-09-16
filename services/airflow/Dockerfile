FROM apache/airflow:2.10.5-python3.12

# Tasks must run with the EXACT library versions the model was trained with
# (sklearn 1.9.1 etc.), independent of the Airflow runtime. A dedicated venv
# keeps the two stacks fully isolated (no constraint conflicts).
USER root
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        numpy==2.5.3 \
        pandas==3.0.5 \
        scikit-learn==1.9.1 \
        joblib==1.6.0 \
        mlflow==3.16.0 \
    && chmod -R a+rX /opt/venv
USER airflow
