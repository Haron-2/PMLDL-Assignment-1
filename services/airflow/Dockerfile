FROM apache/airflow:2.10.5-python3.12

# The deploy task must manage the host Docker daemon through the mounted
# socket: docker CLI + compose plugin only (no daemon inside the container).
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Tasks must run with the EXACT library versions the model was trained with
# (sklearn 1.9.1 etc.), independent of the Airflow runtime. A dedicated venv
# keeps the two stacks fully isolated (no constraint conflicts).
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        numpy==2.5.3 \
        pandas==3.0.5 \
        scikit-learn==1.9.1 \
        joblib==1.6.0 \
        mlflow==3.16.0 \
    && chmod -R a+rX /opt/venv

# Container stays root: /var/run/docker.sock (Docker Desktop) is only
# accessible to root inside the airflow container. Local single-user setup.
USER root

# Defense-in-depth: forbid the airflow user-site (/home/airflow/.local) from
# being picked up as a user site dir. The image entrypoint still injects
# PYTHONPATH=/home/airflow/.local/... at runtime (a Dockerfile ENV cannot
# prevent that), so task commands additionally strip PYTHONPATH via
# `env -u PYTHONPATH` (see dags/wine_quality_pipeline.py ENV_GUARD).
ENV PYTHONNOUSERSITE=1
