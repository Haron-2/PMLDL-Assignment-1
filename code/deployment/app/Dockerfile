FROM python:3.13-slim

WORKDIR /app

COPY code/deployment/app/requirements.txt /app/code/deployment/app/requirements.txt
RUN pip install --no-cache-dir -r /app/code/deployment/app/requirements.txt

COPY code/deployment/app /app/code/deployment/app

EXPOSE 8501

CMD ["streamlit", "run", "/app/code/deployment/app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
