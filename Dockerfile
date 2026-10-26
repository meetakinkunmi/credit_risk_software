FROM python:3.12-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements-docker.txt
COPY src ./src
COPY app ./app
# artifacts/ and reports/ are produced by `python run_pipeline.py` and mounted at run time
EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
