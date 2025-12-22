FROM python:3.10-slim

WORKDIR /app

RUN pip install poetry==1.8.2

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY app/ ./app/
COPY dashboard/ ./dashboard/
# COPY dvc.yaml .dvc/ ./

RUN mkdir -p /app/data/datasets /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]