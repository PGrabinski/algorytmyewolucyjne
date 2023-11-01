FROM python:3.11.6-slim-bookworm

COPY pyproject.toml poetry.lock ./

RUN pip install poetry

CMD poetry shell