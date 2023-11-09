FROM python:3.11.6-slim-bookworm

RUN apt-get -y update && apt-get -y install git

COPY pyproject.toml poetry.lock ./

RUN pip install poetry

CMD poetry shell