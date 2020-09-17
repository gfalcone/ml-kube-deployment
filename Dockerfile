FROM python:3.7-buster

COPY requirements.txt /app/
COPY api.py /app/
COPY lr.joblib /app/

WORKDIR /app/

RUN pip install -r requirements.txt

CMD uvicorn api:app --host 0.0.0.0
