FROM python:3.11.12-slim

WORKDIR /EvalAutomatique

copy . /EvalAutomatique
copy requirementsForEvaluation.txt ./EvalAutomatique

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirementsForEvaluation.txt

ENTRYPOINT ["python3", "evaluateFunction.py"]