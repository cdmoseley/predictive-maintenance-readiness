FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Ensure demo artifacts exist inside the image
RUN chmod +x scripts/bootstrap.sh \
    && python -m predictive_maintenance.data.generate --n-vehicles 120 \
    && python -m predictive_maintenance.ml.train \
    && python -m predictive_maintenance.rag.index

EXPOSE 8507 8700

# Default: Streamlit dashboard on uncommon port 8507
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8507", "--server.address=0.0.0.0"]
