FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY ui/frontend/requirements.txt ./ui/frontend/requirements.txt

RUN pip install --no-cache-dir -r ui/frontend/requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "ui/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
