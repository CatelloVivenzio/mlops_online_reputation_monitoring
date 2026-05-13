FROM python:3.10-slim

WORKDIR /app

# Installazione dei pacchetti di sistema necessari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia dei requisiti e installazione delle librerie Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia delle cartelle del progetto
COPY ./src ./src
COPY ./static ./static

# Esposizione della porta definita nello sviluppo
EXPOSE 8000

# Avvio dell'applicazione FastAPI
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
