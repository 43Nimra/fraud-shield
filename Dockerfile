FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/
COPY mlruns/ ./mlruns/
COPY .env .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
