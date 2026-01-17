# Dockerfile
FROM sentiment-base

WORKDIR /app
COPY src/ src/


CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
