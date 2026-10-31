# Diabetes Prediction System - Streamlit app image
#
# Build:  docker build -t diabetes-prediction:latest .
# Run:    docker run -d --name diabetes-app -p 8501:8501 diabetes-prediction:latest
# Open:   http://localhost:8501

FROM python:3.11-slim

# No .pyc files, unbuffered logs (so `docker logs` / `kubectl logs` show output immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first: this layer is cached and only rebuilt when requirements change
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Copy only what the app needs at runtime
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/figures/ ./reports/figures/
COPY .streamlit/ ./.streamlit/

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Docker marks the container unhealthy if Streamlit stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
