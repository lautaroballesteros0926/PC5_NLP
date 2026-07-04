FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8891

WORKDIR /workspace

# Instalamos únicamente las herramientas del sistema necesarias para compilar y ejecutar Jupyter
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-base.txt requirements-opcional.txt ./

# Instalación limpia de dependencias estrictamente necesarias para el Proyecto 8
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements-base.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements-opcional.txt

EXPOSE 8891

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --allow-root --ServerApp.root_dir=/workspace"]
