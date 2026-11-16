# Dockerfile – API ML Sensores (Producción)

FROM python:3.11-slim

# ── Variables de entorno ──────────────────────────────────────
# DEBIAN_FRONTEND  → evita prompts interactivos en apt
# PYTHONUNBUFFERED → los print() aparecen en logs en tiempo real
# PIP_NO_CACHE_DIR → no guarda caché de pip (imagen más liviana)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Dependencias del sistema ──────────────────────────────────
# build-essential y git son necesarios para compilar ciertas librerías de Python (numpy wheels, etc.)
# Al final limpiamos la caché de apt para no dejar basura con rm -rf.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/* 

# ── Usuario no-root (seguridad) ───────────────────────────────
# Nunca ejecutar la app como root en producción.
# Creamos un usuario "appuser" con su propio home.
RUN useradd -m -d /home/appuser -s /bin/bash appuser

# ── Directorio de trabajo ─────────────────────────────────────
WORKDIR /app

# ── Instalar dependencias Python ──────────────────────────────
# Copiamos requirements.txt primero para aprovechar la caché de capas de Docker: si las dependencias no cambian, esta
# capa no se reconstruye aunque cambie el código fuente.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Crear carpetas de cache─────────────────────────────────
RUN mkdir -p /app/data /app/reports /app/models && \
    chown -R appuser:appuser /app

# ── Copiar código fuente ──────────────────────────────────────
# Usamos --chown para que los archivos pertenezcan a appuser
COPY --chown=appuser:appuser . /app

# ── Pre-entrenar modelo durante el build ──────────────────────
# Ejecutamos train_model.py para que el .pkl ya esté dentro de la imagen. Así el contenedor arranca sin esperas ni
#posibles timeouts al descargar/generar artefactos.
USER appuser
RUN python train_model.py

# ── Puerto de Render ─────────────────────────────
ENV PORT=10000
EXPOSE $PORT

# ── Comando de arranque ───────────────────────────────────────
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
