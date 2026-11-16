# API Sensores Industriales — Semana 3

**Curso:** Despliegue de Modelos de IA & IA en la Nube — EIG Business School

API REST + Dashboard para prediccion de fallas en sensores industriales, usando FastAPI, scikit-learn y Gradio. Preparado para despliegue con Docker en [Render](https://render.com) (free tier).

## Estructura del proyecto

```
semana3_proyecto/
├── Dockerfile          # Imagen Docker lista para Render
├── main.py             # API FastAPI + Dashboard Gradio
├── train_model.py      # Genera datos y entrena el modelo
├── requirements.txt    # Dependencias Python
├── test_api.py         # Tests de los endpoints
├── .dockerignore       # Excluye archivos innecesarios del build
├── .gitignore          # Excluye archivos generados del repo
└── README.md           # Este archivo
```

## Endpoints

| Metodo | Ruta       | Descripcion                          |
|--------|------------|--------------------------------------|
| GET    | `/`        | Status general + enlaces utiles      |
| GET    | `/health`  | Estado del servicio                  |
| POST   | `/predict` | Prediccion individual (temp, vib)    |
| POST   | `/monitor` | Deteccion de drift (regla +/-2sigma) |
| GET    | `/docs`    | Swagger UI (automatico de FastAPI)   |
| GET    | `/ui`      | Dashboard interactivo (Gradio)       |

## Uso local con Docker

```bash
# Construir la imagen
docker build -t sensores-api .

# Ejecutar (mapea el puerto 10000 del contenedor al 8000 local)
docker run -p 8000:10000 sensores-api
```

Luego abre:
- http://localhost:8000/docs — Swagger UI
- http://localhost:8000/ui — Dashboard Gradio

## Uso local sin Docker

```bash
pip install -r requirements.txt
python train_model.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Tests

```bash
# Contra servidor local
python test_api.py

# Contra Render
python test_api.py https://TU-SERVICIO.onrender.com
```

## Despliegue en Render

1. Sube este proyecto a un repositorio publico en GitHub
2. Crea una cuenta en [render.com](https://render.com) (gratis, sin tarjeta)
3. New > Web Service > conecta tu repo > elige **Docker** + **Free**
4. Render detecta el `Dockerfile` y despliega automaticamente

Cada `git push` a `main` redespliega automaticamente (CI/CD).

## Notas

- El modelo se entrena **durante el build** de Docker (`RUN python train_model.py`)
- No necesitas subir archivos `.pkl` ni `.csv` al repositorio
- El free tier de Render tiene cold start (~30-60s tras inactividad)
