"""
Semana 3 – Paso 1: Entrenamiento y serialización del modelo
===========================================================
Genera datos sintéticos propios de sensores industriales (temperatura,
vibración) y entrena un RandomForestClassifier para predecir fallas.

Semilla y parámetros propios del Grupo 6 (distintos al proyecto base),
para que los datos no coincidan con los de la plantilla del curso.
"""

import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# 1) Generar datos simulados
# ──────────────────────────────────────────────
SEMILLA = 123
rng = np.random.default_rng(SEMILLA)

normal = pd.DataFrame({
    "temperatura": rng.normal(loc=68, scale=5, size=900),
    "vibracion":   rng.normal(loc=0.55, scale=0.12, size=900),
    "falla": 0,
})

falla = pd.DataFrame({
    "temperatura": rng.normal(loc=80, scale=6, size=150),
    "vibracion":   rng.normal(loc=0.85, scale=0.15, size=150),
    "falla": 1,
})

df = pd.concat([normal, falla], ignore_index=True).sample(frac=1, random_state=SEMILLA)

# Guardar CSV
data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)
csv_path = data_path / "datos_sensor.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado -> {csv_path}  ({len(df)} filas)")

# ──────────────────────────────────────────────
# 2) Entrenar modelo
# ──────────────────────────────────────────────
X = df[["temperatura", "vibracion"]]
y = df["falla"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEMILLA, stratify=y
)

modelo = RandomForestClassifier(random_state=SEMILLA)
modelo.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 3) Evaluación
# ──────────────────────────────────────────────
# La exactitud sola no alcanza: con ~86% de lecturas normales, un modelo que
# siempre prediga "sin falla" ya acierta ese porcentaje. Por eso se reportan
# también precision, recall, F1 y la matriz de confusión.
y_pred = modelo.predict(X_test)

metricas = {
    "exactitud": round(accuracy_score(y_test, y_pred), 3),
    "exactitud_linea_base": round(float((y_test == 0).mean()), 3),
    "precision": round(precision_score(y_test, y_pred), 3),
    "recall": round(recall_score(y_test, y_pred), 3),
    "f1": round(f1_score(y_test, y_pred), 3),
    "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
    "n_prueba": int(len(y_test)),
}

print("\nMétricas en holdout:")
for nombre, valor in metricas.items():
    print(f"  {nombre:>22}: {valor}")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Sin falla", "Falla"]))

# ──────────────────────────────────────────────
# 4) Guardar modelo serializado + métricas
# ──────────────────────────────────────────────
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)
model_file = models_path / "modelo_sensores.pkl"
joblib.dump(modelo, model_file)
(models_path / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
print(f"\nModelo guardado -> {model_file}")
print(f"Métricas guardadas -> {models_path / 'metricas.json'}")
