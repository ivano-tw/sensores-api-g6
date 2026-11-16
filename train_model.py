"""
Semana 3 – Paso 1: Entrenamiento y serialización del modelo
===========================================================
Genera datos sintéticos de sensores industriales (temperatura, vibración)
y entrena un RandomForestClassifier para predecir retrasos (fallas).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ──────────────────────────────────────────────
# 1) Generar datos simulados
# ──────────────────────────────────────────────
np.random.seed(42)

normal = pd.DataFrame({
    "temperatura": np.random.normal(loc=70, scale=5, size=1000),
    "vibracion":   np.random.normal(loc=0.5, scale=0.1, size=1000),
    "retraso": 0,
})

falla = pd.DataFrame({
    "temperatura": np.random.normal(loc=85, scale=5, size=200),
    "vibracion":   np.random.normal(loc=0.9, scale=0.1, size=200),
    "retraso": 1,
})

df = pd.concat([normal, falla], ignore_index=True).sample(frac=1, random_state=42)

# Guardar CSV
data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)
csv_path = data_path / "datos_sensor.csv"
df.to_csv(csv_path, index=False)
print(f"✅ CSV guardado → {csv_path}  ({len(df)} filas)")

# ──────────────────────────────────────────────
# 2) Entrenar modelo
# ──────────────────────────────────────────────
X = df[["temperatura", "vibracion"]]
y = df["retraso"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 3) Evaluación
# ──────────────────────────────────────────────
y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Precisión en holdout: {acc:.3f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Sin retraso", "Retraso"]))

# ──────────────────────────────────────────────
# 4) Guardar modelo serializado
# ──────────────────────────────────────────────
models_path = Path("models")
models_path.mkdir(parents=True, exist_ok=True)
model_file = models_path / "modelo_sensores.pkl"
joblib.dump(modelo, model_file)
print(f"✅ Modelo guardado → {model_file}")
