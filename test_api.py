"""
Semana 3 – Paso 4: Test de endpoints de la API
================================================
Ejecutar después de levantar el servidor:
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Uso:
  python test_api.py                     # localhost:8000
  python test_api.py https://mi-space.hf.space  # Hugging Face
"""

import sys
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def test_health():
    print("─" * 50)
    print("1. GET /health")
    r = requests.get(f"{BASE}/health")
    print(f"   Status: {r.status_code}")
    print(f"   Body:   {r.json()}")
    assert r.status_code == 200


def test_predict_normal():
    print("─" * 50)
    print("2. POST /predict  (caso normal: temp=70, vib=0.5)")
    r = requests.post(f"{BASE}/predict", params={"temperatura": 70, "vibracion": 0.5})
    print(f"   Status: {r.status_code}")
    print(f"   Body:   {r.json()}")
    assert r.status_code == 200
    assert r.json()["prediccion"] == 0  # no debería haber falla


def test_predict_falla():
    print("─" * 50)
    print("3. POST /predict  (caso falla: temp=90, vib=1.0)")
    r = requests.post(f"{BASE}/predict", params={"temperatura": 90, "vibracion": 1.0})
    print(f"   Status: {r.status_code}")
    print(f"   Body:   {r.json()}")
    assert r.status_code == 200
    assert r.json()["prediccion"] == 1  # se espera falla


def test_monitor():
    print("─" * 50)
    print("4. POST /monitor  (batch con drift)")
    params = {
        "size": 200,
        "t_mean": 90,
        "t_std": 5,
        "v_mean": 1.0,
        "v_std": 0.1,
    }
    r = requests.post(f"{BASE}/monitor", params=params)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Límites:  {data['limits']}")
    print(f"   Actuales: {data['current_means']}")
    print(f"   Alertas:  {data['alerts']}")
    assert r.status_code == 200


if __name__ == "__main__":
    print(f"\n🔗 Base URL: {BASE}\n")
    test_health()
    test_predict_normal()
    test_predict_falla()
    test_monitor()
    print("─" * 50)
    print("\n✅ Todos los tests pasaron correctamente.\n")
