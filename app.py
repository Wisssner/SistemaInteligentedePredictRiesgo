# -*- coding: utf-8 -*-
"""
================================================================================
app.py  --  API REST pura  --  Sistema Inteligente de Riego (Edge AI)
Parques Urbanos  |  Backend de telemetria y control override
================================================================================

El modelo de decision (TinyML) vive dentro del ESP32 como codigo C puro.
Este servidor SOLO recibe telemetria y ejecuta overrides manuales via Firebase.

ENDPOINTS
─────────────────────────────────────────────────────────────────────────────
  POST /api/iot/telemetry   Recibe lectura + decision del ESP32 -> Firebase
  POST /api/iot/override    Envia orden de riego manual forzado -> Firebase
  GET  /api/iot/status      Ultima lectura almacenada en Firebase
  GET  /health              Verificacion de vida del servidor
================================================================================
"""

import os
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Configuracion Firebase ────────────────────────────────────────────────────
FIREBASE_URL = os.getenv("FIREBASE_URL", "").rstrip("/")
# Ejemplo: FIREBASE_URL=https://mi-proyecto-default-rtdb.firebaseio.com

if not FIREBASE_URL:
    import warnings
    warnings.warn(
        "FIREBASE_URL no esta configurada. Define la variable de entorno FIREBASE_URL.",
        stacklevel=1,
    )


def _firebase_write(path: str, data: dict) -> dict:
    """Hace PUT a Firebase Realtime Database y retorna la respuesta."""
    url = f"{FIREBASE_URL}/{path}.json"
    resp = requests.put(url, json=data, timeout=8)
    resp.raise_for_status()
    return resp.json()


def _firebase_push(path: str, data: dict) -> dict:
    """Hace POST (push) a Firebase Realtime Database."""
    url = f"{FIREBASE_URL}/{path}.json"
    resp = requests.post(url, json=data, timeout=8)
    resp.raise_for_status()
    return resp.json()


def _firebase_read(path: str) -> dict | None:
    """Lee un nodo de Firebase; retorna None si esta vacio o falla."""
    url = f"{FIREBASE_URL}/{path}.json"
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    return data if data else None


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Verificacion rapida de disponibilidad del servidor."""
    return jsonify({"status": "ok", "timestamp": _now_iso()})


# ── POST /api/iot/telemetry ───────────────────────────────────────────────────
@app.post("/api/iot/telemetry")
def iot_telemetry():
    """
    El ESP32 envia aqui cada lectura junto con la decision que ya tomo localmente.

    Body JSON esperado:
    {
        "soil_moisture":    float,   // Humedad del suelo (0-100 %)
        "air_temperature":  float,   // Temperatura del aire (grados C)
        "air_humidity":     float,   // Humedad relativa del aire (0-100 %)
        "valve_decision":   int,     // Decision local del ESP32: 0=OFF  1=ON
        "node_id":          string   // Identificador del nodo (opcional)
    }

    Respuesta:
    { "ok": true, "firebase_key": "<key>" }
    """
    body = request.get_json(silent=True) or {}

    required = ["soil_moisture", "air_temperature", "air_humidity", "valve_decision"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"ok": False, "error": f"Faltan campos: {missing}"}), 400

    valve = body["valve_decision"]
    if valve not in (0, 1):
        return jsonify({"ok": False, "error": "valve_decision debe ser 0 o 1"}), 400

    record = {
        "soil_moisture":   float(body["soil_moisture"]),
        "air_temperature": float(body["air_temperature"]),
        "air_humidity":    float(body["air_humidity"]),
        "valve_decision":  int(valve),
        "valve_state":     "ON" if valve == 1 else "OFF",
        "node_id":         str(body.get("node_id", "esp32_nodo_1")),
        "source":          "edge_ai",
        "timestamp":       _now_iso(),
    }

    try:
        # Historial (push agrega registro nuevo)
        result = _firebase_push("telemetry/history", record)
        # Ultimo estado (put sobreescribe)
        _firebase_write("telemetry/latest", record)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    return jsonify({"ok": True, "firebase_key": result.get("name")}), 201


# ── POST /api/iot/override ────────────────────────────────────────────────────
@app.post("/api/iot/override")
def iot_override():
    """
    Envia una orden de control manual forzado a Firebase.
    El ESP32 hace polling de este nodo y obedece la orden.

    Body JSON esperado:
    {
        "command":  int,     // 0 = cerrar valvula  |  1 = abrir valvula
        "duration": int      // Segundos de riego (opcional, default 60)
    }

    Respuesta:
    { "ok": true, "command": 1, "duration": 60 }
    """
    body = request.get_json(silent=True) or {}

    if "command" not in body:
        return jsonify({"ok": False, "error": "Falta el campo 'command'"}), 400

    command = body["command"]
    if command not in (0, 1):
        return jsonify({"ok": False, "error": "command debe ser 0 o 1"}), 400

    duration = int(body.get("duration", 60))
    if not (1 <= duration <= 3600):
        return jsonify({"ok": False, "error": "duration debe estar entre 1 y 3600 segundos"}), 400

    override_record = {
        "command":     int(command),
        "valve_state": "ON" if command == 1 else "OFF",
        "duration":    duration,
        "source":      "dashboard_override",
        "issued_at":   _now_iso(),
        "active":      True,
    }

    try:
        _firebase_write("control/override", override_record)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    return jsonify({
        "ok":       True,
        "command":  command,
        "duration": duration,
        "state":    "ON" if command == 1 else "OFF",
    })


# ── GET /api/iot/status ───────────────────────────────────────────────────────
@app.get("/api/iot/status")
def iot_status():
    """
    Retorna el ultimo estado del nodo IoT y la orden de override activa (si existe).
    Usado por el dashboard Streamlit para refrescar datos.
    """
    try:
        latest   = _firebase_read("telemetry/latest")
        override = _firebase_read("control/override")
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    return jsonify({
        "ok":              True,
        "latest_telemetry": latest,
        "active_override":  override,
    })


# ── Utilidades ────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
