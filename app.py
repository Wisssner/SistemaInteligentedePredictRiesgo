# -*- coding: utf-8 -*-
"""
================================================================================
app.py  --  Orquestador Hibrido  --  Sistema Inteligente de Riego (Edge AI)
Parques Urbanos  |  Firebase  +  Telegram Bot  +  Dashboard Streamlit
================================================================================

El modelo TinyML vive en el ESP32 (modelo_edge.h). Este servidor orquesta:
  - Persistencia de telemetria en Firebase Realtime Database
  - Notificaciones automaticas via Telegram cuando la valvula se activa
  - Recepcion de comandos del bot de Telegram (/regar, /detener)
  - Override manual desde el Dashboard Streamlit

ENDPOINTS
─────────────────────────────────────────────────────────────────────────────
  POST /api/iot/telemetry        ESP32 reporta lectura + decision -> Firebase + Telegram
  POST /api/iot/override         Dashboard envia orden de control -> Firebase
  GET  /api/iot/status           Ultimo estado almacenado en Firebase
  POST /api/telegram/webhook     Telegram envia actualizaciones del bot
  GET  /health                   Verificacion de disponibilidad
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

# ── Configuracion Telegram ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")
TELEGRAM_API_BASE  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

if not FIREBASE_URL:
    import warnings
    warnings.warn("FIREBASE_URL no configurada.", stacklevel=1)

if not TELEGRAM_BOT_TOKEN:
    import warnings
    warnings.warn("TELEGRAM_BOT_TOKEN no configurado. Las notificaciones estaran desactivadas.", stacklevel=1)


# ── Helpers Firebase ──────────────────────────────────────────────────────────

def _firebase_write(path: str, data: dict) -> dict:
    url  = f"{FIREBASE_URL}/{path}.json"
    resp = requests.put(url, json=data, timeout=8)
    resp.raise_for_status()
    return resp.json()


def _firebase_push(path: str, data: dict) -> dict:
    url  = f"{FIREBASE_URL}/{path}.json"
    resp = requests.post(url, json=data, timeout=8)
    resp.raise_for_status()
    return resp.json()


def _firebase_read(path: str):
    url  = f"{FIREBASE_URL}/{path}.json"
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    return data if data else None


# ── Helpers Telegram ──────────────────────────────────────────────────────────

def _telegram_send(text: str, chat_id: str = "") -> bool:
    """Envia un mensaje al chat configurado (o al chat_id indicado)."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    target = chat_id or TELEGRAM_CHAT_ID
    if not target:
        return False
    try:
        resp = requests.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            json={"chat_id": target, "text": text, "parse_mode": "Markdown"},
            timeout=6,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _telegram_reply(chat_id: str, text: str):
    """Responde a un mensaje en el chat de origen."""
    _telegram_send(text, chat_id=chat_id)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({
        "status":            "ok",
        "timestamp":         _now_iso(),
        "telegram_enabled":  bool(TELEGRAM_BOT_TOKEN),
        "firebase_enabled":  bool(FIREBASE_URL),
    })


# ── POST /api/iot/telemetry ───────────────────────────────────────────────────
@app.post("/api/iot/telemetry")
def iot_telemetry():
    """
    El ESP32 reporta cada lectura junto con la decision que tomo localmente.
    El servidor guarda en Firebase y, si la valvula se abrio, notifica por Telegram.

    Body JSON:
    {
        "soil_moisture":   float,   // Humedad del suelo (0-100 %)
        "air_temperature": float,   // Temperatura del aire (grados C)
        "air_humidity":    float,   // Humedad relativa del aire — solo telemetria
        "valve_decision":  int,     // Decision del ESP32: 0=OFF  1=ON
        "node_id":         string   // (opcional)
    }
    """
    body = request.get_json(silent=True) or {}

    required = ["soil_moisture", "air_temperature", "air_humidity", "valve_decision"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"ok": False, "error": f"Faltan campos: {missing}"}), 400

    valve = body["valve_decision"]
    if valve not in (0, 1):
        return jsonify({"ok": False, "error": "valve_decision debe ser 0 o 1"}), 400

    sm   = float(body["soil_moisture"])
    temp = float(body["air_temperature"])
    hum  = float(body["air_humidity"])

    record = {
        "soil_moisture":   sm,
        "air_temperature": temp,
        "air_humidity":    hum,
        "valve_decision":  int(valve),
        "valve_state":     "ON" if valve == 1 else "OFF",
        "node_id":         str(body.get("node_id", "esp32_nodo_1")),
        "source":          "edge_ai",
        "timestamp":       _now_iso(),
    }

    # ── Persistir en Firebase ─────────────────────────────────────────────────
    try:
        result = _firebase_push("telemetry/history", record)
        _firebase_write("telemetry/latest", record)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    # ── Notificacion Telegram (solo cuando la valvula se abre) ────────────────
    telegram_sent = False
    if valve == 1:
        msg = (
            f"*Riego activado* por el ESP32\n"
            f"Humedad de suelo: *{sm:.1f}%*\n"
            f"Temperatura: *{temp:.1f} C*\n"
            f"Humedad del aire: {hum:.1f}%\n"
            f"_Nodo: {record['node_id']} | {record['timestamp']}_"
        )
        telegram_sent = _telegram_send(msg)

    return jsonify({
        "ok":           True,
        "firebase_key": result.get("name"),
        "telegram_sent": telegram_sent,
    }), 201


# ── POST /api/iot/override ────────────────────────────────────────────────────
@app.post("/api/iot/override")
def iot_override():
    """
    Orden de control manual desde el Dashboard Streamlit.
    Escribe en Firebase; el ESP32 hace polling y obedece.

    Body JSON:
    { "command": 1, "duration": 120, "source": "dashboard" }
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

    source = str(body.get("source", "dashboard"))

    override_record = {
        "command":     int(command),
        "valve_state": "ON" if command == 1 else "OFF",
        "duration":    duration,
        "source":      source,
        "issued_at":   _now_iso(),
        "active":      True,
    }

    try:
        _firebase_write("control/override", override_record)
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    # Notificar por Telegram tambien cuando el override es manual
    action = "abierta (riego forzado)" if command == 1 else "cerrada (riego detenido)"
    _telegram_send(f"*Override manual:* valvula {action}\nOrigen: {source} | {override_record['issued_at']}")

    return jsonify({"ok": True, "command": command, "duration": duration, "state": override_record["valve_state"]})


# ── GET /api/iot/status ───────────────────────────────────────────────────────
@app.get("/api/iot/status")
def iot_status():
    """Retorna el ultimo estado del nodo y el override activo."""
    try:
        latest   = _firebase_read("telemetry/latest")
        override = _firebase_read("control/override")
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Firebase error: {exc}"}), 502

    return jsonify({"ok": True, "latest_telemetry": latest, "active_override": override})


# ── POST /api/telegram/webhook ────────────────────────────────────────────────
@app.post("/api/telegram/webhook")
def telegram_webhook():
    """
    Telegram envia aqui cada mensaje que recibe el bot.
    Comandos soportados:
      /regar  [segundos]  -> override command=1 (default 120 s)
      /detener            -> override command=0
      /estado             -> responde con el ultimo estado del nodo

    Para registrar el webhook:
      POST https://api.telegram.org/bot<TOKEN>/setWebhook
      Body: { "url": "https://tu-servidor.com/api/telegram/webhook" }
    """
    update = request.get_json(silent=True) or {}

    message = update.get("message") or update.get("edited_message") or {}
    chat_id = str(message.get("chat", {}).get("id", ""))
    text    = (message.get("text") or "").strip().lower()

    if not chat_id or not text:
        return jsonify({"ok": True}), 200

    # ── /regar ────────────────────────────────────────────────────────────────
    if text.startswith("/regar"):
        parts    = text.split()
        duration = 120
        if len(parts) > 1 and parts[1].isdigit():
            duration = max(10, min(int(parts[1]), 3600))

        override_record = {
            "command":     1,
            "valve_state": "ON",
            "duration":    duration,
            "source":      f"telegram:{chat_id}",
            "issued_at":   _now_iso(),
            "active":      True,
        }
        try:
            _firebase_write("control/override", override_record)
            _telegram_reply(chat_id, f"Riego activado por *{duration} segundos*.")
        except requests.RequestException:
            _telegram_reply(chat_id, "Error al conectar con Firebase.")

    # ── /detener ──────────────────────────────────────────────────────────────
    elif text.startswith("/detener"):
        override_record = {
            "command":     0,
            "valve_state": "OFF",
            "duration":    1,
            "source":      f"telegram:{chat_id}",
            "issued_at":   _now_iso(),
            "active":      True,
        }
        try:
            _firebase_write("control/override", override_record)
            _telegram_reply(chat_id, "Riego detenido. Valvula cerrada.")
        except requests.RequestException:
            _telegram_reply(chat_id, "Error al conectar con Firebase.")

    # ── /estado ───────────────────────────────────────────────────────────────
    elif text.startswith("/estado"):
        try:
            latest = _firebase_read("telemetry/latest") or {}
            if latest:
                msg = (
                    f"*Estado actual del nodo*\n"
                    f"Humedad de suelo: {latest.get('soil_moisture', '?')}%\n"
                    f"Temperatura: {latest.get('air_temperature', '?')} C\n"
                    f"Humedad del aire: {latest.get('air_humidity', '?')}%\n"
                    f"Valvula: *{latest.get('valve_state', '?')}*\n"
                    f"Ultima lectura: {latest.get('timestamp', '?')}"
                )
            else:
                msg = "Sin datos del nodo todavia."
            _telegram_reply(chat_id, msg)
        except requests.RequestException:
            _telegram_reply(chat_id, "Error al leer Firebase.")

    # ── Comando no reconocido ─────────────────────────────────────────────────
    elif text.startswith("/"):
        _telegram_reply(
            chat_id,
            "Comandos disponibles:\n/regar [segundos]\n/detener\n/estado",
        )

    return jsonify({"ok": True}), 200


# ── Utilidades ────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
