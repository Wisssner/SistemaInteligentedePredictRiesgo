# -*- coding: utf-8 -*-
"""
================================================================================
dashboard_streamlit.py  --  Panel Municipal de Telemetria y Control Override
Sistema Inteligente de Riego Edge AI  --  Parques Urbanos
================================================================================
Lee datos en tiempo real desde Firebase Realtime Database.
Permite enviar ordenes de riego manual forzado via API REST.

Compatible con movil (responsive): columnas adaptativas, botones grandes.
================================================================================
"""

import os
import time
from datetime import datetime

import requests
import streamlit as st

# ── Configuracion de pagina (mobile-first) ────────────────────────────────────
st.set_page_config(
    page_title="Riego Urbano — Panel Municipal",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Variables de entorno / configuracion ──────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL",  "http://localhost:5000")
FIREBASE_URL  = os.getenv("FIREBASE_URL",  "").rstrip("/")
AUTO_REFRESH  = 15  # segundos entre recargas automaticas

# ── CSS Mobile-friendly ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fuente base y fondo */
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Tarjetas de metrica */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2d3d 0%, #0f1923 100%);
    border: 1px solid #2d4a6a;
    border-radius: 12px;
    padding: 16px 20px !important;
    margin-bottom: 8px;
}
div[data-testid="metric-container"] label {
    color: #7fb3d3 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4fd !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* Boton de override — grande y visible en movil */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #d62828 0%, #a01010 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 16px 32px !important;
    width: 100% !important;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 15px rgba(214, 40, 40, 0.4);
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(214, 40, 40, 0.6) !important;
    transform: translateY(-1px);
}

/* Boton secundario (apagar riego) */
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    width: 100% !important;
}

/* Separador de seccion */
hr {
    border-color: #2d4a6a !important;
    margin: 1rem 0 !important;
}

/* Alertas / mensajes de estado */
.valve-on {
    background: linear-gradient(135deg, #1a5c2a, #0d3317);
    border-left: 4px solid #2ecc71;
    border-radius: 8px;
    padding: 12px 16px;
    color: #7dff9e;
    font-weight: 600;
    font-size: 1.1rem;
}
.valve-off {
    background: linear-gradient(135deg, #3d1a1a, #200d0d);
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 12px 16px;
    color: #ff9d9d;
    font-weight: 600;
    font-size: 1.1rem;
}
.info-box {
    background: #1a2d40;
    border: 1px solid #2d4a6a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #a8c8e0;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers de datos ──────────────────────────────────────────────────────────

def _get_status() -> dict | None:
    """Consulta el estado actual via API REST."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/iot/status", timeout=6)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _get_firebase_history(limit: int = 20) -> list:
    """Lee el historial de telemetria directamente desde Firebase."""
    if not FIREBASE_URL:
        return []
    try:
        url  = f"{FIREBASE_URL}/telemetry/history.json?orderBy=\"$key\"&limitToLast={limit}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return []
        records = list(data.values())
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return records
    except Exception:
        return []


def _send_override(command: int, duration: int) -> dict | None:
    """Envia orden de override via API REST."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/iot/override",
            json={"command": command, "duration": duration},
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("## 💧 Panel de Riego — Parques Urbanos")
st.markdown(
    '<p class="info-box">Panel municipal de telemetria Edge AI. '
    'El ESP32 toma decisiones autonomas offline. '
    'Este panel muestra lecturas en tiempo real y permite control manual.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Carga de estado ───────────────────────────────────────────────────────────
status_data = _get_status()
latest      = (status_data or {}).get("latest_telemetry") or {}
override    = (status_data or {}).get("active_override")  or {}

soil_moisture   = latest.get("soil_moisture",   "--")
air_temperature = latest.get("air_temperature", "--")
air_humidity    = latest.get("air_humidity",    "--")
valve_state     = latest.get("valve_state",     "DESCONOCIDO")
valve_decision  = latest.get("valve_decision",  -1)
last_seen       = latest.get("timestamp",       "Sin datos")
node_id         = latest.get("node_id",         "esp32_nodo_1")

# ── Estado de la valvula (banner grande) ──────────────────────────────────────
if valve_decision == 1:
    st.markdown(
        '<div class="valve-on">🟢 VALVULA ABIERTA — Riego ACTIVO</div>',
        unsafe_allow_html=True,
    )
elif valve_decision == 0:
    st.markdown(
        '<div class="valve-off">🔴 VALVULA CERRADA — Riego DETENIDO</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Sin datos del nodo todavia. Esperando primera telemetria del ESP32...")

st.markdown("")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    valor_sm = f"{soil_moisture:.1f} %" if isinstance(soil_moisture, float) else soil_moisture
    delta_sm = None
    if isinstance(soil_moisture, (int, float)):
        if soil_moisture < 30:
            delta_sm = "Bajo — regar"
        elif soil_moisture > 70:
            delta_sm = "Optimo"
    st.metric(label="Humedad Suelo", value=valor_sm, delta=delta_sm)

with col2:
    valor_t = f"{air_temperature:.1f} °C" if isinstance(air_temperature, float) else air_temperature
    st.metric(label="Temperatura Aire", value=valor_t)

with col3:
    valor_h = f"{air_humidity:.1f} %" if isinstance(air_humidity, float) else air_humidity
    st.metric(label="Humedad Aire", value=valor_h)

# Info de ultima actualizacion
if last_seen != "Sin datos":
    st.caption(f"Ultimo dato recibido: {last_seen}  |  Nodo: {node_id}")

st.markdown("---")

# ── SECCION: Control Manual (Override) ───────────────────────────────────────
st.markdown("### Control Manual")

with st.expander("Configurar duracion del riego manual", expanded=False):
    dur_min = st.slider("Duracion (minutos)", min_value=1, max_value=60, value=5, step=1)
    duration_sec = dur_min * 60
    st.caption(f"El ESP32 mantendra la valvula abierta {duration_sec} segundos ({dur_min} min).")

col_on, col_off = st.columns(2)

with col_on:
    if st.button("💧 FORZAR RIEGO MANUAL", type="primary", use_container_width=True):
        with st.spinner("Enviando orden a Firebase..."):
            result = _send_override(command=1, duration=duration_sec if "duration_sec" in dir() else 300)
        if result and result.get("ok"):
            st.success(f"Orden enviada: ABRIR valvula por {result.get('duration', 300)} s")
        else:
            err = (result or {}).get("error", "Sin respuesta del servidor")
            st.error(f"Error al enviar override: {err}")

with col_off:
    if st.button("⛔ Detener Riego", type="secondary", use_container_width=True):
        with st.spinner("Enviando orden a Firebase..."):
            result = _send_override(command=0, duration=1)
        if result and result.get("ok"):
            st.success("Orden enviada: CERRAR valvula")
        else:
            err = (result or {}).get("error", "Sin respuesta del servidor")
            st.error(f"Error al enviar override: {err}")

# Estado del override activo
if override and override.get("active"):
    issued = override.get("issued_at", "")
    cmd    = override.get("valve_state", "?")
    dur    = override.get("duration", 0)
    st.info(f"Override activo: valvula **{cmd}** por {dur} s — emitido {issued}")

st.markdown("---")

# ── Historial de telemetria ───────────────────────────────────────────────────
st.markdown("### Ultimas lecturas del nodo")

history = _get_firebase_history(limit=15)

if history:
    import pandas as pd

    rows = []
    for r in history:
        rows.append({
            "Fecha/Hora":   r.get("timestamp", ""),
            "H. Suelo (%)": r.get("soil_moisture", ""),
            "Temp. (°C)":   r.get("air_temperature", ""),
            "H. Aire (%)":  r.get("air_humidity", ""),
            "Valvula":      r.get("valve_state", ""),
            "Fuente":       r.get("source", ""),
        })

    df = pd.DataFrame(rows)

    # Colorear la columna Valvula
    def highlight_valve(val):
        color = "#1a5c2a" if val == "ON" else "#3d1a1a"
        text  = "#7dff9e" if val == "ON" else "#ff9d9d"
        return f"background-color: {color}; color: {text}; font-weight: bold;"

    styled = df.style.applymap(highlight_valve, subset=["Valvula"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.caption("Aun no hay historial en Firebase. Conecta el ESP32 para comenzar.")

st.markdown("---")

# ── Pie de pagina ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="info-box">'
    '<strong>Arquitectura Edge AI:</strong> el modelo TinyML vive dentro del ESP32 '
    'como codigo C puro. Las decisiones se toman de forma autonoma sin internet. '
    'Este panel solo muestra telemetria y permite control de emergencia.</div>',
    unsafe_allow_html=True,
)

# Auto-refresh
col_ref, col_time = st.columns([1, 3])
with col_ref:
    if st.button("Actualizar", use_container_width=True):
        st.rerun()
with col_time:
    st.caption(f"Datos al: {datetime.now().strftime('%H:%M:%S')}  —  Auto-refresh cada {AUTO_REFRESH} s")

# Recarga automatica usando time.sleep + rerun
time.sleep(AUTO_REFRESH)
st.rerun()
