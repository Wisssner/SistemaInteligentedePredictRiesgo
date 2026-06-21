# -*- coding: utf-8 -*-
"""
================================================================================
dashboard_streamlit.py  --  Panel Academico de Riego Edge AI
Sistema Inteligente de Riego  --  Parques Urbanos  --  UNMSM FISI
================================================================================
Panel de presentacion para jurado academico y municipalidad.
Muestra telemetria historica desde Firebase con graficos de lineas y KPIs
en tiempo real. Incluye control de override con notificacion via Telegram.
================================================================================
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Configuracion de pagina ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Riego Edge AI — Panel Academico",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Variables de entorno ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
FIREBASE_URL = os.getenv("FIREBASE_URL", "").rstrip("/")
AUTO_REFRESH = 20

# ── CSS para presentacion academica ──────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', system-ui, sans-serif;
}
/* Tarjetas KPI */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
    border: 1px solid #2a4a6b;
    border-radius: 14px;
    padding: 18px 22px !important;
}
div[data-testid="metric-container"] label {
    color: #7bafd4 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4fd !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}
/* Banner valvula */
.valve-on {
    background: linear-gradient(135deg, #1a5c2a, #0d3317);
    border-left: 5px solid #2ecc71;
    border-radius: 10px;
    padding: 14px 20px;
    color: #7dff9e;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.03em;
}
.valve-off {
    background: linear-gradient(135deg, #3d1a1a, #200d0d);
    border-left: 5px solid #e74c3c;
    border-radius: 10px;
    padding: 14px 20px;
    color: #ff9d9d;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.03em;
}
.arch-box {
    background: #0d1b2a;
    border: 1px solid #2a4a6b;
    border-radius: 10px;
    padding: 16px 20px;
    color: #a8c8e0;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
}
.section-title {
    color: #7bafd4;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
hr { border-color: #2a4a6b !important; margin: 1.2rem 0 !important; }
/* Botones */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5c8a, #0d3357) !important;
    color: white !important;
    border: 1px solid #2a7abf !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers de datos ──────────────────────────────────────────────────────────

def _get_status() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/api/iot/status", timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _get_firebase_history(limit: int = 50) -> list:
    if not FIREBASE_URL:
        return []
    try:
        url  = f'{FIREBASE_URL}/telemetry/history.json?orderBy="$key"&limitToLast={limit}'
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return []
        records = list(data.values())
        records.sort(key=lambda r: r.get("timestamp", ""))
        return records
    except Exception:
        return []


def _send_override(command: int, duration: int, source: str = "dashboard") -> dict:
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/iot/override",
            json={"command": command, "duration": duration, "source": source},
            timeout=8,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Titulo principal ──────────────────────────────────────────────────────────
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.markdown("## Sistema Inteligente de Riego — Edge AI")
    st.caption("Parques Urbanos · Universidad Nacional Mayor de San Marcos · FISI")

st.markdown("---")

# ── Carga de datos ────────────────────────────────────────────────────────────
status_data     = _get_status()
latest          = (status_data or {}).get("latest_telemetry") or {}
override        = (status_data or {}).get("active_override")  or {}

soil_moisture   = latest.get("soil_moisture",   None)
air_temperature = latest.get("air_temperature", None)
air_humidity    = latest.get("air_humidity",    None)
valve_decision  = latest.get("valve_decision",  -1)
last_seen       = latest.get("timestamp",       "Sin datos")
node_id         = latest.get("node_id",         "esp32_nodo_1")

# ── Banner de estado de valvula ───────────────────────────────────────────────
if valve_decision == 1:
    st.markdown('<div class="valve-on">🟢  VALVULA ABIERTA — Riego ACTIVO (decision del ESP32)</div>', unsafe_allow_html=True)
elif valve_decision == 0:
    st.markdown('<div class="valve-off">🔴  VALVULA CERRADA — Riego DETENIDO</div>', unsafe_allow_html=True)
else:
    st.info("Esperando primera telemetria del ESP32...")

st.markdown("")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    val = f"{soil_moisture:.1f} %" if soil_moisture is not None else "--"
    delta = None
    if soil_moisture is not None:
        delta = "Critico — regar" if soil_moisture < 30 else ("Optimo" if soil_moisture > 60 else "Bajo")
    st.metric("Humedad de Suelo", val, delta)

with k2:
    val = f"{air_temperature:.1f} °C" if air_temperature is not None else "--"
    st.metric("Temperatura Aire", val)

with k3:
    val = f"{air_humidity:.1f} %" if air_humidity is not None else "--"
    st.metric("Humedad del Aire", val)

with k4:
    estado = "ON" if valve_decision == 1 else ("OFF" if valve_decision == 0 else "?")
    st.metric("Estado Valvula", estado)

if last_seen != "Sin datos":
    st.caption(f"Ultima lectura: {last_seen}  |  Nodo: {node_id}")

st.markdown("---")

# ── Tabs principales ──────────────────────────────────────────────────────────
tab_hist, tab_ctrl, tab_arch = st.tabs(["Historico y Graficos", "Control Override", "Arquitectura del Sistema"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICO Y GRAFICOS
# ═══════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("##### Telemetria historica desde Firebase")

    history = _get_firebase_history(limit=50)

    if not history:
        st.info("No hay datos historicos aun. El grafico aparecera cuando el ESP32 comience a enviar telemetria.")
    else:
        df_hist = pd.DataFrame(history)

        # Limpiar y convertir tipos
        for col in ["soil_moisture", "air_temperature", "air_humidity"]:
            if col in df_hist.columns:
                df_hist[col] = pd.to_numeric(df_hist[col], errors="coerce")

        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], errors="coerce", utc=True)
        df_hist = df_hist.dropna(subset=["timestamp"]).sort_values("timestamp")
        df_hist["hora"] = df_hist["timestamp"].dt.strftime("%H:%M:%S")

        # ── Grafico principal: Humedad Suelo + Temperatura (doble eje Y) ─────
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df_hist["hora"],
                y=df_hist["soil_moisture"],
                name="Humedad Suelo (%)",
                line=dict(color="#3498db", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.08)",
                mode="lines+markers",
                marker=dict(size=5),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df_hist["hora"],
                y=df_hist["air_temperature"],
                name="Temperatura Aire (°C)",
                line=dict(color="#e67e22", width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

        # Marca los momentos en que la valvula estuvo abierta
        df_on = df_hist[df_hist.get("valve_state", pd.Series(dtype=str)) == "ON"] if "valve_state" in df_hist else pd.DataFrame()
        if not df_on.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_on["hora"],
                    y=df_on["soil_moisture"],
                    mode="markers",
                    name="Riego activado",
                    marker=dict(color="#2ecc71", size=10, symbol="triangle-up"),
                ),
                secondary_y=False,
            )

        fig.update_layout(
            title="Humedad de Suelo vs Temperatura del Aire",
            plot_bgcolor="#0d1b2a",
            paper_bgcolor="#0d1b2a",
            font=dict(color="#a8c8e0"),
            legend=dict(bgcolor="#1b2838", bordercolor="#2a4a6b", borderwidth=1),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig.update_xaxes(title_text="Hora", gridcolor="#1e3050", tickangle=-30)
        fig.update_yaxes(title_text="Humedad Suelo (%)", secondary_y=False, gridcolor="#1e3050")
        fig.update_yaxes(title_text="Temperatura (°C)",  secondary_y=True,  gridcolor="#1e3050")

        st.plotly_chart(fig, use_container_width=True)

        # ── Grafico secundario: Humedad del Aire ─────────────────────────────
        if "air_humidity" in df_hist.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_hist["hora"],
                y=df_hist["air_humidity"],
                name="Humedad Aire (%)",
                line=dict(color="#9b59b6", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(155,89,182,0.08)",
                mode="lines+markers",
                marker=dict(size=5),
            ))
            fig2.update_layout(
                title="Humedad Relativa del Aire",
                plot_bgcolor="#0d1b2a",
                paper_bgcolor="#0d1b2a",
                font=dict(color="#a8c8e0"),
                margin=dict(l=10, r=10, t=50, b=10),
                hovermode="x unified",
            )
            fig2.update_xaxes(gridcolor="#1e3050", tickangle=-30)
            fig2.update_yaxes(gridcolor="#1e3050", range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        # ── Tabla de registros ────────────────────────────────────────────────
        st.markdown("##### Registros recientes")

        col_display = {
            "hora":            "Hora",
            "soil_moisture":   "H. Suelo (%)",
            "air_temperature": "Temp. (°C)",
            "air_humidity":    "H. Aire (%)",
            "valve_state":     "Valvula",
            "source":          "Origen",
        }
        cols_present = [c for c in col_display if c in df_hist.columns]
        df_show = df_hist[cols_present].tail(20).rename(columns=col_display)

        def _color_valve(val):
            if val == "ON":
                return "background-color:#1a5c2a;color:#7dff9e;font-weight:bold"
            if val == "OFF":
                return "background-color:#3d1a1a;color:#ff9d9d;font-weight:bold"
            return ""

        if "Valvula" in df_show.columns:
            styled = df_show.style.applymap(_color_valve, subset=["Valvula"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_show, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — CONTROL OVERRIDE
# ═══════════════════════════════════════════════════════════════════
with tab_ctrl:
    st.markdown("##### Enviar orden de control manual al ESP32")
    st.caption(
        "El ESP32 toma decisiones autonomas con el modelo TinyML. "
        "Estos botones envian un override a Firebase que el nodo consulta periodicamente."
    )

    with st.expander("Duracion del riego manual", expanded=True):
        dur_min      = st.slider("Minutos de riego", 1, 60, 5)
        duration_sec = dur_min * 60
        st.caption(f"La valvula permanecera abierta {duration_sec} s ({dur_min} min).")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("💧 FORZAR RIEGO MANUAL", type="primary", use_container_width=True):
            with st.spinner("Enviando a Firebase y Telegram..."):
                res = _send_override(1, duration_sec)
            if res.get("ok"):
                st.success(f"Orden enviada: valvula ABIERTA por {duration_sec} s")
            else:
                st.error(f"Error: {res.get('error', 'sin respuesta')}")

    with c2:
        if st.button("⛔ Detener Riego", type="secondary", use_container_width=True):
            with st.spinner("Enviando a Firebase y Telegram..."):
                res = _send_override(0, 1)
            if res.get("ok"):
                st.success("Orden enviada: valvula CERRADA")
            else:
                st.error(f"Error: {res.get('error', 'sin respuesta')}")

    st.markdown("---")
    st.markdown("##### Comandos disponibles en Telegram")
    st.code(
        "/regar [segundos]  — Activa el riego por N segundos (default 120)\n"
        "/detener           — Cierra la valvula inmediatamente\n"
        "/estado            — Consulta la ultima lectura del nodo",
        language="text",
    )

    if override and override.get("active"):
        st.info(
            f"Override activo: valvula **{override.get('valve_state')}** "
            f"por {override.get('duration')} s — "
            f"origen: {override.get('source')} — "
            f"emitido: {override.get('issued_at')}"
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ARQUITECTURA DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("##### Arquitectura hibrida Edge AI + Telemetria")

    st.markdown("""
<div class="arch-box">
Sensores (DHT22 + SEN0193)
        |
        v
+--------------------------------------------------+
|             ESP32  (nucleo Edge AI)              |
|                                                  |
|  predict(soil, temp, time)  <-- modelo_edge.h   |
|  Decide OFFLINE, sin internet                    |
|  -> Activa / desactiva valvula                   |
|                                                  |
|  Si WiFi disponible: POST /api/iot/telemetry     |
|  Payload: soil + temp + humidity + decision      |
+--------------------------------------------------+
        |
        v  HTTP
+--------------------------------------------------+
|            Flask API  (orquestador)              |
|                                                  |
|  +-- Firebase Realtime DB                        |
|  |   (historial, ultimo estado, overrides)       |
|  |                                               |
|  +-- API de Telegram                             |
|      (notificacion cuando valvula se abre)       |
+--------------------------------------------------+
        |                          |
        v                          v
  Dashboard Streamlit        Telegram Bot
  (jurado / municipio)       (operador movil)
  Graficos historicos        /regar  /detener
  Override manual            /estado
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("##### Principios de diseno")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Edge AI (ESP32)**
- Modelo: Decision Tree max_depth=8
- Accuracy: 91.08 %
- Features: Soil Moisture, Temperature, Time
- Codigo C puro en `modelo_edge.h`
- 147 nodos, ~16 KB de flash
- Sin conexion a internet: **sigue regando**
""")
    with col_b:
        st.markdown("""
**Capa de telemetria (Flask + Firebase)**
- Solo recibe reportes y overrides
- No toma ninguna decision de riego
- Firebase: persistencia en tiempo real
- Telegram: alertas y control remoto
- Dashboard: visualizacion para jurado
""")

# ── Pie y auto-refresh ────────────────────────────────────────────────────────
st.markdown("---")
cf, ct = st.columns([1, 4])
with cf:
    if st.button("Actualizar ahora", use_container_width=True):
        st.rerun()
with ct:
    st.caption(f"Panel actualizado: {datetime.now().strftime('%H:%M:%S')}  |  Auto-refresh cada {AUTO_REFRESH} s")

time.sleep(AUTO_REFRESH)
st.rerun()
