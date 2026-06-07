# -*- coding: utf-8 -*-
"""
================================================================================
dashboard_streamlit.py
Sistema Inteligente de Riego — Vaccinium corymbosum (Arandano Alto)
Interfaz visual Streamlit  |  Consume la API Flask en app.py

Ejecucion:
    streamlit run dashboard_streamlit.py

Requiere que app.py este corriendo en http://localhost:5000
================================================================================
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema de Riego — Arandano",
    page_icon="🫐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Fondo y tipografia general ---- */
.main { background-color: #0e1117; }
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

/* ---- Tarjetas KPI ---- */
.kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e2538 100%);
    border: 1px solid #2d3650;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.kpi-label {
    color: #8892b0;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-unit {
    color: #8892b0;
    font-size: 0.85rem;
    margin-top: 4px;
}
.kpi-delta-up   { color: #ff6b6b; font-size: 0.8rem; }
.kpi-delta-down { color: #51cf66; font-size: 0.8rem; }
.kpi-delta-ok   { color: #74c0fc; font-size: 0.8rem; }

/* ---- Banner de estado del actuador ---- */
.actuator-on {
    background: linear-gradient(90deg, #0d4f1f, #155724);
    border: 1px solid #28a745;
    border-radius: 10px;
    padding: 12px 20px;
    color: #d4edda;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
}
.actuator-off {
    background: linear-gradient(90deg, #1a1f2e, #1e2538);
    border: 1px solid #2d3650;
    border-radius: 10px;
    padding: 12px 20px;
    color: #8892b0;
    font-size: 1rem;
    font-weight: 600;
    text-align: center;
}
.source-badge-real {
    background: #1a3a2a;
    border: 1px solid #28a745;
    color: #51cf66;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
}
.source-badge-est {
    background: #2a2a1a;
    border: 1px solid #f0ad4e;
    color: #ffd43b;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background-color: #131720;
    border-right: 1px solid #2d3650;
}
.sim-result-ok  { background:#1a3a2a; border:1px solid #28a745;
                  border-radius:8px; padding:12px; color:#d4edda; }
.sim-result-err { background:#3a1a1a; border:1px solid #dc3545;
                  border-radius:8px; padding:12px; color:#f8d7da; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
FLASK_URL  = "http://localhost:5000"
KC_STAGE   = {
    "Germinacion"          : 0.30,
    "Desarrollo_Vegetativo": 0.70,
    "Floracion"            : 1.05,
    "Fructificacion"       : 0.90,
}
STAGE_DISPLAY = {
    "Germinacion"          : "Germinacion",
    "Desarrollo_Vegetativo": "Desarrollo Vegetativo",
    "Floracion"            : "Floracion",
    "Fructificacion"       : "Fructificacion",
}
CROP_OPTIONS  = ["Wheat", "Potato", "Carrot", "Tomato", "Chilli"]
SOIL_OPTIONS  = ["Sandy Soil", "Clay Soil", "Red Soil",
                 "Loam Soil", "Black Soil", "Alluvial Soil", "Chalky Soil"]
STAGE_OPTIONS = list(KC_STAGE.keys())

COLOR_REGAR    = "#51cf66"
COLOR_NO_REGAR = "#ff6b6b"
COLOR_TEMP     = "#ff9f43"
COLOR_HUM      = "#74c0fc"
COLOR_MOI      = "#a29bfe"
COLOR_ETC      = "#00cec9"

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def calcular_etc(temp: float, etapa: str) -> float:
    eto = 0.408 * max(0.0, temp - 2.0) / 25.0 * (1 + temp / 50.0)
    kc  = KC_STAGE.get(etapa, 0.70)
    return round(max(0.0, eto * kc), 4)


def _get(path: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(f"{FLASK_URL}{path}", params=params, timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{FLASK_URL}{path}", json=payload, timeout=8)
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=10)
def fetch_latest() -> dict:
    data = _get("/api/latest-firebase")
    return data or {"temperatura": 22.0, "humedad": 65.0, "moi": 60.0,
                    "moi_category": "Optimo", "timestamp": "",
                    "fuente_moi": "fallback"}


@st.cache_data(ttl=30)
def fetch_history() -> pd.DataFrame:
    data = _get("/api/firebase-data")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


@st.cache_data(ttl=15)
def fetch_predictions() -> pd.DataFrame:
    data = _get("/api/predicciones-historial")
    if not data:
        return pd.DataFrame()
    rows = []
    for p in data:
        ml = p.get("outputs", {}).get("ml", {})
        se = p.get("outputs", {}).get("se", {})
        inp = p.get("inputs", {})
        rows.append({
            "timestamp"     : p.get("timestamp", ""),
            "prediction"    : ml.get("prediction_text", ""),
            "confidence"    : ml.get("confidence", 0),
            "score_mamdani" : se.get("Score_Difuso", 0),
            "nivel"         : se.get("Nivel_Prioridad", ""),
            "moi"           : inp.get("moi", 0),
            "temp"          : inp.get("temp", 0),
            "humidity"      : inp.get("humidity", 0),
            "etapa"         : inp.get("seedling_stage", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)
    return df


@st.cache_data(ttl=5)
def fetch_actuator() -> dict:
    data = _get("/api/iot/actuator-signal")
    return data or {"signal": 0, "confidence": 0.0, "timestamp": ""}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — SIMULADOR DE SENSORES
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/"
        "Blueberries_%28Vaccinium_corymbosum%29.jpg/320px-Blueberries_"
        "%28Vaccinium_corymbosum%29.jpg",
        use_container_width=True,
    )
    st.markdown("### 🫐 Sistema de Riego")
    st.markdown("*Vaccinium corymbosum* — Lima, Peru")
    st.divider()

    # ---- Config ----
    st.markdown("#### ⚙️ Configuracion")
    flask_url_input = st.text_input("URL Flask API", value=FLASK_URL, key="flask_url")
    auto_refresh    = st.toggle("Auto-refresh (10 s)", value=False)
    st.divider()

    # ---- Simulador ----
    st.markdown("#### 🧪 Simulador de Sensores ESP32")
    st.caption("Simula un POST a `/api/iot/sensor-data` mientras el hardware esta en desarrollo.")

    with st.form("simulator_form", clear_on_submit=False):
        sim_node    = st.text_input("Node ID", value="ESP32_SIM_01")

        st.markdown("**Lecturas del sensor**")
        sim_moi  = st.slider("MOI — SEN0193 v1.2 (%)",  0.0, 100.0, 45.0, 0.5,
                             help="Humedad volumetrica del suelo medida por el sensor capacitivo")
        sim_temp = st.slider("Temperatura — DHT22 (°C)", 5.0,  50.0, 24.0, 0.5)
        sim_hum  = st.slider("Humedad — DHT22 (%)",      0.0, 100.0, 65.0, 0.5)

        st.markdown("**Contexto agronomico**")
        sim_crop  = st.selectbox("Cultivo",          CROP_OPTIONS,  index=0)
        sim_soil  = st.selectbox("Tipo de suelo",    SOIL_OPTIONS,  index=0)
        sim_stage = st.selectbox("Etapa fenologica", STAGE_OPTIONS, index=2,
                                 format_func=lambda x: STAGE_DISPLAY[x])

        # Preview ETc en tiempo real
        etc_preview = calcular_etc(sim_temp, sim_stage)
        st.markdown(
            f"<div style='background:#1a1f2e;border-radius:8px;padding:8px 12px;"
            f"font-size:0.82rem;color:#8892b0;margin-top:4px'>"
            f"ETc estimado: <strong style='color:{COLOR_ETC}'>{etc_preview:.4f} mm/d</strong>"
            f"  |  Kc={KC_STAGE[sim_stage]}"
            f"</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("📡 Enviar al Sistema", use_container_width=True,
                                          type="primary")

    if submitted:
        payload = {
            "node_id"       : sim_node,
            "moi"           : sim_moi,
            "temp"          : sim_temp,
            "humidity"      : sim_hum,
            "crop_id"       : sim_crop,
            "soil_type"     : sim_soil,
            "seedling_stage": sim_stage,
            "timestamp"     : datetime.utcnow().isoformat(),
        }
        with st.spinner("Enviando..."):
            resp = _post("/api/iot/sensor-data", payload)

        if resp and "error" not in resp:
            pred_txt = resp.get("prediction_text", "")
            sig      = resp.get("actuator_signal", 0)
            conf     = resp.get("confidence", 0)
            score    = resp.get("score_mamdani", 0)
            nivel    = resp.get("nivel_prioridad", "")
            emoji    = "💧" if sig == 1 else "⏸️"
            color    = "#51cf66" if sig == 1 else "#8892b0"
            st.markdown(
                f"<div class='sim-result-ok'>"
                f"<b>{emoji} {pred_txt}</b><br>"
                f"Confianza ML: {conf*100:.1f}%<br>"
                f"Score Mamdani: {score:.1f}/100<br>"
                f"Nivel: <span style='color:{color}'>{nivel}</span><br>"
                f"Actuador: <b style='color:{color}'>{'ACTIVAR' if sig else 'INACTIVO'}</b>"
                f"</div>", unsafe_allow_html=True)
            # Limpiar cache para reflejar nuevos datos
            fetch_latest.clear()
            fetch_actuator.clear()
        else:
            err = resp.get("error", "Sin respuesta") if resp else "Flask no disponible"
            st.markdown(
                f"<div class='sim-result-err'>❌ Error: {err}</div>",
                unsafe_allow_html=True)

    st.divider()
    st.caption("v2.0 · CRISP-DM · Mamdani · GB+RF")

# ─────────────────────────────────────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#ccd6f6;font-size:1.9rem;margin-bottom:2px'>"
    "🫐 Sistema Inteligente de Riego</h1>"
    "<p style='text-align:center;color:#8892b0;font-size:0.9rem;margin-top:0'>"
    "<em>Vaccinium corymbosum</em> &nbsp;·&nbsp; Costa de Lima, Peru &nbsp;·&nbsp;"
    " Gradient Boosting + Motor Mamdani</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
latest      = fetch_latest()
history_df  = fetch_history()
pred_df     = fetch_predictions()
actuator    = fetch_actuator()

temp        = float(latest.get("temperatura", 22))
hum         = float(latest.get("humedad",     65))
moi         = float(latest.get("moi",         60))
moi_cat     = latest.get("moi_category", "—")
fuente_moi  = latest.get("fuente_moi",   "estimacion_difusa")
ts_str      = latest.get("timestamp",    "")

# ETc para cada etapa (arándano)
etc_actual  = calcular_etc(temp, "Floracion")    # etapa mas representativa
etc_by_stage = {e: calcular_etc(temp, e) for e in KC_STAGE}

# ─────────────────────────────────────────────────────────────────────────────
# BANNER DE ACTUADOR
# ─────────────────────────────────────────────────────────────────────────────
signal     = actuator.get("signal", 0)
act_conf   = actuator.get("confidence", 0.0)
act_ts     = actuator.get("timestamp", "")

if signal == 1:
    st.markdown(
        f"<div class='actuator-on'>💧 ELECTROVALVULA ACTIVA &nbsp;|&nbsp; "
        f"Confianza: {act_conf*100:.1f}%  &nbsp;|&nbsp;  Ultima decision: {act_ts[:19] if act_ts else '—'}"
        f"</div>", unsafe_allow_html=True)
else:
    st.markdown(
        f"<div class='actuator-off'>⏸️ ELECTROVALVULA INACTIVA &nbsp;|&nbsp; "
        f"Confianza: {act_conf*100:.1f}%  &nbsp;|&nbsp;  Ultima decision: {act_ts[:19] if act_ts else '—'}"
        f"</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILA DE KPIs
# ─────────────────────────────────────────────────────────────────────────────
kpi_temp, kpi_hum, kpi_moi, kpi_etc = st.columns(4)

# -- Temperatura
temp_color = "#ff6b6b" if temp > 30 else ("#74c0fc" if temp < 18 else "#51cf66")
temp_note  = "⚠ Calor" if temp > 30 else ("❄ Frio" if temp < 18 else "✓ Optimo")
with kpi_temp:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>🌡 Temperatura</div>"
        f"<div class='kpi-value' style='color:{temp_color}'>{temp:.1f}</div>"
        f"<div class='kpi-unit'>°C  &nbsp; <span style='color:{temp_color}'>{temp_note}</span></div>"
        f"<div style='margin-top:8px;font-size:0.72rem;color:#8892b0'>Optimo: 18–24 °C</div>"
        f"</div>", unsafe_allow_html=True)

# -- Humedad ambiental
hum_color = "#ff9f43" if hum < 40 else ("#74c0fc" if hum > 80 else "#51cf66")
hum_note  = "Seca" if hum < 40 else ("Alta" if hum > 80 else "Adecuada")
with kpi_hum:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>💧 Humedad Ambiental</div>"
        f"<div class='kpi-value' style='color:{hum_color}'>{hum:.1f}</div>"
        f"<div class='kpi-unit'>%  &nbsp; <span style='color:{hum_color}'>{hum_note}</span></div>"
        f"<div style='margin-top:8px;font-size:0.72rem;color:#8892b0'>Sensor DHT22</div>"
        f"</div>", unsafe_allow_html=True)

# -- MOI (real vs estimado)
moi_color = "#ff6b6b" if moi < 40 or moi > 85 else \
            ("#ffd43b" if moi < 60 else "#51cf66")
badge_html = (
    "<span class='source-badge-real'>SEN0193 Real</span>"
    if fuente_moi == "sensor_real_SEN0193"
    else "<span class='source-badge-est'>Estimado</span>"
)
with kpi_moi:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>🌱 MOI — Humedad Suelo</div>"
        f"<div class='kpi-value' style='color:{moi_color}'>{moi:.1f}</div>"
        f"<div class='kpi-unit'>%  &nbsp; {badge_html}</div>"
        f"<div style='margin-top:8px;font-size:0.72rem;color:#8892b0'>"
        f"{moi_cat}  |  Optimo: 60–80 %</div>"
        f"</div>", unsafe_allow_html=True)

# -- ETc actual (floración)
etc_color = "#ff6b6b" if etc_actual > 1.0 else \
            ("#ffd43b" if etc_actual > 0.6 else "#51cf66")
with kpi_etc:
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>☀ ETc — Evapotranspiracion</div>"
        f"<div class='kpi-value' style='color:{etc_color}'>{etc_actual:.4f}</div>"
        f"<div class='kpi-unit'>mm/dia  &nbsp; (Kc=1.05 Floracion)</div>"
        f"<div style='margin-top:8px;font-size:0.72rem;color:#8892b0'>"
        f"Hargreaves-Samani x FAO-56</div>"
        f"</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab_hist, tab_pred, tab_etc, tab_live = st.tabs([
    "📈 Historico de Sensores",
    "🔮 Predicciones",
    "🌿 Analisis ETc",
    "⚡ Estado en Vivo",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICO DE SENSORES
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    if history_df.empty:
        st.info("No hay datos historicos disponibles. Verifica que app.py este corriendo "
                "y Firebase tenga datos.")
    else:
        # Filtro de rango temporal
        col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
        with col_f1:
            n_puntos = st.slider("Ultimas N lecturas", 20, min(500, len(history_df)),
                                 min(100, len(history_df)), 10)
        with col_f2:
            suavizado = st.toggle("Suavizado (rolling 5)", value=False)

        df_plot = history_df.tail(n_puntos).copy()
        if suavizado:
            df_plot["temperatura"] = df_plot["temperatura"].rolling(5, min_periods=1).mean()
            df_plot["humedad"]     = df_plot["humedad"].rolling(5, min_periods=1).mean()

        # ---- Grafico 1: Temperatura + Humedad ----
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Scatter(
            x=df_plot["timestamp"], y=df_plot["temperatura"],
            name="Temperatura (°C)", line=dict(color=COLOR_TEMP, width=2),
            fill="tozeroy", fillcolor="rgba(255,159,67,0.07)",
        ), secondary_y=False)
        fig1.add_trace(go.Scatter(
            x=df_plot["timestamp"], y=df_plot["humedad"],
            name="Humedad (%)", line=dict(color=COLOR_HUM, width=2, dash="dot"),
        ), secondary_y=True)

        # Banda de temperatura optima para arandano
        fig1.add_hrect(y0=18, y1=24, fillcolor="rgba(81,207,102,0.07)",
                       line_width=0, annotation_text="Zona optima temp.",
                       annotation_position="top left",
                       annotation_font_color="#51cf66",
                       secondary_y=False)

        fig1.update_layout(
            title="Temperatura y Humedad Ambiental — DHT22",
            template="plotly_dark", height=350,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=50, b=20),
        )
        fig1.update_yaxes(title_text="Temperatura (°C)", secondary_y=False,
                          color=COLOR_TEMP)
        fig1.update_yaxes(title_text="Humedad (%)", secondary_y=True,
                          color=COLOR_HUM)
        st.plotly_chart(fig1, use_container_width=True)

        # ---- Estadisticas rapidas ----
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Temp. media",   f"{df_plot['temperatura'].mean():.1f} °C")
        col_s2.metric("Temp. max",     f"{df_plot['temperatura'].max():.1f} °C")
        col_s3.metric("Hum. media",    f"{df_plot['humedad'].mean():.1f} %")
        col_s4.metric("N lecturas",    f"{len(df_plot)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    if pred_df.empty:
        st.info("No hay predicciones guardadas todavia. Usa el simulador del panel "
                "lateral o la pagina /prediccion-individual para generar predicciones.")
    else:
        col_p1, col_p2 = st.columns([1, 1])

        # ---- Dona: proporcion regar vs no regar ----
        with col_p1:
            conteo = pred_df["prediction"].value_counts()
            fig_dona = go.Figure(go.Pie(
                labels=conteo.index,
                values=conteo.values,
                hole=0.55,
                marker_colors=[COLOR_REGAR if "Requiere" in l else COLOR_NO_REGAR
                               for l in conteo.index],
                textinfo="label+percent",
                textfont_size=13,
            ))
            fig_dona.update_layout(
                title="Distribucion de Decisiones",
                template="plotly_dark", height=320,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_dona, use_container_width=True)

        # ---- Barras: confianza ultimas predicciones ----
        with col_p2:
            df_recent = pred_df.head(30).copy()
            df_recent["color"] = df_recent["prediction"].apply(
                lambda x: COLOR_REGAR if "Requiere" in str(x) else COLOR_NO_REGAR)
            fig_bar = go.Figure(go.Bar(
                x=df_recent["timestamp"].astype(str).str[:16],
                y=df_recent["confidence"] * 100,
                marker_color=df_recent["color"],
                text=df_recent["confidence"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Confianza ML — Ultimas 30 predicciones",
                template="plotly_dark", height=320,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                xaxis=dict(showticklabels=False),
                yaxis=dict(title="Confianza (%)", range=[0, 110]),
                margin=dict(l=10, r=10, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Score Mamdani vs ML Confidence ----
        st.markdown("##### Comparacion Score Mamdani vs Confianza ML")
        df_scatter = pred_df.dropna(subset=["score_mamdani", "confidence"]).head(100)
        if not df_scatter.empty:
            fig_sc = px.scatter(
                df_scatter,
                x="confidence",
                y="score_mamdani",
                color="prediction",
                color_discrete_map={
                    "Requiere Riego"    : COLOR_REGAR,
                    "No Requiere Riego" : COLOR_NO_REGAR,
                },
                labels={
                    "confidence"    : "Confianza ML (%)",
                    "score_mamdani" : "Score Mamdani (0-100)",
                    "prediction"    : "Decision",
                },
                hover_data=["moi", "temp", "humidity", "etapa"],
                template="plotly_dark",
            )
            fig_sc.add_vline(x=0.5, line_dash="dash", line_color="#8892b0",
                             annotation_text="Umbral ML 50%")
            fig_sc.add_hline(y=50,  line_dash="dash", line_color="#8892b0",
                             annotation_text="Umbral Mamdani 50")
            fig_sc.update_layout(
                height=380,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ---- Tabla detallada ----
        with st.expander("Ver tabla de predicciones recientes"):
            st.dataframe(
                pred_df.head(20)[
                    ["timestamp", "prediction", "confidence",
                     "score_mamdani", "nivel", "moi", "temp", "humidity"]
                ].rename(columns={
                    "timestamp"    : "Fecha",
                    "prediction"   : "Decision ML",
                    "confidence"   : "Confianza",
                    "score_mamdani": "Score Mamdani",
                    "nivel"        : "Nivel Prioridad",
                    "moi"          : "MOI (%)",
                    "temp"         : "Temp (°C)",
                    "humidity"     : "Humedad (%)",
                }),
                use_container_width=True,
                hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALISIS ETc
# ══════════════════════════════════════════════════════════════════════════════
with tab_etc:
    st.markdown("##### ETc por Etapa Fenologica (condiciones actuales)")
    st.caption(f"Temperatura actual: **{temp:.1f} °C** — Metodo Hargreaves-Samani x Kc FAO-56")

    # ---- Barras ETc por etapa ----
    etapas    = list(etc_by_stage.keys())
    etc_vals  = list(etc_by_stage.values())
    kc_vals   = [KC_STAGE[e] for e in etapas]

    fig_etc = go.Figure()
    fig_etc.add_trace(go.Bar(
        name="ETc (mm/dia)",
        x=etapas,
        y=etc_vals,
        marker_color=[COLOR_ETC] * 4,
        text=[f"{v:.4f}" for v in etc_vals],
        textposition="outside",
    ))
    fig_etc.add_trace(go.Scatter(
        name="Kc (FAO-56)",
        x=etapas,
        y=kc_vals,
        mode="markers+lines",
        marker=dict(size=10, color="#ffd43b"),
        line=dict(color="#ffd43b", dash="dot"),
        yaxis="y2",
    ))
    fig_etc.update_layout(
        template="plotly_dark", height=370,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        yaxis=dict(title="ETc (mm/dia)", color=COLOR_ETC),
        yaxis2=dict(title="Kc", overlaying="y", side="right",
                    color="#ffd43b", range=[0, 1.4]),
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=30, b=20),
    )
    st.plotly_chart(fig_etc, use_container_width=True)

    # ---- Gauge de MOI actual vs rangos del arandano ----
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### Indicador MOI — Vaccinium corymbosum")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=moi,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "MOI (%)  |  Optimo: 60–80 %", "font": {"color": "#ccd6f6"}},
            delta={"reference": 70, "increasing": {"color": "#ff6b6b"},
                   "decreasing": {"color": "#51cf66"}},
            gauge={
                "axis"      : {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar"       : {"color": moi_color},
                "bgcolor"   : "#1a1f2e",
                "bordercolor": "#2d3650",
                "steps": [
                    {"range": [0,  40], "color": "rgba(255,107,107,0.25)"},
                    {"range": [40, 60], "color": "rgba(255,212,59,0.15)"},
                    {"range": [60, 80], "color": "rgba(81,207,102,0.25)"},
                    {"range": [80,100], "color": "rgba(116,192,252,0.25)"},
                ],
                "threshold": {
                    "line"  : {"color": "#ffd43b", "width": 3},
                    "thickness": 0.75,
                    "value" : 70,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(color="#ccd6f6"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.markdown("##### Tabla de referencia — Arandano")
        st.markdown("<br>", unsafe_allow_html=True)
        ref_data = {
            "Etapa"          : list(STAGE_DISPLAY.values()),
            "Kc FAO-56"      : [KC_STAGE[e] for e in STAGE_OPTIONS],
            f"ETc @ {temp:.0f}°C (mm/d)": [etc_by_stage[e] for e in STAGE_OPTIONS],
            "MOI min (%)"    : [55, 60, 60, 58],
            "MOI max (%)"    : [75, 80, 80, 78],
        }
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#1a1f2e;border-radius:8px;padding:12px;"
            "font-size:0.82rem;color:#8892b0;border:1px solid #2d3650'>"
            "<b style='color:#ccd6f6'>Fuente MOI actual:</b><br>"
            + (f"<span class='source-badge-real'>SEN0193 v1.2 — Dato real</span>"
               if fuente_moi == "sensor_real_SEN0193"
               else "<span class='source-badge-est'>Estimacion difusa (fallback)</span>")
            + "<br><br><b style='color:#ccd6f6'>Timestamp:</b><br>"
            + f"<code>{ts_str[:19] if ts_str else 'N/A'}</code>"
            + "</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ESTADO EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.markdown("##### Lectura mas reciente del sistema")
        st.markdown("<br>", unsafe_allow_html=True)

        def _badge(label, value, unit, color):
            return (f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;padding:10px 16px;"
                    f"border-bottom:1px solid #2d3650'>"
                    f"<span style='color:#8892b0;font-size:0.85rem'>{label}</span>"
                    f"<span style='color:{color};font-weight:700;font-size:1rem'>"
                    f"{value} <span style='font-size:0.75rem;color:#8892b0'>{unit}</span>"
                    f"</span></div>")

        st.markdown(
            f"<div style='background:#1a1f2e;border-radius:10px;"
            f"border:1px solid #2d3650;overflow:hidden'>"
            + _badge("Temperatura",      f"{temp:.1f}",      "°C",   COLOR_TEMP)
            + _badge("Humedad ambiental",f"{hum:.1f}",       "%",    COLOR_HUM)
            + _badge("MOI (humedad suelo)",f"{moi:.1f}",     "%",    moi_color)
            + _badge("ETc (Floracion)",  f"{etc_actual:.4f}","mm/d", COLOR_ETC)
            + _badge("Fuente MOI",
                     "SEN0193 Real" if fuente_moi=="sensor_real_SEN0193" else "Estimado",
                     "",
                     "#51cf66" if fuente_moi=="sensor_real_SEN0193" else "#ffd43b")
            + _badge("Ultima lectura",   ts_str[:19] if ts_str else "—", "", "#8892b0")
            + "</div>", unsafe_allow_html=True)

    with col_v2:
        st.markdown("##### Endpoints disponibles para el equipo hardware")
        st.markdown("<br>", unsafe_allow_html=True)

        endpoints = [
            ("POST", "/api/iot/sensor-data",
             "Enviar lecturas SEN0193 + DHT22. Devuelve senal de actuacion inmediata."),
            ("GET",  "/api/iot/actuator-signal",
             "Polling: consultar si debe activarse la electrovalvula (senal 0/1)."),
            ("POST", "/api/iot/heartbeat",
             "Registro de disponibilidad del nodo ESP32."),
        ]
        for method, path, desc in endpoints:
            color_m = "#51cf66" if method == "GET" else "#ff9f43"
            st.markdown(
                f"<div style='background:#1a1f2e;border-radius:8px;"
                f"border:1px solid #2d3650;padding:10px 14px;margin-bottom:8px'>"
                f"<span style='background:{color_m};color:#000;border-radius:4px;"
                f"padding:2px 8px;font-size:0.72rem;font-weight:700'>{method}</span>"
                f"&nbsp; <code style='color:#ccd6f6'>{path}</code>"
                f"<div style='color:#8892b0;font-size:0.78rem;margin-top:4px'>{desc}</div>"
                f"</div>", unsafe_allow_html=True)

        # Test de conectividad con Flask
        st.markdown("##### Estado de la API Flask")
        api_ok = _get("/api/latest-firebase") is not None
        if api_ok:
            st.success(f"Flask API activa en `{FLASK_URL}`")
        else:
            st.error(f"Flask API no disponible en `{FLASK_URL}`. "
                     "Ejecuta `python app.py` primero.")

# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8892b0;font-size:0.78rem'>"
    "Sistema Inteligente de Riego &nbsp;·&nbsp; <em>Vaccinium corymbosum</em> "
    "&nbsp;·&nbsp; UNMSM — Facultad de Ingenieria de Sistemas e Informatica"
    "<br>Gradient Boosting F1=0.9922 &nbsp;|&nbsp; Motor Mamdani (22 reglas) "
    "&nbsp;|&nbsp; Split 70/30 estratificado por etapa fenologica"
    "</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()
