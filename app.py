# -*- coding: utf-8 -*-
"""
================================================================================
app.py  —  Servidor Flask  —  Sistema Inteligente de Riego
Vaccinium corymbosum (Arandano Alto) — Costa de Lima, Peru

Endpoints disponibles
─────────────────────
PAGINAS WEB
  GET  /                           Pagina principal
  GET  /prediccion-individual      Formulario de prediccion manual
  GET  /prediccion-masiva          Carga de Excel batch
  GET  /dashboard                  Historico Firebase
  GET  /analisis-vivo              Monitoreo en tiempo real
  GET  /agente-recomendaciones     Agente de recomendaciones agricolas
  GET  /agente-optimizacion        Agente de optimizacion de riego

API REST — PREDICCION
  POST /api/predict                Prediccion individual (GB + SE + Gemini)
  POST /api/predict-batch          Prediccion masiva (archivo Excel)

API REST — IOT HARDWARE (para equipo ESP32)
  POST /api/iot/sensor-data        Recibe lecturas SEN0193 + DHT22 del nodo
  GET  /api/iot/actuator-signal    Polling: retorna senal de actuacion (0/1)
  POST /api/iot/heartbeat          Registro de disponibilidad del nodo

API REST — FIREBASE / DATOS
  GET  /api/firebase-data          Historico de lecturas IoT
  GET  /api/predicciones-historial Ultimas 50 predicciones guardadas
  GET  /api/latest-firebase        Ultima lectura Firebase

API REST — MONITOREO EN TIEMPO REAL
  POST /api/start-monitoring       Inicia hilo de monitoreo
  POST /api/stop-monitoring        Detiene hilo de monitoreo
  GET  /api/last-prediction        Ultima prediccion del monitor

API REST — AGENTES INTELIGENTES
  POST /api/agente-recomendaciones Recomendaciones agricolas personalizadas
  POST /api/agente-optimizacion    Calendario de riego optimizado
  POST /api/analisis-completo      Analisis completo ambos agentes
================================================================================
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import threading
import time
from zoneinfo import ZoneInfo

from agentes_inteligentes import (AgenteRecomendaciones, AgenteOptimizacion,
                                  obtener_recomendaciones_completas)
from expert_system import evaluate_expert_system
from guardar_predicciones import guardar_prediccion_firestore

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION DE LA APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# URLS FIREBASE
# ─────────────────────────────────────────────────────────────────────────────
FIREBASE_URL      = 'https://si-grupo2-default-rtdb.firebaseio.com/iot_lecturas_clima.json'
PREDICCIONES_URL  = 'https://si-grupo2-default-rtdb.firebaseio.com/predicciones.json'
SENSOR_REAL_URL   = 'https://si-grupo2-default-rtdb.firebaseio.com/sensor_esp32.json'

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (Gradient Boosting — arandano)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_FILE = 'modelos_guardados1/best_model_Gradient_Boosting_arandano_20260607_084930.pkl'
with open(_MODEL_FILE, 'rb') as f:
    model_package = pickle.load(f)

model    = model_package['model']
scaler   = model_package['scaler']
encoders = model_package['label_encoders']

# Mapa de etapas: nombres externos -> etapas del arandano codificadas
STAGE_MAP = model_package.get('stage_map', {})
KC_ARANDANO = model_package.get('kc_arandano', {
    'Germinacion': 0.30, 'Desarrollo_Vegetativo': 0.70,
    'Floracion': 1.05,   'Fructificacion': 0.90,
})

# ─────────────────────────────────────────────────────────────────────────────
# AGENTES INTELIGENTES
# ─────────────────────────────────────────────────────────────────────────────
agente_recomendaciones = AgenteRecomendaciones()
agente_optimizacion    = AgenteOptimizacion()

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO GLOBAL — SENSOR REAL DEL ESP32
# ─────────────────────────────────────────────────────────────────────────────
# Almacena el ultimo dato recibido por POST /api/iot/sensor-data
last_real_sensor: dict = {}
_sensor_lock = threading.Lock()

# Estado del actuador (senal enviada al hardware)
actuator_state: dict = {'signal': 0, 'confidence': 0.0, 'timestamp': ''}

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def calcular_etc(temp: float, etapa: str) -> float:
    """Proxy ETc (Hargreaves-Samani simplificado x Kc FAO-56)."""
    eto = 0.408 * max(0.0, float(temp) - 2.0) / 25.0 * (1 + float(temp) / 50.0)
    # Normalizar nombre de etapa
    etapa_norm = str(etapa).strip()
    kc = KC_ARANDANO.get(etapa_norm, 0.70)
    return round(max(0.0, eto * kc), 4)


def normalizar_etapa(etapa_externa: str) -> str:
    """
    Convierte el nombre de etapa del formulario/hardware al formato
    interno del modelo (4 etapas del arandano).
    """
    norm = STAGE_MAP.get(etapa_externa, etapa_externa)
    # Si el encoders conoce la etapa ya mapeada, devolverla tal cual
    if norm in encoders['seedling'].classes_:
        return norm
    # Fallback: buscar coincidencia parcial
    for cls in encoders['seedling'].classes_:
        if cls.lower() in etapa_externa.lower() or etapa_externa.lower() in cls.lower():
            return cls
    return encoders['seedling'].classes_[0]


def format_timestamp(ts) -> str:
    """Convierte timestamp ISO 8601 o UNIX a hora de Lima (America/Lima)."""
    try:
        if isinstance(ts, str) and "T" in ts:
            dt = datetime.fromisoformat(ts)
            dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Lima"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("America/Lima"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)
    except Exception:
        return "N/A"


def get_firebase_data() -> dict | None:
    """Lee el nodo iot_lecturas_clima de Firebase Realtime Database."""
    try:
        r = requests.get(FIREBASE_URL, timeout=5)
        if r.status_code == 200 and r.json():
            return r.json()
    except Exception:
        pass
    return None


def get_latest_values() -> dict:
    """
    Devuelve los valores mas recientes de temperatura, humedad y MOI.

    Prioridad:
      1. Lectura real del sensor SEN0193 recibida en los ultimos 60 segundos
         por POST /api/iot/sensor-data (MOI fisico real).
      2. Ultima lectura de Firebase (temperatura y humedad del DHT22),
         usando la estimacion de MOI por logica difusa como fallback.
    """
    # ---- Prioridad 1: sensor real reciente ----
    with _sensor_lock:
        if last_real_sensor:
            ts_str = last_real_sensor.get('timestamp', '')
            try:
                ts_dt = datetime.fromisoformat(ts_str)
                if (datetime.utcnow() - ts_dt) < timedelta(seconds=60):
                    return {
                        'temperatura'   : last_real_sensor['temp'],
                        'humedad'       : last_real_sensor['humidity'],
                        'moi'           : last_real_sensor['moi'],
                        'moi_category'  : _get_moi_category(last_real_sensor['moi']),
                        'timestamp'     : ts_str,
                        'fuente_moi'    : 'sensor_real_SEN0193',
                    }
            except Exception:
                pass

    # ---- Prioridad 2: Firebase + estimacion difusa ----
    data = get_firebase_data()
    if data:
        last_key   = list(data.keys())[-1]
        last_entry = data[last_key]
        temp = last_entry.get('temperatura', 22)
        hum  = last_entry.get('humedad',     65)
        moi  = _moi_from_fuzzy(temp, hum)
        return {
            'temperatura' : temp,
            'humedad'     : hum,
            'moi'         : moi,
            'moi_category': _get_moi_category(moi),
            'timestamp'   : last_entry.get('timestamp', ''),
            'fuente_moi'  : 'estimacion_difusa',
        }

    # ---- Fallback absoluto ----
    return {
        'temperatura': 22, 'humedad': 65, 'moi': 60,
        'moi_category': 'Optimo', 'timestamp': '',
        'fuente_moi': 'fallback',
    }


def _moi_from_fuzzy(temperatura: float, humedad: float) -> float:
    """
    Estimacion heuristica de MOI cuando no hay sensor real.
    Solo se usa como fallback si el ESP32 no ha enviado datos recientes.
    """
    moi_base = 50.0
    temp_f   = (temperatura - 13) / (46 - 13)
    hum_f    = humedad / 100.0
    if temperatura > 35 and humedad < 40:
        return max(10.0, moi_base - 30)
    elif temperatura > 30 and humedad < 50:
        return max(20.0, moi_base - 20)
    elif temperatura < 20 and humedad > 70:
        return min(90.0, moi_base + 30)
    elif temperatura < 25 and humedad > 60:
        return min(80.0, moi_base + 20)
    adj = (temp_f - 0.5) * 20 - (hum_f - 0.5) * 30
    return round(max(10.0, min(90.0, moi_base + adj)), 2)


def _get_moi_category(moi: float) -> str:
    if moi < 20:   return "Seco Critico"
    if moi < 40:   return "Seco"
    if moi < 60:   return "Intermedio"
    if moi < 80:   return "Optimo"
    return "Humedo / Saturado"


def prepare_features(crop_id: str, soil_type: str, seedling_stage: str,
                     moi: float, temp: float, humidity: float) -> np.ndarray:
    """
    Construye y escala el vector de features para el modelo.
    Incluye ETc y los nuevos features de arandano.
    """
    etapa_norm = normalizar_etapa(seedling_stage)

    crop_enc     = encoders['crop'].transform([crop_id])[0]
    soil_enc     = encoders['soil'].transform([soil_type])[0]
    seedling_enc = encoders['seedling'].transform([etapa_norm])[0]

    hum_clean = min(float(humidity), 100.0)
    etc       = calcular_etc(temp, etapa_norm)

    features = np.array([[
        crop_enc, soil_enc, seedling_enc,
        moi, temp, hum_clean,
        etc,
        temp / (hum_clean + 1),         # temp_humidity_ratio
        moi * temp,                      # moi_temp_interaction
        temp ** 2,                       # temp_squared
        hum_clean ** 2,                  # humidity_squared
        moi ** 2,                        # moi_squared
        max(0.0, 60.0 - moi),            # moi_deficit  (umbral arandano)
        max(0.0, moi - 80.0),            # moi_exceso   (umbral arandano)
        moi * etc,                       # moi_etc_interaction
        etc / (hum_clean + 1),           # etc_humidity_ratio
    ]])

    return scaler.transform(features)


def predict_single(crop_id: str, soil_type: str, seedling_stage: str,
                   moi: float, temp: float, humidity: float) -> dict:
    """Ejecuta la prediccion y devuelve resultado + probabilidades."""
    feat  = prepare_features(crop_id, soil_type, seedling_stage, moi, temp, humidity)
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return {
        'prediction'     : int(pred),
        'prediction_text': 'Requiere Riego' if pred == 1 else 'No Requiere Riego',
        'probability_no' : float(proba[0]),
        'probability_yes': float(proba[1]),
        'confidence'     : float(max(proba)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — PAGINAS WEB
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediccion-individual')
def prediccion_individual():
    latest = get_latest_values()
    return render_template('prediccion_individual.html',
                           temperatura  = latest['temperatura'],
                           humedad      = latest['humedad'],
                           timestamp    = format_timestamp(latest['timestamp']),
                           moi          = latest['moi'],
                           moi_category = latest['moi_category'])


@app.route('/prediccion-masiva')
def prediccion_masiva():
    return render_template('prediccion_masiva.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/analisis-vivo')
def analisis_vivo():
    return render_template('analisis_vivo.html')


@app.route('/agente-recomendaciones')
def agente_recomendaciones_page():
    latest = get_latest_values()
    return render_template('agente_recomendaciones.html',
                           temperatura  = latest['temperatura'],
                           humedad      = latest['humedad'],
                           moi          = latest['moi'],
                           moi_category = latest['moi_category'],
                           timestamp    = format_timestamp(latest['timestamp']))


@app.route('/agente-optimizacion')
def agente_optimizacion_page():
    latest = get_latest_values()
    return render_template('agente_optimizacion.html',
                           temperatura = latest['temperatura'],
                           humedad     = latest['humedad'],
                           moi         = latest['moi'],
                           timestamp   = format_timestamp(latest['timestamp']))


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST: PREDICCION
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Prediccion individual combinando tres modulos:
      - ML      : Gradient Boosting
      - SE      : Sistema Experto Mamdani
      - Gemini  : Agente IA generativa
    """
    data = request.json

    result_ml = predict_single(
        data['crop_id'], data['soil_type'], data['seedling_stage'],
        float(data['moi']), float(data['temp']), float(data['humidity'])
    )

    result_se = evaluate_expert_system(
        float(data['temp']), float(data['humidity']),
        data['crop_id'],     data['soil_type'],
        data['seedling_stage'], float(data['moi'])
    )

    result_gemini = agente_recomendaciones.evaluate_with_gemini(
        float(data['temp']),    float(data['humidity']),
        float(data['moi']),     data['crop_id'],
        data['soil_type'],      data['seedling_stage']
    )

    inputs  = {k: data[k] for k in
               ['crop_id', 'soil_type', 'seedling_stage', 'moi', 'temp', 'humidity']}
    outputs = {'ml': result_ml, 'se': result_se, 'gemini': result_gemini}

    guardar_prediccion_firestore(inputs, outputs)

    # Actualizar estado del actuador con la decision ML
    with _sensor_lock:
        actuator_state['signal']     = result_ml['prediction']
        actuator_state['confidence'] = result_ml['confidence']
        actuator_state['timestamp']  = datetime.utcnow().isoformat()

    return jsonify({'ml': result_ml, 'se': result_se, 'gemini': result_gemini})


@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    """Prediccion masiva desde archivo Excel."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        df = pd.read_excel(file)
        required = ['crop ID', 'soil_type', 'Seedling Stage', 'MOI', 'temp', 'humidity']
        if not all(c in df.columns for c in required):
            return jsonify({'error': f'Columnas requeridas: {required}'}), 400

        predictions = []
        for idx, row in df.iterrows():
            result = predict_single(
                row['crop ID'], row['soil_type'], row['Seedling Stage'],
                float(row['MOI']), float(row['temp']), float(row['humidity'])
            )
            predictions.append({
                'row'       : idx + 1,
                'crop'      : row['crop ID'],
                'prediction': result['prediction_text'],
                'confidence': f"{result['confidence']*100:.2f}%",
            })

        return jsonify({'predictions': predictions, 'total': len(predictions)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST: IOT HARDWARE (equipo ESP32)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/iot/sensor-data', methods=['POST'])
def api_iot_sensor_data():
    """
    Recibe lecturas en tiempo real del nodo ESP32 (SEN0193 + DHT22).

    Body JSON esperado:
    {
        "moi"             : <float 0-100>,   # SEN0193 v1.2
        "temp"            : <float>,          # DHT22 temperatura (°C)
        "humidity"        : <float 0-100>,    # DHT22 humedad relativa (%)
        "node_id"         : <str>,            # identificador del nodo ESP32
        "timestamp"       : <str ISO 8601>,   # timestamp del nodo
        "crop_id"         : <str>,            # (opcional) tipo de cultivo
        "soil_type"       : <str>,            # (opcional) tipo de suelo
        "seedling_stage"  : <str>             # (opcional) etapa fenologica
    }

    Respuesta:
    {
        "received"        : true,
        "prediction"      : 0|1,
        "prediction_text" : "Requiere Riego" | "No Requiere Riego",
        "actuator_signal" : 0|1,
        "confidence"      : <float>,
        "score_mamdani"   : <float>,
        "server_timestamp": <str>
    }
    """
    data = request.json
    if not data:
        return jsonify({'error': 'Body JSON requerido'}), 400

    required = ['moi', 'temp', 'humidity']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Campo requerido faltante: {field}'}), 400

    try:
        moi      = max(0.0, min(100.0, float(data['moi'])))
        temp     = max(0.0, min(50.0,  float(data['temp'])))
        humidity = max(0.0, min(100.0, float(data['humidity'])))
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Valor numerico invalido: {e}'}), 400

    node_id       = data.get('node_id', 'ESP32_unknown')
    crop_id       = data.get('crop_id',       'Wheat')
    soil_type     = data.get('soil_type',     'Sandy Soil')
    seedling_stage= data.get('seedling_stage','Floracion')
    node_ts       = data.get('timestamp',     datetime.utcnow().isoformat())

    # Prediccion ML
    result_ml = predict_single(crop_id, soil_type, seedling_stage,
                                moi, temp, humidity)

    # Inferencia Mamdani
    result_se = evaluate_expert_system(temp, humidity, crop_id, soil_type,
                                        seedling_stage, moi)

    actuator = result_ml['prediction']

    # Guardar dato real del sensor en memoria
    with _sensor_lock:
        last_real_sensor.update({
            'moi'      : moi,
            'temp'     : temp,
            'humidity' : humidity,
            'node_id'  : node_id,
            'timestamp': node_ts,
        })
        actuator_state['signal']     = actuator
        actuator_state['confidence'] = result_ml['confidence']
        actuator_state['timestamp']  = datetime.utcnow().isoformat()

    # Persistir en Firebase
    guardar_prediccion_firestore(
        inputs={
            'node_id': node_id, 'moi': moi, 'temp': temp,
            'humidity': humidity, 'crop_id': crop_id,
            'soil_type': soil_type, 'seedling_stage': seedling_stage,
        },
        outputs={'ml': result_ml, 'se': result_se},
    )

    return jsonify({
        'received'         : True,
        'node_id'          : node_id,
        'prediction'       : result_ml['prediction'],
        'prediction_text'  : result_ml['prediction_text'],
        'actuator_signal'  : actuator,
        'confidence'       : result_ml['confidence'],
        'score_mamdani'    : result_se['Score_Difuso'],
        'nivel_prioridad'  : result_se['Nivel_Prioridad'],
        'server_timestamp' : datetime.utcnow().isoformat(),
    })


@app.route('/api/iot/actuator-signal', methods=['GET'])
def api_iot_actuator_signal():
    """
    Endpoint de polling para el nodo ESP32.
    El microcontrolador consulta periodicamente si debe activar la electrovalvula.

    Query params:
      node_id  (opcional) — identificador del nodo

    Respuesta:
    {
        "signal"    : 0|1,          # 1 = activar electrovalvula
        "confidence": <float>,
        "timestamp" : <str ISO>
    }
    """
    node_id = request.args.get('node_id', 'unknown')
    with _sensor_lock:
        return jsonify({
            'signal'    : actuator_state.get('signal', 0),
            'confidence': actuator_state.get('confidence', 0.0),
            'timestamp' : actuator_state.get('timestamp', ''),
            'node_id'   : node_id,
        })


@app.route('/api/iot/heartbeat', methods=['POST'])
def api_iot_heartbeat():
    """
    Registro de disponibilidad del nodo ESP32.
    El firmware envia este ping periodicamente para confirmar conectividad.

    Body JSON:
    {
        "node_id"         : <str>,
        "firmware_version": <str>,
        "uptime_s"        : <int>
    }
    """
    data = request.json or {}
    return jsonify({
        'ack'          : True,
        'node_id'      : data.get('node_id', 'unknown'),
        'server_time'  : datetime.utcnow().isoformat(),
        'actuator_signal': actuator_state.get('signal', 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST: FIREBASE / DATOS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/firebase-data')
def api_firebase_data():
    data = get_firebase_data()
    if data:
        entries = []
        for key, value in data.items():
            entries.append({
                'timestamp'  : format_timestamp(value.get('timestamp', '')),
                'temperatura': value.get('temperatura', 0),
                'humedad'    : value.get('humedad', 0),
            })
        return jsonify(entries)
    return jsonify([])


@app.route('/api/predicciones-historial')
def api_predicciones_historial():
    try:
        r = requests.get(PREDICCIONES_URL, timeout=5)
        if r.status_code != 200:
            return jsonify([])
        data = r.json()
        if not data:
            return jsonify([])

        predicciones = []
        for key, value in data.items():
            predicciones.append({
                'id'       : key,
                'timestamp': format_timestamp(value.get('timestamp', '')),
                'inputs'   : value.get('inputs', {}),
                'outputs'  : {
                    'ml'    : value.get('outputs', {}).get('ml', {}),
                    'gemini': value.get('outputs', {}).get('gemini', {}),
                    'se'    : value.get('outputs', {}).get('se', {}),
                },
            })

        predicciones.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify(predicciones[:50])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/latest-firebase')
def api_latest_firebase():
    latest = get_latest_values()
    latest['timestamp'] = format_timestamp(latest['timestamp'])
    return jsonify(latest)


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST: MONITOREO EN TIEMPO REAL
# ─────────────────────────────────────────────────────────────────────────────

monitoring       = False
last_prediction  = None


def monitor_firebase():
    global last_prediction, monitoring
    last_data = None

    while monitoring:
        try:
            data = get_firebase_data()
            if data:
                last_key = list(data.keys())[-1]
                current  = data[last_key]

                if last_data != current:
                    last_data = current
                    temp = current.get('temperatura', 22)
                    hum  = current.get('humedad',     65)
                    moi  = _moi_from_fuzzy(temp, hum)

                    result = predict_single('Wheat', 'Sandy Soil', 'Floracion',
                                           moi, temp, hum)

                    last_prediction = {
                        'timestamp'   : format_timestamp(current.get('timestamp', '')),
                        'temperatura' : temp,
                        'humedad'     : hum,
                        'moi'         : moi,
                        'moi_category': _get_moi_category(moi),
                        'prediction'  : result['prediction_text'],
                        'confidence'  : result['confidence'],
                    }
            time.sleep(2)
        except Exception:
            time.sleep(5)


@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    global monitoring
    if not monitoring:
        monitoring = True
        t = threading.Thread(target=monitor_firebase, daemon=True)
        t.start()
    return jsonify({'status': 'started'})


@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring
    monitoring = False
    return jsonify({'status': 'stopped'})


@app.route('/api/last-prediction')
def api_last_prediction():
    return jsonify(last_prediction or {})


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — API REST: AGENTES INTELIGENTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/agente-recomendaciones', methods=['POST'])
def api_agente_recomendaciones():
    data = request.json
    try:
        score, estado = agente_recomendaciones.calcular_score_salud(
            data['crop_id'], float(data['temp']),
            float(data['humidity']), float(data['moi'])
        )
        recomendaciones = agente_recomendaciones.generar_recomendaciones(
            data['crop_id'], data['soil_type'], data['seedling_stage'],
            float(data['moi']), float(data['temp']), float(data['humidity'])
        )
        return jsonify({
            'success'        : True,
            'score'          : score,
            'estado'         : estado,
            'recomendaciones': recomendaciones,
            'timestamp'      : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agente-optimizacion', methods=['POST'])
def api_agente_optimizacion():
    data = request.json
    try:
        calendario_data = agente_optimizacion.generar_calendario_riego(
            data['crop_id'], data['seedling_stage'],
            float(data['moi']), float(data['temp']), float(data['humidity']),
            dias=int(data.get('dias', 7))
        )
        prediccion_riego = 1 if float(data['moi']) < 45 else 0
        alertas  = agente_optimizacion.generar_alertas(
            data['crop_id'], float(data['moi']),
            float(data['temp']), float(data['humidity']), prediccion_riego
        )
        metricas = agente_optimizacion.calcular_metricas_eficiencia(
            calendario_data['calendario']
        )
        return jsonify({
            'success'    : True,
            'calendario' : calendario_data['calendario'],
            'estadisticas': calendario_data['estadisticas'],
            'alertas'    : alertas,
            'metricas'   : metricas,
            'timestamp'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analisis-completo', methods=['POST'])
def api_analisis_completo():
    data = request.json
    try:
        resultado = obtener_recomendaciones_completas(
            data['crop_id'], data['soil_type'], data['seedling_stage'],
            float(data['moi']), float(data['temp']), float(data['humidity'])
        )
        return jsonify({'success': True, 'resultado': resultado})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# INICIO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
