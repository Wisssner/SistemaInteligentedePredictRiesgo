from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import os
import threading
import time
from agentes_inteligentes import AgenteRecomendaciones, AgenteOptimizacion, obtener_recomendaciones_completas
from datetime import datetime
from zoneinfo import ZoneInfo  # zona horaria oficial
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

FIREBASE_URL = 'https://si-grupo2-default-rtdb.firebaseio.com/iot_lecturas_clima.json'

with open('modelos_guardados/best_model_Gradient_Boosting_20251126_085140.pkl', 'rb') as f:
    model_package = pickle.load(f)
    
model = model_package['model']
scaler = model_package['scaler']
encoders = model_package['label_encoders']

# Inicializar agentes inteligentes
agente_recomendaciones = AgenteRecomendaciones()
agente_optimizacion = AgenteOptimizacion()

def moi_fuzzy_logic(temperatura, humedad):
    temp_factor = (temperatura - 13) / (46 - 13)
    hum_factor = humedad / 100
    
    moi_base = 50
    
    if temperatura > 35 and humedad < 40:
        return max(10, moi_base - 30)
    elif temperatura > 30 and humedad < 50:
        return max(20, moi_base - 20)
    elif temperatura < 20 and humedad > 70:
        return min(90, moi_base + 30)
    elif temperatura < 25 and humedad > 60:
        return min(80, moi_base + 20)
    else:
        adjustment = (temp_factor - 0.5) * 20 - (hum_factor - 0.5) * 30
        return max(10, min(90, moi_base + adjustment))

def get_moi_category(moi):
    if moi < 20:
        return "Seco"
    elif moi < 40:
        return "Casi Seco"
    elif moi < 60:
        return "Intermedio"
    elif moi < 80:
        return "Casi Mojado"
    else:
        return "Mojado"

def get_firebase_data():
    try:
        response = requests.get(FIREBASE_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data
        return None
    except:
        return None

def get_latest_values():
    data = get_firebase_data()
    if data:
        last_key = list(data.keys())[-1]
        last_entry = data[last_key]
        temp = last_entry.get('temperatura', 28)
        hum = last_entry.get('humedad', 63)
        timestamp = last_entry.get('timestamp', '')
        
        return {
            'temperatura': temp,
            'humedad': hum,
            'timestamp': timestamp,
            'moi': moi_fuzzy_logic(temp, hum),
            'moi_category': get_moi_category(moi_fuzzy_logic(temp, hum))
        }
    return {'temperatura': 28, 'humedad': 63, 'timestamp': '', 'moi': 50, 'moi_category': 'Intermedio'}

def format_timestamp(ts):
    try:
        # Si es timestamp numérico
        if isinstance(ts, (int, float)):
            # Convertir **a Perú**
            dt = datetime.fromtimestamp(ts, tz=ZoneInfo("America/Lima"))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return ts
    except:
        return 'N/A'

def prepare_features(crop_id, soil_type, seedling_stage, moi, temp, humidity):
    crop_encoded = encoders['crop'].transform([crop_id])[0]
    soil_encoded = encoders['soil'].transform([soil_type])[0]
    seedling_encoded = encoders['seedling'].transform([seedling_stage])[0]
    
    humidity_cleaned = min(humidity, 100)
    temp_humidity_ratio = temp / (humidity_cleaned + 1)
    moi_temp_interaction = moi * temp
    temp_squared = temp ** 2
    humidity_squared = humidity_cleaned ** 2
    moi_squared = moi ** 2
    
    features = np.array([[
        crop_encoded, soil_encoded, seedling_encoded, moi,
        temp, humidity_cleaned, temp_humidity_ratio,
        moi_temp_interaction, temp_squared, humidity_squared, moi_squared
    ]])
    
    return scaler.transform(features)

def predict_single(crop_id, soil_type, seedling_stage, moi, temp, humidity):
    features_scaled = prepare_features(crop_id, soil_type, seedling_stage, moi, temp, humidity)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'prediction_text': 'Requiere Riego' if prediction == 1 else 'No Requiere Riego',
        'probability_no': float(probability[0]),
        'probability_yes': float(probability[1]),
        'confidence': float(max(probability))
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediccion-individual')
def prediccion_individual():
    latest = get_latest_values()
    return render_template('prediccion_individual.html', 
                         temperatura=latest['temperatura'],
                         humedad=latest['humedad'],
                         timestamp=format_timestamp(latest['timestamp']),
                         moi=latest['moi'],
                         moi_category=latest['moi_category'])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    
    result = predict_single(
        data['crop_id'],
        data['soil_type'],
        data['seedling_stage'],
        float(data['moi']),
        float(data['temp']),
        float(data['humidity'])
    )
    
    return jsonify(result)

@app.route('/prediccion-masiva')
def prediccion_masiva():
    return render_template('prediccion_masiva.html')

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        df = pd.read_excel(file)
        
        required_cols = ['crop ID', 'soil_type', 'Seedling Stage', 'MOI', 'temp', 'humidity']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': 'Missing required columns'}), 400
        
        predictions = []
        for idx, row in df.iterrows():
            result = predict_single(
                row['crop ID'],
                row['soil_type'],
                row['Seedling Stage'],
                float(row['MOI']),
                float(row['temp']),
                float(row['humidity'])
            )
            predictions.append({
                'row': idx + 1,
                'crop': row['crop ID'],
                'prediction': result['prediction_text'],
                'confidence': f"{result['confidence']*100:.2f}%"
            })
        
        return jsonify({'predictions': predictions, 'total': len(predictions)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/firebase-data')
def api_firebase_data():
    data = get_firebase_data()
    if data:
        entries = []
        for key, value in data.items():
            ts = value.get('timestamp', '')
            entries.append({
                'timestamp': format_timestamp(ts),
                'temperatura': value.get('temperatura', 0),
                'humedad': value.get('humedad', 0)
            })
        return jsonify(entries)
    return jsonify([])

@app.route('/analisis-vivo')
def analisis_vivo():
    return render_template('analisis_vivo.html')

@app.route('/api/latest-firebase')
def api_latest_firebase():
    latest = get_latest_values()
    latest['timestamp'] = format_timestamp(latest['timestamp'])
    return jsonify(latest)

monitoring = False
last_prediction = None

def monitor_firebase():
    global last_prediction, monitoring
    last_data = None
    
    while monitoring:
        try:
            data = get_firebase_data()
            if data:
                last_key = list(data.keys())[-1]
                current = data[last_key]
                
                if last_data != current:
                    last_data = current
                    
                    temp = current.get('temperatura', 28)
                    hum = current.get('humedad', 63)
                    moi = moi_fuzzy_logic(temp, hum)
                    
                    result = predict_single(
                        'Wheat',
                        'Black Soil',
                        'Flowering',
                        moi,
                        temp,
                        hum
                    )
                    
                    last_prediction = {
                        'timestamp': format_timestamp(current.get('timestamp', '')),
                        'temperatura': temp,
                        'humedad': hum,
                        'moi': moi,
                        'moi_category': get_moi_category(moi),
                        'prediction': result['prediction_text'],
                        'confidence': result['confidence']
                    }
            
            time.sleep(2)
        except:
            time.sleep(5)

@app.route('/api/start-monitoring', methods=['POST'])
def start_monitoring():
    global monitoring
    if not monitoring:
        monitoring = True
        thread = threading.Thread(target=monitor_firebase, daemon=True)
        thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop-monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring
    monitoring = False
    return jsonify({'status': 'stopped'})

@app.route('/api/last-prediction')
def api_last_prediction():
    if last_prediction:
        return jsonify(last_prediction)
    return jsonify({})

# ============================================================================
# RUTAS DE AGENTES INTELIGENTES
# ============================================================================

@app.route('/agente-recomendaciones')
def agente_recomendaciones_page():
    """Página del Agente de Recomendaciones Agrícolas"""
    latest = get_latest_values()
    return render_template('agente_recomendaciones.html',
                         temperatura=latest['temperatura'],
                         humedad=latest['humedad'],
                         moi=latest['moi'],
                         moi_category=latest['moi_category'],
                         timestamp=format_timestamp(latest['timestamp']))

@app.route('/agente-optimizacion')
def agente_optimizacion_page():
    """Página del Agente de Optimización de Riego"""
    latest = get_latest_values()
    return render_template('agente_optimizacion.html',
                         temperatura=latest['temperatura'],
                         humedad=latest['humedad'],
                         moi=latest['moi'],
                         timestamp=format_timestamp(latest['timestamp']))

@app.route('/api/agente-recomendaciones', methods=['POST'])
def api_agente_recomendaciones():
    """API para obtener recomendaciones del agente"""
    data = request.json
    
    try:
        # Calcular score de salud
        score, estado = agente_recomendaciones.calcular_score_salud(
            data['crop_id'],
            float(data['temp']),
            float(data['humidity']),
            float(data['moi'])
        )
        
        # Generar recomendaciones
        recomendaciones = agente_recomendaciones.generar_recomendaciones(
            data['crop_id'],
            data['soil_type'],
            data['seedling_stage'],
            float(data['moi']),
            float(data['temp']),
            float(data['humidity'])
        )
        
        return jsonify({
            'success': True,
            'score': score,
            'estado': estado,
            'recomendaciones': recomendaciones,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/agente-optimizacion', methods=['POST'])
def api_agente_optimizacion():
    """API para obtener calendario de riego optimizado"""
    data = request.json
    
    try:
        # Generar calendario de riego
        calendario_data = agente_optimizacion.generar_calendario_riego(
            data['crop_id'],
            data['seedling_stage'],
            float(data['moi']),
            float(data['temp']),
            float(data['humidity']),
            dias=int(data.get('dias', 7))
        )
        
        # Generar alertas
        prediccion_riego = 1 if float(data['moi']) < 45 else 0
        alertas = agente_optimizacion.generar_alertas(
            data['crop_id'],
            float(data['moi']),
            float(data['temp']),
            float(data['humidity']),
            prediccion_riego
        )
        
        # Calcular métricas de eficiencia
        metricas = agente_optimizacion.calcular_metricas_eficiencia(
            calendario_data['calendario']
        )
        
        return jsonify({
            'success': True,
            'calendario': calendario_data['calendario'],
            'estadisticas': calendario_data['estadisticas'],
            'alertas': alertas,
            'metricas': metricas,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analisis-completo', methods=['POST'])
def api_analisis_completo():
    """API para obtener análisis completo de ambos agentes"""
    data = request.json
    
    try:
        resultado = obtener_recomendaciones_completas(
            data['crop_id'],
            data['soil_type'],
            data['seedling_stage'],
            float(data['moi']),
            float(data['temp']),
            float(data['humidity'])
        )
        
        return jsonify({
            'success': True,
            'resultado': resultado
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
