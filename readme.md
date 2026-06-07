# Sistema Inteligente de Riego — Vaccinium corymbosum

**Universidad Nacional Mayor de San Marcos — FISI**
Practicas Pre Profesionales · Seccion 1 · Lima, Peru

---

## Descripcion general

Sistema de clasificacion binaria con Machine Learning para la gestion optima del riego en el cultivo de **arandano alto** (*Vaccinium corymbosum*) en la region costera de Lima. Determina en tiempo real si el cultivo **requiere riego (1) o no (0)** a partir de variables edafoclimaticas capturadas por sensores IoT.

| Componente | Descripcion |
|---|---|
| Algoritmo principal | Gradient Boosting (F1 = 0.9922, Accuracy = 99.37 %) |
| Validador ML | Random Forest (F1 = 0.9904, Accuracy = 99.22 %) |
| Sistema experto | Motor de Inferencia Difusa Mamdani (22 reglas) |
| Dataset | 15 288 registros, split 70/30 estratificado por etapa fenologica |
| Features | 16 (incluye ETc Hargreaves-Samani x Kc FAO-56 y umbrales arandano) |
| Agente IA | Gemini AI (recomendaciones agricolas contextuales) |
| IoT | Firebase Realtime Database + ESP32 via ESP-NOW |

---

## Estructura del proyecto

```
SistemaInteligentedePredictRiesgo/
|
|-- app.py                          # Servidor Flask + API REST completa
|-- agentes_inteligentes.py         # Agente de Recomendaciones + Agente de Optimizacion
|-- expert_system.py                # Motor Mamdani genuino (fuzzificacion + centroide)
|-- proyecto_ml_completo.py         # Pipeline CRISP-DM: datos, ETc, entrenamiento, guardado
|-- guardar_predicciones.py         # Persistencia en Firebase Realtime Database
|-- dashboard_streamlit.py          # Dashboard visual interactivo (Streamlit + Plotly)
|
|-- dataSalvadora.xlsx              # Dataset de entrenamiento (15 288 registros)
|-- dataSalvadorasintarget.xlsx     # Dataset sin variable objetivo (prediccion batch)
|-- requirements.txt                # Dependencias del proyecto
|-- diagnostico.md                  # Analisis de cumplimiento del documento academico
|
|-- modelos_guardados1/
|   |-- best_model_Gradient_Boosting_arandano_<timestamp>.pkl   # Modelo principal (GB)
|   |-- best_model_Gradient_Boosting_arandano_<timestamp>.joblib
|   |-- best_model_Random_Forest_arandano_<timestamp>.pkl       # Validador (RF)
|   |-- best_model_Random_Forest_arandano_<timestamp>.joblib
|   `-- model_comparison_results.csv
|
|-- templates/                      # Interfaces HTML (Flask)
|   |-- base.html
|   |-- index.html
|   |-- prediccion_individual.html
|   |-- prediccion_masiva.html
|   |-- dashboard.html
|   |-- analisis_vivo.html
|   |-- agente_recomendaciones.html
|   `-- agente_optimizacion.html
|
|-- static/css/style.css
|-- visualizaciones1/               # Graficos generados por el pipeline ML
`-- uploads/                        # Archivos Excel subidos para prediccion batch
```

---

## Instalacion

```bash
pip install -r requirements.txt
```

> Requiere Python 3.10 o superior.

---

## Inicio rapido

El sistema tiene **dos interfaces** que corren en paralelo:

### 1. API Flask (backend + interfaz HTML)

```bash
python app.py
```

Disponible en `http://localhost:5000`

### 2. Dashboard Streamlit (interfaz visual principal)

```bash
streamlit run dashboard_streamlit.py
```

Disponible en `http://localhost:8501`

> El dashboard consume la API Flask, por lo que `app.py` debe estar corriendo primero.

---

## Reentrenar el modelo

```bash
python proyecto_ml_completo.py
```

Ejecuta el pipeline CRISP-DM completo:
- Remapeo de etapas fenologicas (8 genericas -> 4 del arandano)
- Calculo de ETc (Hargreaves-Samani x Kc FAO-56)
- Ingenieria de features (16 variables, incluyendo moi_deficit y moi_exceso)
- Particion 70/30 estratificada por etapa fenologica
- Entrenamiento y comparacion de GB, RF, Decision Tree y Regresion Logistica
- Guardado del modelo en `modelos_guardados1/`

Despues de reentrenar, actualiza el nombre del archivo en `app.py` linea:
```python
_MODEL_FILE = 'modelos_guardados1/best_model_Gradient_Boosting_arandano_<timestamp>.pkl'
```

---

## Arquitectura del sistema

```
+-------------------+     ESP-NOW      +------------------+
|  Nodos sensores   | ---------------> |  ESP32 coordinador|
|  SEN0193 + DHT22  |                  |  (Edge AI / nodo) |
+-------------------+                  +--------+---------+
                                                 |
                                         WiFi / Firebase
                                                 |
                                     +-----------v----------+
                                     |  Firebase Realtime DB |
                                     +-----------+----------+
                                                 |
                             +-------------------v--------------------+
                             |           app.py  (Flask API)          |
                             |                                        |
                             |  +------------+   +----------------+  |
                             |  | Gradient   |   | Motor Mamdani  |  |
                             |  | Boosting   |   | (22 reglas)    |  |
                             |  +------------+   +----------------+  |
                             |                                        |
                             |  +------------+   +----------------+  |
                             |  | Agente     |   | Agente         |  |
                             |  | Recomend.  |   | Optimizacion   |  |
                             |  +------------+   +----------------+  |
                             |                                        |
                             |  Endpoints IoT:                        |
                             |  POST /api/iot/sensor-data             |
                             |  GET  /api/iot/actuator-signal         |
                             |  POST /api/iot/heartbeat               |
                             +-------------------+--------------------+
                                                 |
                             +-------------------v--------------------+
                             |     dashboard_streamlit.py             |
                             |     KPIs · Historico · ETc · Mamdani  |
                             +----------------------------------------+
```

---

## Endpoints de la API Flask

### Prediccion

| Metodo | Ruta | Descripcion |
|---|---|---|
| POST | `/api/predict` | Prediccion individual (GB + Mamdani + Gemini) |
| POST | `/api/predict-batch` | Prediccion masiva desde archivo Excel |

### IoT Hardware (ESP32)

| Metodo | Ruta | Descripcion |
|---|---|---|
| POST | `/api/iot/sensor-data` | Recibe lecturas SEN0193 + DHT22. Devuelve senal de actuacion inmediata |
| GET | `/api/iot/actuator-signal` | Polling: consulta si debe activarse la electrovalvula (0 o 1) |
| POST | `/api/iot/heartbeat` | Registro de disponibilidad del nodo ESP32 |

### Datos y monitoreo

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/api/firebase-data` | Historico de lecturas IoT |
| GET | `/api/predicciones-historial` | Ultimas 50 predicciones guardadas |
| GET | `/api/latest-firebase` | Ultima lectura (prioriza MOI real si ESP32 activo) |
| POST | `/api/start-monitoring` | Inicia hilo de monitoreo en tiempo real |
| GET | `/api/last-prediction` | Ultima prediccion del monitor |

### Agentes inteligentes

| Metodo | Ruta | Descripcion |
|---|---|---|
| POST | `/api/agente-recomendaciones` | Score de salud + recomendaciones agricolas |
| POST | `/api/agente-optimizacion` | Calendario de riego + alertas + metricas |
| POST | `/api/analisis-completo` | Ambos agentes en una sola llamada |

---

## Ejemplo: enviar datos desde ESP32

```bash
curl -X POST http://localhost:5000/api/iot/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "node_id"        : "ESP32_CAMPO_01",
    "moi"            : 42.5,
    "temp"           : 27.3,
    "humidity"       : 61.0,
    "crop_id"        : "Wheat",
    "soil_type"      : "Sandy Soil",
    "seedling_stage" : "Floracion",
    "timestamp"      : "2026-06-07T10:30:00"
  }'
```

Respuesta:
```json
{
  "received"        : true,
  "prediction"      : 1,
  "prediction_text" : "Requiere Riego",
  "actuator_signal" : 1,
  "confidence"      : 0.9987,
  "score_mamdani"   : 62.4,
  "nivel_prioridad" : "MEDIA"
}
```

El ESP32 luego consulta la senal de actuacion:
```bash
curl http://localhost:5000/api/iot/actuator-signal?node_id=ESP32_CAMPO_01
```

---

## Modulos de IA

### Gradient Boosting (modelo principal)
- 200 estimadores, learning rate 0.08, max_depth 5
- Entrenado con 16 features incluyendo ETc y umbrales especificos del arandano
- F1-Score: **0.9922** | Accuracy: **99.37 %** | Falsos negativos: **12**

### Random Forest (validador complementario)
- 200 arboles, max_features='sqrt', class_weight='balanced'
- F1-Score: **0.9904** | Accuracy: **99.22 %** | Falsos negativos: **7**

### Motor Mamdani (sistema experto)
- Variables de entrada: MOI, temperatura, humedad ambiental, ETc
- 5 conjuntos difusos por variable (funciones triangulares y trapezoidales)
- 22 reglas SI-ENTONCES calibradas para *Vaccinium corymbosum*
- Defuzzificacion por centroide sobre universo [0, 100]
- Umbral de decision: score > 50 → Requiere Riego

### Agente de Recomendaciones
- Score de salud del cultivo (0-100)
- Recomendaciones de fertilizacion por etapa fenologica
- Alertas de plagas segun condiciones climaticas
- Enriquecimiento opcional con Gemini AI

### Agente de Optimizacion
- Calendario de riego proyectado (7, 14 o 21 dias)
- Calculo de volumen de agua por sesion (litros/m2)
- Alertas predictivas de condiciones criticas
- Metricas de eficiencia hidrica

---

## Variables del modelo

| Feature | Descripcion |
|---|---|
| `crop_encoded` | Tipo de cultivo (Label Encoding) |
| `soil_encoded` | Tipo de suelo (Label Encoding) |
| `seedling_encoded` | Etapa fenologica del arandano (4 clases) |
| `MOI` | Indice de humedad del suelo (%) — SEN0193 v1.2 |
| `temp` | Temperatura ambiental (°C) — DHT22 |
| `humidity_cleaned` | Humedad relativa del aire (%) — DHT22 |
| `etc` | ETc = Kc x ETo (Hargreaves-Samani, mm/dia) |
| `temp_humidity_ratio` | temp / (humidity + 1) |
| `moi_temp_interaction` | MOI x temp |
| `temp_squared` | temp^2 |
| `humidity_squared` | humidity^2 |
| `moi_squared` | MOI^2 |
| `moi_deficit` | max(0, 60 - MOI) — deficit bajo umbral arandano |
| `moi_exceso` | max(0, MOI - 80) — exceso sobre umbral arandano |
| `moi_etc_interaction` | MOI x ETc |
| `etc_humidity_ratio` | ETc / (humidity + 1) |

---

## Etapas fenologicas del arandano

| Etapa interna | Etapas originales mapeadas | Kc FAO-56 |
|---|---|---|
| Germinacion | Germination, Seedling Stage | 0.30 |
| Desarrollo_Vegetativo | Vegetative Growth / Root or Tuber Development | 0.70 |
| Floracion | Flowering, Pollination | 1.05 |
| Fructificacion | Fruit/Grain/Bulb Formation, Maturation, Harvest | 0.90 |

---

## Variables de entorno (opcional)

Crea un archivo `.env` en la raiz del proyecto:

```
GOOGLE_AI_API_KEY=tu_clave_de_gemini
GOOGLE_AI_MODEL=gemini-2.5-flash
GOOGLE_AI_ENABLED=true
```

Sin estas variables el sistema funciona completo; solo el modulo Gemini queda desactivado.

---

## Solucion de problemas

**`FileNotFoundError: best_model_Gradient_Boosting_arandano_...`**
Ejecuta `python proyecto_ml_completo.py` para generar el modelo y actualiza el nombre del archivo en la linea `_MODEL_FILE` de `app.py`.

**`ModuleNotFoundError`**
Ejecuta `pip install -r requirements.txt`.

**Flask API no disponible (dashboard muestra error)**
Asegurate de correr `python app.py` antes de lanzar el dashboard.

**Puerto 5000 ocupado**
Cambia el puerto en la ultima linea de `app.py`: `app.run(port=5001)` y actualiza `FLASK_URL` en `dashboard_streamlit.py`.

---

## Metricas finales del pipeline

| Modelo | Accuracy | F1-Score | ROC-AUC | Falsos neg. |
|---|---|---|---|---|
| Gradient Boosting | 99.37 % | 0.9922 | 0.9998 | 12 |
| Random Forest | 99.22 % | 0.9904 | 0.9993 | 7 |
| Decision Tree | 99.06 % | 0.9885 | 0.9993 | 19 |
| Logistic Regression | 94.53 % | 0.9321 | 0.9880 | 145 |

Particion: 70 % entrenamiento (10 701 registros) / 30 % prueba (4 587 registros)
Estratificacion: por etapa fenologica del arandano
