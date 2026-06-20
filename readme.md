# Sistema Inteligente de Riego — Edge AI (TinyML)

**Universidad Nacional Mayor de San Marcos — FISI**
Practicas Pre-Profesionales · Lima, Peru

---

## Concepto central: el modelo vive en el ESP32

El nucleo del sistema es un **Decision Tree Classifier** (max_depth=4) entrenado
con datos de sensores reales (TARP.csv) y exportado como **codigo C puro** al
archivo `modelo_edge.h`. Ese archivo se sube directamente al microcontrolador
ESP32, que ejecuta la inferencia de forma completamente offline.

```
Sensores (DHT22 + SEN0193)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   ESP32  (Edge AI)                      │
│                                                         │
│  float sm   = leerHumedadSuelo();                       │
│  float temp = dht.readTemperature();                    │
│  float hum  = dht.readHumidity();                       │
│                                                         │
│  int decision = predict(sm, temp, hum);   // modelo_edge.h │
│  digitalWrite(VALVE_PIN, decision);                     │
│                                                         │
│  // Si hay WiFi: reporta telemetria via HTTP            │
│  // Si no hay WiFi: sigue funcionando igual             │
└─────────────────────────────────────────────────────────┘
        │  (WiFi, cuando disponible)
        ▼
  Flask API REST  ──►  Firebase Realtime DB
        │
        ▼
  Dashboard Streamlit  (panel municipal movil)
```

**Por que TinyML en el ESP32?**

| Escenario              | Resultado |
|------------------------|-----------|
| WiFi disponible        | Riega + reporta telemetria |
| Corte de internet      | **Riega igual** — decision autonoma |
| Servidor Flask caido   | **Riega igual** — no hay dependencia |
| Firebase inaccesible   | **Riega igual** — solo pierde historial |

---

## Arquitectura de archivos

```
├── proyecto_ml_completo.py   # Entrena el DT y genera modelo_edge.h
├── modelo_edge.h             # Codigo C puro para el ESP32 (auto-generado)
├── app.py                    # API REST de telemetria y override
├── dashboard_streamlit.py    # Panel municipal (Firebase, override manual)
├── TARP.csv                  # Dataset de entrenamiento (no incluido en repo)
└── requirements.txt
```

---

## modelo_edge.h — como funciona

El archivo generado contiene una funcion `predict()` en C++ puro:

```cpp
#include "modelo_edge.h"

void loop() {
    float sm   = leerHumedadSuelo();        // 0-100 %
    float temp = dht.readTemperature();     // grados C
    float hum  = dht.readHumidity();        // 0-100 %

    int decision = predict(sm, temp, hum);
    // decision == 1 → abrir valvula (regar)
    // decision == 0 → cerrar valvula

    digitalWrite(VALVE_PIN, decision);

    // Telemetria opcional (no bloquea el riego)
    if (WiFi.isConnected()) {
        enviarTelemetria(sm, temp, hum, decision);
    }
}
```

La funcion interna es un bloque de `if/else` anidados — sin matrices,
sin memoria dinamica, sin dependencias. Ocupa menos de 2 KB de flash.

---

## Entrenamiento del modelo

### Requisitos

```
pip install scikit-learn pandas numpy micromlgen
```

### Ejecutar el pipeline

```bash
python proyecto_ml_completo.py
```

El script:
1. Carga `TARP.csv` (features: Soil Moisture, Air temperature (C), Air humidity (%))
2. Mapea Status: ON→1, OFF→0
3. Entrena un Decision Tree con max_depth=4
4. Imprime Accuracy, Precision, Recall, F1
5. Exporta el modelo a `modelo_edge.h`

---

## API REST (app.py)

### Instalacion y arranque

```bash
pip install flask flask-cors requests
export FIREBASE_URL=https://tu-proyecto-default-rtdb.firebaseio.com
export FLASK_DEBUG=false
python app.py
```

### Endpoints

#### `POST /api/iot/telemetry`
El ESP32 reporta cada lectura y la decision que tomo localmente.

```json
// Request
{
    "soil_moisture":   45.2,
    "air_temperature": 22.1,
    "air_humidity":    60.5,
    "valve_decision":  1,
    "node_id":         "parque_central_01"
}

// Response 201
{ "ok": true, "firebase_key": "-NxxxxxYYYY" }
```

#### `POST /api/iot/override`
Envia una orden de control manual forzado. El ESP32 hace polling de
`/control/override` en Firebase y obedece el comando.

```json
// Request
{ "command": 1, "duration": 120 }

// Response 200
{ "ok": true, "command": 1, "duration": 120, "state": "ON" }
```

| command | Efecto |
|---------|--------|
| `1`     | Abrir valvula (forzar riego) |
| `0`     | Cerrar valvula (detener riego) |

#### `GET /api/iot/status`
Retorna el ultimo estado del nodo y el override activo.

```json
{
    "ok": true,
    "latest_telemetry": { ... },
    "active_override":  { ... }
}
```

#### `GET /health`
Verificacion de disponibilidad del servidor.

---

## Dashboard Streamlit

El panel municipal lee datos desde Firebase y muestra:

- **Humedad de suelo** actual (con indicador de nivel)
- **Temperatura del aire**
- **Humedad relativa del aire**
- **Estado de la valvula** (banner verde/rojo)
- **Historial de las ultimas lecturas** con tabla coloreada
- **Boton "FORZAR RIEGO MANUAL"** — envia override via `/api/iot/override`
- **Boton "Detener Riego"** — envia comando 0

### Arranque

```bash
pip install streamlit requests pandas
export API_BASE_URL=http://localhost:5000
export FIREBASE_URL=https://tu-proyecto-default-rtdb.firebaseio.com
streamlit run dashboard_streamlit.py
```

El dashboard se auto-refresca cada 15 segundos.

---

## Flujo completo de datos

```
1. ESP32 lee sensores cada N segundos
2. Llama a predict(sm, temp, hum) — decision local, sin red
3. Activa/desactiva la valvula
4. Si hay WiFi: POST /api/iot/telemetry  →  Flask  →  Firebase
5. ESP32 consulta Firebase control/override periodicamente
6. Si hay override activo: lo ejecuta (abre o cierra valvula)
7. Dashboard municipal lee Firebase y muestra estado en tiempo real
8. Operador puede pulsar "Forzar Riego Manual" en el dashboard
```

---

## Variables de entorno

| Variable       | Descripcion                         | Ejemplo |
|----------------|-------------------------------------|---------|
| `FIREBASE_URL` | URL base de Firebase Realtime DB    | `https://xxx-rtdb.firebaseio.com` |
| `API_BASE_URL` | URL del servidor Flask (dashboard)  | `http://localhost:5000` |
| `PORT`         | Puerto del servidor Flask           | `5000` |
| `FLASK_DEBUG`  | Modo debug Flask (`true`/`false`)   | `false` |

---

## Dataset TARP.csv

El modelo fue entrenado con el dataset TARP (Tree Autonomous Riego Predictor).
Features utilizadas (de las 15 columnas disponibles):

| Feature              | Descripcion                  | Rango tipico |
|----------------------|------------------------------|--------------|
| `Soil Moisture`      | Humedad del suelo            | 0-100 %      |
| `Air temperature (C)`| Temperatura del aire         | 10-45 °C     |
| `Air humidity (%)`   | Humedad relativa del aire    | 20-100 %     |

Target: `Status` — `ON` (regar) o `OFF` (no regar).

Se usan solo estas 3 features porque son exactamente las que mide el nodo
IoT con los sensores SEN0193 (suelo) y DHT22 (aire).

---

## Dependencias principales

```txt
flask
flask-cors
requests
scikit-learn
pandas
numpy
streamlit
micromlgen          # pip install micromlgen  (opcional, para export alternativo)
```
