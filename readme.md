# Sistema Inteligente de Riego — Edge AI (TinyML)

**Universidad Nacional Mayor de San Marcos — FISI**
Practicas Pre-Profesionales · Lima, Peru

---

## Concepto central: el modelo vive en el ESP32

El nucleo del sistema es un **Decision Tree Classifier** (max_depth=8) entrenado
con datos reales del dataset TARP (100,000 muestras) y exportado como **codigo C
puro** al archivo `modelo_edge.h`. Ese archivo se sube directamente al ESP32,
que ejecuta la inferencia de forma completamente offline.

```
Sensores (DHT22 + SEN0193 + RTC)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   ESP32  (Edge AI)                      │
│                                                         │
│  float sm   = leerHumedadSuelo();   // SEN0193          │
│  float temp = dht.readTemperature();// DHT22            │
│  float t    = horaDelDia();         // RTC / millis()   │
│                                                         │
│  int decision = predict(sm, temp, t); // modelo_edge.h  │
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

## Metricas del modelo

| Metrica | Valor |
|---------|-------|
| **Accuracy** | **91.08 %** |
| Precision OFF | 0.907 |
| Recall OFF | 0.900 |
| F1-score OFF | 0.903 |
| Precision ON | 0.914 |
| Recall ON | 0.920 |
| F1-score ON | 0.917 |
| Nodos del arbol | **147** |
| Muestras de entrenamiento | 70,000 |
| Muestras de test | 30,000 |

### Por que Decision Tree y no Gradient Boosting o Random Forest

Durante el desarrollo se compararon seis configuraciones de modelo:

| Modelo | Accuracy | Flash ESP32 | Exportacion a C |
|--------|----------|-------------|-----------------|
| DT max_depth=4 | 87.19 % | ~2 KB | Manual (sin deps) |
| **DT max_depth=8** | **91.08 %** | **~8 KB** | **Manual (sin deps)** |
| RF 100 arboles depth=8 | 91.02 % | ~500 KB | Requiere micromlgen |
| RF 200 arboles depth=10 | 91.09 % | ~1 MB | Requiere micromlgen |
| GB 100 est. depth=4 | 91.07 % | ~300 KB | Requiere micromlgen |
| GB 200 est. depth=5 | 91.49 % | ~800 KB | Requiere micromlgen |

**Decision:** el GB ganador supera al DT(8) en solo 0.4 pp pero requeriria cientos
de KB de flash adicional y una libreria externa para la exportacion. Para un
sistema de riego de parque urbano, ese margen no justifica el coste en recursos.
El DT(8) con 91% de accuracy es la opcion pragmatica: mas ligero, exportable en
C puro y completamente funcional en un ESP32 de bajo costo.

### Features utilizadas

| Feature | Correlacion con Status | Sensor en ESP32 |
|---------|----------------------|-----------------|
| `Soil Moisture` | r = -0.32 | SEN0193 |
| `Temperature` | r = +0.30 | DHT22 |
| `Time` | r = -0.26 | RTC DS3231 / millis() |

Estas tres features son los predictores con mayor correlacion real en el
dataset TARP. Features como `Air temperature (C)` o `Air humidity (%)` tienen
correlacion casi nula (r < 0.01) y se descartaron.

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

El archivo generado contiene una funcion `predict()` en C++ puro con 17 nodos
`if/else` anidados. Sin matrices, sin memoria dinamica, sin librerias externas:

```cpp
#include "modelo_edge.h"

void loop() {
    float sm   = leerHumedadSuelo();        // 0-100 %
    float temp = dht.readTemperature();     // grados C
    float t    = (float)hora_actual;        // hora del dia (0-23) o millis()/1000

    int decision = predict(sm, temp, t);
    // decision == 1 → abrir valvula (regar)
    // decision == 0 → cerrar valvula

    digitalWrite(VALVE_PIN, decision);

    // Telemetria opcional — no bloquea el riego si no hay WiFi
    if (WiFi.isConnected()) {
        enviarTelemetria(sm, temp, t, decision);
    }
}
```

---

## Entrenamiento del modelo

### Requisitos

```bash
pip install scikit-learn pandas numpy
# Opcional para exportacion alternativa:
pip install micromlgen
```

### Ejecutar el pipeline

```bash
python proyecto_ml_completo.py
```

El script:
1. Carga `TARP.csv` (features: Soil Moisture, Temperature, Time)
2. Mapea Status: ON → 1, OFF → 0
3. Split 70/30 estratificado
4. Entrena Decision Tree con max_depth=8
5. Imprime Accuracy, Precision, Recall, F1 y matriz de confusion
6. Exporta el modelo a `modelo_edge.h`

---

## API REST (app.py)

### Instalacion y arranque

```bash
pip install flask flask-cors requests
export FIREBASE_URL=https://tu-proyecto-default-rtdb.firebaseio.com
python app.py
```

### Endpoints

#### `POST /api/iot/telemetry`
El ESP32 reporta cada lectura y la decision que ya tomo localmente.

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
| `1` | Abrir valvula (forzar riego) |
| `0` | Cerrar valvula (detener riego) |

#### `GET /api/iot/status`
Retorna el ultimo estado del nodo y el override activo.

#### `GET /health`
Verificacion de disponibilidad del servidor.

---

## Dashboard Streamlit

Panel municipal mobile-friendly que lee Firebase y muestra:

- **Humedad de suelo** actual con indicador de nivel
- **Temperatura** del ambiente
- **Estado de la valvula** (banner verde = ON / rojo = OFF)
- **Historial** de las ultimas 15 lecturas con tabla coloreada
- **Boton "FORZAR RIEGO MANUAL"** — envia override via `/api/iot/override`
- **Boton "Detener Riego"** — envia comando 0
- Auto-refresh cada 15 segundos

### Arranque

```bash
pip install streamlit requests pandas
export API_BASE_URL=http://localhost:5000
export FIREBASE_URL=https://tu-proyecto-default-rtdb.firebaseio.com
streamlit run dashboard_streamlit.py
```

---

## Flujo completo de datos

```
1. ESP32 lee sensores cada N segundos
2. Llama a predict(sm, temp, time) — decision local, sin red
3. Activa o desactiva la valvula fisicamente
4. Si hay WiFi: POST /api/iot/telemetry → Flask → Firebase
5. ESP32 consulta Firebase /control/override periodicamente
6. Si hay override activo: lo ejecuta (abre o cierra valvula)
7. Dashboard municipal lee Firebase y muestra estado en tiempo real
8. Operador pulsa "Forzar Riego Manual" en el dashboard si es necesario
```

---

## Variables de entorno

| Variable | Descripcion | Ejemplo |
|----------|-------------|---------|
| `FIREBASE_URL` | URL base de Firebase Realtime DB | `https://xxx-rtdb.firebaseio.com` |
| `API_BASE_URL` | URL del servidor Flask (para el dashboard) | `http://localhost:5000` |
| `PORT` | Puerto del servidor Flask | `5000` |
| `FLASK_DEBUG` | Modo debug (`true`/`false`) | `false` |

---

## Dataset TARP.csv

100,000 muestras con 14 features de sensores agricolas y campo objetivo `Status` (ON/OFF).
De las 14 features disponibles, el modelo usa las 3 con mayor poder predictivo:

| Feature | Descripcion | Rango tipico |
|---------|-------------|--------------|
| `Soil Moisture` | Humedad del suelo | 0-100 % |
| `Temperature` | Temperatura ambiente | 10-45 °C |
| `Time` | Ciclo temporal / hora del dia | 0-143 |

---

## Dependencias

```
flask
flask-cors
requests
scikit-learn
pandas
numpy
streamlit
micromlgen   # opcional — pip install micromlgen
```
