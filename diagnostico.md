# Diagnostico: Proyecto vs. Documento de Investigacion

## Resumen Ejecutivo

El proyecto tiene una **base funcional solida** — el modelo ML principal cumple exactamente las metricas del documento — pero presenta **brechas criticas** en varios componentes clave, especialmente en la capa de hardware, la especificidad del cultivo objetivo y la implementacion matematica del sistema experto.

---

## Características IMPLEMENTADAS correctamente

### 1. Modelo de Gradient Boosting (nucleo del sistema)

El modelo entrenado en `modelos_guardados1/` reporta exactamente las metricas que cita el documento:

- **Accuracy: 98.76%** — `model_comparison_results.csv` linea 2: `0.9875735...`
- **F1-Score: 0.9849** — `model_comparison_results.csv`: `0.9848726...`
- **ROC-AUC: 0.9992** — cumple y supera lo requerido

### 2. Ingenieria de caracteristicas (`proyecto_ml_completo.py:127-132`)

Implementada correctamente:

- `temp_humidity_ratio`, `moi_temp_interaction`
- `temp_squared`, `humidity_squared`, `moi_squared`
- Variables categoricas: crop, soil, seedling stage (Label Encoding)

### 3. Plataforma Web Flask (`app.py`)

- API REST funcional con rutas para prediccion individual, masiva y dashboard
- Integracion con **Firebase Realtime Database** para lectura de datos IoT y persistencia de predicciones (`guardar_predicciones.py`)
- Monitoreo en tiempo real con hilo de polling (`monitor_firebase()`)

### 4. Sistema Experto (`expert_system.py`)

Implementado como sistema de puntuacion ponderada con 15 reglas contextuales. Usa funciones de pertenencia triangular/trapezoidal definidas. Pesos alineados con el documento (MOI: 45%, Temperatura: 22%, Humedad: 15%, Cultivo: 8%, Suelo: 6%, Etapa: 4%).

### 5. Agentes Inteligentes (`agentes_inteligentes.py`)

Dos agentes funcionales **adicionales** no descritos en el documento original:

- `AgenteRecomendaciones`: score de salud, recomendaciones de fertilizacion/plagas
- `AgenteOptimizacion`: calendario de riego semanal, metricas de eficiencia
- Integracion con **Gemini AI** para enriquecimiento contextual (caracteristica extra)

### 6. Prediccion masiva por Excel

`/api/predict-batch` acepta archivos `.xlsx` y devuelve predicciones en lote.

---

## Características FALTANTES o INCORRECTAS

### CRITICO 1 — El cultivo objetivo (arandano) no existe en el sistema

El documento describe **exclusivamente** el arandano alto (*Vaccinium corymbosum*) con parametros especificos:

- Humedad de suelo: 60–80% de la capacidad de campo
- Potencial matrico: −20 a −30 kPa
- Sensibilidad extrema a anegamiento y desecacion

**En el codigo:** La base de conocimiento de `agentes_inteligentes.py:52-113` solo tiene Wheat, Rice, Maize, Sugarcane, Cotton. El `expert_system.py:33-46` mapea cultivos genericos. **El arandano no esta configurado en ningun modulo.**

---

### CRITICO 2 — El sistema experto NO es logica difusa Mamdani

El documento (seccion 4.3 y OE4) describe explicitamente:

> *"sistema experto con logica difusa Mamdani como modulo de validacion cruzada"*

**En el codigo (`expert_system.py`):** Se definen funciones `triangular()` y `trapezoidal()` (lineas 11-28) pero **no se usan** en el flujo principal. La funcion `evaluate_expert_system()` implementa un sistema de **puntuacion ponderada lineal** con reglas SI-ENTONCES, no un motor de inferencia Mamdani genuino. Un sistema Mamdani requiere:

1. Fuzzificacion de todas las entradas con conjuntos difusos
2. Evaluacion de reglas con operadores min/max
3. Agregacion de consecuentes difusos
4. Defuzzificacion (metodo del centroide)

Nada de esto esta implementado.

---

### CRITICO 3 — Capa Edge AI (ESP32/TinyML) completamente ausente

El documento dedica una seccion entera (10.3, OE3, H3) a la arquitectura Edge AI:

> *"latencia total inferior a 500 ms, independientemente de la disponibilidad de conectividad a internet"*

**En el codigo:** No existe ningun archivo `.ino`, `.c`, ni codigo de firmware para ESP32. No hay serializacion del modelo para TinyML (emlearn, micromlgen, etc.). El sistema actual es 100% dependiente de un servidor web Python — exactamente lo opuesto al paradigma Edge AI descrito.

---

### CRITICO 4 — Protocolo ESP-NOW ausente

El documento (seccion 10.4) describe comunicacion peer-to-peer entre nodos sensores sin router. **No existe codigo de firmware ni descripcion de nodos en el repositorio.** Los datos de temperatura/humedad llegan por Firebase desde una fuente externa no documentada en el codigo.

---

### IMPORTANTE 5 — MOI no se lee de sensor real

En `app.py:35-51`, la funcion `moi_fuzzy_logic(temperatura, humedad)` **calcula el MOI** a partir de temperatura y humedad ambiental con una heuristica simple. El documento requiere que el MOI se mida con el **sensor SEN0193 v1.2** (el instrumento de mayor peso predictivo, 45%). En la plataforma actual, el MOI nunca proviene de un sensor de suelo real.

---

### IMPORTANTE 6 — Random Forest no esta en el pipeline de entrenamiento

El documento presenta Random Forest como algoritmo principal junto a Gradient Boosting. En `proyecto_ml_completo.py:213-220`, el diccionario de modelos incluye: LogisticRegression, GradientBoosting, SVM, KNN, NaiveBayes, AdaBoost — **Random Forest no esta incluido**.

---

### IMPORTANTE 7 — ETc (Evapotranspiracion del cultivo) no se calcula

El documento menciona varias veces el "calculo dinamico de la Evapotranspiracion (ETc)" como variable derivada usada en el modelo. **No existe ninguna funcion de calculo de ETc en el codigo** (requeriria la formula de Penman-Monteith o similar).

---

### MENOR 8 — Particion del dataset no coincide

| | Documento | Codigo |
|---|---|---|
| Entrenamiento | 70% (10,701 registros) | 80% |
| Prueba | 30% (4,587 registros) | 20% |
| Estratificacion | Por etapa fenologica | Solo por target (`stratify=y`) |

---

### MENOR 9 — Etapas fenologicas del arandano

El documento define: Germinacion, Desarrollo vegetativo, Floracion, Fructificacion. El codigo usa etapas genericas de otros cultivos (Germination, Vegetative Growth, Flowering, Harvest/Ripening). "Fructificacion" como etapa especifica del arandano no esta mapeada.

---

### MENOR 10 — Control de electrovalvulas ausente

No hay modulo de control del modulo de rele para activar/desactivar las electrovalvulas (seccion 10.5 del documento). La plataforma web muestra la prediccion, pero no actua sobre el hardware.

---

## Tabla resumen

| Componente | Estado |
|---|---|
| Modelo Gradient Boosting (F1=0.9849) | Implementado |
| Ingenieria de features (cuadraticos + interacciones) | Implementado |
| Flask API REST + Dashboard | Implementado |
| Firebase Realtime Database | Implementado |
| Prediccion individual y masiva | Implementado |
| Monitoreo en tiempo real | Implementado |
| Gemini AI (extra, no requerido) | Implementado |
| **Arandano como cultivo objetivo** | Ausente |
| **Sistema experto Mamdani genuino** | No implementado (sistema de scoring, no Mamdani) |
| **Edge AI en ESP32 / TinyML** | Completamente ausente |
| **Protocolo ESP-NOW** | Completamente ausente |
| **Sensor SEN0193 (MOI real)** | Ausente (MOI calculado por heuristica) |
| **Random Forest en entrenamiento** | No incluido |
| **Calculo de ETc** | Ausente |
| **Control de electrovalvulas** | Ausente |
| Particion 70/30 estratificada | Parcial (es 80/20, estratificacion solo por target) |

---

## Conclusion

El proyecto cumple con la **capa de software (ML + web)** del sistema descrito en el documento, y las metricas del modelo principal son exactamente las reportadas. Sin embargo, lo que el documento describe como su aporte mas novedoso — la **arquitectura Edge AI sobre ESP32 con ESP-NOW** y el **sistema experto Mamdani** — no esta implementado. Ademas, el sistema nunca fue adaptado al cultivo objetivo real (arandano), usando en cambio cultivos genericos del dataset original.
